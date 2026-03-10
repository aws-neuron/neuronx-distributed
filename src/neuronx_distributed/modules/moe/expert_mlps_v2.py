import math
from typing import Optional, Any, Dict, Tuple, List

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.modules.moe.model_utils import (
    ACT2FN,
    GLUType,
    ACTFunc,
    DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    get_kernel_activation_func_id,
)
from neuronx_distributed.modules.moe.blockwise import (
    BlockwiseMatmulNKIFunc,
    can_use_blockwise_matmul_nki,
    TorchBlockwiseTraining,
    ExpertAffinityScaleMode,
    SkipMode
)
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig, BlockwiseMatmulConfig
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers import parallel_state, comm
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.utils.tensor_utils import cumsum
from neuronx_distributed.parallel_layers.parallel_state import (
    get_expert_model_parallel_group,
    get_tensor_model_parallel_group,
    rmsg,
)
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.quantization.quantization_config import QuantizationType
from neuronx_distributed.modules.moe.blockwise import augment_inputs_for_padded_blockwise_matmul
from neuronx_distributed.modules.moe.moe_process_group import (
    get_moe_tp_ep_group,
    get_moe_ep_group,
)
import neuronxcc.nki.language as nl
from neuronx_distributed.kernels.find_nonzero_indices import find_nonzero_indices
from neuronx_distributed.kernels.indexed_flatten import indexed_flatten

logger = get_logger()

class ExpertMLPsV2(torch.nn.Module):
    """Class which obtains the output from passing the token hidden states through the assigned expert(s).

    Arguments:
        routed_experts_mlp_config: routed expert configs. Details are in neuronx_distributed.modules.moe.model_utils
        blockwise_matmul_config: blockwise matmul configs. Details are in neuronx_distributed.modules.moe.model_utils
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        dtype: Datatype for the layer weights.
        sequence_parallel_enabled: Whether sequence parallel is enabled.
        device: Device for the layer weights.
        tkg_tensor_model_parallel_group: when hybrid sharding is enabled, this will be the tensor_model_parallel_group for decode
        tkg_expert_model_parallel_group: when hybrid sharding is enabled, this will be the expert_model_parallel_group for decode
        cte_tensor_model_parallel_group: when hybrid sharding is enabled, this will be the tensor_model_parallel_group for prefill
        cte_expert_model_parallel_group: when hybrid sharding is enabled, this will be the expert_model_parallel_group for prefill
        enabled_hybrid_sharding: flag to enable hybrid sharding feature. In hybrid sharding expert_mlps will have different sharding strategy used for prefill and decode
    """
    def __init__(
        self,
        routed_experts_mlp_config: RoutedExpertsMLPOpsConfig,
        blockwise_matmul_config: BlockwiseMatmulConfig = BlockwiseMatmulConfig.default(),
        return_bias: bool = False,
        dtype: torch.dtype = torch.float32,
        sequence_parallel_enabled: bool = False,
        device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        expert_model_parallel_group: Optional[ProcessGroup] = None,
        # The below 4 model_parallel_group will be needed for hybrid sharding
        # For TKG we need 1 set of process group (tensor_model_parallel_group and expert_model_parallel_group)
        # when sharding and doing collectives
        # For CTE, we will need another set of process group for the same purpose
        tkg_tensor_model_parallel_group: Optional[ProcessGroup] = None,
        tkg_expert_model_parallel_group: Optional[ProcessGroup] = None,
        cte_tensor_model_parallel_group: Optional[ProcessGroup] = None,
        cte_expert_model_parallel_group: Optional[ProcessGroup] = None,
        is_prefill = True,
        enabled_hybrid_sharding=False,
        # spmd_rank will be removed once we support ReplicaID (P87857655)
    ):
        super().__init__()
        self.routed_experts_mlp_config = routed_experts_mlp_config
        self.blockwise_matmul_config = blockwise_matmul_config
        self.sequence_parallel_enabled = sequence_parallel_enabled
        skip_dma = SkipMode(blockwise_matmul_config.skip_dma_token,blockwise_matmul_config.skip_dma_weight)
        setattr(self.blockwise_matmul_config,"skip_dma",skip_dma)
        self.validate_routed_experts_configs(routed_experts_mlp_config)
        self.enabled_hybrid_sharding = enabled_hybrid_sharding

        if return_bias:
            raise NotImplementedError("Returning bias is currently unsupported for MoE")
        self.return_bias = return_bias

        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else get_tensor_model_parallel_group()
        self.expert_model_parallel_group = expert_model_parallel_group if \
            expert_model_parallel_group is not None else get_expert_model_parallel_group()

        if enabled_hybrid_sharding:
            assert tkg_tensor_model_parallel_group is not None
            assert tkg_expert_model_parallel_group is not None
            assert cte_tensor_model_parallel_group is not None
            assert cte_expert_model_parallel_group is not None
        else:
            # This is added for old modeling code that does not pass in
            # cte_tensor_model_parallel_group and cte_expert_model_parallel_group
            cte_tensor_model_parallel_group = self.tensor_parallel_group if cte_tensor_model_parallel_group is None else cte_tensor_model_parallel_group
            cte_expert_model_parallel_group = self.expert_model_parallel_group if cte_expert_model_parallel_group is None else cte_expert_model_parallel_group
            tkg_tensor_model_parallel_group = self.tensor_parallel_group if tkg_tensor_model_parallel_group is None else tkg_tensor_model_parallel_group
            tkg_expert_model_parallel_group = self.expert_model_parallel_group if tkg_expert_model_parallel_group is None else tkg_expert_model_parallel_group

        self.dtype = dtype
        self.device = device
        self.is_prefill = is_prefill
        self.enabled_hybrid_sharding = enabled_hybrid_sharding
        # In current state, selective loading is not in EP.
        if routed_experts_mlp_config.enable_spmd_rank:
            # spmd_rank will be removed once we support ReplicaID (P87857655)
            self.spmd_rank = SPMDRank(
                world_size=parallel_state.get_world_group().size(),
                tensor_model_parallel_size=cte_tensor_model_parallel_group.size(),
            )
            if enabled_hybrid_sharding:
                self.spmd_rank_tkg = SPMDRank(
                world_size=parallel_state.get_world_group().size(),
                tensor_model_parallel_size=tkg_tensor_model_parallel_group.size(),
                )

        if cte_expert_model_parallel_group.size() > 1:
            logger.warning(f"enable_spmd_rank set to {self.routed_experts_mlp_config.enable_spmd_rank}, enable_spmd_rank must be set to True when using expert parallelism under SPMD flow")

        self.mlp_op = self.initialize_mlp_op(cte_tensor_model_parallel_group, cte_expert_model_parallel_group, True)
        if cte_expert_model_parallel_group.size() > 1:
            self.spmd_rank.initialize_expert_indices(num_local_experts=self.mlp_op.gate_up_proj._n_local_experts)

        if enabled_hybrid_sharding:
            self.mlp_op_tkg = self.initialize_mlp_op(tkg_tensor_model_parallel_group, tkg_expert_model_parallel_group, False)
            if tkg_expert_model_parallel_group.size() > 1:
                self.spmd_rank_tkg.initialize_expert_indices(num_local_experts=self.mlp_op_tkg.gate_up_proj._n_local_experts)

        self.moe_tensor_model_parallel_group = cte_tensor_model_parallel_group if is_prefill else tkg_tensor_model_parallel_group
        self.moe_expert_model_parallel_group = cte_expert_model_parallel_group if is_prefill else tkg_expert_model_parallel_group

    def initialize_mlp_op(self, tensor_model_parallel_group: ProcessGroup, expert_model_parallel_group: ProcessGroup, is_prefill: bool):
        # increase clamping limit by hidden_act_bias because normally clamping is done before hidden act bias is added
        if self.routed_experts_mlp_config.up_clamp_upper_limit:
            self.routed_experts_mlp_config.up_clamp_upper_limit += self.routed_experts_mlp_config.hidden_act_bias
        if self.routed_experts_mlp_config.up_clamp_lower_limit:
            self.routed_experts_mlp_config.up_clamp_lower_limit += self.routed_experts_mlp_config.hidden_act_bias
        expert_distribution = self.routed_experts_mlp_config.expert_distribution if is_prefill else None
        mlp_op = Experts(
            num_experts=self.routed_experts_mlp_config.num_experts,
            hidden_size=self.routed_experts_mlp_config.hidden_size,
            intermediate_size=self.routed_experts_mlp_config.intermediate_size,
            glu=self.routed_experts_mlp_config.glu_mlp,
            activation_fn=ACT2FN[self.routed_experts_mlp_config.hidden_act],
            dtype=self.dtype,
            device=self.device,
            bias=self.routed_experts_mlp_config.bias,
            glu_type=self.routed_experts_mlp_config.glu_type,
            hidden_act_scaling_factor=self.routed_experts_mlp_config.hidden_act_scaling_factor,
            gate_clamp_upper_limit=self.routed_experts_mlp_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=self.routed_experts_mlp_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=self.routed_experts_mlp_config.up_clamp_upper_limit,
            up_clamp_lower_limit=self.routed_experts_mlp_config.up_clamp_lower_limit,
            input_layer_init_method=self.routed_experts_mlp_config.input_layer_init_method,
            output_layer_init_method=self.routed_experts_mlp_config.output_layer_init_method,
            tensor_model_parallel_group=tensor_model_parallel_group,
            expert_model_parallel_group=expert_model_parallel_group,
            is_prefill=is_prefill,
            expert_distribution=expert_distribution,
        )
        return mlp_op

    def get_mlp_op(self):
        """
        This function selects the correct weight for mlp_op when hybrid_sharding is enabled
        """
        if self.is_prefill or not self.enabled_hybrid_sharding:
            return self.mlp_op
        else:
            return self.mlp_op_tkg

    def get_spmd_rank(self):
        """
        This function selects the correct weight for spmd_rank when hybrid_sharding is enabled
        """
        if self.is_prefill or not self.enabled_hybrid_sharding:
            return self.spmd_rank
        else:
            return self.spmd_rank_tkg

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        prefix = prefix.removesuffix("weight")
        create_spmd_ranks(
            model_state_dict=model_state_dict,
            prefix=prefix,
            world_size=parallel_state.get_world_group().size(),
            n_routed_experts=self.routed_experts_mlp_config.num_experts,
            expert_model_parallel_group=self.moe_expert_model_parallel_group,
            spmd_rank_name="spmd_rank",
            expert_distribution=self.routed_experts_mlp_config.expert_distribution
        )

        if self.routed_experts_mlp_config.bias:
            # currently kernel only supports hidden_act_bias=0, so we preprocess the checkpoint to
            # make sure that flat compiler and kernel code paths are the same by adding hidden_act_bias
            # to the up_proj portion (second half) of the concatenated bias
            gate_up_proj_biases = model_state_dict[f"{prefix}mlp_op.gate_up_proj.bias"].clone()  # clone to avoid modifying in place for tensors that have grad
            intermediate_size = gate_up_proj_biases.shape[-1] // 2
            if self.dtype == torch.float32: 
                # Fix for preshard hook added +1 to bias in BF16 which had 
                # precision issues compared to +1 in code which in CPU tests is FP32
                gate_up_proj_biases = gate_up_proj_biases.to(torch.float32)

            # Skip up_bias += hidden_act_bias when I dim is shuffled, as this preprocessing has already happened upstream pre-shuffle.
            if not self.routed_experts_mlp_config.is_intermediate_dim_shuffled:
                if self.routed_experts_mlp_config.intermediate_size_actual:
                    gate_up_proj_biases[:, intermediate_size:intermediate_size + self.routed_experts_mlp_config.intermediate_size_actual] += self.routed_experts_mlp_config.hidden_act_bias
                else:
                    gate_up_proj_biases[:, intermediate_size:] += self.routed_experts_mlp_config.hidden_act_bias

            model_state_dict[f"{prefix}mlp_op.gate_up_proj.bias"] = gate_up_proj_biases
            # RPL bias needs to be divided by tp_degree because we do all-reduce at the end of MoE, we
            # divide bias by TP here to avoid needing to separately add down_proj bias after all-reduce
            model_state_dict[f"{prefix}mlp_op.down_proj.bias"] = model_state_dict[f"{prefix}mlp_op.down_proj.bias"] / self.moe_tensor_model_parallel_group.size()

        # In hybrid sharding new keys will be created for gate_up_proj and down_proj for different sharding strategy
        if self.enabled_hybrid_sharding:
            old_prefix_down_proj_weight = f"{prefix}mlp_op.down_proj.weight"
            new_prefix_down_proj_weight = f"{prefix}mlp_op_tkg.down_proj.weight"
            old_prefix_gate_up_proj_weight = f"{prefix}mlp_op.gate_up_proj.weight"
            new_prefix_gate_up_proj_weight = f"{prefix}mlp_op_tkg.gate_up_proj.weight"
            duplicate_and_replace_prefixes(old_prefix_down_proj_weight, new_prefix_down_proj_weight, model_state_dict)
            duplicate_and_replace_prefixes(old_prefix_gate_up_proj_weight, new_prefix_gate_up_proj_weight, model_state_dict)

            # create another spmd_rank for decode
            create_spmd_ranks(
                model_state_dict=model_state_dict,
                prefix=prefix,
                world_size=parallel_state.get_world_group().size(),
                n_routed_experts=self.routed_experts_mlp_config.num_experts,
                expert_model_parallel_group=get_moe_ep_group(prefill=False),
                spmd_rank_name="spmd_rank_tkg",
            )
            if self.routed_experts_mlp_config.bias:
                old_prefix_down_proj_bias = f"{prefix}mlp_op.down_proj.bias"
                new_prefix_down_proj_bias = f"{prefix}mlp_op_tkg.down_proj.bias"
                old_prefix_gate_up_proj_bias = f"{prefix}mlp_op.gate_up_proj.bias"
                new_prefix_gate_up_proj_bias = f"{prefix}mlp_op_tkg.gate_up_proj.bias"
                duplicate_and_replace_prefixes(old_prefix_down_proj_bias, new_prefix_down_proj_bias, model_state_dict)
                duplicate_and_replace_prefixes(old_prefix_gate_up_proj_bias, new_prefix_gate_up_proj_bias, model_state_dict)
                # divide down_proj bias by tp_degree for decode as well
                model_state_dict[f"{prefix}mlp_op_tkg.down_proj.bias"] = model_state_dict[f"{prefix}mlp_op_tkg.down_proj.bias"] / get_moe_tp_ep_group(prefill=False).size()

    @staticmethod
    def validate_routed_experts_configs(routed_experts_mlp_config: RoutedExpertsMLPOpsConfig):
        if not (0 < routed_experts_mlp_config.top_k <= routed_experts_mlp_config.num_experts):
            raise ValueError(f"Invalid top_k={routed_experts_mlp_config.top_k} for num_experts={routed_experts_mlp_config.num_experts}")
        if routed_experts_mlp_config.hidden_act not in ACT2FN:
            raise ValueError(f"Unknown activation: {routed_experts_mlp_config.hidden_act} ; Supported: {list(ACT2FN.keys())}")
        if (routed_experts_mlp_config.capacity_factor is None
                or routed_experts_mlp_config.capacity_factor >= routed_experts_mlp_config.num_experts / routed_experts_mlp_config.top_k):
            routed_experts_mlp_config.capacity_factor = None  # Denotes full capacity
        if routed_experts_mlp_config.normalize_top_k_affinities and routed_experts_mlp_config.top_k == 1:
            raise ValueError("top_k must be greater than 1 for normalizing top-k expert affinities")

    @staticmethod
    def get_expert_mask(expert_index, num_experts):
        """Helper function which computes top_k-hot encoded expert_mask from the given expert_index.

        Arguments:
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
        Returns:
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
        """

        # Construct expert_mask from expert_index, using efficient version of one-hot encoding for xla device
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # (Use float dtype to perform computations in the vector engine for efficiency)
        # expert_mask: top_k-hot encoded expert assignment per token -> (T, E)
        expert_mask = torch.zeros(
            expert_index.shape[0], num_experts, device=expert_index.device, dtype=torch.float64
        )
        expert_num_idx_arr = torch.arange(num_experts, device=expert_index.device, dtype=torch.float64)
        top_k = expert_index.shape[1]
        for e in range(top_k):
            expert_mask += (expert_index[:, e].unsqueeze(1) == expert_num_idx_arr).to(torch.float64)

        return expert_mask

    @staticmethod
    def get_expert_affinities_masked(expert_affinities, expert_mask, normalize_top_k_affinities):
        """Helper function which computes the masked expert_affinities by selecting the chosen experts for each token,
        and normalizes the affinities if needed.

        Arguments:
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
            normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
        Returns:
            expert_affinities_masked: Tensor of shape (T, E) containing the affinities of just the chosen experts for
                                      each token (after normalization if required).
        """

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (T, E)
        expert_affinities_masked = expert_affinities.masked_fill(torch.eq(expert_mask, 0), 0)
        if normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        return expert_affinities_masked

    def get_sp_expert_masks_index(self, expert_affinities_masked: torch.tensor, expert_index: torch.tensor):
            """Gather for Inference when sequence parallel enabled
            Gathers expert affinities mask and indices from sequence parallel region and computes expert mask.

            Args:
                expert_affinities_masked: Tensor of shape (T, E) with masked expert affinities.
                expert_index: Tensor of shape (T, top_k) with chosen expert indices.

            Returns:
                Tuple of (expert_affinities_masked, expert_mask, expert_index) after gathering from sequence parallel region.
            """

            expert_affinities_masked, expert_index = [
                    mappings.gather_from_sequence_parallel_region(
                        tensor,
                        sequence_dimension=0,
                        to_model_parallel=False,
                        process_group=self.tensor_parallel_group,
                    ) for tensor in (expert_affinities_masked, expert_index)
                ]
            expert_mask = (expert_affinities_masked > 0).to(torch.float64)
            return expert_affinities_masked, expert_mask, expert_index

    def mask_padding_tokens(self, expert_mask: torch.tensor, expert_affinities_masked: torch.tensor, padding_mask: torch.tensor) -> tuple[Optional[torch.tensor], torch.tensor]:
        """Masks padding tokens in expert assignments and affinities based on padding_mask.

        Args:
            expert_mask: Tensor of shape (B*S, E) with expert assignments, or None.
            expert_affinities_masked: Tensor of shape (B*S, E) with masked expert affinities.
            padding_mask: Tensor of shape (B, S) with mask for padded tokens, or None to skip masking.
        
        Returns:
            Tuple of (expert_mask, expert_affinities_masked) with padding tokens masked out.
        """
        if padding_mask is None:
            return expert_mask, expert_affinities_masked

        padding_mask = padding_mask.view(-1).unsqueeze(1)

        expert_affinities_masked = expert_affinities_masked * padding_mask
        if expert_mask is not None:
            expert_mask = expert_mask * padding_mask

        return expert_mask, expert_affinities_masked

    def forward_all_experts(self, hidden_states, expert_affinities, expert_index, chosen_expert_indices=None):
        """Forward pass where all tokens are computed by all experts.
        This is equivalent to running forward_capacity_factor with full capacity (i.e. no token dropping), but
        by avoiding the permute/unpermute overhead.
        """
        mlp_op = self.get_mlp_op()
        num_experts, expert_mask, expert_affinities_masked, mlp_input = self.setup_all_experts(hidden_states, expert_affinities, expert_index, chosen_expert_indices=chosen_expert_indices)

        # Pass all tokens through all experts
        # gate_up_proj: (1, T, H) @ (E, H, I) -> (E, T, I)
        # down_proj: (E, T, I) @ (E, I, H) -> (E, T, H)
        mlp_output = mlp_op(mlp_input.unsqueeze(0), expert_indices=chosen_expert_indices)

        output = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Scale by expert affinity and combine output
        if self.routed_experts_mlp_config.early_expert_affinity_modulation:
            for e in range(num_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * expert_mask[:, e].unsqueeze(1)
        else:
            for e in range(num_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * expert_affinities_masked[:, e].unsqueeze(1)

        return output

    def forward_all_experts_EP(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        expert_index: torch.Tensor,
        chosen_expert_indices: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Forward pass where all tokens are computed by all experts.
        This is equivalent to running forward_capacity_factor with full capacity (i.e. no token dropping), but
        by avoiding the permute/unpermute overhead.

        Args:
        hidden_states: Input tensor of shape (T, H) where T is batch_size * sequence_length and H is hidden dimension
        expert_affinities: Expert routing weights of shape (T, E) where E is number of experts
        expert_index: Selected expert indices of shape (T, TopK)
        chosen_expert_indices: Optional tensor for chosen expert specific compute

        """
        assert not (self.routed_experts_mlp_config.early_expert_affinity_modulation and self.routed_experts_mlp_config.top_k > 1), \
            "Early expert affinity modulation is not compatible with top_k > 1."
        mlp_op = self.get_mlp_op()
        spmd_rank = self.get_spmd_rank()
        _, expert_mask, expert_affinities_masked, mlp_input = self.setup_all_experts(hidden_states, expert_affinities, expert_index, chosen_expert_indices=chosen_expert_indices)
        T = hidden_states.shape[0]
        num_local_experts = mlp_op.gate_up_proj._n_local_experts
        # [T, E/ep_size]
        local_expert_indices = spmd_rank.get_local_expert_indices()
        broadcasted_local_expert_indices = torch.broadcast_to(local_expert_indices, (T, num_local_experts))
        local_expert_affinities_masked = torch.gather(expert_affinities_masked, 1, broadcasted_local_expert_indices)
        local_expert_mask = torch.gather(expert_mask, 1, broadcasted_local_expert_indices)

        # Pass all tokens through all experts
        # gate_up_proj: (1, T, H) @ (E/Ep, H, I) -> (E/Ep, T, I)
        # down_proj: (E/Ep, T, I) @ (E/Ep, I, H) -> (E/Ep, T, H)
        mlp_output = mlp_op(mlp_input.unsqueeze(0), expert_indices=chosen_expert_indices)

        output = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Scale by expert affinity and combine output
        if self.routed_experts_mlp_config.early_expert_affinity_modulation:
            for e in range(num_local_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * local_expert_mask[:, e].unsqueeze(1)
        else:
            for e in range(num_local_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * local_expert_affinities_masked[:, e].unsqueeze(1)

        return output

    def setup_all_experts(
        self,
        hidden_states: torch.Tensor,
        expert_affinities: torch.Tensor,
        expert_index: torch.Tensor,
        chosen_expert_indices: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        This is the common setup for forward_all_expert and forward_all_expert_EP, this function returns the global expert_mask,
        expert_affinity_mask and mlp_input
        """
        num_experts = expert_affinities.shape[1]
        if chosen_expert_indices is None:
            assert num_experts == self.routed_experts_mlp_config.num_experts
        else:
            assert num_experts == chosen_expert_indices.shape[0]

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index, num_experts)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities,
        )

        if self.sequence_parallel_enabled and not self.training:
            expert_affinities_masked, expert_mask, expert_index = self.get_sp_expert_masks_index(expert_affinities_masked, expert_index)

        if self.routed_experts_mlp_config.early_expert_affinity_modulation:
            mlp_input = torch.zeros_like(hidden_states)
            # dense_expert_affinities_masked: (T, 1)
            # This will only work for TopK = 1
            dense_expert_affinities_masked = torch.gather(expert_affinities_masked, 1, expert_index)
            mlp_input = hidden_states * dense_expert_affinities_masked
        else:
            mlp_input = hidden_states
        return num_experts, expert_mask, expert_affinities_masked, mlp_input

    def forward_capacity_factor(self, hidden_states, expert_affinities, expert_index):
        """Forward pass for performing Expert MLP computations, where each expert has a fixed 'expert capacity',
        i.e. maximum number of tokens that it can process. This is necessary for maintaining static shapes in the
        compilation graph, but may lead to dropped tokens in the computation.

        Expert capacity C is defined as:
            C = min(total_tokens, (total_tokens * top_k * capacity_factor) / num_experts)
        Note that when capacity_factor >= num_experts / top_k, C = total_tokens (i.e. each expert can hold all
        input tokens, and therefore no tokens are dropped).
        """
        mlp_op = self.get_mlp_op()
        total_tokens = hidden_states.shape[0]

        # compute expert capacity C = (total_tokens * top_k * Cf) / E
        expert_capacity = math.ceil(total_tokens * self.routed_experts_mlp_config.top_k * self.routed_experts_mlp_config.capacity_factor / self.routed_experts_mlp_config.num_experts)
        # expert_capacity can be upper bounded by total number of tokens, for the case when every token is routed to an expert
        expert_capacity = min(expert_capacity, total_tokens)

        if self.sequence_parallel_enabled and not self.training:
            expert_affinities, expert_index = [
                    mappings.gather_from_sequence_parallel_region(
                        tensor,
                        sequence_dimension=0,
                        to_model_parallel=False,
                        process_group=self.tensor_parallel_group,
                    ) for tensor in (expert_affinities, expert_index)
            ]

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)

        # Compute the position of each token in experts, by a cumulative sum over the T dimension
        # position_in_expert: (T, E)
        position_in_expert = cumsum(expert_mask)

        # Update expert_mask by accounting for capacity factor (i.e. tokens exceeding capacity are dropped)
        expert_mask.masked_fill_(torch.gt(position_in_expert, expert_capacity), 0)

        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities
        )

        # Add expert offset to the position_in_expert
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # expert_index_offsets: (E, )
        expert_index_offsets = (
            torch.arange(self.routed_experts_mlp_config.num_experts, device=hidden_states.device, dtype=torch.float64) * expert_capacity
        )
        # position_in_expert_with_offset: (T, E)
        position_in_expert_with_offset = position_in_expert + expert_index_offsets
        # Apply expert_mask
        position_in_expert_with_offset.masked_fill_(torch.eq(expert_mask, 0), 0)

        # Determine the index (with offset) of each token
        # total_tokens_idx: (T, 1)
        total_tokens_idx = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long).unsqueeze(1)
        # token_permutation_idx: (T, top_k)
        token_permutation_idx = position_in_expert_with_offset[total_tokens_idx, expert_index].to(dtype=torch.long)
        # token_permutation_idx is 1-indexed (contains 0 for dropped tokens)

        # token_assignments: (C*E+1, )
        # Account for the 1-indexed token_permutation_idx by adding an extra row
        token_assignments = torch.zeros(
            expert_capacity * self.routed_experts_mlp_config.num_experts + 1, device=hidden_states.device, dtype=torch.long
        )
        # Perform a broadcasted assignment to map to token_idx
        token_assignments[token_permutation_idx] = total_tokens_idx + 1
        # Drop the first row (which was added to account for the 1-indexed token_permutation_idx)
        token_assignments = token_assignments[1:]
        # token_assignments: (E, C)
        token_assignments = token_assignments.view(self.routed_experts_mlp_config.num_experts, expert_capacity)
        # token_assignments is 1-indexed (contains 0 for dropped tokens)

        # Convert token_permutation_idx and token_assignments to be 0-indexed
        token_permutation_idx = token_permutation_idx - 1
        token_assignments = token_assignments - 1
        # They now contain '-1' for dropped tokens, enforce non-negative indices to avoid potential runtime OOB
        zero_tensor = torch.zeros(1, device=hidden_states.device, dtype=torch.long)
        token_permutation_idx = torch.max(token_permutation_idx, zero_tensor)
        token_assignments = torch.max(token_assignments, zero_tensor)
        # Indexing using these will result in the first token (index 0) being loaded in place of dropped tokens
        # However, the output from these will get masked out in the affinity scaling step

        # Permute hidden_states using token_assignments to get expert_aligned_hidden_states
        # expert_aligned_hidden_states: (E, C, H)
        expert_aligned_hidden_states = hidden_states[token_assignments, :]

        # Perform MLP operations
        # expert_aligned_output: (E, C, H)
        expert_aligned_output = mlp_op(expert_aligned_hidden_states)

        # convert back (E, C, H) into (C*E, H)
        permuted_output = expert_aligned_output.view(expert_capacity * self.routed_experts_mlp_config.num_experts, -1)

        # output: (T, H)
        output = torch.zeros(
            total_tokens, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for k in range(self.routed_experts_mlp_config.top_k):
            # Unpermute output from the kth chosen expert for each token using token_permutation_idx
            output_k = permuted_output[token_permutation_idx[:, k]]
            expert_affinities_k = expert_affinities_masked[total_tokens_idx, expert_index[:, k].unsqueeze(1)]
            # Multiplying with the expert_affinities masks out the output of dropped tokens
            # (T, H) * (T, 1)
            output += output_k * expert_affinities_k

        return output

    def forward_selective_loading(self, hidden_states, expert_affinities, expert_index):
        """Forward pass which selectively loads only the experts chosen for each input token, during token generation."""
        mlp_op = self.get_mlp_op()
        T = hidden_states.shape[0]

        # chosen_expert_affinities: (T, top_k)
        chosen_expert_affinities = expert_affinities[
            torch.arange(T, device=hidden_states.device).unsqueeze(1), expert_index
        ]
        if self.routed_experts_mlp_config.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            chosen_expert_affinities = F.normalize(chosen_expert_affinities, p=1.0, dim=1)

        output_list = []
        for t in range(T):
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                weighted_hidden = hidden_states[t].unsqueeze(0) * chosen_expert_affinities[t].unsqueeze(1)
                mlp_output_t = mlp_op(weighted_hidden.unsqueeze(1), expert_indices=expert_index[t])
                output_t = torch.sum(mlp_output_t.squeeze(1), dim=0)
            else:
                # gate_up_proj: (1, 1, H) @ (top_k, H, I) -> (top_k, 1, I)
                # down_proj: (top_k, 1, I) @ (top_k, I, H) -> (top_k, 1, H)
                mlp_output_t = mlp_op(hidden_states[t].unsqueeze(0).unsqueeze(1), expert_indices=expert_index[t])
                # output_t: sum((top_k, H) * (top_k, 1), dim=0) -> H
                output_t = torch.sum(mlp_output_t.squeeze(1) * chosen_expert_affinities[t].unsqueeze(1), dim=0)
            output_list.append(output_t)

        # output: (T, H)
        output = torch.stack(output_list, dim=0)

        return output    

    def use_index_calc_kernel(self, total_tokens):
        if self.training:
            # TODO: enable index calc kernel for training
            return False
        if not self.is_prefill:
            # No index calculation needed for TKG
            return False
        if not self.routed_experts_mlp_config.use_index_calc_kernel:
            # The use_index_calc_kernel config can be used to turn the kernel off.
            return False
        if not self.routed_experts_mlp_config.enable_spmd_rank:
            return False

        mlp_op = self.get_mlp_op()
        assert mlp_op.gate_up_proj._n_local_experts == mlp_op.down_proj._n_local_experts
        local_experts = mlp_op.gate_up_proj._n_local_experts

        use_index_calc_kernel = can_use_find_index_kernel(
            T=total_tokens,
            block_size=self.blockwise_matmul_config.block_size,
            E_local=local_experts,
            logical_nc_config=self.blockwise_matmul_config.logical_nc_config,
            tp_size=self.moe_tensor_model_parallel_group.size(),
            ep_size=self.moe_expert_model_parallel_group.size(),
        )

        return use_index_calc_kernel

    def get_full_expert_affinities_masked(self, expert_affinities, expert_index):
        # expert_mask: (T/SP, E). Happens in SP on ALL the experts.
        expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities,
        )

        # Gather expert affinities from SP
        expert_affinities_masked = mappings.gather_from_sequence_parallel_region(
            expert_affinities_masked,
            sequence_dimension=0,
            to_model_parallel=False,
            process_group=self.tensor_parallel_group,
        )

        return expert_affinities_masked

    def maybe_get_expert_affinities_masked(self, expert_index, expert_affinities, expert_affinities_masked_full=None, padding_mask=None):
        if expert_affinities_masked_full is not None:
            # This means router is in SP.
            return expert_affinities_masked_full
        
        # Router is not in SP.
        # expert_mask: (T, E). Happens on all the experts, not just the local experts
        expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities,
        )
        return expert_affinities_masked

    def forward_blockwise(self, hidden_states, expert_affinities, expert_index, expert_affinities_masked_full=None, padding_mask=None):
        """
        Forward pass which implements the blockwise matmul approach for expert computations without
        dropping tokens.

        The tokens are assembled into fixed sized 'blocks', each of which is assigned to a single expert.
        The outputs across blocks are un-permuted and combined to obtain the output corresponding to
        each input token.

        The 'block size', i.e. the number of tokens in each block, is a hyper-parameter that must be chosen
        carefully. Larger block sizes result in better hardware utilization, but may also lead to large
        padding overheads (especially at smaller sequence lengths).
        """
        mlp_op = self.get_mlp_op()
        spmd_rank = self.get_spmd_rank() if self.routed_experts_mlp_config.enable_spmd_rank else None
        total_tokens, hidden_size = hidden_states.shape

        # num_blocks N = CEIL(((T*top_k)-(E-1))/ B) + (E-1)
        # N is the number of blocks needed in the worst case distribution of tokens to experts. Intuition below.
        # With top_k = 1, let E-1 experts be assigned 1 token each. They require 1 block each, and (E-1) total. The remaining
        # tokens are all assigned the same experts, and require CEIL((T-(E-1))/B) blocks. The formula holds for top_k > 1 also.
        if self.moe_expert_model_parallel_group.size() > 1:
            assert mlp_op.gate_up_proj._n_local_experts == mlp_op.down_proj._n_local_experts
            local_experts = mlp_op.gate_up_proj._n_local_experts
        else:
            local_experts = self.routed_experts_mlp_config.num_experts

        num_blocks = math.ceil((total_tokens * self.routed_experts_mlp_config.top_k - (local_experts - 1)) / self.blockwise_matmul_config.block_size) + local_experts - 1
        # Handle case where T*top_k is smaller than E. We will need atmost T*top_k blocks.
        num_blocks = min(num_blocks, total_tokens * self.routed_experts_mlp_config.top_k)
        # Padding num_blocks to even (TODO: currently only for MXFP4 BWMM kernel to support dynamic while)

        pad_num_blocks_to_even = self.blockwise_matmul_config.pad_num_blocks_to_even
        if pad_num_blocks_to_even:
            num_blocks += num_blocks % 2

        # Get num_static_block from blockwise_matmul_config, if not set, the default num_static_blocks will be computed as
        # NUM_STATIC_BLOCK = T * TopK / (EP_degree * B)
        # this estimation represent a perfect balance workload between workers.
        if self.blockwise_matmul_config.num_static_blocks is not None:
            num_static_blocks = self.blockwise_matmul_config.num_static_blocks
        else:
            num_static_blocks = math.ceil(math.ceil((total_tokens * self.routed_experts_mlp_config.top_k) / self.moe_expert_model_parallel_group.size() ) / self.blockwise_matmul_config.block_size)

        use_index_calc_kernel = self.use_index_calc_kernel(total_tokens)
        if use_index_calc_kernel:
            expert_affinities_masked = self.maybe_get_expert_affinities_masked(
                expert_index, expert_affinities, expert_affinities_masked_full, padding_mask
            )
            _, expert_affinities_masked = self.mask_padding_tokens(None, expert_affinities_masked, padding_mask)
            if self.moe_expert_model_parallel_group.size() > 1:
                # Slice to get EP-local expert affinities
                local_expert_indices = spmd_rank.get_local_expert_indices()
                broadcasted_local_expert_indices = torch.broadcast_to(local_expert_indices, (total_tokens, local_experts))
                local_expert_affinities_masked = torch.gather(expert_affinities_masked, 1, broadcasted_local_expert_indices)
            else:
                local_expert_affinities_masked = expert_affinities_masked

            # Enable kernel flow only for EP + inference (cte) cases.
            # Note that when the kernel is used, we don't need to materialize `expert_mask`, `local_expert_mask`, or 'expert_index'.
            # Pass expert_affinities_masked directly
            block_to_expert, token_position_to_id = self.get_blockwise_expert_and_token_mapping_kernel(
                num_blocks=num_blocks,
                expert_affinities_masked=expert_affinities_masked,
                block_size=self.blockwise_matmul_config.block_size,
                device=hidden_states.device,
                spmd_rank = spmd_rank,
                tensor_parallel_group=self.moe_tensor_model_parallel_group,
                expert_parallel_group=self.moe_expert_model_parallel_group,
                logical_nc_config=self.blockwise_matmul_config.logical_nc_config,
                pad_num_blocks_to_even=pad_num_blocks_to_even
            )
        else:
            # expert_mask: (T, E). Still happens on all the experts, not just the local experts
            expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)
            # expert_affinities_masked: (T, E)
            expert_affinities_masked = self.get_expert_affinities_masked(
                expert_affinities,
                expert_mask,
                self.routed_experts_mlp_config.normalize_top_k_affinities,
            )

            if self.sequence_parallel_enabled and not self.training:
                expert_affinities_masked, expert_mask, expert_index = self.get_sp_expert_masks_index(expert_affinities_masked, expert_index)

            expert_mask, expert_affinities_masked = self.mask_padding_tokens(expert_mask, expert_affinities_masked, padding_mask)

            if self.moe_expert_model_parallel_group.size() > 1:
                # [T, E/ep_size]
                local_expert_indices = spmd_rank.get_local_expert_indices()
                broadcasted_local_expert_indices = torch.broadcast_to(local_expert_indices, (total_tokens, local_experts))

                local_expert_mask = torch.gather(expert_mask, 1, broadcasted_local_expert_indices)
                local_expert_affinities_masked = torch.gather(expert_affinities_masked, 1, broadcasted_local_expert_indices)
                if self.routed_experts_mlp_config.expert_distribution:
                    expert_parallel_rank = spmd_rank.get_rank() // self.moe_tensor_model_parallel_group.size()
                    expert_start_idx, expert_end_idx = self.allocate_token_blocks(
                        torch.tensor(
                            self.routed_experts_mlp_config.local_redudancy_degree,
                            dtype=torch.int32, device=local_expert_mask.device
                        ), local_expert_mask.shape[0])
                    local_expert_mask = self.generate_local_expert_mask_with_redundancy(
                        local_expert_mask,
                        local_expert_indices,
                        expert_start_idx,
                        expert_end_idx,
                        self.routed_experts_mlp_config.num_experts,
                        expert_parallel_rank,
                    )
                    local_expert_affinities_masked = self.get_expert_affinities_masked(
                        local_expert_affinities_masked, # (T, E')
                        local_expert_mask, # (T, E')
                        False,
                    )
                # TODO: make EP work with optimized_block_to_token_mapping in the future, currently not supported.
                ues_optimized_block_to_token_mapping = False
                logger.info("Expert Parallel Enabled for forward_blockwise\n" +
                            f"broadcasted_local_expert_indices: {broadcasted_local_expert_indices.shape}\n" +
                            f"expert_mask: {expert_mask.shape}\n" +
                            f"expert_affinities_masked: {expert_affinities_masked.shape}\n" +
                            f"local_expert_mask: {local_expert_mask.shape}\n" +
                            f"local_expert_affinities_masked: {local_expert_affinities_masked.shape}\n")
            else:
                # [T, E]
                local_expert_mask = expert_mask
                local_expert_affinities_masked = expert_affinities_masked
                ues_optimized_block_to_token_mapping = self.blockwise_matmul_config.optimized_block_to_token_mapping,

            block_to_expert, token_position_to_id = self.get_blockwise_expert_and_token_mapping(
                total_tokens=total_tokens,
                num_blocks=num_blocks,
                expert_mask=local_expert_mask,
                expert_index=expert_index,
                block_size=self.blockwise_matmul_config.block_size,
                device=hidden_states.device,
                enable_spmd_rank=self.routed_experts_mlp_config.enable_spmd_rank,
                spmd_rank=spmd_rank if self.routed_experts_mlp_config.enable_spmd_rank else None,
                tensor_parallel_group=self.moe_tensor_model_parallel_group,
                optimized_block_to_token_mapping=ues_optimized_block_to_token_mapping,
                parallelize_token_to_block_mapping=self.blockwise_matmul_config.parallelize_token_to_block_mapping,
                pad_num_blocks_to_even=pad_num_blocks_to_even,
            )

        if self.blockwise_matmul_config.use_shard_on_block_dynamic_while:
            # shard-on-block dynamic while kernel will only use even-numbered idx element in conditions.
            # Each loop iteration will process 2 blocks in shard-on-block dynamic while kernel.
            conditions = self.get_block_conditions(self.blockwise_matmul_config.block_size, num_blocks, token_position_to_id)[::2]
        elif self.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while:
            conditions = self.get_block_conditions(self.blockwise_matmul_config.block_size, num_blocks, token_position_to_id)
        else:
            conditions = None

        use_blockwise_matmul_nki = can_use_blockwise_matmul_nki(
            hidden_size=hidden_size,
            intermediate_size_tp=mlp_op.down_proj.weight.shape[1],
            block_size=self.blockwise_matmul_config.block_size,
            glu_mlp=self.routed_experts_mlp_config.glu_mlp,
            glu_type=self.routed_experts_mlp_config.glu_type,
            act_fn=self.routed_experts_mlp_config.hidden_act,
            device=hidden_states.device,
            logical_nc_config=self.blockwise_matmul_config.logical_nc_config,
            use_block_parallel=self.blockwise_matmul_config.use_block_parallel,
            use_shard_on_intermediate=self.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while,
            use_shard_on_block_dynamic_while=self.blockwise_matmul_config.use_shard_on_block_dynamic_while,
            use_torch_block_wise=self.blockwise_matmul_config.use_torch_block_wise,
            use_bias=self.routed_experts_mlp_config.bias,
            scaling_factor=self.routed_experts_mlp_config.hidden_act_scaling_factor,
            gate_clamp_upper_limit=self.routed_experts_mlp_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=self.routed_experts_mlp_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=self.routed_experts_mlp_config.up_clamp_upper_limit,
            up_clamp_lower_limit=self.routed_experts_mlp_config.up_clamp_lower_limit,
        )
        #TODO: Have the blockwise matmul kernel support I_TP sizes that are not divisible by 16
        if use_blockwise_matmul_nki:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

            blockwise_nki_autograd_cls = self.blockwise_matmul_config.blockwise_nki_autograd_cls \
                if self.blockwise_matmul_config.blockwise_nki_autograd_cls \
                else BlockwiseMatmulNKIFunc

            gate_up_proj_scale, down_proj_scale = getattr(mlp_op.gate_up_proj, "scale", None), getattr(mlp_op.down_proj, "scale", None)
            if (gate_up_proj_scale is not None or down_proj_scale is not None) and \
            QuantizationType.EXPERT_WISE_PER_CHANNEL_SYMMETRIC == mlp_op.gate_up_proj.quantization_type:
                assert self.blockwise_matmul_config.logical_nc_config == 2, "EXPERT_WISE_PER_CHANNEL_SYMMETRIC only supported for LNC=2"

            # This will retrieve the activation function id based on glu_type and act_func
            # Currently, SiLU should be used when glu_type == GLU_TYPE.GLU in NKI kernel
            # Sigmoid should be used when glu_type == GLU_TYPE.SWIGLU
            kernel_act_fn_id = get_kernel_activation_func_id(ACTFunc.validate(self.routed_experts_mlp_config.hidden_act), self.routed_experts_mlp_config.glu_type)
            return blockwise_nki_autograd_cls.apply(
                hidden_states,
                local_expert_affinities_masked,
                mlp_op.gate_up_proj.weight,
                mlp_op.down_proj.weight,
                token_position_to_id,
                block_to_expert,
                gate_up_proj_scale,
                down_proj_scale,
                self.training,
                self.blockwise_matmul_config,
                self.routed_experts_mlp_config.top_k > 1,
                expert_affinities_scaling_mode,
                conditions,
                num_static_blocks,
                self.routed_experts_mlp_config.gate_clamp_upper_limit,
                self.routed_experts_mlp_config.gate_clamp_lower_limit,
                self.routed_experts_mlp_config.up_clamp_upper_limit,
                self.routed_experts_mlp_config.up_clamp_lower_limit,
                mlp_op.gate_up_proj.bias if self.routed_experts_mlp_config.bias else None,
                mlp_op.down_proj.bias if self.routed_experts_mlp_config.bias else None,
                kernel_act_fn_id,
            )
        elif self.training:
            return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
        else:
            return self.torch_blockwise_matmul_inference(
                block_size=self.blockwise_matmul_config.block_size,
                num_blocks=num_blocks,
                hidden_states=hidden_states,
                expert_affinities_masked=local_expert_affinities_masked,
                token_position_to_id=token_position_to_id,
                block_to_expert=block_to_expert,
                pad_inputs_for_matmul=not self.blockwise_matmul_config.skip_dma_token,
            )

    @staticmethod
    def allocate_token_blocks(local_redudancy_degree, tokens_per_expert):
        """
        Helper function that calculates the start position id and the end position id
        corresponding to each expert based on the expert distribution across all EP groups.

        Args:
        local_redudancy_degree : The redundancy degree of experts within each EP group.
        tokens_per_expert: Total number of input tokens.

        Returns:
        start_indices: Tensor of shape [num_ep_group, experts] denoting start position ids for each expert.
        end_indices: Tensor of shape [num_ep_group, experts] denoting end position ids for each expert.
        """
        # Get global counts for each expert
        global_expert_counts = local_redudancy_degree.sum(dim=0, dtype=torch.int32)  # Shape: [num_experts]

        # Get cumulative counts up to (but not including) each EP group
        cumsum = torch.cumsum(local_redudancy_degree, dim=0, dtype=torch.int32)
        prev_counts = torch.zeros_like(local_redudancy_degree, dtype=torch.int32)
        prev_counts[1:] = cumsum[:-1]  # Shape: [ep_group, num_experts]

        # Calculate base block size for each expert
        block_size = tokens_per_expert // global_expert_counts
        remainder = tokens_per_expert % global_expert_counts
        block_size = block_size.to(dtype=torch.int32)
        remainder = remainder.to(dtype=torch.int32)

        # Calculate start and end indices
        start_indices = (prev_counts * block_size.unsqueeze(0)).to(dtype=torch.int32)
        end_indices = ((prev_counts + local_redudancy_degree) * block_size.unsqueeze(0)).to(dtype=torch.int32) - 1

        # Add remainder to the last block for each expert
        is_last = (cumsum == global_expert_counts.unsqueeze(0))
        end_indices = end_indices + (is_last * remainder.unsqueeze(0)).to(dtype=torch.int32)
        return start_indices, end_indices

    @staticmethod
    def generate_local_expert_boolean_mask(local_expert_indices, num_experts):
        """
        Helper function which converts the local_expert_indices into a mask of logical experts to local physical experts.
        Example:
        local_expert_indices : [5, 5, 0, 1, 0], num_experts = 8
        result [[F F T F T], [F F F T F], [F F F F F ], [F, F, F, F, F], [F, F, F, F, F], [T T F F F]], [F, F, F, F, F], [F, F, F, F, F]
        """
        indices = torch.arange(num_experts, dtype=torch.int32, device=local_expert_indices.device).unsqueeze(1)
        result = (indices == local_expert_indices)
        return result

    @staticmethod
    def generate_mask_with_no_local_redundancy(mask):
        """
        Helper function which masks out the duplicate expert Ids within the same ep group.
        Example:
        mask [[F F T F T], [F F F T F], [F F F F F ], [F, F, F, F, F], [F, F, F, F, F], [T T F F F]], [F, F, F, F, F], [F, F, F, F, F]
        result [[F F T F F], [F F F T F], [F F F F F ], [F, F, F, F, F], [F F F F F ], [T F F F F]], [F F F F F ], [F F F F F ],
        """
        t_mask = torch.transpose(mask, 0, 1).to(dtype=torch.int32)
        mask_cumsum = cumsum(t_mask)
        mask_first_occurance = mask_cumsum == 1
        mask_first_occurance = torch.transpose(mask_first_occurance, 0, 1)
        return torch.logical_and(mask_first_occurance, mask)

    @staticmethod
    def generate_local_expert_id_no_local_redundancy(mask, device):
        """
        Helper function that takes in a ogical experts to local physical experts mask and converts to
        local_expert_indices with -1s for locally duplicated experts.
        """
        temp = torch.arange(mask.shape[0], device=device, dtype=torch.int32).unsqueeze(1).expand_as(mask)
        temp_pad = torch.full(mask.shape, fill_value=-1, device=device, dtype=torch.int32)
        expert_indices = torch.where(mask, temp, temp_pad)
        expert_per_slot = torch.max(expert_indices, dim=0).values
        return expert_per_slot

    def generate_local_expert_mask_with_redundancy(self, local_expert_mask, local_expert_indices, expert_start_ids, expert_end_ids, num_experts, rank):
        """
        Generates local expert mask where tokens are divided among redundant experts in equal chunks based on degree of redundancy to
        avoid duplicate computation for tokens routed to redundant experts.

        Eg:

        For EP degree is 2, and expert distribution as [[0, 2, 4, 5], [1, 3, 5, 4]] with num_experts as 6
        If the expert_mask is as shown below
        [ E0 E1 E2 E3 E4 E5
        [1, 0, 0, 0, 0, 1],  # Token0
        [0, 0, 1, 0, 1, 0],  # Token1
        [0, 0, 0, 1, 1, 0],  # Token2
        [0, 1, 0, 0, 0, 1]   # Token3
        ]
        then the local expert mask for EP Group0 is
        E0 E2 E4 E5
        [[1, 0, 0, 1],  # Token0
        [0, 1, 1, 0],  # Token1
        [0, 0, 1, 0],  # Token2
        [0, 0, 0, 1],  # Token3
        ]

        and the local expert mask for EP Group1 is
        E1 E3 E5 E4
        [[0, 0, 1, 0],  # Token0
        [0, 0, 0, 1],  # Token1
        [0, 1, 0, 1],  # Token2
        [1, 0, 1, 0],  # Token3
        ]

        The output local expert mask for EP Group0 will be
        E0 E2 E4 E5
        [[1, 0, 0, 1],  # Token0
        [0, 1, 1, 0],  # Token1
        [0, 0, 0, 0],  # Token2
        [0, 0, 0, 0],  # Token3
        ]

        The output local expert mask for EP Group0 will be
        E1 E3 E5 E4
        [[0, 0, 0, 0],  # Token0
        [0, 0, 0, 0],  # Token1
        [0, 1, 0, 1],  # Token2
        [1, 0, 1, 0],  # Token3
        ]

        Args:
        local_expert_mask: Local expert mask of shape [T, Local experts]
        local_expert_indices: Local expert indices tensor of shape [1, Local experts]
        expert_start_ids: The start indices of tokens for each experts as per the expert distribution
        expert_end_ids: The end indices of tokens for each experts as per the expert distribution
        num_experts: Total number of logical routed experts.
        rank: A tensor of shape [1,] indicating the current expert parallel rank.

        Returns:
        A tensor of shape [T, Local experts] ensuring no duplicate processing across physical experts when redundancy is applicable.
        """
        local_expert_indices = local_expert_indices.squeeze(0)
        total_tokens = local_expert_mask.shape[0]
        local_start_ids = expert_start_ids[rank, local_expert_indices][None, :]
        loacl_end_ids = expert_end_ids[rank, local_expert_indices][None, :]
        token_range = torch.arange(total_tokens, device=local_expert_mask.device)[:, None]
        from_start = token_range >= local_start_ids
        before_end = token_range <= loacl_end_ids
        mask = torch.logical_and(from_start, before_end)
        final_mask = local_expert_mask.masked_fill(~mask, 0)
        local_expert_boolean_masked =  self.generate_local_expert_boolean_mask(local_expert_indices, num_experts)
        local_mask_with_no_redundancy = self.generate_mask_with_no_local_redundancy(local_expert_boolean_masked)
        local_expert_indices_unique = self.generate_local_expert_id_no_local_redundancy(local_mask_with_no_redundancy, local_expert_mask.device) # -1 where it is duplicated
        local_redundancy_mask = local_expert_indices_unique != -1
        return final_mask.masked_fill(~local_redundancy_mask, 0)

    @staticmethod
    def get_block_conditions(
        block_size,
        num_blocks,
        token_position_to_id
    ):
        # Reshape the token positions into blocks
        blocks = token_position_to_id.view(num_blocks, block_size)
        # Check each block for non padded tokens (any position != -1)
        conditions = torch.any(blocks != -1, dim=1).to(torch.int32)
        return conditions

    @staticmethod
    def get_blockwise_expert_and_token_mapping_kernel(
        num_blocks: int,
        expert_affinities_masked: torch.tensor,
        block_size: int,
        device: torch.device,
        spmd_rank: SPMDRank,
        tensor_parallel_group: ProcessGroup,
        expert_parallel_group: ProcessGroup,
        logical_nc_config: int,
        pad_num_blocks_to_even: bool = False,
    ):
        '''
        Equivalent function of get_blockwise_expert_and_token_mapping, but nstead of torch code,
        this function uses a kernel to perform the index mapping.

        Note, `expert_affinities_masked` is the full (T,E) shaped expert affinities masked w/o sharding.

        Args:
            num_blocks: int, total number of blocks on this rank.
            expert_affinities_masked: Tensor of shape (T,E), containing masked expert affinities
                - Note for performance reasons, this input should be full (unsharded).
                - The kernel will read the corresponding mask chunk it processes.
            block_size: int, block size of the blockwise matmul kernel.
            device: device
            spmd_rank: SPMDRank object for the local rank.
            tensor_parallel_group: MoE's TP replica group.
            expert_parallel_group: MoE's EP replica group.
            logical_nc_config: LNC config
        Returns:
            block_to_expert: Tensor of shape (num_blocks,), indicating which expert each block belongs to.
            token_position_to_id: Tensor of shape (num_blocks*block_size,), mapping positions of tokens in
                the flattened blocks to their indices in T.
        '''
        T, E = expert_affinities_masked.shape

        global_rank = spmd_rank.get_rank().to(device)
        tp_size = tensor_parallel_group.size()
        tp_rank = torch.remainder(global_rank, tp_size)
        ep_size = expert_parallel_group.size()
        ep_rank = global_rank // tp_size
        max_chunk_size = 16384

        # Every EP rank needs the information about its local expert (`E_local` of those).
        E_local = E // ep_size
        if E_local % tp_size != 0:
            # Replicate index calc kernel on TP ranks in the EP group.
            logger.warning(f"Replicating index calc kernel on TP ranks, because {E_local=} is not divisible by {tp_size=}.")
            E_kernel = E_local
            # Process E_local by setting:
            #  - `row_start_id` to the start of the expert on this EP rank.
            #  - `n_rows` to the number of experts on this EP rank.
            indices, nonzero_counts = find_nonzero_indices[nl.nc(logical_nc_config)](
                input_tensor=expert_affinities_masked.to(torch.float32),
                row_start_id=ep_rank*E_kernel,
                n_rows=E_kernel,
                chunk_size=min(T, max_chunk_size),
                index_dtype=nl.int32,
            )
        else:
            # Shard index calc kernel further along expert dimension in the TP group.
            # i.e. the index calc is sharded globally in EP.
            E_kernel = E_local // tp_size
            # TP ranks shards the processing of E_local by starting at different `row_start_id`,
            # and calculating fewer experts `E_kernel`.
            indices, nonzero_counts = find_nonzero_indices[nl.nc(logical_nc_config)](
                input_tensor=expert_affinities_masked.to(torch.float32),
                row_start_id=global_rank*E_kernel,
                n_rows=E_kernel,
                chunk_size=min(T, max_chunk_size),
                index_dtype=nl.int32,
            )
            # Gather nonzero_counts: [E_kernel,] --> [E/EP,]
            nonzero_counts = mappings.gather_from_tensor_model_parallel_region(
                nonzero_counts, process_group=tensor_parallel_group,
            )

        # Get number of blocks and cumulative number of blocks per expert.
        blocks_per_expert = ((nonzero_counts + block_size - 1) // block_size).to(dtype=torch.long)  # (E_EP,)

        # Calculate padding blocks needed and add to last expert
        if pad_num_blocks_to_even:
            total_needed_blocks = torch.sum(blocks_per_expert)
            padding_blocks = num_blocks - total_needed_blocks
            blocks_per_expert[-1] += padding_blocks

        blocks_per_expert_expanded = blocks_per_expert.unsqueeze(1)  # (E_EP, 1)
        cum_blocks_per_expert = cumsum(blocks_per_expert_expanded)  # (E_EP, 1)
        cum_blocks_per_expert[1:] = cum_blocks_per_expert[:-1]
        cum_blocks_per_expert[0] = 0

        # Convert [E_kernel, T] indices into [N*B+T] token_position_to_id_padded
        # Each partition writes `f_len` elements on the free dimension.
        f_len = min(128, T // 16)
        row_offsets = cum_blocks_per_expert * (block_size // f_len)
        if E_local % tp_size != 0:
            # [E/EP, T] --> [num_blocks * block_size,]
            token_position_to_id_padded = indexed_flatten[nl.nc(logical_nc_config)](
                input_tensor = indices,
                f_len = f_len,
                output_len=num_blocks*block_size + T,
                row_offsets= row_offsets.reshape(-1).to(torch.int32),
                row_offsets_start = 0,
            )
            token_position_to_id = token_position_to_id_padded[:num_blocks * block_size]
        else:
            # [E/(EP*TP), T] --> [num_blocks * block_size,]
            token_position_to_id_padded = indexed_flatten[nl.nc(logical_nc_config)](
                input_tensor = indices,
                f_len = f_len,
                output_len=num_blocks*block_size + T,
                row_offsets= row_offsets.reshape(-1).to(torch.int32),
                row_offsets_start = tp_rank*E_kernel,
            )
            # Aggregate information across TP ranks.
            token_position_to_id = mappings._reduce(
                token_position_to_id_padded[:num_blocks * block_size], computation=xm.REDUCE_MAX, process_group=tensor_parallel_group,
            )

        # Get the block to expert mapping.
        block_ids = torch.arange(num_blocks, device=device, dtype=torch.long)  # (N, )
        block_to_expert = torch.sum(block_ids.unsqueeze(0) >= cum_blocks_per_expert[1:], dim=0).to(torch.long)  # (N, )

        return block_to_expert, token_position_to_id


    @staticmethod
    def get_blockwise_expert_and_token_mapping(
        total_tokens,
        num_blocks,
        expert_mask,
        expert_index,
        block_size,
        device,
        enable_spmd_rank,
        spmd_rank,
        tensor_parallel_group,
        optimized_block_to_token_mapping=True,
        parallelize_token_to_block_mapping=False,
        pad_num_blocks_to_even=False,
    ):
        """
        Token position: position in blocks.
        E.g. given block_size=2, num_token=6. The following expert_mask
            [1, 0, 0],  # First token for expert 0
            [0, 1, 0],  # First token for expert 1
            [1, 0, 0],  # Second token for expert 0
            [0, 0, 1],  # First token for expert 2
            [1, 0, 0],  # Third token for expert 0
            [0, 1, 0],  # Second token for expert 1
        would put tokens into blocks as follows:
            Block 0 (expert 0)
                0
                2
            Block 1 (expert 0)
                4
                -1
            Block 2 (expert 1)
                1
                5
            Block 3 (expert 2)
                3
                -1
        This would result in block to expert mappping:
            0 -> 0
            1 -> 0
            2 -> 1
            3 -> 2
        and token position to id mapping:
            0 -> 0
            1 -> 2
            2 -> 4
            3 -> -1
            4 -> 1
            5 -> 5
            6 -> 3
            7 -> -1
        """

        # tokens_per_expert: (E, )
        tokens_per_expert = torch.sum(expert_mask, dim=0)
        # blocks_per_expert: (E, )
        blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(dtype=torch.long)

        # Calculate padding blocks needed and add to last expert
        if pad_num_blocks_to_even:
            total_needed_blocks = torch.sum(blocks_per_expert)
            padding_blocks = num_blocks - total_needed_blocks
            blocks_per_expert[-1] += padding_blocks

        # block_to_expert: (N, ). Block id to expert id mapping.
        # The simplest way to do this is to use repeat_interleave after padding blocks_per_expert with unassigned blocks.
        # But this op is not lowered to xla with vector 'repeats', so we use the equivalent implementation below.
        block_ids = torch.arange(num_blocks, device=device, dtype=torch.long)  # (N, )
        cumulative_blocks_per_expert = cumsum(blocks_per_expert.unsqueeze(1)).squeeze(1)  # (E, )
        block_to_expert = torch.sum(block_ids.unsqueeze(1) >= cumulative_blocks_per_expert[:-1], dim=1).to(torch.long)

        # token_position_by_id_and_expert: (T, E)
        token_position_by_id_and_expert = cumsum(expert_mask)
        # Tokens assigned to a given expert are assembled in consecutive blocks (and will have consecutive positions)
        # The block position for a token for an expert depends on the number of blocks assigned to all previous experts.
        # Compute and add this offset for each expert
        expert_block_offsets = cumulative_blocks_per_expert * block_size
        token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
        # Apply expert_mask
        token_position_by_id_and_expert = token_position_by_id_and_expert.masked_fill(torch.eq(expert_mask, 0), 0).to(dtype=torch.long)

        # Invert token_position_by_id_and_expert to obtain token_position_to_id
        # token_position_to_id: (N*B+1,)
        # Initialize with -1 indices (which will remain as the indices of padding tokens)
        token_position_to_id = -1 * torch.ones(num_blocks * block_size + 1, device=device, dtype=torch.long)

        if total_tokens % tensor_parallel_group.size() != 0:
            # Pad token_position_by_id_and_expert
            num_pad = (-total_tokens) % tensor_parallel_group.size()
            token_position_by_id_and_expert = F.pad(token_position_by_id_and_expert, (0, 0, 0, num_pad))
            tokens_idx = torch.arange(total_tokens + num_pad, device=device, dtype=torch.long)
        else:
            tokens_idx = torch.arange(total_tokens, device=device, dtype=torch.long)
        # Further reduce token_position_by_id_and_expert to only keep topk experts' token positions
        if optimized_block_to_token_mapping:
            token_position_by_id_and_expert = torch.gather(token_position_by_id_and_expert, dim=1,
                                                           index=expert_index)
        if not parallelize_token_to_block_mapping:
            # token_position_to_id is a flattened array that contains the mapping of token indices for each block
            # token_position_to_id contains -1 as the index of 'padding' tokens
            token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
            token_position_to_id = token_position_to_id[1:]
        # Distribute computation by splitting token_position_by_id_and_expert and tokens_idx across TP ranks
        # The same token_position_by_id_and_expert and tokens_idx is present at all ranks, so we can use any of MIN/MAX/AVG to as the reduce operation
        else:
            if enable_spmd_rank:
                # use rank information is available at runtime in inference
                # get tp_rank from global rank
                # note: we use `get_tensor_model_parallel_group()` here to parallelize within a single node
                #       but may get replicated in multi-node case
                tp_rank = torch.remainder(spmd_rank.get_rank(), tensor_parallel_group.size())
                token_position_by_id_and_expert = mappings.scatter_to_process_group_spmd(
                    token_position_by_id_and_expert, partition_dim=0, rank=tp_rank, process_group=tensor_parallel_group,
                )
                tokens_idx = mappings.scatter_to_process_group_spmd(
                    tokens_idx, partition_dim=0, rank=tp_rank, process_group=tensor_parallel_group,
                )
                # Assemble token_position_to_id using chunk of token_position_by_id_and_expert and tokens_idx at each TP rank
                # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
                token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
                token_position_to_id = token_position_to_id[1:]
                # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
                token_position_to_id = mappings._reduce(
                    token_position_to_id, computation=xm.REDUCE_MAX, process_group=tensor_parallel_group,
                )
            else:
                # To avoid rank-specific computations, we do a reduce-scatter instead of a simple split
                token_position_by_id_and_expert = mappings._reduce_scatter_along_first_dim(
                    token_position_by_id_and_expert, computation=xm.REDUCE_MIN, process_group=tensor_parallel_group,
                )
                tokens_idx = mappings._reduce_scatter_along_first_dim(
                    tokens_idx, computation=xm.REDUCE_MIN, process_group=tensor_parallel_group,
                )
                # Assemble token_position_to_id using chunk of token_position_by_id_and_expert and tokens_idx at each TP rank
                # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
                token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
                token_position_to_id = token_position_to_id[1:]
                # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
                token_position_to_id = mappings._reduce(
                    token_position_to_id, computation=xm.REDUCE_MAX, process_group=tensor_parallel_group,
                )
        return block_to_expert, token_position_to_id

    def torch_blockwise_matmul_inference(
        self,
        block_size,
        num_blocks,
        hidden_states,
        expert_affinities_masked,
        token_position_to_id,
        block_to_expert,
        pad_inputs_for_matmul = False,
    ):
        """
        PyTorch implementation of the blockwise matmul.

        This is used when running on GPU, or when the blockwise NKI kernel is not compatible with the model
        configuration.
        """
        mlp_op = self.get_mlp_op()
        total_tokens, hidden_size = hidden_states.shape
        output = torch.zeros(total_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        # this simulates the case (when using blockwise nki kernel) when skip_tokens in skip_dma is false,
        # where -1s in token_position_to_id needs to be converted to total_tokens, hidden_states etc needs to be padded by
        # one extra row
        if pad_inputs_for_matmul:
            output, hidden_states, token_position_to_id, expert_affinities_masked = augment_inputs_for_padded_blockwise_matmul(output, hidden_states, token_position_to_id, expert_affinities_masked)
        else:
            # the extra row is used to store the output for the padded indices in the blocks which will be discarded
            # as the outputs for padding indices should not contribute to the final output
            output = torch.cat([output, torch.zeros(1, hidden_size, device=output.device, dtype=output.dtype)])
        # block_to_token_indices: (N, B)
        block_to_token_indices = token_position_to_id.view(num_blocks, block_size)
        for block_idx in range(num_blocks):
            block_token_indices = block_to_token_indices[block_idx]
            block_expert_idx = block_to_expert[block_idx]
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                # Scale the input hidden states before MLP
                block_hidden_states = (
                    hidden_states[block_token_indices]
                    * expert_affinities_masked[block_token_indices, block_expert_idx.unsqueeze(0)].unsqueeze(1)).unsqueeze(0)
                block_output = mlp_op(block_hidden_states, expert_indices=block_expert_idx.unsqueeze(0)).squeeze(0)
            else:
                # block_hidden_states: (1, B, H)
                block_hidden_states = hidden_states[block_token_indices].unsqueeze(0)
                # block_mlp_output: (B, H)
                block_mlp_output = mlp_op(block_hidden_states, expert_indices=block_expert_idx.unsqueeze(0)).squeeze(0)
                # block_output: (B, H)
                # FIXME: remove unsqueeze(0) from block_expert_idx.unsqueeze(0) would OOM
                block_output = block_mlp_output * expert_affinities_masked[block_token_indices, block_expert_idx.unsqueeze(0)].unsqueeze(
                    1
                )
            # Update the tokens computed by the block
            output[block_token_indices] += block_output

        # Drop the last row that store the output from the padded indices in the block
        output = output[:total_tokens, :]

        return output

    def forward(self, hidden_states, expert_affinities, expert_index, seq_len, padding_mask=None, expert_affinities_masked_full=None):
        """Forward pass of the ExpertMLPs.

        For training:
        1. If capacity_factor is None (full capacity), run forward_all_experts().
        2. Else if capacity_factor is smaller than or equal to zero, run forward_blockwise().
        3. Else run forward_capacity_factor().

        For inference:
        1. If token generation (seq len == 1):
            Run forward_selective_loading() or forward_all_experts() depending on the following logic.
            Let T be the total number of tokens. Using selective loading, T*top_k experts will be loaded.
            If (T*top_k/num_experts) is less than SELECTIVE_LOADING_THRESHOLD, then we use selective loading.
            Otherwise, we use forward_all_experts (for better performance).
        2. Else (seq len > 1):
            a. If capacity factor is not None, run forward_capacity_factor().
            b. If the number of tokens is very small such that the number of experts loaded is less
               than SELECTIVE_LOADING_THRESHOLD, use selective loading. This is useful for the speculation use case.
            c. If number of tokens * top_k is less than the block size, run forward_all_experts().
            c. Otherwise, run forward_blockwise().

        Note on the SELECTIVE_LOADING_THRESHOLD:
        This parameter determines when forward_selective_loading is used for token-gen (in favor of
        forward_all_experts), and should be a float <= 1.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Tensor of shape (T, H).
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
            seq_len: Sequence length S. Used to infer context encoding vs token generation in inference.
            padding_mask: Padding mask to mask out padded tokens

            If sequence_parallel_enabled:
                hidden_states: Tensor of shape (T, H).
                expert_affinities: Tensor of shape (T//tp_degree, E)
                expert_index: Tensor of shape (T//tp_degree, top_k)

        Returns:
            output: Output tensor of the same shape as hidden_states, obtained by passing each token through its assigned experts,
                    combined with the corresponding expert affinities.
        """

        if self.training:
            # Training flow
            if self.routed_experts_mlp_config.capacity_factor is None:
                return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
            elif self.routed_experts_mlp_config.capacity_factor <= 0:
                act_fn = ACT2FN[self.routed_experts_mlp_config.hidden_act]
                if act_fn != F.silu:
                    logger.info(rmsg(f"{act_fn=}. Blockwise training only supports SiLU activation, falling back to full capacity"))
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                if not self.routed_experts_mlp_config.glu_mlp:
                    logger.info(rmsg(f"{self.routed_experts_mlp_config.glu_mlp=}. Blockwise training only supports glu_mlp=True, falling back to full capacity"))
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                return self.forward_blockwise(hidden_states, expert_affinities, expert_index)
            else:
                return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
        else:
            # Inference flow
            total_tokens = hidden_states.shape[0]
            if seq_len == 1:
                # Token generation
                perc_experts_loaded = total_tokens * self.routed_experts_mlp_config.top_k / self.routed_experts_mlp_config.num_experts
                if perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD:
                    if self.moe_expert_model_parallel_group.size() > 1:
                        return self.forward_all_experts_EP(hidden_states, expert_affinities, expert_index)
                    else:
                        return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                else:
                    if self.moe_expert_model_parallel_group.size() > 1:
                        raise NotImplementedError("Selective Loading with Expert parallelism is not supported in token generation.")
                    else:
                        return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
            else:
                # Context Encoding / Speculative Decoding
                if self.routed_experts_mlp_config.capacity_factor is None:
                    perc_experts_loaded = total_tokens * self.routed_experts_mlp_config.top_k / self.routed_experts_mlp_config.num_experts
                    if perc_experts_loaded < DEFAULT_SELECTIVE_LOADING_THRESHOLD:
                        # Use selective loading for small speculation lengths
                        return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
                    elif total_tokens * self.routed_experts_mlp_config.top_k < self.blockwise_matmul_config.block_size:
                        # Use all experts for large speculation lengths, and small context encoding prompt sizes
                        # (more efficient to run all_experts instead of blockwise - equivalent in FLOPs, lower memory bandwidth usage)
                        return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                    else:
                        # Use blockwise for dropless context encoding
                        return self.forward_blockwise(hidden_states, expert_affinities, expert_index, expert_affinities_masked_full, padding_mask=padding_mask)
                else:
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
    n_routed_experts: int,
    expert_model_parallel_group: ProcessGroup,
    spmd_rank_name: str,
    expert_distribution=None,
):
    # add weight for spmd rank
    model_state_dict[f"{prefix}{spmd_rank_name}.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    if expert_model_parallel_group.size() > 1:
        expert_indices = []
        for rank in range(world_size):
            curr_expert_rank = parallel_state.get_expert_parallel_rank_from_global_rank(
                rank=rank, expert_parallel_group=expert_model_parallel_group
            )
            curr_expert_indices = parallel_state.get_experts_for_expert_parallel_rank(
                curr_expert_rank,
                total_number_of_experts=n_routed_experts,
                expert_model_parallel_size=expert_model_parallel_group.size(),
                expert_distribution=expert_distribution,
            )
            expert_indices.append(curr_expert_indices)

        model_state_dict[f"{prefix}{spmd_rank_name}.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )

def duplicate_and_replace_prefixes(old_prefix: str, new_prefix: str, model_state_dict: Dict[str, Any]):
    """
    This function is used by hybrid sharding to duplicate weight with key "old_prefix" with new_key "new_prefix"
    in model_state_dict. The duplicated weight will be sharded with different sharding strategy.
    """
    old_keys = []
    new_keys = []
    for key in model_state_dict.keys():
        if old_prefix in key:
            new_key = key.replace(old_prefix, new_prefix)
            new_keys.append(new_key)
            old_keys.append(key)

    for key_index in range(len(old_keys)):
        model_state_dict[new_keys[key_index]] = model_state_dict[old_keys[key_index]]


def can_use_find_index_kernel(
    T: int,
    block_size: int,
    E_local: int,
    logical_nc_config: int,
    tp_size: int,
    ep_size: int,
) -> bool:
    """
    Checks whether the index calculation kernel can be used with the given configuration.

    Args:
        T: total number of tokens
        block_size: block size of the blockwise matmul
        E_local: number of experts on each EP rank
        logical_nc_config: LNC setting.
        tp_size: TP size
        ep_size: EP size
    """
    # The kernel outputs (E/EP, T) and we massage the data to (N*B,) for BWMM kernel input.
    # If T is not divisible by the block_size, the kernel path is not supported.
    if T % block_size != 0:
        logger.warning(f"{T=} not divisible by {block_size=}, cannot use index calc kernel.")
        return False
    # The kernel currently runs on LNC2, sharding along the expert dimension.
    # This means each rank needs to have at least 2 experts
    if logical_nc_config != 2:
        logger.warning(f"{logical_nc_config=} must be 2 to use the index calc kernel.")
        return False
    if not (E_local % logical_nc_config == 0):
        logger.warning(f"{E_local=} not divisible by {logical_nc_config=}, cannot use index calc kernel.")
        return False
    if tp_size == 8 and ep_size == 8:
        logger.warning("TP=8 and EP=8 is not supported by the index calc kernel, because TP ranks are not contiguous")
        return False

    return True

