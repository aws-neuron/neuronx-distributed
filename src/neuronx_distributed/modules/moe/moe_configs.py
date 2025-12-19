import importlib
import torch
from typing import Union, Optional, List

from neuronxcc.nki._private_kernels.blockwise_mm import BlockShardStrategy

from neuronx_distributed.modules.moe.model_utils import GLUType, ACTFunc
from neuronx_distributed.modules.moe.model_utils import DEFAULT_BLOCK_SIZE, DEFAULT_SKIP_MODE, DEFAULT_LNC_SIZE
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()

def to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    assert dtype_str in dtype_mapping, f"Unsupported dtype: {dtype_str}"
    return dtype_mapping[dtype_str]

class BlockwiseMatmulConfig:
    """
    Class that contains configs for blockwise matmul
    Arguments:
        block_size: block size used for blockwise matmul
        blockwise_nki_autograd_cls: NKI function that implements blockwise matmul for expert MLPs which will default to BlockwiseMatmulNKIFunc
                                    when specified None. Currently only BlockwiseMatmulNKIFunc is supported.

        use_torch_block_wise: Force using torch implementation of blockwise matmul for expert MLPs instead of invoking NKI kernel.
        logical_nc_config: lnc_size (1 or 2). Default to 1 on trn1, and 2 on trn2.
        parallelize_token_to_block_mapping: parallel computation of block position to token indices mapping. Enabled by default, can be disabled in testing
                                            to rule out collectives issue.
        optimized_block_to_token_mapping: If enabled, token position in blocks will only include top k experts.
        use_block_parallel: Enable calling block parallel blockwise matmuk nki kernel
        block_sharding_strategy: corresponds to different block parallel blockwise matmul kernel
        skip_dma: kernel optimizations for skip tokens and skip weights. When skip token is true, inputs to blockwise kernel do not need to be padded.
                  always_augment_inputs_for_blockwise_matmul: always pad the inputs to blockwise kernel regardless of the value of skip dma.
        use_shard_on_intermediate_dynamic_while: flag to enable shard-on-intermediate dynamic while kernel in forward_blockwise
        use_shard_on_block_dynamic_while: flag to enable shard-on-block dynamic while kernel in forward_blockwise
        num_static_blocks: number of static blocks will be computed in dynamic kernel. The dynamic kernel will have static block and dynamic block where static
                           block computation will be fixed, and dynamic block computation can be skipped
    """
    def __init__(self,
                 block_size: int,
                 use_block_parallel: bool,
                 block_sharding_strategy: BlockShardStrategy,
                 skip_dma_token: bool,
                 skip_dma_weight: bool,
                 logical_nc_config: int,
                 blockwise_nki_autograd_cls,
                 use_torch_block_wise: bool,
                 parallelize_token_to_block_mapping:bool,
                 optimized_block_to_token_mapping: bool,
                 always_augment_inputs_for_blockwise_matmul: bool,
                 use_shard_on_intermediate_dynamic_while: bool,
                 use_shard_on_block_dynamic_while: bool,
                 num_static_blocks: int,
                 ):
        self.block_size = block_size
        self.logical_nc_config = logical_nc_config
        self.use_block_parallel = use_block_parallel
        self.block_sharding_strategy = block_sharding_strategy
        self.use_torch_block_wise = use_torch_block_wise
        self.skip_dma_token = skip_dma_token
        self.skip_dma_weight = skip_dma_weight
        self.optimized_block_to_token_mapping = optimized_block_to_token_mapping
        self.blockwise_nki_autograd_cls = blockwise_nki_autograd_cls
        self.parallelize_token_to_block_mapping = parallelize_token_to_block_mapping
        self.always_augment_inputs_for_blockwise_matmul = always_augment_inputs_for_blockwise_matmul
        self.use_shard_on_intermediate_dynamic_while = use_shard_on_intermediate_dynamic_while
        self.use_shard_on_block_dynamic_while = use_shard_on_block_dynamic_while
        self.num_static_blocks = num_static_blocks

    #TODO: refactor this function
    @staticmethod
    def from_kwargs(**kwargs):
        block_size = kwargs.pop("block_size", DEFAULT_BLOCK_SIZE)
        use_block_parallel = kwargs.pop("use_block_parallel", False)
        logical_nc_config = kwargs.pop("logical_nc_config", DEFAULT_LNC_SIZE)
        blockwise_nki_autograd_cls = kwargs.pop("blockwise_nki_autograd_cls", None)
        if isinstance(blockwise_nki_autograd_cls, dict):
            # TODO: Need a better solution to avoid arbitrary imports
            blockwise_nki_autograd_cls_name = blockwise_nki_autograd_cls["__name__"]
            blockwise_nki_autograd_module_name = blockwise_nki_autograd_cls["__module__"]
            blockwise_nki_autograd_module = importlib.import_module(blockwise_nki_autograd_module_name)
            blockwise_nki_autograd_cls = getattr(blockwise_nki_autograd_module, blockwise_nki_autograd_cls_name)
        # TODO: need a cost model to decide on the default and what to pass as block_sharding_strategy
        block_sharding_strategy = kwargs.pop("block_sharding_strategy", BlockShardStrategy.HI_LO)
        skip_dma_token = kwargs.pop("skip_dma_token", DEFAULT_SKIP_MODE[0])
        skip_dma_weight = kwargs.pop("skip_dma_weight", DEFAULT_SKIP_MODE[1])
        optimized_block_to_token_mapping = kwargs.pop("optimized_block_to_token_mapping", True)
        use_torch_block_wise = kwargs.pop("use_torch_block_wise", False)
        parallelize_token_to_block_mapping = kwargs.pop("parallelize_token_to_block_mapping", True)
        always_augment_inputs_for_blockwise_matmul = kwargs.pop(
            "always_augment_inputs_for_blockwise_matmul", False)
        use_shard_on_intermediate_dynamic_while = kwargs.pop(
            "use_shard_on_intermediate_dynamic_while", False)
        use_shard_on_block_dynamic_while = kwargs.pop(
            "use_shard_on_block_dynamic_while", False)
        num_static_blocks = kwargs.pop(
            "num_static_blocks", None)
        if isinstance(block_sharding_strategy, str):
            if block_sharding_strategy == "HI_LO":
                block_sharding_strategy = BlockShardStrategy.HI_LO
            elif block_sharding_strategy == "PING_PONG":
                block_sharding_strategy = BlockShardStrategy.PING_PONG
            else:
                raise ValueError(f"Unsupported block_sharding_strategy: {block_sharding_strategy}")

        return BlockwiseMatmulConfig(block_size=block_size,
                                    use_block_parallel=use_block_parallel,
                                    logical_nc_config=logical_nc_config,
                                    block_sharding_strategy=block_sharding_strategy,
                                    skip_dma_token=skip_dma_token,
                                    skip_dma_weight=skip_dma_weight,
                                    blockwise_nki_autograd_cls=blockwise_nki_autograd_cls,
                                    optimized_block_to_token_mapping=optimized_block_to_token_mapping,
                                    use_torch_block_wise=use_torch_block_wise,
                                    parallelize_token_to_block_mapping=parallelize_token_to_block_mapping,
                                    always_augment_inputs_for_blockwise_matmul=always_augment_inputs_for_blockwise_matmul,
                                    use_shard_on_intermediate_dynamic_while=use_shard_on_intermediate_dynamic_while,
                                    use_shard_on_block_dynamic_while=use_shard_on_block_dynamic_while,
                                    num_static_blocks=num_static_blocks,
                                    )

    @staticmethod
    def default():
        return BlockwiseMatmulConfig.from_kwargs()

class RoutedExpertsMLPOpsConfig:
    """
    Configuration for routed experts
    Arguments:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        hidden_size: Hidden dimension.
        intermediate_size: Intermediate dimension used in the MLPs.
        hidden_act: Activation function. See ACT2FN for supported activations.
        hidden_size_actual: Actual hidden size without padding the weights.
        intermediate_size_actual: Acutal hidden size without padding the weights.
        is_hidden_dim_shuffled: Whether the hidden dimension of the weights has been shuffled.
        is_intermediate_dim_shuffled: Whether the hidden dimension of the weights has been shuffled.
        glu_mlp: Whether to use the Gated Linear Unit in the MLP. If True, then a combination of gate and up projection is performed in the MLP.
                 Otherwise, a simple up projection is performed.
        capacity_factor: Hyperparameter which controls the expert capacity, and determines the rate of token dropping.
                         If None, then assumed to be running with 'full capacity' (i.e. no tokens dropped).
        use_index_calc_kernel: Whether to use the find index kernel for index calculation for the blockwise matmul kernel.
                               This field is only relevant when the blockwise matmul kernel is used.
        gate_clamp_upper_limit: upper bound value to clamp on expert MLP gate projection results, does not perform clamping if the value is set to None.
        gate_clamp_lower_limit: lower bound value to clamp on expert MLP gate projection results, does not perform clamping if the value is set to None.
        up_clamp_upper_limit: upper bound value to clamp on expert MLP up projection results, does not perform clamping if the value is set to None.
        up_clamp_lower_limit: lower bound value to clamp on expert MLP up projection results, does not perform clamping if the value is set to None.
        normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
                                    Should be used only with top_k > 1.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        output_layer_init_method: Function used for initializing the down projection linear layer weights.
        enable_spmd_rank: use rank information available at runtime in inference i.e., get tp_rank from global rank
        expert_distribution: Specify the expert distribution in each expert parallel group for weight shuffling.
    """
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        hidden_act: str,
        glu_mlp: bool,
        bias: bool = False,
        glu_type: Optional[GLUType] = GLUType.GLU,
        hidden_act_scaling_factor: float = 1.,
        hidden_act_bias: float = 0.,
        hidden_size_actual: Optional[int] = None,
        intermediate_size_actual: Optional[int] = None,
        is_hidden_dim_shuffled: Optional[bool] = None,
        is_intermediate_dim_shuffled: Optional[bool] = None,
        use_index_calc_kernel: bool = True,
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        normalize_top_k_affinities: bool = False,
        early_expert_affinity_modulation: bool = False,
        input_layer_init_method = None,
        output_layer_init_method = None,
        capacity_factor: Union[None, float] = None,
        enable_spmd_rank = False,
        is_prefill = None,
        expert_distribution: Optional[List[List[int]]] = None,
        ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_size_actual = hidden_size_actual
        self.intermediate_size_actual = intermediate_size_actual
        self.is_hidden_dim_shuffled = is_hidden_dim_shuffled
        self.is_intermediate_dim_shuffled = is_intermediate_dim_shuffled
        self.glu_mlp = glu_mlp
        if self.glu_mlp:
            glu_type = GLUType.validate(glu_type)
        self.glu_type = glu_type
        self.hidden_act_scaling_factor = hidden_act_scaling_factor
        self.hidden_act_bias = hidden_act_bias
        self.use_index_calc_kernel = use_index_calc_kernel
        self.gate_clamp_upper_limit = gate_clamp_upper_limit
        self.gate_clamp_lower_limit = gate_clamp_lower_limit
        self.up_clamp_upper_limit = up_clamp_upper_limit
        self.up_clamp_lower_limit = up_clamp_lower_limit
        self.input_layer_init_method = input_layer_init_method
        self.output_layer_init_method = output_layer_init_method
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self.bias = bias
        self.normalize_top_k_affinities = normalize_top_k_affinities
        self.early_expert_affinity_modulation = early_expert_affinity_modulation
        self.enable_spmd_rank = enable_spmd_rank
        self.is_prefill = is_prefill
        self.expert_distribution = expert_distribution
        if self.expert_distribution:
            self.local_redudancy_degree = self.bincount_2d(expert_distribution, num_experts)
            assert len(self.local_redudancy_degree) == len(self.expert_distribution)
            assert len(self.local_redudancy_degree[0]) == num_experts

    @staticmethod
    def bincount_2d(expert_distribution, num_experts):
        """
        Given an expert distribution as input, this generates a count of logical experts
        within each EP group.

        Args:
        expert_distribution:  Expert distribution config in each expert parallel group.
        num_experts: Total number of logical experts across all EP groups.

        Returns:
        local_redudancy_degree: [EP_Groups, num_experts] denoting count of experts within each group.
        """
        local_redudancy_degree = []
        for row in expert_distribution:
            local_redudancy_degree_row = [0]* num_experts
            for val in row:
                local_redudancy_degree_row[val] += 1
            local_redudancy_degree.append(local_redudancy_degree_row)
        return local_redudancy_degree


class RouterConfig:
    """
    Configuration for router
    Arguments:
        act_fn: Activation function. See ACT2FN for supported activations.
        dtype: Router dtype
    """
    def __init__(
            self,
            act_fn: str = "softmax",
            dtype: torch.dtype = torch.float32):
        self.act_fn = act_fn
        self.dtype = dtype

    @staticmethod
    def from_kwargs(**kwargs):
        act_fn = kwargs.pop("router_act_fn", "softmax")
        dtype = kwargs.pop("router_dtype", torch.float32)
        if isinstance(dtype, str):
            dtype = to_torch_dtype(dtype)
        return RouterConfig(act_fn=act_fn, dtype=dtype)

class MoEFusedTKGConfig:
    def __init__(
        self,
        quantized: bool,
        moe_fused_kernel_enabled: Optional[bool] = None,
        router_topk_kernel_enabled: Optional[bool] = None,
        expert_mlp_kernel_enabled: Optional[bool] = None,
        shared_mlp_kernel_enabled: Optional[bool] = None,
        norm_topk_prob: bool = False, # Boolean to normalize top k expert affinities, defaults to no normalization
        is_mxfp4_compute: Optional[bool] = None
    ):
        self.quantized = quantized
        self.moe_fused_kernel_enabled = moe_fused_kernel_enabled
        self.router_topk_kernel_enabled = router_topk_kernel_enabled
        self.expert_mlp_kernel_enabled = expert_mlp_kernel_enabled
        self.shared_mlp_kernel_enabled = shared_mlp_kernel_enabled
        self.norm_topk_prob = norm_topk_prob
        self.is_mxfp4_compute = is_mxfp4_compute
