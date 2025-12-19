# Standard Library
import dataclasses
import itertools
import os
import unittest
from types import SimpleNamespace

# Third Party
import torch
import torch_neuronx.testing
from neuronx_distributed.modules.moe import (
    load_balancing_loss_func as neuron_load_balancing_loss_func,
)
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from parameterized import parameterized
from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

from . import llama4_moe as llama4_moe
from . import loss_fn_correctness_test_helper as lch
from . import mixtral_model as m_mixtral
from . import sbase_model as m_sbase
from . import utils_testing as ut
from .deepseek_v3_router import DeepSeekV3HFMoEGate
from .utils_testing import ExptCfg

if not torch.distributed.is_initialized():
    # Simulate torchrun (required because MoE uses parallel layers for TP)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
ut.nxd_init(tp_degree=1, ep_degree=1, token_shuffle_group_size=1, seed=0)


# IMPORTANT:
# Running with bf16 creates discrepancies in the expert assignment between the sbase and neuron implementations, causing test failures.
# Therefore, we design our tests to ignore (minor) differences in expert assignment when comparing outputs.
# We permit discrepancies in the expert assignment upto BF16_EXPERT_ASSIGNMENT_DIFF_TOL when using bf16.
BF16_EXPERT_ASSIGNMENT_DIFF_TOL = 0.05

TEST_TOLS = {
    "atol": 5e-3,
    "rtol": 1e-2,
}


def get_impl_correctness_test_configs(test_modes):
    test_modes = set(test_modes)
    assert (
        len(test_modes) > 0 and len(test_modes - {"training", "inference"}) == 0
    ), f"Unknown test modes: {str(test_modes)}"

    GLU_MLP_ARGS = [True, False]
    DTYPE_ARGS = [torch.float32, torch.bfloat16]

    test_configs = []

    # S-BASE test cases
    sbase_test_configs = []
    for glu_mlp, dtype in itertools.product(GLU_MLP_ARGS, DTYPE_ARGS):
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": glu_mlp,
            "implementation": "sbase",
        }

        # Training tests
        test_cfg["test_mode"] = "training"
        sbase_test_configs.extend(
            [
                # Test forward_all_experts (full capacity)
                ExptCfg(seq_len=128, batch_size=1, hidden_size=384, num_experts=2, capacity_factor=None, **test_cfg),
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=None, **test_cfg),
                # Test forward_capacity_factor
                ExptCfg(seq_len=128, batch_size=1, hidden_size=384, num_experts=4, capacity_factor=2.0, **test_cfg),
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=1.0, **test_cfg),
            ]
        )
        # Inference tests
        test_cfg["test_mode"] = "inference"
        sbase_test_configs.extend(
            [
                # Test context encoding
                ExptCfg(seq_len=128, batch_size=1, hidden_size=384, num_experts=2, capacity_factor=None, **test_cfg),
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=None, **test_cfg),
                ExptCfg(seq_len=128, batch_size=1, hidden_size=384, num_experts=4, capacity_factor=2.0, **test_cfg),
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=1.0, **test_cfg),
                # Test forward_capacity_factor equivalent to forward_all_expert: capacity_factor >= num_experts / top_k
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=8.0, **test_cfg),
                # Test forward_selective_loading with cte: seq_len*top_k < num_experts
                ExptCfg(seq_len=32, batch_size=4, hidden_size=384, num_experts=64, capacity_factor=None, **test_cfg),
                # Test forward_all_experts with cte: seq_len*top_k < 512
                ExptCfg(seq_len=256, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=None, **test_cfg),
                # Test forward_blockwise with cte: seq_len*top_k > 512
                ExptCfg(seq_len=1024, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=None, **test_cfg),
                #ExpertMLPsV2
                ExptCfg(seq_len=128, batch_size=4, hidden_size=384, num_experts=8, capacity_factor=None, use_expert_mlps_v2=True, **test_cfg),
                ExptCfg(seq_len=128, batch_size=1, hidden_size=384, num_experts=4, capacity_factor=2.0, use_expert_mlps_v2=True, **test_cfg),
                # Test token generation
                # Test forward_all_experts with tkg: batch_size < self.num_experts
                ExptCfg(seq_len=1, batch_size=1, hidden_size=384, num_experts=4, capacity_factor=None, **test_cfg),
                ExptCfg(seq_len=1, batch_size=2, hidden_size=384, num_experts=4, capacity_factor=None, **test_cfg),
                # Test forward_all_experts with tkg: batch_size > self.num_experts
                ExptCfg(seq_len=1, batch_size=8, hidden_size=960, num_experts=4, capacity_factor=None, **test_cfg),
                # ExpertMLPsV2
                ExptCfg(seq_len=1, batch_size=2, hidden_size=384, num_experts=4, capacity_factor=None, use_expert_mlps_v2=True, **test_cfg),
                ExptCfg(seq_len=1, batch_size=8, hidden_size=960, num_experts=4, capacity_factor=None, use_expert_mlps_v2=True, **test_cfg),
            ]
        )

    # Test each configuration on 2 random activation functions
    for test_no, cfg in enumerate(sbase_test_configs):
        for hidden_act in ut.get_random_activations(num=2, seed=test_no):
            test_configs.append(dataclasses.replace(cfg, hidden_act=hidden_act))

    # Mixtral test cases
    # Only fp32 testing with full capacity is supported for mixtral because we havent hacked the golden implementation
    test_cfg = {
        "dtype": torch.float32,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "topk",
        "capacity_factor": None,
    }

    # Training tests
    test_cfg["test_mode"] = "training"
    test_configs.extend(
        [
            ExptCfg(
                seq_len=128,
                batch_size=1,
                hidden_size=384,
                num_experts=2,
                top_k=1,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=2,
                hidden_size=384,
                num_experts=4,
                top_k=2,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=4,
                hidden_size=384,
                num_experts=4,
                top_k=4,
                **test_cfg,
            ),
        ]
    )

    # Inference tests
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            # Test context encoding
            ExptCfg(
                seq_len=128,
                batch_size=1,
                hidden_size=384,
                num_experts=2,
                top_k=1,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=2,
                hidden_size=384,
                num_experts=4,
                top_k=2,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=4,
                hidden_size=384,
                num_experts=4,
                top_k=4,
                **test_cfg,
            ),
            # Test forward_selective_loading with cte: seq_len*top_k < num_experts
            ExptCfg(
                seq_len=32,
                batch_size=1,
                hidden_size=384,
                num_experts=64,
                top_k=1,
                **test_cfg,
            ),
            # Test forward_all_experts with cte: seq_len*top_k < 512
            ExptCfg(
                seq_len=128,
                batch_size=1,
                hidden_size=384,
                num_experts=64,
                top_k=1,
                **test_cfg,
            ),
            # Test forward_blockwise with cte: seq_len*top_k > 512
            ExptCfg(
                seq_len=1024,
                batch_size=1,
                hidden_size=384,
                num_experts=64,
                top_k=1,
                **test_cfg,
            ),
            # Test token generation
            ExptCfg(
                seq_len=1,
                batch_size=1,
                hidden_size=384,
                num_experts=4,
                top_k=2,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=2,
                hidden_size=384,
                num_experts=4,
                top_k=4,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                hidden_size=384,
                num_experts=8,
                top_k=4,
                **test_cfg,
            ),
            # Test forward_all_experts with tkg: batch_size * self.top_k > self.num_experts
            ExptCfg(
                seq_len=128,
                batch_size=4,
                hidden_size=384,
                num_experts=4,
                top_k=4,
                use_expert_mlps_v2=True,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                hidden_size=384,
                num_experts=8,
                top_k=4,
                **test_cfg,
            ),
            # Test forward_selective_loading with tkg: batch_size * self.top_k < self.num_experts
            ExptCfg(
                seq_len=1,
                batch_size=1,
                hidden_size=384,
                num_experts=8,
                top_k=4,
                use_expert_mlps_v2=True,
                **test_cfg,
            ),
            # Test early_expert_affinity
            ExptCfg(
                seq_len=1,
                batch_size=1,
                hidden_size=384,
                num_experts=8,
                top_k=1,
                early_expert_affinity_modulation=True,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                hidden_size=384,
                num_experts=8,
                top_k=1,
                early_expert_affinity_modulation=True,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1024,
                batch_size=1,
                hidden_size=384,
                num_experts=8,
                top_k=1,
                early_expert_affinity_modulation=True,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=1,
                hidden_size=384,
                num_experts=8,
                top_k=1,
                early_expert_affinity_modulation=True,
                **test_cfg,
            ),
         ]
     )

    test_cfg = {
        "dtype": torch.float32,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "llama4",
        "capacity_factor": None,
    }
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            ExptCfg(
                seq_len=1,
                batch_size=1,
                hidden_size=5120,
                intermediate_size=7192,
                num_experts=16,
                top_k=1,
                num_shared_experts=1,
                fused_gate_up_shared_expert=False,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=128,
                batch_size=1,
                hidden_size=5120,
                intermediate_size=7192,
                num_experts=16,
                top_k=1,
                num_shared_experts=1,
                fused_gate_up_shared_expert=False,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=1,
                hidden_size=5120,
                intermediate_size=7192,
                num_experts=16,
                top_k=1,
                num_shared_experts=1,
                moe_fused_tkg_enabled=True,
                **test_cfg,
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                hidden_size=5120,
                intermediate_size=7192,
                num_experts=16,
                top_k=1,
                num_shared_experts=1,
                moe_fused_tkg_enabled=True,
                **test_cfg,
            ),
        ]
    )

    # Add full capacity tests for forward_capacity_factor
    forward_capacity_factor_full_capacity_test_configs = []
    eps = 10**-6
    for cfg in test_configs:
        if cfg.seq_len == 1:
            # Skip for token generation configs
            continue

        if cfg.capacity_factor is None:
            # Set capacity_factor = full_capacity_factor - eps
            full_cf_eps = float(cfg.num_experts / cfg.top_k) - eps
            full_cf_eps_cfg = dataclasses.replace(
                cfg,
                capacity_factor=full_cf_eps,
            )
            forward_capacity_factor_full_capacity_test_configs.append(full_cf_eps_cfg)

    test_configs.extend(forward_capacity_factor_full_capacity_test_configs)

    test_mode_configs = [cfg for cfg in test_configs if cfg.test_mode in test_modes]

    return test_mode_configs


def initialize_neuron_and_golden_models(cfg):
    # Initialize model_neuron (seed and default_dtype are set within this function)
    model_neuron = ut.initialize_neuron_model(cfg)

    # Initialize model_golden (seed and default_dtype are set within this function)
    if cfg.implementation == "sbase":
        model_golden = m_sbase.initialize_sbase_model(cfg)
    elif cfg.implementation == "topk":
        model_golden = m_mixtral.initialize_mixtral_model(cfg)
    elif cfg.implementation == "llama4":
        model_golden = llama4_moe.initialize_llama4_text_moe(cfg)
    else:
        raise ValueError(f"Unknown implementation: {cfg.implementation}")

    # Force model_neuron to have the same weights as model_golden
    missing_keys, unexpected_keys = model_neuron.load_state_dict(
        convert_golden_to_neuron_state_dict(model_golden.state_dict(), cfg)
    )
    assert len(missing_keys) == 0, "missing_keys: %s" % str(missing_keys)
    assert len(unexpected_keys) == 0, "unexpected_keys: %s" % str(unexpected_keys)

    return model_neuron, model_golden


def convert_golden_to_neuron_state_dict(golden_state_dict, cfg):
    if cfg.implementation == "sbase":
        convert_golden_to_neuron_state_dict_func = m_sbase.convert_sbase_to_neuron_state_dict
    elif cfg.implementation == "topk":
        convert_golden_to_neuron_state_dict_func = m_mixtral.convert_mixtral_to_neuron_state_dict
    elif cfg.implementation == "llama4":
        convert_golden_to_neuron_state_dict_func = llama4_moe.convert_llama4_moe_to_neuron_state_dict
    else:
        raise ValueError(f"Unknown implementation: {cfg.implementation}")

    neuron_state_dict = convert_golden_to_neuron_state_dict_func(golden_state_dict, cfg)
    return neuron_state_dict


def get_expected_dropped_token_indices(expert_ind, cfg):
    # Compute the indices of tokens which will be dropped due to exceeding expert_capacity
    # Currently only supports top-1 routing
    assert cfg.top_k == 1
    expert_capacity = ut.get_expert_capacity(cfg)
    expert_counts = {e: 0 for e in range(cfg.num_experts)}
    expected_dropped_token_indices = []
    for token_idx in range(len(expert_ind)):
        expert = expert_ind[token_idx].item()
        expert_counts[expert] += 1
        if expert_counts[expert] > expert_capacity:
            expected_dropped_token_indices.append(token_idx)
    return expected_dropped_token_indices


def _matching_neuron_golden_shared_expert_weights(golden_shared_expert, neuron_shared_expert):
    """
      Match weights between golden and neuron shared experts by copying weights from golden expert to neuron expert.

      Args:
          golden_shared_expert: Source expert model containing the original weights
          neuron_shared_expert: Target expert model where weights will be copied to

      Notes:
          - For fused gate_up projection, concatenates gate_proj and up_proj weights
          - For non-fused version, copies gate_proj and up_proj weights separately
          - Down projection weights are copied directly

      The function handles two cases:
      1. Fused gate-up projection: Concatenates gate and up projection weights
      2. Non-fused: Copies gate and up projection weights separately
      """
    if neuron_shared_expert.fused_gate_up_projection:
        neuron_shared_expert.gate_up_proj.weight.data = torch.cat((golden_shared_expert.gate_proj.weight,
                   golden_shared_expert.up_proj.weight))
    else:
        neuron_shared_expert.gate_proj.weight = golden_shared_expert.gate_proj.weight
        neuron_shared_expert.up_proj.weight = golden_shared_expert.up_proj.weight
    neuron_shared_expert.down_proj.weight = golden_shared_expert.down_proj.weight

def _generate_shared_expert_test_configs():
    test_config1 = {"num_shared_experts": 1,
                    "intermediate_size": 2048,
                    "hidden_size": 1024,
                    "hidden_act": "silu",
                    "seq_len": 1,
                    "fused_gate_up_projection": False,
                    }
    test_config2 = {"num_shared_experts": 2,
                    "intermediate_size": 2048,
                    "hidden_size": 1024,
                    "hidden_act": "silu",
                    "seq_len": 16,
                    "fused_gate_up_projection": True,
                    }
    test_config3 = {"num_shared_experts": 4,
                    "intermediate_size": 2048,
                    "hidden_size": 1024,
                    "hidden_act": "silu",
                    "seq_len": 1,
                    "fused_gate_up_projection": False,
                    }

    test_configs = []
    test_configs.append(test_config1)
    test_configs.append(test_config2)
    test_configs.append(test_config3)
    return test_configs

def _generate_router_test_configs():
    test_config1 = {"bsz": 2,
                    "seq_len": 2,
                    "hidden_size": 4,
                    "n_group": 2,
                    "top_k": 3,
                    "n_routed_experts": 4,
                    "topk_group": 2,
                    "routed_scaling_factor": 2.5
                    }

    test_config2 = {"bsz": 1,
                    "seq_len": 8,
                    "hidden_size": 16,
                    "n_group": 4,
                    "top_k": 1,
                    "n_routed_experts": 8,
                    "topk_group": 2,
                    "routed_scaling_factor": 2.5
                    }

    test_config3 = {"bsz": 1,
                    "seq_len": 16,
                    "hidden_size": 64,
                    "n_group": 8,
                    "top_k": 2,
                    "n_routed_experts": 16,
                    "topk_group": 4,
                    "routed_scaling_factor": 2.5
                    }

    test_config4 = {"bsz": 1,
                    "seq_len": 1,
                    "hidden_size": 7068,
                    "n_group": 8,
                    "top_k": 8,
                    "n_routed_experts": 256,
                    "topk_group": 4,
                    "routed_scaling_factor": 1
                    }

    test_config5 = {"bsz": 1,
                    "seq_len": 16,
                    "hidden_size": 7068,
                    "n_group": 8,
                    "top_k": 8,
                    "n_routed_experts": 256,
                    "topk_group": 4,
                    "routed_scaling_factor": 2.5
                    }

    test_configs = []
    test_configs.append(test_config1)
    test_configs.append(test_config2)
    test_configs.append(test_config3)
    test_configs.append(test_config4)
    test_configs.append(test_config5)
    return test_configs

class TestImplCorrectness(unittest.TestCase):
    @parameterized.expand(
        get_impl_correctness_test_configs(test_modes=["training", "inference"]), name_func=ut.custom_name_func
    )
    def test_fwd_correctness(self, cfg):
        model_neuron, model_golden = initialize_neuron_and_golden_models(cfg)

        model_neuron.return_expert_index = True
        if cfg.implementation == "sbase":
            model_golden.return_expert_index = True

        is_token_gen = cfg.seq_len == 1
        if cfg.test_mode == "inference":
            model_neuron.eval()
            model_golden.eval()
        else:
            model_neuron.train()
            model_golden.train()

        with torch.no_grad():
            for it in range(cfg.num_iters):
                if model_neuron.sequence_dimension == 1:
                    ip = torch.randn(
                        cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device
                    )
                else:
                    ip = torch.randn(
                        cfg.seq_len, cfg.batch_size, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device
                    )

                if cfg.implementation == "topk":
                    # Run fwd on both the Neuron and Mixtral HF model
                    op_neuron, router_logits_neuron, exp_ind_neuron = model_neuron(ip)
                    op_mixtral, router_logits_mixtral = model_golden(ip)

                    # Check that router logits and outputs match
                    ut.check_tensors(
                        router_logits_neuron, router_logits_mixtral, **TEST_TOLS, additional_msg=f"Iteration {it}"
                    )
                    ut.check_tensors(op_neuron, op_mixtral, **TEST_TOLS, additional_msg=f"Iteration {it}")

                elif cfg.implementation == "llama4":
                    # Run fwd on both the Neuron and llama4 HF model
                    op_neuron = model_neuron(ip)[0]
                    op_llama4 = model_golden(ip)[0]
                    # Check that router logits and outputs match
                    ut.check_tensors(op_neuron.view(-1, cfg.hidden_size), op_llama4, **TEST_TOLS, additional_msg=f"Iteration {it}")

                elif cfg.implementation == "sbase":
                    # Run fwd on both the Neuron and S-BASE model
                    op_neuron, _, exp_ind_neuron = model_neuron(ip)
                    op_sbase, _, exp_ind_sbase = model_golden(ip)

                    if cfg.dtype == torch.bfloat16:
                        # Skip this check for token-gen (because perc_discrepancy may be large since S*B is small)
                        if not is_token_gen:
                            # Permit minor discrepancies for bf16
                            perc_discrepancy = 1 - torch.mean(
                                torch.isclose(exp_ind_neuron, exp_ind_sbase, **TEST_TOLS).to(torch.float32)
                            )
                            assert (
                                perc_discrepancy.item() < BF16_EXPERT_ASSIGNMENT_DIFF_TOL
                            ), f" diff is {perc_discrepancy}"
                    else:
                        # Check that the initial expert assignments were identical for fp32
                        ut.check_tensors(
                            exp_ind_neuron, exp_ind_sbase, **TEST_TOLS, additional_msg=f"Iteration {it}"
                        )

                    # Token-gen is dropless
                    if not is_token_gen:
                        # Get the indices of the tokens which should have been dropped by the model_neuron
                        expected_dropped_token_indices = get_expected_dropped_token_indices(exp_ind_neuron, cfg)

                        # Manually simulate the dropping of tokens in op_sbase
                        op_sbase = ut.drop_tokens_in_tensor(op_sbase, expected_dropped_token_indices)

                    if cfg.dtype == torch.bfloat16:
                        # Simulate dropping of tokens in op_neuron and op_sbase where the expert assignments are not matching with neuron
                        expert_mismatch_indices = torch.where(exp_ind_neuron != exp_ind_sbase)[0].tolist()
                        op_sbase = ut.drop_tokens_in_tensor(op_sbase, expert_mismatch_indices)
                        op_neuron = ut.drop_tokens_in_tensor(op_neuron, expert_mismatch_indices)

                    # Check that op_neuron matches the op_sbase with the dropped tokens
                    ut.check_tensors(op_neuron, op_sbase, **TEST_TOLS, additional_msg=f"Iteration {it}")

                else:
                    raise ValueError(f"Unknown implementation: {cfg.implementation}")

    def test_group_limited_router(self):
        test_configs = _generate_router_test_configs()
        for test_config in test_configs:
            bsz = test_config["bsz"]
            seq_len = test_config["seq_len"]
            hidden_size = test_config["hidden_size"]
            n_group = test_config["n_group"]
            top_k = test_config["top_k"]
            n_routed_experts = test_config["n_routed_experts"]
            topk_group = test_config["topk_group"]
            routed_scaling_factor = test_config["routed_scaling_factor"]
            hidden_states = torch.rand((bsz, seq_len, hidden_size))
            norm_topk_prob = True
            torch.manual_seed(42)
            router = GroupLimitedRouter(num_experts=n_routed_experts,
                                       topk_group=topk_group,
                                       top_k=top_k,
                                       dtype=torch.float32,
                                       hidden_size=hidden_size,
                                       n_group=n_group).eval()
            shared_correctional_bias = torch.nn.Parameter(torch.zeros((n_routed_experts)), requires_grad=False)
            router.e_score_correction_bias = shared_correctional_bias
            router_logits, expert_weights, topk_idx = router(hidden_states)
            config = SimpleNamespace(
                n_routed_experts=n_routed_experts,
                topk_group=topk_group,
                num_experts_per_tok=top_k,
                hidden_size=hidden_size,
                n_group=n_group,
                norm_topk_prob=norm_topk_prob,
                dtype=torch.float32,
                neuron_config=SimpleNamespace(torch_dtype=torch.float32, rpl_reduce_dtype=torch.float32),
                routed_scaling_factor=routed_scaling_factor,
                scoring_func="sigmoid",
                seq_aux="",
                topk_method="noaux_tc",
            )
            hf_router = DeepSeekV3HFMoEGate(config=config).eval()
            hf_router.weight = router.linear_router.weight
            hf_router.e_score_correction_bias = shared_correctional_bias
            hf_router.training = False
            hf_topk_idx, hf_expert_weights = hf_router(hidden_states)
            # NxD's expert implementation expects full expert weights output, so the operation below is to match HF's
            # router implementation
            expert_weights_topk = expert_weights.gather(1, topk_idx)
            if norm_topk_prob and top_k > 1:
                denominator = expert_weights_topk.sum(dim=-1, keepdim=True) + 1e-20
                expert_weights_topk = expert_weights_topk / denominator
            expert_weights_topk *= config.routed_scaling_factor
            # HF expert weights are not sorted
            hf_expert_weights_sorted, _ = torch.sort(hf_expert_weights, descending=True)
            torch_neuronx.testing.assert_close(expert_weights_topk, hf_expert_weights_sorted)

    @parameterized.expand(get_impl_correctness_test_configs(test_modes=["training"]), name_func=ut.custom_name_func)
    def test_bwd_correctness(self, cfg):
        model_neuron, model_golden = initialize_neuron_and_golden_models(cfg)

        model_neuron.return_expert_index = True
        if cfg.implementation == "sbase":
            model_golden.return_expert_index = True

        # Set models to train mode
        model_neuron.train()
        model_golden.train()

        optimizer_neuron = torch.optim.Adadelta(model_neuron.parameters())
        optimizer_golden = torch.optim.Adadelta(model_golden.parameters())
        mse_loss = torch.nn.MSELoss()

        for it in range(cfg.num_iters):
            # Generate random input tensor
            if model_neuron.sequence_dimension == 1:
                ip = torch.randn(
                    cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device
                )
            else:
                ip = torch.randn(
                    cfg.seq_len, cfg.batch_size, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device
                )

            if cfg.dtype == torch.bfloat16:
                # Simulate dropping of tokens in input where the expert assignments are not matching with neuron
                assert cfg.implementation == "sbase"
                with torch.no_grad():
                    op_neuron, _, exp_ind_neuron = model_neuron(ip)
                    op_sbase, _, exp_ind_sbase = model_golden(ip)
                    expert_mismatch_indices = torch.where(exp_ind_neuron != exp_ind_sbase)[0].tolist()
                    ip = ut.drop_tokens_in_tensor(ip, expert_mismatch_indices)

            # Run forward pass on model_neuron
            op_neuron, _, exp_ind_neuron = model_neuron(ip)

            if cfg.implementation == "sbase":
                # Get the indices of the tokens which should have been dropped by the model_neuron
                expected_dropped_token_indices = get_expected_dropped_token_indices(exp_ind_neuron, cfg)
                # Manually simulate the dropping of tokens in the input passed
                ip = ut.drop_tokens_in_tensor(ip.clone().detach(), expected_dropped_token_indices)

            # Run forward pass on model_golden
            if cfg.implementation == "sbase":
                op_golden, _, _ = model_golden(ip)
            elif cfg.implementation == "topk":
                op_golden, _ = model_golden(ip)
            else:
                raise ValueError(f"Unknown implementation: {cfg.implementation}")

            # Compute MSE loss wrt which we get the gradients
            targets = torch.zeros_like(ip, device=cfg.device, dtype=torch.float32)
            loss_neuron = mse_loss(op_neuron.to(torch.float32), targets)
            loss_golden = mse_loss(op_golden.to(torch.float32), targets)
            ut.check_tensors(loss_neuron, loss_golden, **TEST_TOLS, additional_msg=f"Iteration {it}")

            # Run backward pass to compute gradients
            loss_neuron.backward()
            loss_golden.backward()

            # Compare gradients
            grads_neuron = ut.get_model_grads_dict(model_neuron)
            grads_golden = convert_golden_to_neuron_state_dict(ut.get_model_grads_dict(model_golden), cfg=cfg)
            assert set(grads_neuron.keys()) == set(grads_golden.keys())
            for key in grads_neuron:
                ut.check_tensors(
                    grads_neuron[key], grads_golden[key], **TEST_TOLS, additional_msg=f"Iteration: {it}, key: {key}"
                )

            # Zero out gradients before next iteration
            optimizer_neuron.zero_grad()
            optimizer_golden.zero_grad()

    @parameterized.expand(
        lch.get_loss_fn_correctness_test_configs(dtypes=[torch.bfloat16, torch.float32]), name_func=ut.custom_name_func
    )
    def test_loss_fn_correctness(self, cfg):
        # Set random seed for reproducibility
        torch.manual_seed(cfg.num_experts)
        with torch.no_grad():
            for it in range(cfg.num_iters):
                test_gate_logits = [
                    torch.randn(cfg.batch_size * cfg.seq_len, cfg.num_experts, device=cfg.device, dtype=cfg.dtype)
                    for _ in range(cfg.num_layers)
                ]
                test_gate_logits = tuple(test_gate_logits)
                hf_loss = lch.hf_load_balancing_loss_func(test_gate_logits, cfg.num_experts, cfg.top_k)
                concatenated_test_gate_logits = torch.cat([layer_gate for layer_gate in test_gate_logits], dim=0)
                neuron_loss = neuron_load_balancing_loss_func(
                    concatenated_test_gate_logits, cfg.num_experts, cfg.top_k
                )
                assert neuron_loss.dtype == hf_loss.dtype
                test_tols = lch.FP32_TEST_TOLS if cfg.dtype == torch.float32 else lch.BF16_TEST_TOLS
                ut.check_tensors(neuron_loss, hf_loss, **test_tols, additional_msg=f"Iteration {it}")

    def test_shared_expert_correctness(self):
        """
        Test the correctness of shared expert implementation by comparing outputs
        between golden and neuron models across different configurations.

        The test:
        1. Generates various test configurations
        2. Creates both golden and neuron models for each config
        3. Matches weights between models
        4. Compares output results
        """

        def _create_golden_model(config):
            torch.manual_seed(42)
            model = Llama4TextMLP(config, config.intermediate_size * config.num_shared_experts)
            model.eval()
            return model

        def _create_neuron_model(test_config):
            model = SharedExperts(
                hidden_size=test_config["hidden_size"],
                intermediate_size=test_config["intermediate_size"],
                num_shared_experts=test_config["num_shared_experts"],
                fused_gate_up_projection=test_config["fused_gate_up_projection"],
                hidden_act=test_config["hidden_act"],
            )
            model.eval()
            return model

        def _create_config(test_config):
            return SimpleNamespace(
                num_shared_experts=test_config["num_shared_experts"],
                intermediate_size=test_config["intermediate_size"],
                hidden_size=test_config["hidden_size"],
                hidden_act=test_config["hidden_act"],
            )

        # Test each configuration
        test_configs = _generate_shared_expert_test_configs()
        for test_config in test_configs:
            # Create configuration object
            config = _create_config(test_config)

            # Initialize models
            golden_shared_expert = _create_golden_model(config)
            neuron_shared_expert = _create_neuron_model(test_config)

            # Generate input data
            hidden_states = torch.rand((1, test_config["seq_len"], test_config["hidden_size"]))

            # Match weights between models
            _matching_neuron_golden_shared_expert_weights(golden_shared_expert, neuron_shared_expert)

            # Compare outputs
            golden_output = golden_shared_expert(hidden_states)
            neuron_output = neuron_shared_expert(hidden_states, test_config["seq_len"])
            torch_neuronx.testing.assert_close(golden_output, neuron_output)



if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
