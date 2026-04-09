import dataclasses
from typing import List
import torch
import torch_xla.runtime as xr  # TRN enablement
from nkilib.core.moe.moe_cte.moe_cte_utils import BlockShardStrategy, SkipMode

from neuronx_distributed.utils.model_utils import get_platform_lnc, LogicalNCConfig
# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
from utils_testing import ExptCfg, get_random_activations

GLU_MLP_ARGS = [True, False]


TEST_MODEL_CONFIGS = {
    "sbase-small": {
        "hidden_size": 4096,
        "intermediate_size": 10944,
        "num_experts": 4,
        "top_k": 1,
    },
    "sbase-large": {
        "hidden_size": 8192,
        "intermediate_size": 20480,
        "num_experts": 24,
        "top_k": 1,
    },
    "mixtral": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_experts": 8,
        "top_k": 2,
    },
    "dbrx": {
        "hidden_size": 6144,
        "intermediate_size": 10752,
        "num_experts": 16,
        "top_k": 4,
    },
    # not full name due to ip scanner requirement
    "llama4-400b": {
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "num_experts": 128,
        "top_k": 1,
        "num_shared_experts": 1
    },
    "llama4-100b": {
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "num_experts": 16,
        "top_k": 1,
        "num_shared_experts": 1
    },
    "ds-v3-reduced-experts": {
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 64,
        "top_k": 8,
        "num_shared_experts": 1
    },
    "ds-v3-reduced-experts-inference": {
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 64,
        "num_shared_experts": 1,
        "top_k": 4,
        "enable_spmd_rank": True,
        "optimized_block_to_token_mapping": False,
        "parallelize_token_to_block_mapping": False,
    },
    "deepseek_V3": {
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 64,
        "top_k": 4,
        "enable_spmd_rank": True,
        "optimized_block_to_token_mapping": False,
        "parallelize_token_to_block_mapping": False,
    },
    "qwen3-reduced-experts": {
        "hidden_size": 4096,
        "intermediate_size": 1536,
        "num_experts": 64,
        "num_shared_experts": 1,
        "top_k": 4,
        "enable_spmd_rank": True,
        "optimized_block_to_token_mapping": False,
        "parallelize_token_to_block_mapping": False,
    },
    "qwen3-experts-ep": {
        "hidden_size": 4096,
        "intermediate_size": 1536,
        "num_experts": 128,
        "num_shared_experts": 0,
        "top_k": 4,
        "enable_spmd_rank": True,
        "optimized_block_to_token_mapping": False,
        "parallelize_token_to_block_mapping": False,
    }
}

def get_device_correctness_test_configs_EP(dtype) -> List[ExptCfg]:
    test_configs = []

    test_cfg = {
        "dtype": dtype,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "topk",
        "num_iters": 1,
    }
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            # Context-encoding
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("deepseek_V3", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("ds-v3-reduced-experts-inference", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("qwen3-reduced-experts", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("ds-v3-reduced-experts-inference", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("qwen3-reduced-experts", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1024,
                batch_size=1,
                capacity_factor=None,
                hidden_size=4096,
                intermediate_size=1024,
                num_experts=128,
                top_k=8,
                num_shared_experts=0,
                enable_spmd_rank=True,
                use_index_calc_kernel=True,
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=128,
                batch_size=1,
                capacity_factor=None,
                hidden_size=4096,
                intermediate_size=1024,
                num_experts=128,
                top_k=8,
                num_shared_experts=0,
                enable_spmd_rank=True,
                use_index_calc_kernel=True,
                **test_cfg
            ),
        ])

    return test_configs

def get_model_config(model_name, scale_down_factor=1):

    assert model_name in TEST_MODEL_CONFIGS
    config_dict = TEST_MODEL_CONFIGS[model_name].copy()
    config_dict.update({
        "hidden_size": int(config_dict["hidden_size"] / scale_down_factor),
        "intermediate_size": int(config_dict["intermediate_size"] / scale_down_factor),
    })
    return config_dict


def get_neuron_cc_flags(test_dtype):
    cc_flags = [
        "--model-type=transformer",
        "--enable-saturate-infinity",  #clip matmul transpose input to [-MAX, MAX] to avoid nans (0*INF)
        "--retry_failed_compilation",
    ]
    if test_dtype == torch.float32:
        # Disable auto-casting
        cc_flags.append("--auto-cast=none")
    return " ".join(cc_flags)


def get_device_correctness_test_configs(dtype) -> List[ExptCfg]:
    """
    Get device correctness test configs.
    
    Note: Inference tests using beta1 blockwise kernels (blockwise_mm_baseline, 
    blockwise_mm_baseline_shard_hidden) have been removed. Only training tests remain.
    """
    test_configs = []

    # S-BASE test cases
    sbase_test_configs = []
    for glu_mlp in GLU_MLP_ARGS:
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": glu_mlp,
            "num_iters": 25,
            "implementation": "sbase",
        }

        # Training tests
        test_cfg["test_mode"] = "training"
        sbase_test_configs.extend(
            [
                # Test forward_all_experts (full capacity)
                ExptCfg(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=None,
                    **get_model_config("sbase-small", scale_down_factor=4),
                    **test_cfg
                ),
                # Test forward_capacity_factor
                ExptCfg(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=2.0,
                    **get_model_config("sbase-large", scale_down_factor=8),
                    **test_cfg
                ),
            ]
        )

        # Inference tests removed - used beta1 blockwise_mm_baseline_shard_hidden kernel

    # Test each S-BASE configuration on 2 random activation functions
    for test_no, cfg in enumerate(sbase_test_configs):
        for hidden_act in get_random_activations(num=2, seed=test_no):
            test_configs.append(dataclasses.replace(cfg, hidden_act=hidden_act))

    # TopK test cases
    test_cfg = {
        "dtype": dtype,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "topk",
    }

    # Training tests
    test_cfg["test_mode"] = "training"
    test_configs.extend(
        [
            # Test forward_all_experts (full capacity)
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("mixtral", scale_down_factor=4),
                **test_cfg
            ),
            # This configuration fails for BF16 on TRN2 but passes for FP32,  and passes BF16 on TRN1
            # ExptCfg(
            #     seq_len=256,
            #     batch_size=1,
            #     capacity_factor=None,
            #     **get_model_config("dbrx", scale_down_factor=4),
            #     **test_cfg
            # ),
            # Test forward_capacity_factor
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=2.0,
                **get_model_config("mixtral", scale_down_factor=4),
                **test_cfg
            ),
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=2.0,
                **get_model_config("dbrx", scale_down_factor=4),
                **test_cfg
            ),
        ]
    )
    # This configurations fails for BF16 on TRN2 but passes for FP32,  and passes BF16 on TRN1 
    if get_platform_lnc() == LogicalNCConfig.LNC_1:
        test_configs.append(
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("dbrx", scale_down_factor=4),
                **test_cfg
            )
        )

    # Inference tests removed - used beta1 blockwise_mm_baseline_shard_hidden kernel

    return test_configs


@dataclasses.dataclass
class ExptCfgParallel(ExptCfg):
    # Default values must be over-ridden
    tp_degree: int = 0
    ep_degree: int = 0
    token_shuffle_group_size: int = 1
    sequence_parallel_enabled: bool = False


def get_device_correctness_parallel_test_configs(dtype, test_mode, tp_degree, ep_degree, token_shuffle_group_size, zero1, test_modules=None):
    """
    Get device correctness parallel test configs.
    
    Note: Inference tests using beta1 blockwise kernels (blockwise_mm_baseline, 
    blockwise_mm_baseline_shard_hidden, blockwise_mm_baseline_block_parallel) have been removed.
    Only training tests remain.
    """
    assert test_mode in {"training", "inference"}

    test_configs = []

    # S-BASE test cases

    # All test cases use glu_mlp = True (glu_mlp = False tested in single-core test)
    # All test cases use "silu" (other activations tested in the single-core test)
    test_cfg = {
        "dtype": dtype,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "sbase",
        "num_iters": 1,
        "zero1": zero1,
    }

    # Training tests
    test_cfg["test_mode"] = "training"
    test_configs.extend(
        [
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=2.0,
                **get_model_config("sbase-large", scale_down_factor=8),
                **test_cfg
            ),
        ]
    )

    # Inference tests removed - used beta1 blockwise_mm_baseline_shard_hidden kernel

    # TopK test cases
    test_cfg = {
        "dtype": dtype,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "topk",
        "num_iters": 1,
        "zero1": zero1,
    }

    # Training tests
    test_cfg["test_mode"] = "training"
    test_configs.extend(
        [
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=2.0,
                **get_model_config("mixtral", scale_down_factor=8),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=2.0,
                **get_model_config("dbrx", scale_down_factor=8),
                **test_cfg
            ),
        ]
    )

    # Inference tests removed - used beta1 blockwise_mm_baseline_shard_hidden and 
    # blockwise_mm_baseline_block_parallel kernels
    
    # Training with blockwise kernel tests for TRN2 with LNC2
    if get_platform_lnc() == LogicalNCConfig.LNC_2:
        # Llama4 and Dsv3 test cases - reuses same implementation as llama4
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": True,
            "hidden_act": "silu",
            "implementation": "llama4",
            "num_iters": 1,
            "zero1": zero1,
        }
        test_cfg["test_mode"] = "training"
        test_configs.extend(
            [
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=0.0,
                    **get_model_config("ds-v3-reduced-experts", scale_down_factor=1),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=0.0,
                    fused_gate_up_shared_expert=True,
                    **get_model_config("ds-v3-reduced-experts", scale_down_factor=1),
                    **test_cfg
                ),
                # run_shared_expert_in_sequence_parallel tests are flaky and failing non-deterministically
                # ExptCfgParallel(
                #     seq_len=1024,
                #     batch_size=1,
                #     capacity_factor=0.0,
                #     fused_gate_up_shared_expert=True,
                #     run_shared_expert_in_sequence_parallel=True,
                #     enable_spmd_rank_shared_expert=True,
                #     **get_model_config("ds-v3-reduced-experts", scale_down_factor=1),
                #     **test_cfg
                # ),
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=0.0,
                    **get_model_config("llama4-100b", scale_down_factor=1),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=0.0,
                    fused_gate_up_shared_expert=True,
                    **get_model_config("llama4-100b", scale_down_factor=1),
                    **test_cfg
                ),
                # run_shared_expert_in_sequence_parallel tests are flaky and failing non-deterministically
                # ExptCfgParallel(
                #     seq_len=1024,
                #     batch_size=1,
                #     capacity_factor=0.0,
                #     run_shared_expert_in_sequence_parallel=True,
                #     enable_spmd_rank_shared_expert=True,
                #     **get_model_config("llama4-100b", scale_down_factor=1),
                #     **test_cfg
                # ),
            ]
        )

    # Inference tests removed - used beta1 blockwise_mm_baseline_shard_hidden kernel

    # Add tp_degree, sequence_parallel_enabled to config
    test_configs_parallel = []
    if ep_degree > 1 and test_mode == "inference":
        dtype = torch.bfloat16
        EP_test_config = get_device_correctness_test_configs_EP(dtype)
        for cfg in EP_test_config:
            cfg_parallel = dataclasses.replace(
                    cfg,
                    tp_degree=tp_degree,
                    ep_degree=ep_degree,
                    token_shuffle_group_size=token_shuffle_group_size,
                    sequence_parallel_enabled=False,
                )
            test_configs_parallel.append(cfg_parallel)
    else:
        for cfg in test_configs:
            # Filter to required test_mode
            if cfg.test_mode != test_mode:
                continue
            if test_modules:
                if cfg.test_module not in test_modules:
                    continue
            # EP + token-gen not supported
            if ep_degree > 1 and cfg.seq_len == 1 and cfg.test_mode == "inference":
                continue

            # Run training test in SP, test both with and without SP for inference
            if test_mode == "training":
                sp_modes = [True]
            else:
                if get_platform_lnc() == LogicalNCConfig.LNC_2:
                    sp_modes = [False]
                else:
                    sp_modes = [True, False]

            for sp_mode in sp_modes:
                if cfg.seq_len == 1 and sp_mode:
                    # Token gen cannot be run in SP
                    continue
                if cfg.shared_experts_sequence_parallel_enabled and not sp_mode and cfg.seq_len > 1:
                    # Running shared expert in SP for context encoding is not available when input is not in SP
                    continue
                cfg_parallel = dataclasses.replace(
                    cfg,
                    tp_degree=tp_degree,
                    ep_degree=ep_degree,
                    token_shuffle_group_size=token_shuffle_group_size,
                    sequence_parallel_enabled=sp_mode,
                )
                test_configs_parallel.append(cfg_parallel)


    return test_configs_parallel

def get_device_correctness_parallel_test_configs_EP(dtype, test_mode, tp_degree, ep_degree, token_shuffle_group_size, zero1, test_modules=None):	
    assert test_mode in {"training", "inference"}	
    test_configs = []	
    if get_platform_lnc() == LogicalNCConfig.LNC_2:	
        test_cfg = {	
            "dtype": dtype,	
            "glu_mlp": True,	
            "hidden_act": "silu",	
            "implementation": "llama4",	
            "num_iters": 1,	
        }	
        	
        test_cfg["test_mode"] = "training"	
        	
        test_configs.extend(	
            [	
                ExptCfgParallel(	
                    seq_len=1024,	
                    batch_size=1,	
                    capacity_factor=0.0,	
                    fused_gate_up_shared_expert=False,	
                    **get_model_config("qwen3-experts-ep", scale_down_factor=1),	
                    **test_cfg	
                ),	
                	
            ]	
        )	
        	
    # Add tp_degree, sequence_parallel_enabled to config	
    test_configs_parallel = []	
    if test_mode == "training":
        for cfg in test_configs:			
            sp_modes = [True]	
            for sp_mode in sp_modes:	
                if cfg.seq_len == 1 and sp_mode:	
                    # Token gen cannot be run in SP	
                    continue	
                if cfg.shared_experts_sequence_parallel_enabled and not sp_mode and cfg.seq_len > 1:	
                    # Running shared expert in SP for context encoding is not available when input is not in SP	
                    continue	
                cfg_parallel = dataclasses.replace(	
                    cfg,	
                    tp_degree=tp_degree,	
                    ep_degree=ep_degree,	
                    token_shuffle_group_size=token_shuffle_group_size,	
                    sequence_parallel_enabled=sp_mode,	
                )	
                test_configs_parallel.append(cfg_parallel)	
    return test_configs_parallel
