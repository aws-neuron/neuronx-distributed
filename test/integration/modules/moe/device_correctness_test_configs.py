import dataclasses
from typing import List
import torch
from neuronxcc.nki._private_kernels.blockwise_mm import BlockShardStrategy, SkipMode

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
    "llama4-m": {
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "num_experts": 128,
        "top_k": 1,
        "num_shared_experts": 1
    },
    "llama4-s": {
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "num_experts": 16,
        "top_k": 1,
        "num_shared_experts": 1
    }
}


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

        # Inference tests
        test_cfg["test_mode"] = "inference"
        sbase_test_configs.extend(
            [
                # Context encoding
                ExptCfg(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=None,
                     **get_model_config("sbase-small", scale_down_factor=4),
                    **test_cfg
                ),
                # Token-generation
                ExptCfg(
                    seq_len=1,
                    batch_size=1,
                    capacity_factor=None,
                    **get_model_config("sbase-large", scale_down_factor=8),
                    **test_cfg
                ),
                ExptCfg(
                    seq_len=1,
                    batch_size=4,
                    capacity_factor=None,
                    **get_model_config("sbase-large", scale_down_factor=8),
                    **test_cfg
                ),
            ]
        )

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
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("dbrx", scale_down_factor=4),
                **test_cfg
            ),
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

    # Inference tests
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            # Context-encoding
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("mixtral", scale_down_factor=4),
                **test_cfg
            ),
            # Context-encoding for blockwise
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                num_iters=2, # increase n_iter leads to OOM
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfg(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("dbrx", scale_down_factor=4),
                **test_cfg
            ),
            # Token-generation
            ExptCfg(
                seq_len=1,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("mixtral", scale_down_factor=4),
                **test_cfg
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                capacity_factor=None,
                **get_model_config("dbrx", scale_down_factor=4),
                **test_cfg
            ),
            #shared_experts
            ExptCfg(
                seq_len=128,
                batch_size=1,
                capacity_factor=None,
                num_shared_experts=1,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
        ]
    )

    return test_configs


@dataclasses.dataclass
class ExptCfgParallel(ExptCfg):
    # Default values must be over-ridden
    tp_degree: int = 0
    ep_degree: int = 0
    token_shuffle_group_size: int = 1
    sequence_parallel_enabled: bool = False


def get_device_correctness_parallel_test_configs(dtype, test_mode, tp_degree, ep_degree, token_shuffle_group_size, zero1):
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

    # Inference tests
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            # Context-encoding
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("sbase-large", scale_down_factor=4),
                **test_cfg
            ),
            # Token-generation
            ExptCfgParallel(
                seq_len=1,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("sbase-large", scale_down_factor=4),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1,
                batch_size=4,
                capacity_factor=None,
                **get_model_config("sbase-large", scale_down_factor=4),
                **test_cfg
            ),
        ]
    )

    if get_platform_lnc() == LogicalNCConfig.LNC_2:
        test_configs.extend(
            [
                # non block parallel basic
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=128,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.PING_PONG,
                    **get_model_config("sbase-large", scale_down_factor=4),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=128,
                    block_sharding_strategy=BlockShardStrategy.HI_LO,
                    **get_model_config("sbase-large", scale_down_factor=4),
                    **test_cfg
                ),
                # block parallel with different block size
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=512,
                    use_block_parallel=True,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.PING_PONG,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=512,
                    use_block_parallel=True,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.HI_LO,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
                # block size 256
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=256,
                    use_block_parallel=True,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.PING_PONG,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=256,
                    use_block_parallel=True,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.HI_LO,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
                # block_size 128 failed with sequence parallel on bf16 trn2
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=128,
                    use_block_parallel=True,
                    use_expert_mlps_v2=True,
                    block_sharding_strategy=BlockShardStrategy.HI_LO,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=512,
                    batch_size=1,
                    capacity_factor=None,
                    block_size=128,
                    use_torch_block_wise=True,
                    use_expert_mlps_v2=True,
                    **get_model_config("sbase-large", scale_down_factor=2),
                    **test_cfg
                ),
            ]
        )

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

    # Inference tests
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            #Context-encoding
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                skip_dma=SkipMode(True, True),
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                skip_dma=SkipMode(True, False),
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                skip_dma=SkipMode(True, False),
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                skip_dma=SkipMode(False, True),
                always_augment_inputs_for_blockwise_matmul=True,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                use_expert_mlps_v2=True,
                skip_dma=SkipMode(True, True),
                always_augment_inputs_for_blockwise_matmul=True,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                # using torch blockwise matmul
                use_torch_block_wise=True,
                # Testing without parallelizing token to block mapping
                parallelize_token_to_block_mapping=False,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                use_torch_block_wise=True,
                parallelize_token_to_block_mapping=False,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),
            # test early_expert_affinity_modulation
            # fix: this is failing on Mismatched elements: 2082 / 16384 (12.7%) with kernel, but pass with torch blockwise
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                early_expert_affinity_modulation=True,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                early_expert_affinity_modulation=True,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                early_expert_affinity_modulation=True,
                use_torch_block_wise=True,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=256,
                batch_size=1,
                capacity_factor=None,
                parallelize_token_to_block_mapping=False,
                early_expert_affinity_modulation=True,
                use_torch_block_wise=True,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),

            # Token-generation
            ExptCfgParallel(
                seq_len=1,
                batch_size=1,
                capacity_factor=None,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1,
                batch_size=4,
                capacity_factor=None,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=1,
                batch_size=1,
                capacity_factor=None,
                early_expert_affinity_modulation=True,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            # fix: this is failing on Mismatched elements: 2 / 12288 (0.0%)
            ExptCfgParallel(
                seq_len=1,
                batch_size=4,
                capacity_factor=None,
                early_expert_affinity_modulation=True,
                **get_model_config("dbrx", scale_down_factor=2),
                **test_cfg
            ),

            ExptCfgParallel(
                seq_len=128,
                batch_size=1,
                capacity_factor=None,
                num_shared_experts=1,
                **get_model_config("mixtral", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=128,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("llama4-m", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("llama4-m", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=128,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("llama4-s", scale_down_factor=1),
                **test_cfg
            ),
            ExptCfgParallel(
                seq_len=512,
                batch_size=1,
                capacity_factor=None,
                fused_gate_up_shared_expert=True,
                **get_model_config("llama4-s", scale_down_factor=1),
                **test_cfg
            ),
        ]
    )

    # Add tp_degree, sequence_parallel_enabled to config
    test_configs_parallel = []
    for cfg in test_configs:
        # Filter to required test_mode
        if cfg.test_mode != test_mode:
            continue

        # EP + token-gen not supported
        if ep_degree > 1 and cfg.seq_len == 1 and cfg.test_mode == "inference":
            continue

        # Run training test in SP, test both with and without SP for inference
        if test_mode == "training":
            sp_modes = [True]
        else:
            sp_modes = [True, False]

        for sp_mode in sp_modes:
            if cfg.seq_len == 1 and sp_mode:
                # Token gen cannot be run in SP
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
