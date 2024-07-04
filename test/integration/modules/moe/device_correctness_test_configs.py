import dataclasses
import itertools

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
from utils_testing import (
    ExptCfgCorrectness,
    filter_valid_expt_configs,
    get_random_activations,
)

from neuronx_distributed.modules.moe import MoESequenceParallelMode


@dataclasses.dataclass
class ExptCfgDeviceCorrectness(ExptCfgCorrectness):
    test_mode: str = "training"


def get_device_correctness_test_configs(dtype):
    GLU_MLP_ARGS = [True, False]
    PERMUTE_STRATEGY_ARGS = ["matmul", "index"]

    test_configs = []

    # S-BASE test cases
    sbase_test_configs = []
    for glu_mlp in GLU_MLP_ARGS:
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": glu_mlp,
            "implementation": "sbase",
            "expert_mlps_permute_strategy": "index",
        }

        # Test forward_full_capacity and token-gen
        sbase_test_configs.extend(
            [
                # Training / Context-encoding
                ExptCfgDeviceCorrectness(
                    seq_len=256,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=2560,
                    num_experts=4,
                    capacity_factor=4.0,
                    **test_cfg
                ),
                # Token-generation
                ExptCfgDeviceCorrectness(
                    seq_len=1,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=2560,
                    num_experts=24,
                    capacity_factor=1.0,
                    test_mode="inference",
                    **test_cfg
                ),
                ExptCfgDeviceCorrectness(
                    seq_len=1,
                    batch_size=4,
                    hidden_size=1024,
                    intermediate_size=2560,
                    num_experts=24,
                    capacity_factor=1.0,
                    test_mode="inference",
                    **test_cfg
                ),
            ]
        )

        for permute_strategy in PERMUTE_STRATEGY_ARGS:
            test_cfg["expert_mlps_permute_strategy"] = permute_strategy
            sbase_test_configs.extend(
                [
                    # Training / Context-encoding
                    # capacity_factor such that some tokens may be dropped
                    ExptCfgDeviceCorrectness(
                        seq_len=256,
                        batch_size=1,
                        hidden_size=1024,
                        intermediate_size=2560,
                        num_experts=24,
                        capacity_factor=2.0,
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
        "expert_mlps_permute_strategy": "index",
    }
    # Test forward_full_capacity and token-gen
    test_configs.extend(
        [
            # Training / Context-encoding
            # capacity_factor = num_experts/top_k to ensure no dropped tokens
            ExptCfgDeviceCorrectness(
                seq_len=256,
                batch_size=1,
                hidden_size=1024,
                intermediate_size=3584,
                num_experts=8,
                top_k=2,
                capacity_factor=4.0,
                **test_cfg
            ),
            # Token-generation
            ExptCfgDeviceCorrectness(
                seq_len=1,
                batch_size=1,
                hidden_size=1024,
                intermediate_size=3584,
                num_experts=8,
                capacity_factor=1.0,
                top_k=2,
                test_mode="inference",
                **test_cfg
            ),
            ExptCfgDeviceCorrectness(
                seq_len=1,
                batch_size=4,
                hidden_size=768,
                intermediate_size=2688,
                num_experts=16,
                capacity_factor=1.0,
                top_k=4,
                test_mode="inference",
                **test_cfg
            ),
        ]
    )

    for permute_strategy in PERMUTE_STRATEGY_ARGS:
        test_cfg["expert_mlps_permute_strategy"] = permute_strategy
        test_configs.extend(
            [
                # Training / Context-encoding
                # capacity_factor such that some tokens may be dropped
                ExptCfgDeviceCorrectness(
                    seq_len=256,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=3584,
                    num_experts=8,
                    top_k=2,
                    capacity_factor=2.0,
                    **test_cfg
                ),
                ExptCfgDeviceCorrectness(
                    seq_len=256,
                    batch_size=1,
                    hidden_size=1536,
                    intermediate_size=2688,
                    num_experts=16,
                    top_k=4,
                    capacity_factor=2.0,
                    **test_cfg
                ),
            ]
        )

    return filter_valid_expt_configs(test_configs)


@dataclasses.dataclass
class ExptCfgParallel(ExptCfgDeviceCorrectness):
    # Default values must be over-ridden
    tp_degree: int = 0
    ep_degree: int = 0
    sequence_parallel_mode: MoESequenceParallelMode = -1


def get_device_correctness_parallel_test_configs(dtype, tp_degree, sp_mode):
    GLU_MLP_ARGS = [True, False]
    PERMUTE_STRATEGY_ARGS = ["matmul", "index"]

    test_configs = []

    # S-BASE test cases
    # All test cases use "silu" since other activations are tested in the single-core test
    for test_no, (glu_mlp, permute_strategy) in enumerate(itertools.product(GLU_MLP_ARGS, PERMUTE_STRATEGY_ARGS)):
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": glu_mlp,
            "hidden_act": "silu",
            "implementation": "sbase",
            "expert_mlps_permute_strategy": permute_strategy,
            "num_iters": 5,
        }
        test_configs.extend(
            [
                # Training / Context-encoding
                ExptCfgParallel(
                    seq_len=256, 
                    batch_size=1, 
                    hidden_size=1024, 
                    intermediate_size=2560, 
                    num_experts=24, 
                    capacity_factor=2.0, 
                    **test_cfg
                ),
                # Token-generation
                ExptCfgParallel(
                    seq_len=1,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=2560,
                    num_experts=24,
                    capacity_factor=1.0,
                    test_mode="inference",
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=1,
                    batch_size=4,
                    hidden_size=1024,
                    intermediate_size=2560,
                    num_experts=24,
                    capacity_factor=1.0,
                    test_mode="inference",
                    **test_cfg
                ),
            ]
        )

    # TopK test cases
    for permute_strategy in PERMUTE_STRATEGY_ARGS:
        test_cfg = {
            "dtype": dtype,
            "glu_mlp": True,
            "hidden_act": "silu",
            "implementation": "topk",
            "expert_mlps_permute_strategy": permute_strategy,
            "num_iters": 5,
        }
        test_configs.extend(
            [
                # Training / Context-encoding
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=3584,
                    num_experts=8,
                    top_k=2,
                    capacity_factor=2.0,
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=256,
                    batch_size=1,
                    hidden_size=1536,
                    intermediate_size=2688,
                    num_experts=16,
                    top_k=4,
                    capacity_factor=2.0,
                    **test_cfg
                ),
                # Token-generation
                ExptCfgParallel(
                    seq_len=1,
                    batch_size=1,
                    hidden_size=1024,
                    intermediate_size=3584,
                    num_experts=8,
                    capacity_factor=1.0,
                    top_k=2,
                    test_mode="inference",
                    **test_cfg
                ),
                ExptCfgParallel(
                    seq_len=1,
                    batch_size=4,
                    hidden_size=768,
                    intermediate_size=2688,
                    num_experts=16,
                    capacity_factor=1.0,
                    top_k=4,
                    test_mode="inference",
                    **test_cfg
                ),
            ]
        )

    # Add TP degree, SP mode to config
    test_configs_parallel = []
    for cfg in test_configs:
        cfg_parallel = dataclasses.replace(
            cfg, tp_degree=tp_degree, ep_degree=1, sequence_parallel_mode=sp_mode
        )
        test_configs_parallel.append(cfg_parallel)

    return filter_valid_expt_configs(test_configs_parallel)
