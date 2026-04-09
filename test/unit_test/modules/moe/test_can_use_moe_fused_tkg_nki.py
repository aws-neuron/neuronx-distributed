import dataclasses
import os
from typing import Optional
import unittest

import torch
from neuronx_distributed import parallel_layers

from . import utils_testing as ut


if not torch.distributed.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
    parallel_layers.parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    parallel_layers.parallel_state.initialize_token_shuffle_group(1)

@dataclasses.dataclass
class TestCfg(ut.ExptCfg):
    # Default values must be over-ridden
    test_kernel: Optional[str] = None
    result: Optional[bool] = None

def _generate_test_configs():
    test_configs = []
    cfg = dict(
        seq_len=1,
        hidden_size=128,
        intermediate_size=4096,
        num_experts=16,
        capacity_factor=None,
        dtype=torch.bfloat16,
        test_mode="inference",
        implementation="llama4",
        moe_fused_tkg_enabled=True,
        use_expert_mlps_v2=True,
    )
    cfg["test_kernel"] = "moe_fused"
    test_configs.extend(
        [
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                moe_fused_tkg_kernel_enabled=True,
                device="xla",
                glu_mlp=False,
                normalize_top_k_affinities=False,
                result=True,
            ),
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                moe_fused_tkg_kernel_enabled=False,
                device="xla",
                glu_mlp=True,
                normalize_top_k_affinities=False,
                result=False,
            ),
            # shared experts not supported by moe_block_tkg kernel
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                device="xla",
                glu_mlp=True,
                normalize_top_k_affinities=False,
                result=False,
            ),
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                device="cpu",
                glu_mlp=True,
                normalize_top_k_affinities=False,
                result=False,
            ),
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                device="xla",
                glu_mlp=False,
                normalize_top_k_affinities=False,
                result=False,
            ),
            # shared experts not supported by moe_block_tkg kernel
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=1,
                device="xla",
                glu_mlp=True,
                top_k=2,
                normalize_top_k_affinities=True,
                result=False,
            ),
            TestCfg(
                **cfg,
                batch_size=256,
                num_shared_experts=1,
                device="xla",
                glu_mlp=True,
                normalize_top_k_affinities=False,
                result=False,
            ),
            TestCfg(
                **cfg,
                batch_size=1,
                num_shared_experts=0,
                device="xla",
                glu_mlp=True,
                normalize_top_k_affinities=False,
                result=True,
            ),
        ]
    )

    return test_configs


class MoEFusedTKGNkiAvailabilityTest(unittest.TestCase):
    def test_can_use_moe_fused_tkg_nki(self):
        for cfg in _generate_test_configs():
            hidden_states = torch.rand(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device)
            model = ut.initialize_neuron_model(cfg, move_to_device=False).moe_fused_tkg
            res = model._can_use_nki_kernel(cfg.test_kernel, hidden_states)
            assert res == cfg.result, f"Test failed for {cfg}"

if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
