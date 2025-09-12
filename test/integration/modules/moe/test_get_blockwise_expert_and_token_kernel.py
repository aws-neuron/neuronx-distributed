import os
import torch
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import numpy as np
import math
from types import SimpleNamespace
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.parallel_layers.layers import SPMDRank

from integration.kernels.test_find_index import routing_setup_random_topk

def get_test_configs():
    return [
        {
            "T": 8192,
            "E": 128,
            "tp_degree": 4,
            "ep_degree": 16,
            "top_k": 8,
            "block_size": 128,
        },
        {
            "T": 8192,
            "E": 128,
            "tp_degree": 1,
            "ep_degree": 64,
            "top_k": 8,
            "block_size": 128,
        }
    ]


def main():
    np.random.seed(42)
    dist.init_process_group("xla")
    device = xm.xla_device()

    test_configs = get_test_configs()
    for cfg in test_configs:
        cfg = SimpleNamespace(**cfg)

        # Parallel state set up
        parallel_state.destroy_model_parallel()
        assert not parallel_state.model_parallel_is_initialized()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=cfg.tp_degree,
            expert_model_parallel_size=cfg.ep_degree,
            lnc_size=2,
        )

        rank = torch.distributed.get_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        spmd_rank = SPMDRank(
            world_size=parallel_state.get_world_group().size(),
            tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_group().size(),
        )
        spmd_rank.rank.data[:] = torch.tensor([rank], dtype=torch.int32)

        # EP mask setup
        expert_mask, _ = routing_setup_random_topk(cfg.T, cfg.top_k, cfg.E)
        E_local = cfg.E // cfg.ep_degree
        expert_mask = expert_mask[:, E_local*ep_rank:E_local*(ep_rank+1)]
        expert_mask = torch.tensor(expert_mask, dtype=torch.float64).to(device)

        num_blocks = math.ceil((cfg.T * cfg.top_k - (E_local - 1)) / cfg.block_size) + E_local - 1

        block_to_expert_kernel, token_position_to_id_kernel = ExpertMLPsV2.get_blockwise_expert_and_token_mapping_kernel(
            num_blocks=num_blocks,
            expert_mask=expert_mask,
            block_size=cfg.block_size,
            device=device,
            spmd_rank = spmd_rank,
            tensor_parallel_group=parallel_state.get_tensor_model_parallel_group(),
            logical_nc_config=2,
        )
        block_to_expert_kernel_np = block_to_expert_kernel.cpu().detach().numpy()
        token_position_to_id_kernel_np = token_position_to_id_kernel.cpu().detach().numpy()

        block_to_expert_fw, token_position_to_id_fw = ExpertMLPsV2.get_blockwise_expert_and_token_mapping(
            total_tokens=cfg.T,
            num_blocks=num_blocks,
            expert_mask=expert_mask,
            expert_index=None,
            block_size=cfg.block_size,
            device=device,
            enable_spmd_rank=False,
            spmd_rank=None,
            tensor_parallel_group=parallel_state.get_tensor_model_parallel_group(),
            optimized_block_to_token_mapping=False,
            parallelize_token_to_block_mapping=True,
        )
        block_to_expert_fw_np = block_to_expert_fw.cpu().detach().numpy()
        token_position_to_id_fw_np = token_position_to_id_fw.cpu().detach().numpy()

        assert np.allclose(block_to_expert_kernel_np, block_to_expert_fw_np), "block_to_expert mismatch!"
        assert np.allclose(token_position_to_id_kernel_np, token_position_to_id_fw_np), "block_to_expert mismatch!"


if __name__ == "__main__":
    main()

