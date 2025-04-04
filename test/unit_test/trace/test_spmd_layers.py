import unittest
from collections import OrderedDict

import torch
import torch.distributed
import torch_xla.core.xla_model as xm

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
from neuronx_distributed.parallel_layers.layers import SPMDRank
import neuronx_distributed as nxd

from neuronx_distributed.parallel_layers.utils import is_torch_version_greater_than_2


class TestSPMDLayers(unittest.TestCase):
    @unittest.skipIf(not is_torch_version_greater_than_2(),
                        "ModelBuilder only works for torch-neuronx>=2.*")
    def test_spmd_rank(self):
        world_size = 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spmd_rank = SPMDRank(world_size=world_size)
                self.process_group = nxd.parallel_layers.parallel_state.get_world_group()

            def forward(self, ones: torch.Tensor):
                # using ones as input here to avoid tracing errors
                # gather ranks from all processes and return as output
                return xm.all_gather(
                    self.spmd_rank.get_rank() * ones,
                    dim=0,
                    groups=self.process_group._mesh,
                    pin_layout=False,
                )

        ranks = torch.arange(0, world_size, dtype=torch.int32)
        ones = torch.ones(1, dtype=torch.int32)

        def get_checkpoint():
            ckpt = OrderedDict()
            ckpt['spmd_rank.rank'] = ranks
            return ckpt

        builder = ModelBuilder(router=None, tp_degree=world_size, checkpoint_loader=get_checkpoint)
        builder.add(
            key="main",
            model_instance=BaseModelInstance(Model, input_output_aliases={}),
            example_inputs=[(ones,)],
        )
        model = builder.trace(initialize_model_weights=True)

        output = model(ones)
        self.assertTrue(torch.equal(output, ranks))

    @unittest.skipIf(not is_torch_version_greater_than_2(),
                        "ModelBuilder only works for torch-neuronx>=2.*")
    def test_spmd_scatter(self):
        world_size = 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spmd_rank = SPMDRank(world_size=world_size)

            def forward(self, x):
                x = nxd.parallel_layers.mappings.scatter_to_process_group_spmd(
                    x, 1, self.spmd_rank.get_rank(), nxd.parallel_layers.parallel_state.get_world_group()
                )
                x = x + self.spmd_rank.get_rank()
                out = xm.all_gather(
                    x,
                    dim=1,
                    groups=nxd.parallel_layers.parallel_state.get_world_group(as_list=True),
                    pin_layout=False,
                )
                return out

        split_size = 4
        input_t = torch.empty(1, split_size * world_size, 2)
        input_t = torch.reshape(torch.arange(1, input_t.numel() + 1, dtype=torch.int32), input_t.size())

        expected_tensor_chunks = list(torch.split(input_t, split_size, dim=1))
        for rank in range(world_size):
            expected_tensor_chunks[rank] = expected_tensor_chunks[rank] + rank
        expected_tensor = torch.cat(expected_tensor_chunks, dim=1)

        def get_checkpoint():
            ckpt = OrderedDict()
            ckpt['spmd_rank.rank'] = torch.arange(0, world_size, dtype=torch.int32)
            return ckpt

        builder = ModelBuilder(router=None, tp_degree=world_size, checkpoint_loader=get_checkpoint)
        builder.add(
            key="main",
            model_instance=BaseModelInstance(Model, input_output_aliases={}),
            example_inputs=[(input_t,)],
        )
        model = builder.trace(initialize_model_weights=True)

        output = model(input_t)
        print(output)
        self.assertTrue(torch.equal(output, expected_tensor))



if __name__ == "__main__":
    unittest.main()