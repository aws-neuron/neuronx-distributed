import copy
import unittest

import torch

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.trace.trace import (
    _mock_parallel_state,
    _validate_traceable,
    shard_children,
)


class TestCheckpoint(unittest.TestCase):
    def test_validate_traceable(self):

        class Model(torch.nn.Module):
            def __init__(self, shard_across_embedding):
                super().__init__()
                self.embed_tokens = ParallelEmbedding(10, 10, shard_across_embedding=shard_across_embedding)

            def forward(self):
                pass

        def model_over_vocab():
            return (Model(shard_across_embedding=False), {})

        def model_over_embed():
            return (Model(shard_across_embedding=True), {})

        with self.assertRaises(ValueError):
            _validate_traceable(model_over_vocab, tp_degree=1)
        _validate_traceable(model_over_embed, tp_degree=1)

        class ModelWithChildren(torch.nn.Module):
            def __init__(self, shard_across_embedding):
                super().__init__()
                self.model = Model(shard_across_embedding)

            def forward(self):
                pass

        def model_over_vocab():
            return (ModelWithChildren(shard_across_embedding=False), {})

        def model_over_embed():
            return (ModelWithChildren(shard_across_embedding=True), {})

        with self.assertRaises(ValueError):
            _validate_traceable(model_over_vocab, tp_degree=1)
        _validate_traceable(model_over_embed, tp_degree=1)

    def test_shard_children(self):

        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if parallel_state.model_parallel_is_initialized():
                    self.embed_tokens = ParallelEmbedding(10, 32, shard_across_embedding=True)
                    self.cpl = ColumnParallelLinear(10, 64)
                    self.rpl = RowParallelLinear(64, 10)
                else:
                    self.embed_tokens = torch.nn.Embedding(10, 32)
                    self.cpl = torch.nn.Linear(10, 64)
                    self.rpl = torch.nn.Linear(64, 10)

            def forward(self):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lay1 = InnerModel()
                if parallel_state.model_parallel_is_initialized():
                    self.embed_tokens = ParallelEmbedding(10, 32, shard_across_embedding=True)
                    self.cpl = ColumnParallelLinear(8, 128)
                    self.rpl = RowParallelLinear(128, 8)
                else:
                    self.embed_tokens = torch.nn.Embedding(8, 32)
                    self.cpl = torch.nn.Linear(8, 128)
                    self.rpl = torch.nn.Linear(128, 8)

            def forward(self):
                pass

        model = Model()
        checkpoint = model.state_dict()

        for tp_degree in [2, 4, 8, 16, 32]:
            for rank in range(0, tp_degree):
                _mock_parallel_state(tp_degree, rank)
                nxd_model = Model()
                sharded_checkpoint = copy.deepcopy(checkpoint)
                shard_children(nxd_model, sharded_checkpoint, "", torch.float32, rank=rank, tp_degree=tp_degree)

                def validate_shard_weight(prefix):
                    embed_shard = sharded_checkpoint[prefix + "embed_tokens.weight"]
                    embed = checkpoint[prefix + "embed_tokens.weight"]
                    assert embed_shard.shape == (embed.shape[0], embed.shape[1] / tp_degree)
                    assert torch.equal(embed_shard, torch.split(embed, embed.shape[1] // tp_degree, dim=1)[rank])

                    rpl_shard = sharded_checkpoint[prefix + "rpl.weight"]
                    rpl = checkpoint[prefix + "rpl.weight"]
                    assert rpl_shard.shape == (rpl.shape[0], rpl.shape[1] / tp_degree)
                    assert torch.equal(rpl_shard, torch.split(rpl, rpl.shape[1] // tp_degree, dim=1)[rank])

                    cpl_shard = sharded_checkpoint[prefix + "cpl.weight"]
                    cpl = checkpoint[prefix + "cpl.weight"]
                    assert cpl_shard.shape == (cpl.shape[0] / tp_degree, cpl.shape[1])
                    assert torch.equal(cpl_shard, torch.split(cpl, cpl.shape[0] // tp_degree, dim=0)[rank])

                validate_shard_weight("")
                validate_shard_weight("lay1.")


if __name__ == "__main__":
    unittest.main()
