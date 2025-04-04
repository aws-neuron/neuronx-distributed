import unittest
import torch
from functools import partial

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

torch.manual_seed(0)

ckpt_path = "/tmp/test_model_builder_shared_tensors_ckpt.pt"

class SharedTensorsModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float32)
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float32)
        # torch shared tensor -- https://huggingface.co/docs/safetensors/en/torch_shared_tensors
        self.lay3 = self.lay1

    def forward(self, x):
        rx = self.lay1(x)
        ry = self.lay2(rx)
        rz = self.lay3(ry)
        return rz

def test_shared_tensors_model():
    hidden_dim = 4
    batch_size = 2
    tp_degree = 2

    model = SharedTensorsModel(hidden_dim=hidden_dim)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(torch.load, ckpt_path),
        compiler_workdir="/tmp/model_builder_shared_tensors_compiler_workdir/"
    )

    x = torch.randn((batch_size, hidden_dim))
    builder.add(
        key="main",
        model_instance=BaseModelInstance(partial(SharedTensorsModel, hidden_dim=hidden_dim), input_output_aliases={}),
        example_inputs=[(x,)],
        compiler_args="--auto-cast=none"
    )

    traced_model = builder.trace(initialize_model_weights=True)

    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

    print("SharedTensorsModel test passed")

if __name__ == "__main__":
    unittest.main(module="test_model_builder_shared_tensors")