import os
import shutil
import torch
import multiprocessing
from functools import partial

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from torch_neuronx import BucketModelConfig

from typing import List

ckpt_path = "/tmp/test_model_builder_ckpt.pt"

class CPLOnlyModel(torch.nn.Module):
    def __init__(self,
                 hidden_dim,
                 is_distributed):
        super().__init__()
        if is_distributed:
            self.lay1 = ColumnParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, gather_output=True, dtype=torch.float32)
            self.lay2 = ColumnParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, gather_output=True, dtype=torch.float32)
        else:
            self.lay1 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)
            self.lay2 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)

    def forward(self, x):
        rx = self.lay1(x)
        ry = self.lay2(rx)
        return ry


class CPLRPLModel(torch.nn.Module):
    def __init__(self,
                 hidden_dim,
                 is_distributed):
        super().__init__()
        if is_distributed:
            self.lay1 = ColumnParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, gather_output=False, dtype=torch.float32)
            self.lay2 = RowParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, input_is_parallel=True, dtype=torch.float32)
        else:
            self.lay1 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)
            self.lay2 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)

    def forward(self, x):
        rx = self.lay1(x)
        ry = self.lay2(rx)
        return ry


class StatefulModel(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_dim,
                 is_distributed):
        super().__init__()
        if is_distributed:
            self.lay1 = ColumnParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, gather_output=False, dtype=torch.float32)
            self.lay2 = RowParallelLinear(input_size=hidden_dim, output_size=hidden_dim, bias=False, input_is_parallel=True, dtype=torch.float32)
        else:
            self.lay1 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)
            self.lay2 = torch.nn.Linear(hidden_dim,hidden_dim, bias=False, dtype=torch.float32)

        self.state = torch.nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)

    def forward(self, x):
        rx = self.lay1(x)
        ry = self.lay2(rx)
        return ry + self.state, ry


def checkpoint_loader_fn(ckpt_path):
  model_sd = torch.load(ckpt_path)
  return model_sd


def batch_bucket_model_kernel(inputs: List[torch.Tensor]):
    inp = inputs[0]
    batch_size = inp.shape[0]
    if (batch_size == 1 or batch_size == 4):
        return inputs, torch.tensor(0)
    else:
        return inputs, torch.tensor(1)


def get_bucket_kernel():
    return torch.jit.script(batch_bucket_model_kernel)


def generate_simple_CPL_only_model(batch_size, hidden_dim):
    model = CPLOnlyModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=2,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    x = torch.randn((batch_size, hidden_dim))
    builder.add(key = "main",
                model_instance = BaseModelInstance(partial(CPLOnlyModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=[(x,)],
                compiler_args="--auto-cast=none")
    traced_model = builder.trace(initialize_model_weights=True)

    return model,traced_model


def test_saving_loading_model():
    _,traced_model = generate_simple_CPL_only_model(2,4)
    torch.jit.save(traced_model, "test.pt")
    torch.jit.load("test.pt")
    os.remove("test.pt")
    del traced_model
    torch.classes.neuron.Runtime().unsafe_close()


def test_CPL_only_model():
    hidden_dim=4
    batch_size=2
    model,traced_model = generate_simple_CPL_only_model(batch_size,hidden_dim)
    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)


def test_executing_loaded_model():
    hidden_dim=4
    batch_size=2

    model,traced_model = generate_simple_CPL_only_model(batch_size,hidden_dim)

    torch.jit.save(traced_model, "test.pt")
    del traced_model

    loaded_traced_model = torch.jit.load("test.pt")
    loaded_traced_model.nxd_model.initialize_with_saved_weights()

    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = loaded_traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

    os.remove("test.pt")


def test_CPL_RPL_model():

    hidden_dim=4
    batch_size=2

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=2,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    x = torch.randn((batch_size, hidden_dim))
    builder.add(key = "main",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=[(x,)],
                compiler_args="--auto-cast=none")
    traced_model = builder.trace(initialize_model_weights=True)

    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)


def test_multiple_input_shapes():

    hidden_dim=4
    batch_size=2

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=2,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    x = torch.randn((batch_size, hidden_dim))
    builder.add(key = "ctx",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=[(x,)],
                compiler_args="--auto-cast=none")
    y = torch.randn((batch_size+1, hidden_dim))
    builder.add(key = "tkg",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=[(y,)],
                compiler_args="--auto-cast=none")
    traced_model = builder.trace(initialize_model_weights=True)

    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

        x = torch.randn((batch_size+1, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)


def test_weight_layout_optimization():
    # Currently compiler outputs hlo stub in the current working dir, and it
    # needs to be clean before the execution, so moving it to a new dir
    original_dir = os.getcwd()
    new_dir = os.path.join(original_dir, "wlt")
    print(f"current dir is {original_dir}, will move working dir to {new_dir}")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)
    os.chdir(new_dir)
    assert os.getcwd() == new_dir

    hidden_dim = 4
    batch_size = 2

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(
        router=None,
        tp_degree=2,
        checkpoint_loader=partial(torch.load, ckpt_path),
        compiler_workdir="new_compiler_workdir/",
    )
    x = torch.randn((batch_size, hidden_dim))
    builder.add(
        key="ctx",
        model_instance=BaseModelInstance(
            partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}
        ),
        example_inputs=[(x,)],
    )
    y = torch.randn((batch_size + 1, hidden_dim))
    builder.add(
        key="tkg",
        model_instance=BaseModelInstance(
            partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}
        ),
        example_inputs=[(y,)],
        priority_model_idx=0,
    )
    traced_model = builder.trace(initialize_model_weights=True)

    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

        x = torch.randn((batch_size + 1, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

    os.chdir(original_dir)


def test_weight_layout_optimization_with_serialization():
    # Currently compiler outputs hlo stub in the current working dir, and it
    # needs to be clean before the execution, so moving it to a new dir
    original_dir = os.getcwd()
    new_dir = os.path.join(original_dir, "wlt")
    print(f"current dir is {original_dir}, will move working dir to {new_dir}")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)
    os.chdir(new_dir)
    assert os.getcwd() == new_dir

    hidden_dim = 4
    batch_size = 2
    tp_degree = 2

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(torch.load, ckpt_path),
        compiler_workdir="new_compiler_workdir/",
    )
    x = torch.randn((batch_size, hidden_dim))
    builder.add(
        key="ctx",
        model_instance=BaseModelInstance(
            partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}
        ),
        example_inputs=[(x,)],
    )
    y = torch.randn((batch_size + 1, hidden_dim))
    builder.add(
        key="tkg",
        model_instance=BaseModelInstance(
            partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}
        ),
        example_inputs=[(y,)],
        priority_model_idx=0,
    )
    traced_model = builder.trace(initialize_model_weights=False)

    # Save the traced model 
    torch.jit.save(traced_model, "traced_model.pt")
    del traced_model

    # Shard weights from checkpoint 
    shard_weights_path = "weights/"
    builder.shard_checkpoint(serialize_path=shard_weights_path)
    weights = []
    for rank in range(tp_degree):
        ckpt = torch.load(os.path.join(shard_weights_path, f"tp{rank}_sharded_checkpoint.pt"))
        weights.append(ckpt)
        
    # Load the traced model
    traced_model = torch.jit.load("traced_model.pt")
    print("Done loading serialized model")
    
    # Load new weights
    traced_model.nxd_model.initialize(weights)

    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

        x = torch.randn((batch_size + 1, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)

    os.chdir(original_dir)


class StatefulModelInstance(BaseModelInstance):
    def __init__(self):
        self.module = None
        self.input_output_aliases = None

    def load_module(self):
        self.module = StatefulModel(batch_size=2, hidden_dim=4, is_distributed=True)
        self.input_output_aliases = {self.module.state: 1}

    def get(self, bucket_rank, **kwargs):
        return self.module, self.input_output_aliases


def test_stateful_model():

    hidden_dim=4
    batch_size=2

    model = StatefulModel(batch_size=batch_size, hidden_dim=hidden_dim, is_distributed=False)
    sd = model.state_dict()
    torch.save(sd, ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=2,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    x = torch.randn((batch_size, hidden_dim))
    builder.add(key = "main",
                model_instance = StatefulModelInstance(),
                example_inputs=[(x,)],
                compiler_args="--auto-cast=none")
    traced_model = builder.trace(initialize_model_weights=True)

    model = StatefulModel(batch_size=batch_size, hidden_dim=hidden_dim, is_distributed=False)
    model.load_state_dict(sd)
    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result, new_state = model(x)
        model.state.data = new_state
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)


def test_batch_bucketed_model():
    hidden_dim=4
    batch_sizes_ctx=[1,2]
    batch_sizes_tkg = [4,8]

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=2,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    inps = [(torch.randn((batch_size, hidden_dim)),) for batch_size in batch_sizes_ctx]
    bucket_config = BucketModelConfig(
        get_bucket_kernel
    )

    builder.add(key = "ctx",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=inps,
                bucket_config=bucket_config,
                compiler_args="--auto-cast=none")
    inps = [(torch.randn((batch_size, hidden_dim)),) for batch_size in batch_sizes_tkg]
    builder.add(key = "tkg",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=inps,
                bucket_config=bucket_config,
                compiler_args="--auto-cast=none")
    traced_model = builder.trace(initialize_model_weights=True)

    # Test multiple invocations
    for _ in range(5):
        for batch_size in batch_sizes_ctx:
            x = torch.randn((batch_size, hidden_dim))
            cpu_result = model(x)
            nxd_result = traced_model(x)
            torch.testing.assert_close(cpu_result, nxd_result)
        for batch_size in batch_sizes_tkg:
            x = torch.randn((batch_size, hidden_dim))
            cpu_result = model(x)
            nxd_result = traced_model(x)
            torch.testing.assert_close(cpu_result, nxd_result)


def test_loading_checkpoint():

    hidden_dim=4
    batch_size=2
    tp_degree=2

    model = CPLRPLModel(hidden_dim=hidden_dim, is_distributed=False)
    torch.save(model.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=tp_degree,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir="new_compiler_workdir/")
    x = torch.randn((batch_size, hidden_dim))
    builder.add(key = "main",
                model_instance = BaseModelInstance(partial(CPLRPLModel, hidden_dim=hidden_dim, is_distributed=True), input_output_aliases={}),
                example_inputs=[(x,)],
                compiler_args="--auto-cast=none")
    
    traced_model = builder.trace(initialize_model_weights=False) # stops weight sharding

    # Save the traced model 
    torch.jit.save(traced_model, "traced_model.pt")
    del traced_model

    # Shard weights from checkpoint 
    shard_weights_path = "weights/"
    builder.shard_checkpoint(serialize_path=shard_weights_path)
    weights = []
    for rank in range(tp_degree):
        ckpt = torch.load(os.path.join(shard_weights_path, f"tp{rank}_sharded_checkpoint.pt"))
        weights.append(ckpt)
        
    # Load the traced model
    traced_model = torch.jit.load("traced_model.pt")
    
    # Load new weights
    traced_model.nxd_model.initialize(weights)
        
    # Test multiple invocations
    for _ in range(5):
        x = torch.randn((batch_size, hidden_dim))
        cpu_result = model(x)
        nxd_result = traced_model(x)
        torch.testing.assert_close(cpu_result, nxd_result)


if __name__ == "__main__":
    test_list = [
        test_saving_loading_model, 
        test_CPL_only_model,
        test_executing_loaded_model,
        test_CPL_RPL_model,
        test_multiple_input_shapes,
        test_weight_layout_optimization,
        test_weight_layout_optimization_with_serialization,
        test_stateful_model,
        test_batch_bucketed_model,
        test_loading_checkpoint,
    ]
    # Run tests in a separate process so it can init and release runtime properly
    for test in test_list:
        print(f"Starting test: {test.__name__}")
        p = multiprocessing.Process(target=test)
        p.start()
        p.join()
        if p.exitcode == 0:
            print(f"Test succeeded: {test.__name__}\n")
        else:
            raise Exception(f"Test failed: {test.__name__}\n")
    print(f"All {len(test_list)} tests on ModelBuilder succeeded!")
    
