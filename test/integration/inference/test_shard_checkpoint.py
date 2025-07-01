import os
import tempfile
import pytest
import torch
import torch_neuronx
from neuronx_distributed.trace.model_builder import shard_checkpoint
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.parallel_layers import ColumnParallelLinear, parallel_state
from safetensors.torch import load_file

VALID_DEVICE_TYPES = {"privateuseone", "neuron"}

class SimpleMod(torch.nn.Module):
    def __init__(self, is_distributed=False):
        super().__init__()
        if is_distributed:
            self.lin1 = ColumnParallelLinear(128, 128, bias=False, gather_output=True)
        else:
            self.lin1 = torch.nn.Linear(128, 128, bias=False)

    def forward(self, x):
        return self.lin1(x)

def validate_sharded_checkpoint(orig_checkpoint, sharded_checkpoint, world_size, start_rank, end_rank, ondevice):
    if start_rank is None:
        start_rank = 0
    if end_rank is None:
        end_rank = world_size - 1
    assert len(sharded_checkpoint) == end_rank - start_rank + 1
    for shard in sharded_checkpoint:
        assert len(shard.keys()) == len(orig_checkpoint.keys())
        assert set(shard.keys()) - set(orig_checkpoint.keys()) == set()

    for key in orig_checkpoint.keys():
        tensors_to_concat = []
        full_tensor = orig_checkpoint[key]
        full_shape = full_tensor.shape
        concat_dimension = None
        for i, shard in enumerate(sharded_checkpoint):
            shard_tensor = shard[key]
            if ondevice:
                assert shard_tensor.device.type in VALID_DEVICE_TYPES
                assert shard_tensor.device.index == i + start_rank
                shard_tensor = shard_tensor.cpu()
            tensors_to_concat.append(shard_tensor)
            if concat_dimension is None:
                shard_shape = shard_tensor.shape
                dim_match = [dim1 == dim2 for dim1, dim2 in zip(full_shape, shard_shape)]
                concat_dimension = dim_match.index(False)

        if end_rank - start_rank + 1 != world_size:
            chunk_size = full_shape[concat_dimension] // world_size
            full_tensor = torch.ops.aten.slice(
                full_tensor,
                dim=concat_dimension,
                start=(start_rank*chunk_size),
                end=((end_rank+1)*chunk_size)
            )
        assert torch.equal(
            full_tensor,
            torch.cat(
                tensors_to_concat,
                dim=concat_dimension
            )
        )

@pytest.mark.parametrize("world_size,start_rank,end_rank,save_checkpoint,ondevice",
    [
        (2,None,None,False,False),(2,None,None,False,True),(2,None,None,True,False),(2,None,None,True,True), # basic case 1
        (2,0,0,False,False),(2,0,0,False,True),(32,0,0,True,False),(2,0,0,True,True), # partial sharding from 0th rank
        (2,1,None,False,False),(2,1,None,False,True),(2,1,None,True,False),(2,1,None,True,True), # partial sharding from tp/2 rank
    ]
)
def test_shard_checkpoint(world_size, start_rank, end_rank, save_checkpoint, ondevice):
    unsharded_mod = SimpleMod()
    orig_checkpoint = unsharded_mod.state_dict()

    with mock_distributed(world_size): # this doesn't affect single rank models
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=world_size)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size, skip_collective_init=True)
        mod = SimpleMod(is_distributed=True)

        with tempfile.TemporaryDirectory() as fp:
            sharded_checkpoint = shard_checkpoint(
                orig_checkpoint.copy(),
                mod,
                start_rank=start_rank,
                end_rank=end_rank,
                load_on_device=ondevice,
                serialize_path=fp if save_checkpoint else None
            )

            if start_rank is None:
                start_rank = 0
            if end_rank is None:
                end_rank = world_size - 1
            if ondevice and save_checkpoint:
                for i,shard in enumerate(sharded_checkpoint):
                    for key in shard:
                        assert shard[key].device.type in VALID_DEVICE_TYPES
                        assert shard[key].device.index == i + start_rank
                ondevice = False # already checked device
            if save_checkpoint:
                weight_files = sorted(os.listdir(fp), key=lambda x: int(x[2:x.find("_")])) # listdir is not guaranteed to load files in sequential order
                assert len(weight_files) == (end_rank - start_rank + 1)
                sharded_checkpoint = []
                for weight_file in weight_files:
                    sharded_checkpoint.append(load_file(
                        os.path.join(fp, weight_file)
                    ))

        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    validate_sharded_checkpoint(orig_checkpoint, sharded_checkpoint, world_size, start_rank, end_rank, ondevice)
