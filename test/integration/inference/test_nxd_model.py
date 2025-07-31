from functools import partial
import random
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import pytest
import torch

from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2
from torch_neuronx.xla_impl.trace import get_torch_dtype

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, parallel_state
from neuronx_distributed.trace.nxd_model import NxDModel
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.trace.model_builder import trace, compile, compile_wlo, compile_layout_transformer
from neuronx_distributed.trace.hlo_utils import mark_weights_for_wlo, apply_layout_transformation
from neuronx_distributed.trace.model_builder_utils import TraceArtifacts, CompilationArtifacts, WLOArtifacts, ModelParamInfo

MultiInputType = Dict[
    str,
    Tuple[
        Union[torch.Tensor,Tuple[torch.Tensor]],
        Optional[Dict[str, torch.Tensor]]
    ]
]

VALID_DEVICE_TYPES = {'privateuseone', 'neuron'}

class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x
    def forward_ranked(self, x):
        return x
    def forward_async(self, x):
        return x

def generate_nxdmodel_with_mock_spmdmodels(tp_degree, num_models, fake_spmdmodel_obj=True):
    nxd_model = NxDModel(tp_degree)
    for i in range(num_models):
        # create mock metaneff
        mock_metaneff = metaneff_pb2.MetaNeff()
        mock_metaneff.name = f"mock_metaneff{i}".encode()

        # create mock hlo
        mock_hlo = hlo_pb2.HloModuleProto()
        mock_hlo.name = f"mock_hlo{i}" # dtype in hlo.proto is string not bytes
        mock_metaneff.serialized_graph_def = mock_hlo.SerializeToString()

        if fake_spmdmodel_obj:
            nxd_model.spmd_models[f'key{i}'] = IdentityModule()
        else:
            nxd_model.spmd_models[f'key{i}'] = torch.classes.neuron.SPMDModel(
                f"mock_neff{i}".encode(),
                mock_metaneff.SerializeToString(),
                tp_degree,
                tp_degree
            )

    return nxd_model

def generate_example_input(shapes, kwarg_names=None):
    if kwarg_names is not None:
        if not isinstance(shapes, list):
            shapes = [shapes]
        return (None,{
            kwarg_name: torch.rand(shape)
            for kwarg_name,shape
            in zip(kwarg_names, shapes)
        })

    if isinstance(shapes, tuple):
        return (torch.rand(shapes), None) # use single tensor

    return (tuple(torch.rand(shape) for shape in shapes), None) # use tuple of tensors

class SimpleMod(torch.nn.Module):
    def __init__(self, is_distributed=False):
        super().__init__()
        if is_distributed:
            self.lin1 = ColumnParallelLinear(10, 4, bias=False, gather_output=True)
        else:
            self.lin1 = torch.nn.Linear(10,4)

    def forward(self, x):
        return self.lin1(x)

class MultiInpMod(torch.nn.Module):
    def __init__(self, is_distributed=False):
        super().__init__()
        self.smod = SimpleMod(is_distributed)

    def forward(self, x, y):
        return self.smod(x) + self.smod(y)

class DenseMLP(torch.nn.Module):
    def __init__(self, is_distributed=False):
        super().__init__()
        if is_distributed:
            self.lin1 = RowParallelLinear(10, 1024, bias=False, input_is_parallel=False)
            self.lin2 = ColumnParallelLinear(10, 1024, bias=False, gather_output=True)
        else:
            self.lin1 = torch.nn.Linear(10, 1024, bias=False)
            self.lin2 = torch.nn.Linear(1024, 2, bias=False)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        return x

class SimpleAliasedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.tensor(0.0), persistent=True)
    def forward(self, x):
        self.counter = self.counter + torch.max(x)
        return x + 1

def trace_and_compile_model(
    mod,
    inputs: MultiInputType,
    world_size: int = 1,
    use_wlo: bool = False
):
    with mock_distributed(world_size): # this doesn't affect single rank models
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=world_size)
        parallel_state.initialize_model_parallel(world_size, skip_collective_init=True)
        mod = mod()
        trace_artifacts: Dict[str, TraceArtifacts] = {
            key:trace(mod, args=inputs[key][0], kwargs=inputs[key][1])
            for key in inputs
        }

        compile_artifacts: Dict[str, Union[CompilationArtifacts, WLOArtifacts]] = {}
        layout_transformer = None
        if use_wlo:
            # Let 'key1' always be the priority model for test simplicity
            mark_weights_for_wlo(
                priority_model_trace_hlo=trace_artifacts['key1'].hlo,
                priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
            )
            wlo_artifacts = compile_wlo(
                hlo_module=trace_artifacts['key1'].hlo,
                metaneff=trace_artifacts['key1'].metaneff,
                key='key1'
            )
            compile_artifacts['key1'] = wlo_artifacts
            for key in trace_artifacts.keys():
                if key == 'key1':
                    continue
                apply_layout_transformation(
                    hlo_module=trace_artifacts[key].hlo,
                    flattener=trace_artifacts[key].flattener,
                    packer=trace_artifacts[key].packer,
                    metaneff=trace_artifacts[key].metaneff,
                    weight_name_to_idx=trace_artifacts[key].weight_name_to_idx,
                    wlo_artifacts=wlo_artifacts,
                    key=key
                )

                compile_artifacts[key] = compile(
                    hlo_module=trace_artifacts[key].hlo,
                    metaneff=trace_artifacts[key].metaneff,
                    key=key
                )
            layout_transformer = compile_layout_transformer(
                wlo_artifacts=wlo_artifacts,
                priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
            )
        else:
            compile_artifacts = {
                key:compile(
                    trace_artifacts[key].hlo,
                    trace_artifacts[key].metaneff,
                    key=key
                )
                for key in trace_artifacts
            }
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    return trace_artifacts, compile_artifacts, layout_transformer

def build_nxd_model(
    mod,
    inputs: MultiInputType,
    world_size: int = 1,
    use_wlo: bool = False,
):
    trace_artifacts, compile_artifacts, layout_transformer = trace_and_compile_model(
        mod,
        inputs,
        world_size,
        use_wlo
    )
    nxd_model = NxDModel(world_size, layout_transformer=layout_transformer)
    for key in inputs.keys():
        nxd_model.add(
            key,
            trace_artifacts[key],
            compile_artifacts[key]
        ) # this should run with no errors

    return nxd_model

def validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip=None):
    names_to_skip = set() if names_to_skip is None else names_to_skip
    ordered_list, ordered_names = nxd_model.convert_dict_to_ordered_list(
        kwargs,
        len(args)
    )
    ordered_list = args + ordered_list

    assert len(ordered_list) == len(model_params) - len(names_to_skip)
    assert len(ordered_names) == len(model_params) - len(names_to_skip)
    skipped = 0
    for i,(expected_name,is_positional) in enumerate(model_params):
        if expected_name in names_to_skip:
            skipped += 1
            continue
        tensor = ordered_list[i - skipped]
        actual_name = ordered_names[i - skipped]
        assert expected_name == actual_name
        if is_positional:
            try:
                assert torch.equal(tensor, args[i]) # not subtracting skipped here, because pos args are never skipped
            except IndexError:
                assert torch.equal(tensor, kwargs[actual_name]) # it could be that pos arg was passed as kwarg
        else:
            try:
                assert torch.equal(tensor, kwargs[actual_name])
            except KeyError:
                assert torch.equal(tensor, args[i]) # it could be that kwarg was passed as positional arg

@pytest.mark.parametrize('on_cpu,layout_opt,replace_weights',
    [
        (True,False,False),
        (True, True,False),
        (False,False,False),
        (True,False,True),
        (True, True,True),
        (False,False,True)
    ]
)
def test_weight_setting(on_cpu, layout_opt, replace_weights):
    inputs = {
        'key1': (torch.rand(1,10), None),
    }
    world_size = 2
    nxd_model = build_nxd_model(
        partial(DenseMLP, True),
        inputs,
        world_size,
        use_wlo=layout_opt
    )

    get_device = lambda rank: 'cpu' if on_cpu else f'privateuseone:{rank}'  # noqa: E731
    my_new_weights = [
        {
            'lin1.weight': torch.zeros((1024,10 // world_size)).to(get_device(rank)),
            'lin2.weight': torch.zeros((1024 // world_size, 10)).to(get_device(rank))
        }
        for rank in range(world_size)
    ]
    nxd_model.set_weights(my_new_weights)

    def check_weights(source_weights):
        for rank in range(world_size):
            for weight_key in ['lin1.weight', 'lin2.weight']:
                assert nxd_model.weights[rank][weight_key].device.type in VALID_DEVICE_TYPES
                assert nxd_model.weights[rank][weight_key].device.index == rank
                # value comparison is not possible at this time since we don't know how to undo a transformation
                if not layout_opt:
                    assert torch.equal(
                        source_weights[rank][weight_key].cpu(),
                        nxd_model.weights[rank][weight_key].cpu()
                    )

    check_weights(my_new_weights)

    nxd_model.to_neuron() # should not have any error
    assert nxd_model.loaded_on_neuron

    # Check replace weights functionality
    if replace_weights:
        my_replacement_weights = [
            {
                'lin1.weight': torch.ones((1024,10 // world_size)).to(get_device(rank)),
                'lin2.weight': torch.ones((1024 // world_size, 10)).to(get_device(rank))
            }
            for rank in range(world_size)
        ]
        nxd_model.replace_weights(my_replacement_weights)
        assert nxd_model.loaded_on_neuron
        check_weights(my_replacement_weights)

    torch.classes.neuron.Runtime().unsafe_close()

def test_add_basic():
    inputs = {
        'key1': (torch.rand(1,10), None),
    }
    world_size = 1
    nxd_model = build_nxd_model(
        partial(SimpleMod, False),
        inputs,
        world_size
    ) # this should run with no errors

    assert nxd_model.state_initializer is None
    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.parametrize(
    'use_kwargs, multi_input, shuffle',
    [
        (False, False, False), # pos arg
        (True, False, False), # single kwarg
        (False, True, False), # multi pos args
        (True, True, False), # multi kwargs
        (True, True, True) # multi shuffled kwargs
    ]
)
def test_add_multiple(use_kwargs, multi_input, shuffle):
    kwarg_names = ['x'] if use_kwargs else None
    if multi_input and use_kwargs:
        if shuffle:
            kwarg_names.append('y')
        else:
            kwarg_names = ['y'] + kwarg_names
    num_shapes = 2 if multi_input else 1
    inputs = {
        'key1': generate_example_input([(1,10) for _ in range(num_shapes)],kwarg_names),
        'key2': generate_example_input([(2,10) for _ in range(num_shapes)],kwarg_names),
        'key3': generate_example_input([(3,10) for _ in range(num_shapes)],kwarg_names),
    }
    world_size = 2
    nxd_model = build_nxd_model(
        partial(SimpleMod if not multi_input else MultiInpMod, True),
        inputs,
        world_size
    ) # this should run with no errors
    assert nxd_model.state_initializer is None

    # the below should be true regardless of shuffle setting
    available_keys = nxd_model.input_shape_map.keys()
    if not multi_input:
        assert "(x: [1, 10])" in available_keys
        assert "(x: [2, 10])" in available_keys
        assert "(x: [3, 10])" in available_keys
    else:
        assert "(x: [1, 10], y: [1, 10])" in available_keys
        assert "(x: [2, 10], y: [2, 10])" in available_keys
        assert "(x: [3, 10], y: [3, 10])" in available_keys

    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.parametrize(
    'use_kwargs', (False, True)
)
def test_add_aliased_model(use_kwargs):
    kwarg_names = ['x'] if use_kwargs else None
    inputs = {
        'key1': generate_example_input((1,1), kwarg_names),
        'key2': generate_example_input((2,1), kwarg_names),
        'key3': generate_example_input((3,1), kwarg_names),
    }
    world_size = 1
    
    # Build model and extract artifacts for inspection
    trace_artifacts, compile_artifacts, _ = trace_and_compile_model(
        SimpleAliasedModel,
        inputs,
        world_size
    )
    
    nxd_model = NxDModel(world_size)
    keys = ['key1', 'key2', 'key3']

    for key in keys:
        nxd_model.add(key, trace_artifacts[key], compile_artifacts[key])

    assert nxd_model.state_initializer is not None

    available_keys = nxd_model.input_shape_map.keys()
    assert "(x: [1, 1])" in available_keys
    assert "(x: [2, 1])" in available_keys
    assert "(x: [3, 1])" in available_keys

    for key in keys:
        # Verify metaneff structure
        metaneff = trace_artifacts[key].metaneff
        assert len(metaneff.output_tensors) > 1, "Expected multiple output tensors in metaneff"
        assert len(metaneff.output_aliases_to) > 0, "Expected at least one output alias"
        
        # Verify ``reserved_example_outputs`` has correct number of tensors (excluding aliases)
        assert key in nxd_model.reserved_example_outputs
        num_expected_outputs = len(metaneff.output_tensors) - len(metaneff.output_aliases_to)
        assert len(nxd_model.reserved_example_outputs[key]) == num_expected_outputs, \
            f"Expected {num_expected_outputs} reserved output tensors (aliased outputs should be skipped)"

        # Verify the shapes and dtypes of the ``reserved_example_outputs``
        for i, reserved_output in enumerate(nxd_model.reserved_example_outputs[key]):
            output_idx = 0
            for j, out_tensor in enumerate(metaneff.output_tensors):
                if j not in metaneff.output_aliases_to:
                    if output_idx == i:
                        expected_shape = list(out_tensor.shape)
                        expected_dtype = get_torch_dtype(out_tensor.data_type)
                        break
                    output_idx += 1

            assert list(reserved_output.shape) == expected_shape, \
                f"Output {i}: Expected shape {expected_shape}, got {list(reserved_output.shape)}"
            assert reserved_output.dtype == expected_dtype, \
                f"Output {i}: Expected dtype {expected_dtype}, got {reserved_output.dtype}"

    torch.classes.neuron.Runtime().unsafe_close()

def test_get_neff():
    num_keys = 4
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, num_keys, fake_spmdmodel_obj=False)
    random_selection = random.randint(0, num_keys - 1)
    assert nxd_model.get_neff(f"key{random_selection}") == f"mock_neff{random_selection}".encode()
    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.xfail(raises=KeyError)
def test_get_nonexistent_neff():
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, 4)
    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.get_neff("key10")

def test_get_metaneff():
    num_keys = 4
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, num_keys, fake_spmdmodel_obj=False)
    random_selection = random.randint(0, num_keys - 1)
    assert nxd_model.get_metaneff(f"key{random_selection}").name == f"mock_metaneff{random_selection}".encode()
    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.xfail(raises=KeyError)
def test_get_nonexistent_metaneff():
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, 4)
    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.get_metaneff("key10")

def test_get_hlo():
    num_keys = 4
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, num_keys, fake_spmdmodel_obj=False)
    random_selection = random.randint(0, num_keys - 1)
    assert nxd_model.get_hlo(f"key{random_selection}").name == f"mock_hlo{random_selection}"
    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.xfail(raises=KeyError)
def test_get_nonexistent_hlo():
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(32, 4)
    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.get_hlo("key10")

@pytest.mark.parametrize('buffer_type', [('weights',),('states',)])
def test_read_from_neuron_buffer(buffer_type: str):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1, 1)
    nxd_model.loaded_on_neuron = True # needed for mocking
    if buffer_type == 'weights':
        nxd_model.weights = [{
            'lin1.weight': torch.zeros(10,2).to('privateuseone:0')
        }]
    else:
        nxd_model.states = [{
            'lin1.weight': torch.zeros(10,2).to('privateuseone:0')
        }]

    retrieved_weight = nxd_model.read_from_neuron_buffer('lin1.weight', 0)
    assert torch.equal(retrieved_weight, torch.zeros(10,2))
    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.xfail(raises=KeyError)
@pytest.mark.parametrize('buffer_type', [('weights',),('states',)])
def test_read_from_nonexistent_neuron_buffer(buffer_type: str):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1, 1)
    nxd_model.loaded_on_neuron = True # needed for mocking
    if buffer_type == 'weights':
        nxd_model.weights = [{
            'lin1.weight': torch.zeros(10,2).to('privateuseone:0')
        }]
    else:
        nxd_model.states = [{
            'lin1.weight': torch.zeros(10,2).to('privateuseone:0')
        }]

    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.read_from_neuron_buffer('lin2.weight', 0)

@pytest.mark.xfail(raises=AssertionError)
def test_read_from_uninitialized_neuron_model():
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1, 1)
    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.read_from_neuron_buffer('lin1.weight', 0)

@pytest.mark.parametrize('buffer_type', [('weights',),('states',)])
def test_write_to_neuron_buffer(buffer_type: str):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1, 1)
    nxd_model.loaded_on_neuron = True # needed for mocking
    if buffer_type == 'weights':
        nxd_model.weights = [{
            'lin1.weight': torch.rand(10,2).to('privateuseone:0')
        }]
    else:
        nxd_model.states = [{
            'lin1.weight': torch.rand(10,2).to('privateuseone:0')
        }]

    new_weight = torch.zeros(10,2)
    nxd_model.write_to_neuron_buffer(new_weight, 'lin1.weight', 0)
    if buffer_type == 'weights':
        assert torch.equal(nxd_model.weights[0]['lin1.weight'].cpu(), torch.zeros(10,2))
    else:
        assert torch.equal(nxd_model.states[0]['lin1.weight'].cpu(), torch.zeros(10,2))

    torch.classes.neuron.Runtime().unsafe_close()

@pytest.mark.xfail(raises=KeyError)
@pytest.mark.parametrize('buffer_type', [('weights',),('states',)])
def test_write_to_nonexistent_neuron_buffer(buffer_type: str):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1, 1)
    nxd_model.loaded_on_neuron = True # needed for mocking
    if buffer_type == 'weights':
        nxd_model.weights = [{
            'lin1.weight': torch.rand(10,2).to('privateuseone:0')
        }]
    else:
        nxd_model.states = [{
            'lin1.weight': torch.rand(10,2).to('privateuseone:0')
        }]

    new_weight = torch.zeros(10,2)
    torch.classes.neuron.Runtime().unsafe_close()
    nxd_model.write_to_neuron_buffer(new_weight, 'lin2.weight', 0)

def test_convert_dict_to_ordered_list_basic():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', False), ('b', False), ('c', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = []
    kwargs = {
        'b': torch.rand(1,2),
        'a': torch.rand(1,2),
        'c': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs)

def test_convert_dict_to_ordered_list_with_pos_args():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = [torch.rand(1,2)]
    kwargs = {
        'c': torch.rand(1,2),
        'b': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs)

def test_convert_dict_to_ordered_list_skip_kwargs():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', False), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = []
    kwargs = {
        'a': torch.rand(1,2),
        'd': torch.rand(1,2),
        'c': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip={'b'})

def test_convert_dict_to_ordered_list_skip_kwargs_and_with_pos_args():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = [torch.rand(1,2)]
    kwargs = {
        'b': torch.rand(1,2),
        'd': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip={'c'})

def test_convert_dict_to_ordered_list_kwargs_as_pos_args():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = [torch.rand(1,2), torch.rand(1,2), torch.rand(1,2)]
    kwargs = {
        'd': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs)

def test_convert_dict_to_ordered_list_pos_args_as_kwargs():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = []
    kwargs = {
        'b': torch.rand(1,2),
        'd': torch.rand(1,2),
        'a': torch.rand(1,2),
        'c': torch.rand(1,2)
    }

    validate_ordering(nxd_model, model_params, args, kwargs)

def test_convert_dict_to_ordered_list_pos_args_as_kwargs_with_kwarg_skip():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = []
    kwargs = {
        'c': torch.rand(1,2),
        'd': torch.rand(1,2),
        'a': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip={'b'})

@pytest.mark.xfail(raises=AssertionError)
def test_convert_dict_to_ordered_list_invalid_pos_args():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = []
    kwargs = {
        'b': torch.rand(1,2),
        'd': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip={'c'})

@pytest.mark.xfail(raises=AssertionError)
def test_convert_dict_to_ordered_list_invalid_unrecognized_kwarg():
    nxd_model = NxDModel(1)

    # create mock model params
    model_params = [('a', True), ('b', False), ('c', False), ('d', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]

    args = [torch.rand(1,2)]
    kwargs = {
        'random': torch.rand(1,2),
        'd': torch.rand(1,2),
    }

    validate_ordering(nxd_model, model_params, args, kwargs, names_to_skip={'c'})

def test_router_basic():
    nxd_model = NxDModel(1)
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key1'],
    }

    assert nxd_model.router(
        [torch.rand(1,2), torch.rand(1,2)],
        ['a', 'b']
    )[0] == 'key1'

@pytest.mark.xfail(raises=KeyError)
def test_router_no_route_available():
    nxd_model = NxDModel(1)
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key1'],
    }

    nxd_model.router(
        [torch.rand(1,5), torch.rand(1,2)],
        ['a', 'c']
    )

@pytest.mark.parametrize('mode',['default','ranked','async',])
def test_forward_basic(mode):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1,1)

    # mock nxdmodel settings
    nxd_model.loaded_on_neuron = True
    model_params = [('a', True), ('b', True)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key0'],
    }
    nxd_model.flattener_map['key0'] = IdentityModule()
    nxd_model.packer_map['key0'] = IdentityModule()

    a = torch.rand(1,2)
    b = torch.rand(1,2)

    if mode == 'default':
        output = nxd_model(a,b,forward_mode=mode)
        assert torch.equal(output[0], a)
        assert torch.equal(output[1], b)
    else:
        output = nxd_model([a],[b],forward_mode=mode)
        assert torch.equal(output[0][0], a)
        assert torch.equal(output[0][1], b)

@pytest.mark.parametrize('mode',['default','ranked','async',])
def test_forward_kwargs(mode):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1,1)

    # mock nxdmodel settings
    nxd_model.loaded_on_neuron = True
    model_params = [('a', False), ('b', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key0'],
    }
    nxd_model.flattener_map['key0'] = IdentityModule()
    nxd_model.packer_map['key0'] = IdentityModule()

    a = torch.rand(1,2)
    b = torch.rand(1,2)

    if mode == 'default':
        output = nxd_model(b=b, a=a, forward_mode=mode)
        assert torch.equal(output[0], a)
        assert torch.equal(output[1], b)
    else:
        output = nxd_model(b=[b],a=[a], forward_mode=mode)
        assert torch.equal(output[0][0], a)
        assert torch.equal(output[0][1], b)

@pytest.mark.parametrize('mode',['default','ranked','async',])
def test_forward_pos_and_kwargs(mode):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1,1)

    # mock nxdmodel settings
    nxd_model.loaded_on_neuron = True
    model_params = [('a', True), ('b', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key0'],
    }
    nxd_model.flattener_map['key0'] = IdentityModule()
    nxd_model.packer_map['key0'] = IdentityModule()

    a = torch.rand(1,2)
    b = torch.rand(1,2)

    if mode == 'default':
        output = nxd_model(a, b=b, forward_mode=mode)
        assert torch.equal(output[0], a)
        assert torch.equal(output[1], b)
    else:
        output = nxd_model([a],b=[b], forward_mode=mode)
        assert torch.equal(output[0][0], a)
        assert torch.equal(output[0][1], b)

@pytest.mark.parametrize('mode',['default','ranked','async',])
def test_forward_specific_model_name(mode):
    nxd_model = generate_nxdmodel_with_mock_spmdmodels(1,2)

    # mock nxdmodel settings
    nxd_model.loaded_on_neuron = True
    model_params = [('a', True), ('b', False)]
    nxd_model.model_params = [ModelParamInfo(*model_param) for model_param in model_params]
    nxd_model.input_shape_map = {
        '(a: [1, 2], b: [1, 2])': ['key0', 'key1'],
    }
    nxd_model.flattener_map['key0'] = IdentityModule()
    nxd_model.flattener_map['key1'] = IdentityModule()
    nxd_model.packer_map['key0'] = IdentityModule()
    nxd_model.packer_map['key1'] = IdentityModule()
    def mock_model(tensors):
        if isinstance(tensors[0], list):
            return [
                [tensor+1 for tensor in tensor_collection]
                for tensor_collection in tensors
            ]
        return [tensor+1 for tensor in tensors]
    setattr(mock_model, 'forward', mock_model)
    setattr(mock_model, 'forward_ranked', mock_model)
    setattr(mock_model, 'forward_async', mock_model)
    nxd_model.spmd_models['key1'] = mock_model

    a = torch.rand(1,2)
    b = torch.rand(1,2)

    if mode == 'default':
        output = nxd_model(a, b=b, model_name='key1', forward_mode=mode)
        assert torch.equal(output[0], a + 1)
        assert torch.equal(output[1], b + 1)
    else:
        output = nxd_model([a],b=[b], model_name='key1', forward_mode=mode)
        assert torch.equal(output[0][0], a + 1)
        assert torch.equal(output[0][1], b + 1)


def test_nxd_model_load_dtype_conversion():
    # Model with mixed dtypes
    class MixedDtypeAliasedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.register_buffer('float32_counter', torch.tensor([0.0], dtype=torch.float32), persistent=True)
            self.register_buffer('int32_counter', torch.tensor([0], dtype=torch.int32), persistent=True)

        def forward(self, x):
            self.float32_counter = self.float32_counter + torch.max(x)
            self.int32_counter = self.int32_counter + torch.tensor(x.shape[0], dtype=torch.int32)

            return x + 1

    inputs = {
        'key1': (torch.rand(1, 10), None),
    }
    world_size = 1

    original_model = build_nxd_model(
        MixedDtypeAliasedModel,
        inputs,
        world_size
    )

    # Save and load the model
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        original_model.save(tmp.name)
        loaded_model = NxDModel.load(tmp.name)

        assert loaded_model.state_initializer is not None

        expected_dtypes = {
            'float32_counter': torch.float32,
            'int32_counter': torch.int32,
        }

        for name, dtype in expected_dtypes.items():
            assert name in loaded_model.state_initializer.dtypes
            assert loaded_model.state_initializer.dtypes[name] == dtype
