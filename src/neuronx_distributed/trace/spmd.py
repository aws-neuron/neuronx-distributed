import logging
import math
from typing import List, Dict, Tuple

from neuronx_distributed.parallel_layers.utils import is_torch_version_greater_than_2
import torch
from torch_neuronx.xla_impl import structure

logger = logging.getLogger("Neuron")

def default_bucket_kernel(inputs: List[torch.Tensor]):
    return inputs, torch.tensor(0).to(torch.int)


class SPMDBucketModelScript(torch.nn.Module):
    """
    BucketModelScript mostly remains the same. Just that he data needs to be passed in.
    """
    def __init__(self, compiled_models: List[torch.classes.neuron.SPMDModel]): # type: ignore[name-defined, torch.classes.neuron.SPMDModel]
        super().__init__()
        self.models = compiled_models

    def forward(
        self,
        inputs: List[torch.Tensor],
        bucket_idx_tensor: torch.Tensor,
    ):
        bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
        initialized = self.models[bucket_idx].is_initialized()

        if initialized:
            output = self.models[bucket_idx].forward(inputs)
            return output
        else:
            raise ValueError("This model is not initialized, please call traced_model.nxd_model.initialize(sharded_checkpoint) or traced_model.nxd_model.initialize_with_saved_weights()")

    @torch.jit.export
    def forward_async(
        self,
        input_collection: List[List[torch.Tensor]],
        bucket_idx_tensor: torch.Tensor
    ):
        bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
        initialized = self.models[bucket_idx].is_initialized()

        if initialized:
            output = self.models[bucket_idx].forward_async(input_collection)
            return output
        else:
            raise ValueError("This model is not initialized, please call traced_model.nxd_model.initialize(sharded_checkpoint) or traced_model.nxd_model.initialize_with_saved_weights()")

    @torch.jit.export
    def forward_ranked(
        self,
        input_collection: List[List[torch.Tensor]],
        bucket_idx_tensor: torch.Tensor
    ):
        bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
        initialized = self.models[bucket_idx].is_initialized()

        if initialized:
            output = self.models[bucket_idx].forward_ranked(input_collection)
            return output
        else:
            raise ValueError("This model is not initialized, please call traced_model.nxd_model.initialize(sharded_checkpoint) or traced_model.nxd_model.initialize_with_saved_weights()")


class StateInitializer(torch.nn.Module):
    kv_cache_keys_map: Dict[str, Tuple[str, str]]
    state_keys: List[str]

    # torchscript cannot script dict of with values of different types
    # so we store shapes and dtypes in separate dicts
    def __init__(self, shapes, dtypes, local_ranks_size, combine_kv_on_device=False):
        super().__init__()
        self.shapes = shapes
        self.dtypes = dtypes
        self.local_ranks_size = local_ranks_size
        self.kv_cache_keys_map = {}
        self.state_keys = []

        kv_cache_keys = []
        for key in self.shapes.keys():
            if combine_kv_on_device and ".past_key_values." in key:
                kv_cache_keys.append(key)
            else:
                self.state_keys.append(key)

        # Iterate through kv_cache_keys in pairs.
        it = iter(kv_cache_keys)
        for idx, (k_key, v_key) in enumerate(zip(it, it)):
            k_shape = self.shapes[k_key]
            v_shape = self.shapes[v_key]
            k_dtype = self.dtypes[k_key]
            v_dtype = self.dtypes[v_key]
            if k_dtype != v_dtype or math.prod(k_shape) != math.prod(v_shape):
                raise ValueError("Could not combine KV allocations due to incompatible dtype or shape")

            kv_key = k_key.rsplit(".", 1)[0] + f".combined.{idx}"
            kv_shape = [2, *k_shape]
            logger.debug(f"Will combine {k_key} and {v_key} into {kv_key}, shape: {kv_shape}")
            # Insert combined kv_key and kv_shape into state_keys and shapes so they are created on device.
            self.state_keys.append(kv_key)
            self.dtypes[kv_key] = k_dtype
            self.shapes[kv_key] = kv_shape
            # Map the combined kv_key back to (k_key, v_key) so they can be split once on device.
            self.kv_cache_keys_map[kv_key] = (k_key, v_key)

    def forward(self):
        cpu_states : Dict[str, torch.Tensor] = {}

        # Create states on CPU
        for key in self.state_keys:
            dtype = self.dtypes[key]
            shape = self.shapes[key]
            if is_torch_version_greater_than_2() and dtype == torch.float8_e4m3fn:
                val = torch.zeros(shape, dtype=torch.bfloat16).to(dtype=dtype)
            else:
                val = torch.zeros(shape, dtype=dtype)
            cpu_states[key] = val

        results : List[Dict[str, torch.Tensor]]= []

        # Copy CPU states to all ranks.
        for rank in range(0, self.local_ranks_size):
            states = {}

            for key, val in cpu_states.items():
                val = val.to(device=f"privateuseone:{rank}")
                states[key] = val

                # Split K and V on device.
                if key in self.kv_cache_keys_map:
                    k_key, v_key = self.kv_cache_keys_map[key]
                    states[k_key] = val[0].view(self.shapes[k_key])
                    states[v_key] = val[1].view(self.shapes[v_key])

            results.append(states)

        return results

class NxDModel(torch.nn.Module):
    """
    NxDModel runs houses multiple SPMD bucket models that share the same weights.
    Note: The weights must be sharded the same way across all SPMD bucket models.
    """

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        flattener_map: torch.nn.ModuleDict,
        input_shape_map: Dict[str, Tuple[str, int]],
        packer: structure.Packer,
        state_initializer: StateInitializer,
        weight_loader: torch.classes.neuron.LayoutTransformation, # type: ignore[name-defined, torch.classes.neuron.LayoutTransformation]
        start_rank_id:int = 0,
    ):
        super().__init__()
        self.models = models
        self.flattener_map = flattener_map
        self.packer = packer
        self.input_shape_map = input_shape_map
        self.state_initializer = state_initializer
        self.weight_loader = weight_loader

        # default values only used for scripting to prevent it complaining about empty lists
        self.weights: List[Dict[str, torch.Tensor]] = [{'__neuronprivatetensor__':torch.tensor(0)}]
        self.state: List[Dict[str, torch.Tensor]] = [{'__neuronprivatetensor__':torch.tensor(0)}]
        self.start_rank_id:int = start_rank_id

    def initialize_spmd_models(
        self,
        states: List[Dict[str, torch.Tensor]],
        weights: List[Dict[str, torch.Tensor]],
        start_rank_id: int,
    ):
        for bucket_model in self.models.values():
            for model in bucket_model.models:
                model.initialize(states, weights, start_rank_id)

    @torch.jit.export
    def initialize(self, checkpoint: List[Dict[str, torch.Tensor]], start_rank_tensor: torch.Tensor):
        start_rank = torch.ops.aten.Int(start_rank_tensor)
        if self.weight_loader is not None:
            self.weights = self.weight_loader.forward(checkpoint, False)
        else:
            self.weights = torch.ops.neuron._parallel_load(checkpoint)

        if (self.state_initializer is not None):
            self.state = self.state_initializer()
            self.initialize_spmd_models(self.state, self.weights, start_rank)
        else:
            self.initialize_spmd_models([],self. weights, start_rank)

    @torch.jit.export
    def initialize_with_saved_weights(self, start_rank_tensor: torch.Tensor):
        start_rank = torch.ops.aten.Int(start_rank_tensor)
        if (self.state_initializer is not None):
            self.state = self.state_initializer()
            self.initialize_spmd_models(self.state,self.weights, start_rank)
        else:
            self.initialize_spmd_models([],self.weights, start_rank)

    @torch.jit.unused
    def mock_initialization(self, mock: bool):
        """
        This function is only used once for jit tracing,
        and won't be serialized on jit.save
        """
        for script_model in self.models.modules():
            if script_model.original_name == SPMDBucketModelScript.__name__:
                for model in script_model.models:
                    model.set_mock_initialized(mock)

    def router(self, inputs: List[torch.Tensor]) -> Tuple[str, int]:
        actual_shape = str([tensor.shape for tensor in inputs])
        if actual_shape not in self.input_shape_map:
            raise ValueError(f"Input shape {actual_shape} not found in input_shape_map: {self.input_shape_map.keys()}")
        return self.input_shape_map[actual_shape]

    def forward(self, inputs: List[torch.Tensor]):
        """ """
        model_name, bucket_idx = self.router(inputs)

        # Initialize empty tensor to ensure jit.script gets the write type
        flattened_inputs : List[torch.Tensor] = [torch.zeros(0)]

        # torch.jit.script does not allow indexing of ModuleDict
        # so we work around by looping and conditionally executing it
        for name, flattener in self.flattener_map.items():
            if name == f"{model_name}_{bucket_idx}":
                flattened_inputs = flattener(inputs)

        result: List[torch.Tensor] = [torch.zeros(0)]
        for name, model in self.models.items():
            if name == model_name:
                result = model.forward(flattened_inputs, torch.tensor(bucket_idx, dtype=torch.int32))

        result = self.packer(result)
        return result

    @torch.jit.export
    def forward_async(self, input_collection: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        sample_inputs: List[torch.Tensor] = input_collection[0]
        model_name, bucket_idx = self.router(sample_inputs)

        flattened_input_collection: List[List[torch.Tensor]] = []
        for name, flattener in self.flattener_map.items():
            if name == f"{model_name}_{bucket_idx}":
                for inputs in input_collection:
                    flattened_inputs = flattener(inputs)
                    flattened_input_collection.append(flattened_inputs)

        result: List[List[torch.Tensor]] = []
        for name, model in self.models.items():
            if name == model_name:
                result = model.forward_async(flattened_input_collection, torch.tensor(bucket_idx, dtype=torch.int32))

        return result

    @torch.jit.export
    def forward_ranked(self, input_collection: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        sample_inputs: List[torch.Tensor] = input_collection[0]
        model_name, bucket_idx = self.router(sample_inputs)

        flattened_input_collection: List[List[torch.Tensor]] = []
        for name, flattener in self.flattener_map.items():
            if name == f"{model_name}_{bucket_idx}":
                for inputs in input_collection:
                    flattened_inputs = flattener(inputs)
                    flattened_input_collection.append(flattened_inputs)

        result: List[List[torch.Tensor]] = []
        for name, model in self.models.items():
            if name == model_name:
                result = model.forward_ranked(flattened_input_collection, torch.tensor(bucket_idx, dtype=torch.int32))

        return result


class NxDModelExecutor(torch.nn.Module):
    """
    Wraps over jit scripted NxDModel class
    so traced model can be executed like traced_model(*inputs)
    """
    def __init__(self,nxd_model):
        super().__init__()
        self.nxd_model = nxd_model

    def forward(self, *inputs):
        return self.nxd_model(list(inputs))
