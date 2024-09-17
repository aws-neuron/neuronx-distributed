from typing import List, Dict

import torch
from torch_neuronx.xla_impl import structure

def default_bucket_kernel(inputs: List[torch.Tensor]):
    return inputs, torch.tensor(0).to(torch.int)

class SPMDBucketModelScript(torch.nn.Module):
    """
    BucketModelScript mostly remains the same. Just that he data needs to be passed in.
    """

    def __init__(self, compiled_models: List[torch.classes.neuron.SPMDModel]):
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

class SPMDBucketModel(torch.nn.Module):
    """
    This classes implements bucketing with the SPMDModel runtime class.
     The major difference from torch_neuronx's BucketModel class is that the
     weights are not registered in this class but passed in as parameter in the forward.
    """

    def __init__(
        self,
        bucket_kernel,
        bucket_kernel_constant_args,
        bucket_model_executor: SPMDBucketModelScript,
    ):
        super().__init__()
        # bucket kernel & preprocessors goes here
        # weights and states are passed in
        self.bucket_kernel = bucket_kernel
        self.bucket_kernel_constant_args = bucket_kernel_constant_args

        self.bucket_model_executor = bucket_model_executor

    def forward(
        self,
        inputs: List[torch.Tensor],
    ):
        preprocessed_inputs, bucket_idx_tensor = self.bucket_kernel(
            inputs, *self.bucket_kernel_constant_args
        )

        return self.bucket_model_executor(preprocessed_inputs, bucket_idx_tensor)

class StateInitializer(torch.nn.Module):
    # torchscript cannot script dict of with values of different types
    # so we store shapes and dtypes in separate dicts
    def __init__(self, shapes, dtypes, tp_degree):
        super().__init__()
        self.shapes = shapes
        self.dtypes = dtypes
        self.tp_degree = tp_degree

    def forward(self):
        results : List[Dict[str, torch.Tensor]]= []
        for rank in range(0, self.tp_degree):
            states = {}
            for key in self.shapes.keys():
                states[key] = torch.zeros(self.shapes[key], dtype=self.dtypes[key], device=f"privateuseone:{rank}")
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
        tp_degree: int,
        flattener_map: torch.nn.ModuleDict,
        input_shape_map: Dict[str, str],
        packer: structure.Packer,
        state_initializer: StateInitializer,
        weight_loader: torch.classes.neuron.LayoutTransformation
    ):
        super().__init__()
        self.models = models
        self.flattener_map = flattener_map
        self.packer = packer
        self.input_shape_map =input_shape_map
        self.state_initializer = state_initializer
        self.tp_degree = tp_degree
        self.weight_loader = weight_loader

        # default values only used for scripting to prevent it complaining about empty lists
        self.weights: List[Dict[str, torch.Tensor]] = [{'__neuronprivatetensor__':torch.tensor(0)}]
        self.state: List[Dict[str, torch.Tensor]] = [{'__neuronprivatetensor__':torch.tensor(0)}]

    def initialize_spmd_models(
        self,
        states: List[Dict[str, torch.Tensor]],
        weights: List[Dict[str, torch.Tensor]]
    ):
        for bucket_model in self.models.values():
            for model in bucket_model.bucket_model_executor.models:
                model.initialize(states, weights)

    @torch.jit.export
    def initialize(self, checkpoint: List[Dict[str, torch.Tensor]]):
        if self.weight_loader is not None:
            self.weights = self.weight_loader.forward(checkpoint, False)
        else:
            self.weights = torch.ops.neuron._parallel_load(checkpoint)
    
        if (self.state_initializer is not None):
            self.state = self.state_initializer()
            self.initialize_spmd_models(self.state,self.weights)
        else:
            self.initialize_spmd_models([],self.weights)

    @torch.jit.export
    def initialize_with_saved_weights(self):
        if (self.state_initializer is not None):
            self.state = self.state_initializer()
            self.initialize_spmd_models(self.state,self.weights)
        else:
            self.initialize_spmd_models([],self.weights)

    @torch.jit.unused
    def mock_initialization(self, mock: bool):
        """
        This function is only used once for jit tracing,
        and won't be serialized on jit.save
        """
        for script_model in self.models.modules():
            if script_model.original_name == SPMDBucketModel.__name__:
                for model in script_model.bucket_model_executor.models:
                    model.set_mock_initialized(mock)

    def router(self, inputs: List[torch.Tensor]) -> str:
        actual_shape = str([tensor.shape for tensor in inputs])
        return self.input_shape_map[actual_shape]

    def forward(self, inputs: List[torch.Tensor]):
        """ """
        model_name = self.router(inputs)

        # Initialize empty tensor to ensure jit.script gets the write type
        flattened_inputs : List[torch.Tensor] = [torch.zeros(0)]

        # torch.jit.script does not allow indexing of ModuleDict
        # so we work around by looping and conditionally executing it
        for name, flattener in self.flattener_map.items():
            if name == model_name:
                flattened_inputs = flattener(inputs)

        result: List[torch.Tensor] = [torch.zeros(0)]
        for name, model in self.models.items():
            if name == model_name:
                result = model.forward(flattened_inputs)

        result = self.packer(result)
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
