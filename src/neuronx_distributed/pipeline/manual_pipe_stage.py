import torch
from typing import Iterator, Union, List, Tuple, Dict, Optional, Callable
import operator

from torch.nn.parameter import Parameter

WEIGHT_SHARING_ATTR_NAME = "weight_sharing"


class PipelineStageModule(torch.nn.Module):

    def __init__(
        self,
        layers: Union[List, Tuple],
        num_stages: int,
        stage_index: int,
        partition_fn: Optional[Callable] = None,
    ):
        """
        This module is used for constructing the pipeline stage running in pipeline parallelism.
        It currently cannot handle skip connections

        Args:
            layers: list or tuple of torch.nn.Module, that represents the model in a linearized form
            num_stages: the number of pipeline stages
            stage_index: the index of the current pipeline stage
            partition_fn: a function that takes in layers and num_stages, and returns a dictionary.
                the dictionary maps every layer to the index of the pipeline stage it belongs to

        """
        super().__init__()

        self.layers = layers
        assert isinstance(self.layers, (list, tuple)), "layers must be a list or tuple"

        self.num_stages = num_stages
        self.stage_index = stage_index
        self.partition_fn = partition_fn

        # put the layers belonging to current pipeline stage to the stage_modules
        self.stage_modules = self._partition(
            self.layers, self.num_stages, self.stage_index, self.partition_fn
        )

    @staticmethod
    def _partition_evenly(
        layers: Union[List, Tuple], num_stages: int
    ) -> Dict[object, int]:
        """This function partitions the layers evenly to the `num_stage` stages

        Args:
            layers: list or tuple of torch.nn.Module, that represents the model in a linearized form
            num_stages: the number of pipeline stages

        Returns:
            a dictionary that maps every layer to the index of the pipeline stage it belongs to
        """
        nlayers = len(layers)
        chunk_size = nlayers // num_stages
        assert chunk_size > 0, "num_stages must be smaller than the number of layers"

        stage_idx = 0
        rst = {}
        for i, layer in enumerate(layers):
            if i > 0 and i % chunk_size == 0 and i // chunk_size < num_stages:
                stage_idx += 1
            rst[layer] = stage_idx

        assert (
            stage_idx == num_stages - 1
        ), f"must have {num_stages}, but got {stage_idx + 1}"

        return rst

    @staticmethod
    def _partition(
        layers: Union[List, Tuple],
        num_stages: int,
        stage_idx: int,
        partition_fn: Optional[Callable] = None,
    ) -> List[torch.nn.Module]:
        """This function partitions the layers to the `num_stage` stages based on the partition_fn

        Args:
            layers: list or tuple of torch.nn.Module, that represents the model in a linearized form
            num_stages: the number of pipeline stages
            stage_idx: the index of the current pipeline stage
            partition_fn: a function that takes in layers and num_stages, and returns a dictionary.
                the dictionary maps every layer to the index of the pipeline stage it belongs to

        Returns:
            a list of torch.nn.Module, that represents the layers belonging to the current pipeline
        """
        if partition_fn is not None:
            stage_idx_mapping = partition_fn(layers, num_stages)
        else:
            stage_idx_mapping = PipelineStageModule._partition_evenly(
                layers, num_stages
            )

        stage_modules = []
        for layer in layers:
            if stage_idx_mapping[layer] == stage_idx:
                stage_modules.append(layer)

        return stage_modules

    def forward(self, x):
        output = x
        for mod in self.stage_modules:
            output = mod(output)

        return output

    @staticmethod
    def mark_weight_sharing(
        layer_and_weight_path: List[Tuple[torch.nn.Module, str]],
        sharing_group_name: Optional[str] = "default",
    ):
        """This function marks the weight params shared across different layers.
        This function should be called before constructing the pipeline stage module.
        For example in the following model represented in a linearized form as a list
        of layers, we mark the first layer and the last layer to share the same weight:
        ```
        layers = [
            Embedding(8, 8),
            Linear(8, 16),
            Linear(16, 8),
            Linear(8, 8)
        ]

        mark_weight_sharing(
            [(layers[0], "weight"), (layers[-1], "weight")],
            "shared_first_last_layer_weight",
        )
        ```

        Args:
            layer_and_weight_path: a list of tuples, where each tuple contains the layer and
                the weight path, that is to obtain the weight tensor for sharing.
            sharing_group_name: the name of the weight sharing group, this can help to
                differentiate multiple sharing groups in the same model.

        """
        shared_weight_shape = None
        for layer, weight_path in layer_and_weight_path:
            if not hasattr(layer, WEIGHT_SHARING_ATTR_NAME):
                setattr(layer, WEIGHT_SHARING_ATTR_NAME, dict())

            shared_weights = getattr(layer, WEIGHT_SHARING_ATTR_NAME)
            if sharing_group_name in shared_weights:
                raise RuntimeError(
                    f'WARNING: weight sharing group named "{sharing_group_name}" already exists, overwriting'
                )
            shared_weights[sharing_group_name] = weight_path

            if shared_weight_shape is None:
                weight = operator.attrgetter(weight_path)(layer)
                assert isinstance(
                    weight, torch.Tensor
                ), f"Expected to get a torch.Tensor, but got {type(weight)}"
                shared_weight_shape = weight.shape

            assert (
                shared_weight_shape == operator.attrgetter(weight_path)(layer).shape
            ), "All shared weights must have the same shape"

            setattr(layer, WEIGHT_SHARING_ATTR_NAME, shared_weights)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({[mod.__str__() for mod in self.stage_modules]})"

    def __repr__(self) -> str:
        return self.__str__()

    def to(self, device):
        for layer in self.stage_modules:
            layer.to(device)
        return self

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        if len(prefix) > 0:
            prefix += "."

        for idx, mod in enumerate(self.stage_modules):
            yield f"{prefix}pipeline_stage_modules.{idx}.{mod._get_name()}", mod

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:

        for mod_name, mod in self.named_modules():
            for name, param in mod.named_parameters(
                prefix + mod_name, recurse, remove_duplicate
            ):
                yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, mod in self.named_modules():
            for _, param in mod.named_parameters(recurse=recurse):
                yield param
