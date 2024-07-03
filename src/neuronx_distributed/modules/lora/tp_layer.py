import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

from .config import LoraConfig
from .layer import LoraLinear


class LoraParallelLinear(LoraLinear):
    r"""
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(self, base_layer: nn.Module, lora_config: LoraConfig) -> None:
        super().__init__(base_layer, lora_config)

    def update_layer(self, lora_config, init_method=xavier_normal_):
        input_is_parallel = (
            self.base_layer.input_is_parallel if isinstance(self.base_layer, RowParallelLinear) else True
        )
        gather_output = self.base_layer.gather_output if isinstance(self.base_layer, ColumnParallelLinear) else False

        lora_dropout = lora_config.lora_dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if isinstance(self.base_layer, RowParallelLinear):
            self.lora_A = RowParallelLinear(
                input_size=self.in_features,
                output_size=self.lora_rank,
                bias=False,
                input_is_parallel=input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
            )
            self.lora_B = nn.Linear(
                in_features=self.lora_rank, out_features=self.out_features, bias=False, dtype=torch.float32
            )
        else:
            self.lora_A = nn.Linear(
                in_features=self.in_features, out_features=self.lora_rank, bias=False, dtype=torch.float32
            )
            self.lora_B = ColumnParallelLinear(
                input_size=self.lora_rank,
                output_size=self.out_features,
                bias=False,
                gather_output=gather_output,
                init_method=init_method,
            )

        if lora_config.use_rslora:
            self.scaling = self.lora_alpha / (self.lora_rank**0.5)
        else:
            self.scaling = self.lora_alpha / self.lora_rank

        self.init_lora_parameters(lora_config.init_lora_weights)
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
