import torch
import torch.nn as nn
import warnings
import math
from torch.nn.init import xavier_normal_

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.modules.qkv_linear import (
    GQAQKVColumnParallelLinear,
    gqa_qkv_linear_with_async_allreduce,
    gather_from_tensor_model_parallel_region
)

from .config import LoraConfig
from .layer import LoraLinear, LoraLayer
from typing import Any, Optional, Tuple


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
        base_layer = self.get_base_layer()
        sequence_parallel_enabled = base_layer.sequence_parallel_enabled
        lora_dropout = lora_config.lora_dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if isinstance(self.base_layer, RowParallelLinear):
            self.lora_A = RowParallelLinear(
                input_size=self.in_features,
                output_size=self.lora_rank,
                bias=False,
                input_is_parallel=base_layer.input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
                sequence_parallel_enabled = sequence_parallel_enabled,
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
                gather_output=base_layer.gather_output,
                init_method=init_method,
                sequence_parallel_enabled = sequence_parallel_enabled,
            )

        self.init_lora_parameters(lora_config.init_lora_weights)



class LoraGQAQKVParallelLinear(LoraLayer):
    r"""
    When the target layer parallel_linear is GQAQKVColumnParallelLinear, in order to keep the input and output shapes
    consistent, we perform column segmentation on lora_B, while lora_A is still a complete linear layer.
    """
    def __init__(self, base_layer: nn.Module, lora_config: LoraConfig) -> None:
        super().__init__(base_layer, lora_config)
        self.update_layer(lora_config)

    def update_layer(self, lora_config, init_method=xavier_normal_):
        base_layer = self.get_base_layer()
        lora_dropout = lora_config.lora_dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.sequence_parallel_enabled = base_layer.sequence_parallel_enabled
        self.kv_size_multiplier = base_layer.kv_size_multiplier
        self.gather_output = base_layer.gather_output
        self.lora_A = nn.Linear(
            in_features=self.in_features, out_features=self.lora_rank, bias=False, dtype=torch.float32
        )
        self.lora_B = GQAQKVColumnParallelLinear(
            input_size=self.lora_rank,
            output_sizes=self.out_features,
            bias=False,
            gather_output=self.gather_output,
            dtype=torch.float32,
            init_method=init_method,
            kv_size_multiplier=self.kv_size_multiplier,
            sequence_parallel_enabled = self.sequence_parallel_enabled,
        )

        self.init_lora_parameters(lora_config.init_lora_weights)


    def init_lora_parameters(self, init_lora_weights):
        init_lora_weights = init_lora_weights.lower()
        assert init_lora_weights in ["default", "gaussian"]

        if init_lora_weights == "default":
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.lora_rank)
        else:
            raise ValueError(f"Unknown LoRA parameters initialization with {init_lora_weights}")

        q, k, v = self.get_qkv(self.lora_B)
        nn.init.zeros_(q.data)
        nn.init.zeros_(k.data)
        nn.init.zeros_(v.data)


    def merge(self) -> None:
        """
        Merge the adapter weights into the base weights
        """
        weight_q, weight_k, weight_v = self.get_qkv(self.base_layer)
        delta_weight_q, delta_weight_k, delta_weight_v = self.get_delta_weight()

        weight_q.data += delta_weight_q
        weight_k.data += delta_weight_k
        weight_v.data += delta_weight_v
        self.merged = True


    def unmerge(self) -> None:
        """
        This method unmerges merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        q, k, v = self.get_qkv(self.base_layer)
        delta_weight_q, delta_weight_k, delta_weight_v = self.get_delta_weight()

        q.data -= delta_weight_q
        k.data -= delta_weight_k
        v.data -= delta_weight_v
        self.merged = False


    def get_qkv(self, layer):
        return layer.weight_q, layer.weight_k, layer.weight_v


    def get_delta_weight(self) -> torch.Tensor:
        weight_A = self.lora_A.weight
        q_lora_B, k_lora_B, v_lora_B = self.get_qkv(self.lora_B)

        output_q = (q_lora_B @ weight_A) * self.scaling
        output_k = (k_lora_B @ weight_A) * self.scaling
        output_v = (v_lora_B @ weight_A) * self.scaling

        return output_q, output_k, output_v


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        previous_dtype = x.dtype
        if self.merged:
            output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
        else:
            output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
            lora_A = self.lora_A
            dropout = self.lora_dropout
            scaling = self.scaling
            x = x.to(lora_A.weight.dtype)

            q_lora_B, k_lora_B, v_lora_B = self.get_qkv(self.lora_B)
            dropout_input = lora_A(dropout(x))
            lora_q_output, lora_k_output, lora_v_output = self._lora_forward(dropout_input, q_lora_B, k_lora_B, v_lora_B)

            output_q += lora_q_output * scaling
            output_k += lora_k_output * scaling
            output_v += lora_v_output * scaling

        return output_q.to(previous_dtype), output_k.to(previous_dtype), output_v.to(previous_dtype)


    def _lora_forward(self, input, weight_q, weight_k, weight_v):
        # Matrix multiply.
        output_parallel_q, output_parallel_k, output_parallel_v = gqa_qkv_linear_with_async_allreduce(
            input=input,
            weight_q=weight_q,
            weight_k=weight_k,
            weight_v=weight_v,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            async_grad_allreduce=not self.sequence_parallel_enabled,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            kv_size_multiplier=self.kv_size_multiplier,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output_q = gather_from_tensor_model_parallel_region(output_parallel_q)
            output_k = gather_from_tensor_model_parallel_region(output_parallel_k)
            output_v = gather_from_tensor_model_parallel_region(output_parallel_v)
        else:
            output_q, output_k, output_v = output_parallel_q, output_parallel_k, output_parallel_v
        return output_q, output_k, output_v
