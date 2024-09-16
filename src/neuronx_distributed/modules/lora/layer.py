import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.utils.model_utils import is_hf_transformers_available

from .config import LoraConfig


class LoraLayer(torch.nn.Module, ABC):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("lora_rank", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: torch.nn.Module, lora_config: LoraConfig) -> None:
        if lora_config.lora_rank <= 0:
            raise ValueError(
                f"`lora_rank` should be a positive integer value but the value passed is {lora_config.lora_rank}"
            )

        super().__init__()
        self.lora_rank = lora_config.lora_rank
        self.lora_alpha = lora_config.lora_alpha
        self.base_layer = base_layer
        self.lora_A = None
        self.lora_B = None
        self.lora_embedding_A = None
        self.lora_embedding_B = None
        self.lora_config = lora_config

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # ColumnParallelLinear, RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_sizes"):
            # GQAQKVColumnParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_sizes
        else:
            if is_hf_transformers_available():
                from transformers.pytorch_utils import Conv1D

                if isinstance(base_layer, Conv1D):
                    in_features, out_features = (
                        base_layer.weight.ds_shape
                        if hasattr(base_layer.weight, "ds_shape")
                        else base_layer.weight.shape
                    )
            else:
                raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.merged = False
        if lora_config.use_rslora:
            self.scaling = self.lora_alpha / math.sqrt(self.lora_rank)
        else:
            self.scaling = self.lora_alpha / self.lora_rank

    def get_base_layer(self) -> torch.nn.Module:
        r"""
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. For example, when loading a saved LoRA adapter for serving,
                you can check if this LoRA adapter has NaNs. Defaults to `False`.
        """
        base_layer = self.get_base_layer()
        if safe_merge:
            # Note that safe_merge will be slower than the normal merge
            # because of the copy operation.
            orig_weights = base_layer.weight.data.clone()
            orig_weights += self.get_delta_weight()

            if not torch.isfinite(orig_weights).all():
                raise ValueError("NaNs detected in the merged weights. The adapter seems to be broken")
            base_layer.weight.data = orig_weights
        else:
            base_layer.weight.data += self.get_delta_weight()
        self.merged = True

    def unmerge(self) -> None:
        """
        This method unmerges merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.get_base_layer().weight.data -= self.get_delta_weight()
        self.merged = False

    @abstractmethod
    def update_layer(self):
        r"""
        inject LoRA matrices into the base layer.
        """

    @abstractmethod
    def get_delta_weight(self) -> torch.Tensor:
        r"""
        return the matrix multiplication of A and B in LoRA.
        """

    def init_lora_parameters(self, init_lora_weights):
        init_lora_weights = init_lora_weights.lower()
        assert init_lora_weights in ["default", "gaussian"]

        if self.lora_A:
            if init_lora_weights == "default":
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif init_lora_weights == "gaussian":
                nn.init.normal_(self.lora_A.weight, std=1 / self.lora_rank)
            else:
                raise ValueError(f"Unknown LoRA parameters initialization with {init_lora_weights}")
            nn.init.zeros_(self.lora_B.weight)

        if self.lora_embedding_A is not None:
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A)
            nn.init.normal_(self.lora_embedding_B)

    def transpose(self, weight):
        if isinstance(weight, torch.nn.Parameter):
            return torch.nn.Parameter(weight.T)
        return weight.T

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class LoraLinear(LoraLayer):
    # Lora implemented in a dense layer
    def __init__(self, base_layer, lora_config: LoraConfig, is_conv_1d_layer=False) -> None:
        super().__init__(base_layer, lora_config)
        self.is_conv_1d_layer = is_conv_1d_layer
        self.update_layer(lora_config)

    def update_layer(self, lora_config: LoraConfig):
        lora_dropout = lora_config.lora_dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, self.lora_rank, bias=False)
        self.lora_B = nn.Linear(self.lora_rank, self.out_features, bias=False)

        self.init_lora_parameters(lora_config.init_lora_weights)
        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the matrix multiplication of A and B in LoRA.
        """
        device = self.lora_B.weight.device
        dtype = self.lora_B.weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 or bfloat16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16/bf16.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A.weight
        weight_B = self.lora_B.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if not self.is_conv_1d_layer:
            output_tensor = (weight_B @ weight_A) * self.scaling
        else:
            output_tensor = self.transpose(weight_B @ weight_A) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A.weight.data = weight_A.to(dtype)
            self.lora_B.weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            lora_A = self.lora_A
            lora_B = self.lora_B
            dropout = self.lora_dropout
            scaling = self.scaling
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result


class LoraEmbedding(LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(self, base_layer, lora_config: LoraConfig) -> None:
        super().__init__(base_layer, lora_config)
        self.update_layer(lora_config)

    def update_layer(self, lora_config: LoraConfig):
        lora_dropout = lora_config.lora_dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Actual trainable parameters
        weight_A = torch.randn(self.lora_rank, self.in_features)
        weight_B = torch.randn(self.out_features, self.lora_rank)
        self.lora_embedding_A = nn.Parameter(weight_A)
        self.lora_embedding_B = nn.Parameter(weight_B)

        self.init_lora_parameters(lora_config.init_lora_weights)
        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B.device
        dtype = self.lora_embedding_A.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A
        weight_B = self.lora_embedding_B

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = self.transpose(weight_B @ weight_A) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A = weight_A.to(dtype)
            self.lora_embedding_B = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            embedding_A = self.lora_embedding_A.T
            embedding_B = self.lora_embedding_B.T
            scaling = self.scaling
            after_A = self._embed(x, embedding_A)
            result += (after_A @ embedding_B) * scaling

        result = result.to(previous_dtype)
        return result


class LoraConv2d(LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(self, base_layer, lora_config: LoraConfig) -> None:
        super().__init__(base_layer, lora_config)
        self.update_layer(lora_config)

    def update_layer(self, lora_config: LoraConfig):
        lora_dropout = lora_config.lora_dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A = nn.Conv2d(self.in_features, self.lora_rank, kernel_size, stride, padding, bias=False)
        self.lora_B = nn.Conv2d(self.lora_rank, self.out_features, (1, 1), (1, 1), bias=False)

        self.init_lora_parameters(lora_config.init_lora_weights)
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B.weight.device
        dtype = self.lora_A.weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A.weight
        weight_B = self.lora_B.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A.weight.data = weight_A.to(dtype)
            self.lora_B.weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            lora_A = self.lora_A
            lora_B = self.lora_B
            dropout = self.lora_dropout
            scaling = self.scaling
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result
