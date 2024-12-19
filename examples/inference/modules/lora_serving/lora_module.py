import torch
import torch.nn as nn


from .config import LoraServingConfig
from .lora_layer import (
    MultiLoraLinear,
    MultiLoraConv2d,
    MultiLoraEmbedding,
)


class MultiLoraModule(nn.Module):
    def __init__(self, base_layer: nn.Module, lora_config: LoraServingConfig) -> None:
        if lora_config.max_lora_rank <= 0:
            raise ValueError(
                f"`lora_rank` should be a positive integer value but the value passed is {lora_config.lora_rank}"
            )

        super().__init__()
        self.lora_max_rank = lora_config.max_lora_rank
        self.max_loras = lora_config.max_loras
        self.lora_dtype = lora_config.lora_dtype
        self.base_layer = base_layer
        self.lora_config = lora_config
        self.lora_scaling = [1] * self.max_loras
        self.skip_dtype_convert = False

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # ColumnParallelLinear, RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "num_embeddings") and hasattr(base_layer, "embedding_dim"):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.lora_A, self.lora_B = None, None
        self.create_lora()


    def update_scaling(self, adapter_ids, lora_scaling):
        self.lora_scaling[adapter_ids] = lora_scaling


    def get_scaling(self, adapter_ids):
        return self.lora_scaling[adapter_ids] if adapter_ids is not None else self.lora_scaling[0]


    def create_lora(self):
        r"""
        Create the corresponding LoraAdapter according to its module type, such as nn.Linear and nn.Embedding.
        """
        raise NotImplementedError


    def get_base_layer(self) -> nn.Module:
        r"""
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer


    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        base_layer = self.get_base_layer()
        result = base_layer(x, *args, **kwargs)
        A_result = self.lora_A(x, adapter_ids)
        scaling = self.get_scaling(adapter_ids)
        result = result + self.lora_B(A_result, adapter_ids) * scaling

        if not self.skip_dtype_convert:
            result = result.to(previous_dtype)
        return result


    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep



class MultiLoraModuleLinear(MultiLoraModule):
    def create_lora(self):
        self.lora_A = MultiLoraLinear(self.max_loras, self.in_features, self.lora_max_rank, self.lora_dtype)
        self.lora_B = MultiLoraLinear(self.max_loras, self.lora_max_rank, self.out_features, self.lora_dtype)



class MultiLoraModuleConv2d(MultiLoraModule):
    def create_lora(self):
        base_layer = self.get_base_layer()

        self.lora_A = MultiLoraConv2d(
            self.max_loras,
            self.in_features,
            self.lora_max_rank,
            base_layer.kernel_size,
            base_layer.stride,
            base_layer.padding,
            self.lora_dtype
        )
        self.lora_B = MultiLoraConv2d(
            self.max_loras,
            self.lora_max_rank,
            self.out_features,
            (1, 1),
            (1, 1),
            0,
            self.lora_dtype
        )


class MultiLoraModuleEmbedding(MultiLoraModule):
    def create_lora(self):
        base_layer = self.get_base_layer()
        self.lora_A = MultiLoraEmbedding(
            self.max_loras,
            self.in_features,
            self.lora_max_rank,
            base_layer.padding_idx,
            base_layer.max_norm,
            base_layer.norm_type,
            base_layer.scale_grad_by_freq,
            base_layer.sparse,
            self.lora_dtype,
        )
        self.lora_B = MultiLoraLinear(self.max_loras, self.lora_max_rank, self.out_features, self.lora_dtype)
        self.skip_dtype_convert = True
