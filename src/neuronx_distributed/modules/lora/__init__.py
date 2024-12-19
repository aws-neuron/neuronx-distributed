from .config import LoraConfig
from .model import LoraModel

__all__ = ["LoraConfig", "LoraModel", "get_lora_model"]


def get_lora_model(model, lora_config: LoraConfig):
    if lora_config is None:
        return model

    from neuronx_distributed.trainer.model import NxDModel

    if isinstance(model, NxDModel):
        model.module = LoraModel(model.module, lora_config)
    else:
        model = LoraModel(model, lora_config)
    return model
