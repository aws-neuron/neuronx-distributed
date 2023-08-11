from torch import nn
from torch.nn import functional as F
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.utils.model_utils import is_hf_pretrained_model


def get_number_of_extra_heads(n_head, tp_degree):
    """
    Get number of extra heads needed to pad

    Args:
        n_head (int): number of heads in source/initial model
        tp_degree (int): tensor parallel degree

    Returns:
        int: extra heads
    """
    if n_head % tp_degree == 0:
        extra_heads = 0
    else:
        extra_heads = tp_degree - n_head % tp_degree
    return extra_heads


def pad_model(model, tp_degree, n_heads):
    """
    Pads a generic model to function to a desired tensor parallelism degree by padding the number of attention heads.
    Returns the original model modified with padding.

    Uses 1-axis padding strategy: pads the sharded dim of the ParallelLinear layers to the size it would have been
     for the padded number of heads.

    Usage:
        When modifying the Attention layer, typically you must divide by TP degree like so:
        > self.num_heads = neuronx_dist_utils.divide(self.num_heads, get_tensor_model_parallel_size())

        This line must be modified like so:
        > self.num_heads = neuronx_dist_utils.divide(
            self.num_heads + get_number_of_extra_heads(self.num_heads, get_tensor_model_parallel_size()),
            get_tensor_model_parallel_size())

        Then, after initializing the model, you must call this wrapper:
        > model = get_model(config=desired_config)
        > model = pad_model(model, tp_degree=32, desired_config.num_heads)  # Use the model as desired after this point

    Args:
        model (torch.nn.Module): model to be padded
        tp_degree (int): tensor parallel degree
        n_heads (int): the number of heads the given model to be padded has. This can typically be found in the config

    Returns:
        torch.nn.Module: padded model
    """
    def pad_helper(model, tgt_src_ratio):
        # Recursive helper to not repeat the calculations and not need to pass down the model config
        for _, module in model.named_children():
            pad_helper(module, tgt_src_ratio)

        # Note: many models don't use split_size to split the heads after fusing, but they still keep the field
        if is_hf_model and hasattr(model, "split_size"):
            model.split_size = int(model.split_size * tgt_src_ratio)

        if isinstance(model, ColumnParallelLinear):
            # pad output dim (dim=0)
            size_to_pad = int(model.weight.shape[0] * tgt_src_ratio - model.weight.shape[0])
            model.weight = nn.Parameter(F.pad(model.weight.data, (0, 0, 0, size_to_pad)))
            if model.bias is not None:  # bias may not always exist
                model.bias = nn.Parameter(F.pad(model.bias.data, (0, size_to_pad)))

        elif isinstance(model, RowParallelLinear):
            # pad input dim (dim=1)
            size_to_pad = int(model.weight.shape[1] * tgt_src_ratio - model.weight.shape[1])
            model.weight = nn.Parameter(F.pad(model.weight.data, (0, size_to_pad)))  # along dim = 1
            # ignore bias b/c bias not sharded

        return model

    # must do here b/c attn layer (which contains split size) does not inherit from PreTrainedModel
    is_hf_model = is_hf_pretrained_model(model)

    # We use tgt_src_ratio to figure out how much we have to pad by, but we could also just use n_heads_padded/n_heads?
    n_heads_padded = n_heads + get_number_of_extra_heads(n_heads, tp_degree)

    tgt_src_ratio = n_heads_padded / n_heads

    return pad_helper(model, tgt_src_ratio)
