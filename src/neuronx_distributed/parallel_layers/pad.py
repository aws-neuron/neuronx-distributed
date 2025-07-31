import torch
from torch import nn
from torch.nn import functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target


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


def pad_model(model, tp_degree, n_heads, wrapped_classes=(), pad_hook_fn=None):
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

        You can specify a specific layer or class for your model to pad, so you aren't unnecessarily padding.
        Typically, this layer will be your Attention layer
        > model = pad_model(model, tp_degree=32, desired_config.num_heads, wrapped_classes=[MyAttention])

        You can also specify a pad_hook_fn, to be called whenever encountering an instance of wrapped_class,
        passing in said instance as a parameter, along with the tgt_src_ratio (num_heads_padded / num_heads).
        > def my_hook(attention_to_pad, tgt_src_ratio):
        >   attention_to_pad.split_size = int(model.split_size * tgt_src_ratio)
        > model = pad_model(model, tp_degree=32, desired_config.num_heads, wrapped_classes=[MyAttention],
        >   pad_hook_fn=my_hook)

    Args:
        model (torch.nn.Module): model to be padded
        tp_degree (int): tensor parallel degree
        n_heads (int): the number of heads the given model to be padded has. This can typically be found in the config
        wrapped_classes (Tuple[any], *optional*, defaults to `()`): tuple of classes (and their submodules) which
             should be padded
        pad_hook_fn (Callable[any, float], *optional*, defaults to `None`): a hook function that is called whenever
             encountering a class to pad. Receives an instance of the class to pad and the
             tgt_src_ratio (num_heads_padded / num_heads)as its argument

    Returns:
        torch.nn.Module: padded model
    """

    def pad_helper(model, tgt_src_ratio, should_pad, pad_hook_fn):
        # Recursive helper to not repeat any initial calculations/work and allow us to easily track when to pad.
        for _, module in model.named_children():
            pad_helper(module, tgt_src_ratio, should_pad or isinstance(module, wrapped_classes), pad_hook_fn)

        # Note: many models don't use split_size to split the heads after fusing, but they still keep the field
        if should_pad:
            if pad_hook_fn:
                pad_hook_fn(model, tgt_src_ratio)

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

    # We use tgt_src_ratio to figure out how much we have to pad by, but we could also just use n_heads_padded/n_heads?
    n_heads_padded = n_heads + get_number_of_extra_heads(n_heads, tp_degree)

    tgt_src_ratio = n_heads_padded / n_heads

    wrapped_classes = tuple(wrapped_classes)
    should_pad = not wrapped_classes or isinstance(model, wrapped_classes)

    return pad_helper(model, tgt_src_ratio, should_pad, pad_hook_fn)


def generate_padding_mask(
    num_heads, num_heads_with_pad, num_kv_heads, tp_degree, tp_rank, hardware_type=None
):
    """
    Gets the padding mask for a given attention config, TP degree, TP rank, and the hardware type.
    Different hardware types (e.g. TRN1 vs. TRN2) require different masking, due to
    different K/V head weight layout on TP ranks.

    Args:
        num_heads: number of query heads in the (unsharded) model w/o padding.
        num_heads_with_pad: number of query heads in the (unsharded) model w/ padding.
        num_kv_heads: number of kv heads in the (unsharded) model.
        tp_degree: TP degree.
        tp_rank: TP rank of the worker to generate padding mask for.
        hardware_type: type of the hardware to generate attention mask for.
    Returns:
        `torch.Tensor` 1D mask tensor whose length is the number of heads on each rank.
        1s for original heads and 0s for padded heads.

    For example, a model w/ 48 query heads and 8 kv heads in TP32:
        - The number of query heads is padded from 48 to 64.
        - The number of kv heads is replicated from 8 to 32.
        - Each rank gets 2 query heads and 1 kv head.
    We need to mask out 16 query heads (2 per kv head) to make sure we don't use more
    heads than in the config.
    """
    if hardware_type is None:
        hardware_type = hardware(get_platform_target())
    num_heads_with_pad_per_rank = num_heads_with_pad // tp_degree
    num_heads_per_kv_head = num_heads // num_kv_heads

    if hardware_type == hardware.TRN1:
        # On TRN1, KV heads are replicated in groups.
        # For example, TP32 with 56 query heads, 8 kv heads and kv_replicator=4:
        # (K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7)
        # We need to mask out the 1 out of 2 query heads on each TP rank in the last
        # group of (K0,K1,...K7).
        kv_group_idx = tp_rank // num_kv_heads

        # Low and high query head index of the local query heads among all query heads
        # associated with the KV head.
        low_idx, high_idx = (
            num_heads_with_pad_per_rank * kv_group_idx,
            num_heads_with_pad_per_rank * (kv_group_idx + 1),
        )

        padding_mask = torch.arange(low_idx, high_idx) < num_heads_per_kv_head
    elif hardware_type == hardware.TRN2:
        # On TRN2, a KV head is replicated on adjacent ranks.
        # For example, TP32 with 48 query heads, 8 kv heads and kv_replicator=4:
        # (K0,K0,K0,K0),(K1,K1,K1,K1),...,(K7,K7,K7,K7)
        # We need to mask out the query heads associated with the last replicate
        # kv head in each 4 replicates.
        # Note that LNC is irrelevant to attention head padding / masking.
        kv_replicator = max(tp_degree // num_kv_heads, 1)
        replicate_id = tp_rank % kv_replicator

        # Low and high query head index of the local query heads among all query heads
        # associated with the KV head.
        low_idx, high_idx = (
            num_heads_with_pad_per_rank * replicate_id,
            num_heads_with_pad_per_rank * (replicate_id + 1),
        )

        num_kv_heads_per_rank = (num_kv_heads * kv_replicator) // tp_degree
        padding_mask = torch.arange(low_idx, high_idx) < num_heads_per_kv_head * num_kv_heads_per_rank
    else:
        raise RuntimeError(
            f"Unexpected hardware_type {hardware_type} received in padding mask generation."
        )

    padding_mask.requires_grad = False
    return padding_mask
