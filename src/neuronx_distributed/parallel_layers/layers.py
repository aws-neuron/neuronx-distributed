from typing import Optional, Tuple
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch_xla.core.xla_model as xm
from torch import nn
from torch.nn.parameter import Parameter

from .mappings import (copy_to_tensor_model_parallel_region,
                       gather_from_tensor_model_parallel_region,
                       reduce_from_tensor_model_parallel_region,
                       scatter_to_tensor_model_parallel_region)
from .parallel_state import (get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size)
from .random import get_xla_rng_tracker
from .utils import EmbeddingUtility, divide

if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1
}


def param_is_not_tensor_parallel_duplicate(param: torch.Tensor) -> bool:
    return (hasattr(param, "tensor_model_parallel") and
            param.tensor_model_parallel) or (get_tensor_model_parallel_rank()
                                             == 0)


def set_tensor_model_parallel_attributes(tensor: torch.Tensor,
                                         is_parallel: bool, dim: int,
                                         stride: int) -> None:
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(
        tensor: torch.Tensor) -> None:

    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor: torch.Tensor,
                                          source_tensor: torch.Tensor) -> None:

    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_neuron(weight,
                                     init_method,
                                     partition_dim,
                                     stride=1):
    """Initialize affine weight for model parallel on Neuron device.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
        stride (int):
    """

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_xla_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = Parameter(torch.empty(output_size,
                                input_size,
                                dtype=torch.float,
                                requires_grad=False))
    

    init_method(master_weight)
        
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight,
                              per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_method=init.normal_,
        *,
        params_dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = False,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size(
        )
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.start_index,
            self.end_index,
        ) = EmbeddingUtility.range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = (self.end_index -
                                             self.start_index)

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=params_dtype,
                ))
            _initialize_affine_weight_cpu(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=xm.xla_device(),
                    dtype=params_dtype,
                ))
            _initialize_affine_weight_neuron(self.weight,
                                             init_method,
                                             partition_dim=0,
                                             stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            input_mask = (input_ >= self.start_index) & \
                         (input_ < self.end_index)
            # Mask the input.
            masked_input = input_.clone() - self.start_index
            masked_input = torch.mul(masked_input, input_mask.long())
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input.long(),
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel = torch.mul(output_parallel, \
                            torch.unsqueeze(input_mask.float(), dim=-1))
        # Reduce across all the model parallel Neuron devices.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        async_grad_allreduce: bool,
    ):
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.compute_weight_gradient = weight.requires_grad

        if ctx.compute_weight_gradient:
            ctx.save_for_backward(input, weight)
        else:
            ctx.save_for_backward(weight)

        total_input = input
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None

        use_bias = ctx.use_bias

        handle = None
        if ctx.compute_weight_gradient:
            total_input = input

        grad_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input,
                group=get_tensor_model_parallel_group(),
                async_op=True)

        #if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            if ctx.async_grad_allreduce:
                handle.wait()
            return grad_input, None, None, None, None, None, None

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    async_grad_allreduce: bool,
) -> torch.Tensor:
    args = (
        input,
        weight,
        bias,
        async_grad_allreduce,
    )
    with torch.cuda.amp.autocast(enabled=False):
        return LinearWithAsyncCommunication.apply(*args)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all Neuron devices, otherwise, every Neuron device will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.

    Keyword Arguments:
        no_async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        accumulation_in_fp16:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=None,
        stride=1,
        keep_master_weight_for_test=False,
        *,
        no_async_tensor_model_parallel_allreduce=False,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        def _init_method(weight):
            return nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        if init_method is None: 
            init_method =  _init_method
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(self.output_size_per_partition,
                            self.input_size,
                            dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=xm.xla_device(),
                    dtype=params_dtype,
                ))
            _initialize_affine_weight_neuron(self.weight,
                                             init_method,
                                             partition_dim=0,
                                             stride=stride)

        if bias:
            self.bias_size = self.output_size if self.gather_output else self.output_size_per_partition
            if use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.bias_size,
                                dtype=params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.bias_size,
                        device=xm.xla_device(),
                        dtype=params_dtype,
                    ))
            if init_method.__name__ == "kaiming_uniform_":
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
            else:
                with torch.no_grad():
                    self.bias.zero_()
            if not self.gather_output:
                set_tensor_model_parallel_attributes(self.bias, True, 0, stride)  
        else:
            self.register_parameter("bias", None)

        self.async_tensor_model_parallel_allreduce = (
            not no_async_tensor_model_parallel_allreduce and world_size > 1)

        self._forward_impl = linear_with_async_allreduce

    def forward(
            self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
        """

        if self.async_tensor_model_parallel_allreduce:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
        )
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output = (output + self.bias) if self.bias is not None else output
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the Neuron devices and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    Keyword Arguments:
        params_dtype:
        use_cpu_initialization:
        accumulation_in_fp16:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=None,
        stride=1,
        keep_master_weight_for_test=False,
        *,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        
        def _init_method(weight):
            return nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        if init_method is None: 
            init_method =  _init_method

        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(self.output_size,
                            self.input_size_per_partition,
                            dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=xm.xla_device(),
                    dtype=params_dtype,
                ))
            _initialize_affine_weight_neuron(self.weight,
                                             init_method,
                                             partition_dim=1,
                                             stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=xm.xla_device(),
                        dtype=params_dtype,
                    ))
            if init_method.__name__ == "kaiming_uniform_":
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
            else:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self._forward_impl = linear_with_async_allreduce

    def forward(
            self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=False,
        )
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = (output_ + self.bias) if self.bias is not None else output_
        return output


class ParallelAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(
            self.num_attention_heads, world_size)
        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.attention_head_size


        self.query = ColumnParallelLinear(config.hidden_size,
                                          self.all_head_size,
                                          gather_output=False,
                                          use_cpu_initialization=True)
        self.key = ColumnParallelLinear(config.hidden_size,
                                        self.all_head_size,
                                        gather_output=False,
                                        use_cpu_initialization=True)
        self.value = ColumnParallelLinear(config.hidden_size,
                                          self.all_head_size,
                                          gather_output=False,
                                          use_cpu_initialization=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = RowParallelLinear(config.hidden_size,
                                       config.hidden_size,
                                       input_is_parallel=True,
                                       use_cpu_initialization=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads_per_partition,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions (assuming the past values have already been parallelised)
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1,
                    dtype=torch.long,
                    device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length,
                    dtype=torch.long,
                    device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length,
                                          dtype=torch.long,
                                          device=hidden_states.device).view(
                                              1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )

        if self.is_decoder:
            outputs = outputs + (past_key_value, )

        attention_output = self.dense(outputs[0])
        attention_output = self.dropout_out(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output, ) + outputs[1:]

        return outputs
