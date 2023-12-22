import math
import warnings
from typing import Optional, Tuple

import torch
import torch_xla.core.xla_model as xm
from torch.nn.parameter import Parameter

from neuronx_distributed.parallel_layers.layers import create_local_weight, BaseParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.random import get_xla_rng_tracker
from neuronx_distributed.parallel_layers.utils import (
    divide,
    cast_if_autocast_enabled,
    set_tensor_model_parallel_attributes,
    verify_casted_dtype,
)

_KV_SHARED_GROUP = None
_KV_SHARED_GROUP_SPMD = None
_KV_GROUP_SIZE = None

def _initialize_kv_group(kv_shared_group_size=1):
    # Build the kv-shared model-parallel groups.
    global _KV_SHARED_GROUP
    global _KV_SHARED_GROUP_SPMD
    global _KV_GROUP_SIZE
    all_kv_shred_group_ranks = []
    tensor_model_parallel_size = get_tensor_model_parallel_size()
    world_size = torch.distributed.get_world_size()
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    if _KV_SHARED_GROUP is not None:
        assert kv_shared_group_size == _KV_GROUP_SIZE, "Currently the library supports only single KV size for all layers"
        return
    
    assert tensor_model_parallel_size % kv_shared_group_size == 0, (
                f"kv_shared_group_size: {kv_shared_group_size}, "
                f"should divide tensor model parallel group {tensor_model_parallel_size} "
            )
    _KV_GROUP_SIZE = kv_shared_group_size
    rank = torch.distributed.get_rank()
    if rank == 0:
        print("> initializing kv group with size {}".format(kv_shared_group_size))
    for i in range(num_tensor_model_parallel_groups):
        for j in range(tensor_model_parallel_size//kv_shared_group_size):
            ranks = list(
                range(i * tensor_model_parallel_size+j, (i + 1) * tensor_model_parallel_size, tensor_model_parallel_size//kv_shared_group_size)
            )
            all_kv_shred_group_ranks.append(ranks)
    _KV_SHARED_GROUP_SPMD = all_kv_shred_group_ranks
    for ranks in all_kv_shred_group_ranks:
        pg_options = {'xla_pg_options': {'mesh': _KV_SHARED_GROUP_SPMD}}
        if rank in ranks:
            group = torch.distributed.new_group(ranks, pg_options=pg_options)
            _KV_SHARED_GROUP = group


def get_kv_shared_group(as_list=False):
    """Get the KV shared group the caller rank belongs to."""
    assert _KV_SHARED_GROUP is not None, "kv_shared parallel group is not initialized"
    return _KV_SHARED_GROUP._mesh if as_list else _KV_SHARED_GROUP


def destroy_kv_group():
    global _KV_SHARED_GROUP
    _KV_SHARED_GROUP = None
    global _KV_SHARED_GROUP_SPMD
    _KV_SHARED_GROUP_SPMD = None


def _initialize_affine_weight(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    stride=1,
    tp_size_multiplier=1,
    device=None,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, dim=partition_dim, stride=stride)

    # Initialize master weight
    master_weight = Parameter(
        torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False, device=device)
    )

    if device == "xla":
        with get_xla_rng_tracker().fork():
            init_method(master_weight)
    else:
        init_method(master_weight)

    # here we are repeating along the 0th dim which is the num_heads dim.
    repeated_weight = master_weight.repeat(tp_size_multiplier, 1)

    repeated_weight = repeated_weight.to(dtype=params_dtype)

    create_local_weight(repeated_weight, partition_dim, per_partition_size, stride, out_weight=weight)
    if return_master_weight:
        return master_weight
    return None


def _linear_forward(input, weight, bias):
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


def _compute_gradients(input, weight, grad_output, use_bias):
    """
        This method computes the gradients for the weight and bias, 
        given the output gradient and input.
        grad_weight = grad_output.T*input
        grad_bias = sum(grad_output, dim=0)
    """
    grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
    grad_weight = grad_output.t().matmul(input)

    grad_bias = grad_output.sum(dim=0) if use_bias else None

    return grad_weight, grad_bias


class GQAQKVLinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
        weight_v: torch.Tensor,
        bias_q: Optional[torch.Tensor],
        bias_k: Optional[torch.Tensor],
        bias_v: Optional[torch.Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        kv_size_multiplier: int,
    ):
        ctx.use_bias = bias_q is not None and weight_q.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.compute_weight_gradient = weight_q.requires_grad
        ctx.kv_size_multiplier = kv_size_multiplier

        if ctx.compute_weight_gradient:
            ctx.save_for_backward(input, weight_q, weight_k, weight_v)
        else:
            ctx.save_for_backward(weight_q, weight_k, weight_v)

        if ctx.sequence_parallel_enabled:
            # `input` is supposed to be 3D and its order of dimension is [sequence, batch, hidden]
            total_input = xm.all_gather(
                input,
                groups=get_tensor_model_parallel_group(as_list=True),
                pin_layout=False,
            )
        else:
            total_input = input

        output_q = _linear_forward(total_input, weight_q, bias_q)
        output_k = _linear_forward(total_input, weight_k, bias_k)
        output_v = _linear_forward(total_input, weight_v, bias_v)

        return output_q, output_k, output_v

    @staticmethod
    def backward(ctx, grad_output_q, grad_output_k, grad_output_v):
        if ctx.compute_weight_gradient:
            input, weight_q, weight_k, weight_v = ctx.saved_tensors
        else:
            weight_q, weight_k, weight_v = ctx.saved_tensors[:3]
            input = None

        use_bias = ctx.use_bias

        handle = None
        if ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                total_input = xm.all_gather(
                    input,
                    groups=get_tensor_model_parallel_group(as_list=True),
                    pin_layout=False,
                )
            else:
                total_input = input

        if ctx.kv_size_multiplier > 1:
            # Since we repeat the K and V by a factor of kv_size_multipler, we need to 
            # sum up the gradients from the repeated portions. get_kv_shared_group() 
            # returns the ranks which have the same K and V heads, and hence allows us to
            # sum up from the distributed ranks. 
            handlek = torch.distributed.all_reduce(grad_output_k, group=get_kv_shared_group())
            handlev = torch.distributed.all_reduce(grad_output_v, group=get_kv_shared_group())

        grad_input_q = grad_output_q.matmul(weight_q)
        grad_input_k = grad_output_k.matmul(weight_k)
        grad_input_v = grad_output_v.matmul(weight_v)
        # Here we need to divide the grad_input_k and grad_input_v by a factor of kv_size_multipler,
        # because after this step we are going to do an all-reduce over the entire tp group which
        # would cause the K and V duplicate factor to be counted twice.
        grad_input = grad_input_q + (grad_input_k + grad_input_v)/ctx.kv_size_multiplier

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                        grad_input, group=get_tensor_model_parallel_group())

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                assert not ctx.async_grad_allreduce
                world_size = get_tensor_model_parallel_size()
                shape = list(grad_input.shape)
                shape[0] //= world_size

                sub_grad_input = torch.empty(
                    torch.Size(shape),
                    dtype=grad_input.dtype,
                    device=grad_input.device,
                    requires_grad=False,
                )
                groups = get_tensor_model_parallel_group()._mesh

                xm.reduce_scatter(
                    xm.REDUCE_SUM,
                    grad_input,
                    output=sub_grad_input,
                    groups=groups,
                    shard_count=len(groups[0]),
                    scatter_dim=0,
                    scale=1,
                    pin_layout=False,
                )

                return sub_grad_input, None, None, None, None, None, None
            return grad_input, None, None, None, None, None, None

        # Convert the tensor shapes to 2D for execution compatibility
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.sequence_parallel_enabled:
            assert not ctx.async_grad_allreduce
            sub_grad_input = torch.empty(input.shape, dtype=input.dtype, device=input.device, requires_grad=False)
            groups = get_tensor_model_parallel_group()._mesh
            xm.reduce_scatter(
                xm.REDUCE_SUM,
                grad_input,
                output=sub_grad_input,
                groups=groups,
                shard_count=len(groups[0]),
                scatter_dim=0,
                scale=1,
                pin_layout=False,
            )

        grad_weight_q, grad_bias_q = _compute_gradients(
            total_input, weight_q, grad_output_q, use_bias
        )
        grad_weight_k, grad_bias_k = _compute_gradients(
            total_input, weight_k, grad_output_k, use_bias
        )
        grad_weight_v, grad_bias_v = _compute_gradients(
            total_input, weight_v, grad_output_v, use_bias
        )

        if ctx.sequence_parallel_enabled:
            return (
                sub_grad_input,
                grad_weight_q,
                grad_weight_k,
                grad_weight_v,
                grad_bias_q,
                grad_bias_k,
                grad_bias_v,
                None,
                None,
                None,
                None,
            )

        return (
            grad_input,
            grad_weight_q,
            grad_weight_k,
            grad_weight_v,
            grad_bias_q,
            grad_bias_k,
            grad_bias_v,
            None,
            None,
            None,
            None,
        )


def gqa_qkv_linear_with_async_allreduce(
    input: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    bias_q: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    kv_size_multiplier: int = 1,
) -> torch.Tensor:
    args = cast_if_autocast_enabled(
        input,
        weight_q,
        weight_k,
        weight_v,
        bias_q,
        bias_k,
        bias_v,
        async_grad_allreduce,
        sequence_parallel_enabled,
        kv_size_multiplier,
    )
    verify_casted_dtype(args)
    with torch.cuda.amp.autocast(enabled=False):
        return GQAQKVLinearWithAsyncCommunication.apply(*args)


class GQAQKVColumnParallelLinear(BaseParallelLinear):
    """GQA QKV layer uses column parallel linear layer.

    It allows to replicate the KV heads such that the heads become
    divisible by tp degree.

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be batch,sequence and hidden feature, respectively.
    
    Example usage:
        # here we initialize the kv_shared_group_size to 4. This would replicate
        # KV heads 4 times. The 4 devices that have the same KV head would be put 
        # into the same replica group inside get_kv_shared_group()
        kv_shared_group_size = 4
        tensor_model_parallel_size = 32
        num_heads_kv_group = 8
        num_heads = 32
        parallel_state.initialize_model_parallel(
                                        tensor_model_parallel_size=tensor_model_parallel_size, 
                                        kv_shared_group_size=kv_shared_group_size)
        col_linear = qkv_linear.GQAQKVColumnParallelLinear(
            input_size = hidden_size,
            output_sizes = [num_heads*hidden_size, num_heads_kv_group*hidden_size],
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
            kv_size_multiplier=kv_shared_group_size
        ).to(device)
        q, k, v = col_linear(input)


    Arguments:
        input_size: Hidden size
        output_sizes: List containing Q and KV sizes
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all Neuron devices, otherwise, every Neuron device will have its output
                       which is Y_i = XA_i
        dtype: dtype of the weights
        device: Device on which the weights should be initialized.
        init_method: Initialization method to initialize the weights
        sequence_parallel_enabled: Enable sequence parallelism. It would add the all-gather and reduce-scatter
                                   operations to collect the tensor along sequence dimension.
        kv_size_multiplier: A factor to replicate the KV heads so that they become divisible by tp degree.

    """

    def __init__(
        self,
        input_size: int,
        output_sizes: int,
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        init_method: torch.nn.init = None,
        sequence_parallel_enabled: bool = False,
        keep_master_weight: bool = False,
        kv_size_multiplier: int = 1,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.gather_output = gather_output
        self.arg_init_method = init_method
        world_size = get_tensor_model_parallel_size()
        self.kv_size_multiplier = kv_size_multiplier
        assert world_size % kv_size_multiplier == 0, "tp_world_size should be divisible by kv_size_multiplier"
        assert (output_sizes[1]*kv_size_multiplier) % world_size == 0, "kv_output_dim*kv_size_multiplier should be divisible by tp_world_size"
        _initialize_kv_group(kv_size_multiplier)
        self.q_output_size_per_partition = divide(output_sizes[0], world_size)
        self.kv_output_size_per_partition = divide(output_sizes[1] * kv_size_multiplier, world_size)
        self.dtype = dtype
        self.keep_master_weight = keep_master_weight
        self.device = device
        self.use_bias = bias
        self._create_weights_biases()
        self.initialize_weight_biases()

        self.async_tensor_model_parallel_allreduce = not sequence_parallel_enabled and world_size > 1
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}")
        self.sequence_parallel_enabled = sequence_parallel_enabled

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time."
            )

        self._forward_impl = gqa_qkv_linear_with_async_allreduce

    def _create_weights_biases(self):
        self.weight_q = Parameter(
            torch.empty(self.q_output_size_per_partition, self.input_size, dtype=self.dtype, device=self.device)
        )
        self.weight_k = Parameter(
            torch.empty(self.kv_output_size_per_partition, self.input_size, dtype=self.dtype, device=self.device)
        )
        self.weight_v = Parameter(
            torch.empty(self.kv_output_size_per_partition, self.input_size, dtype=self.dtype, device=self.device)
        )
        if self.use_bias:
            bias_size = self.output_sizes[0] if self.gather_output else self.q_output_size_per_partition
            self.bias_q = Parameter(torch.empty(bias_size, device=self.device, dtype=self.dtype))
            bias_size = self.output_sizes[1] if self.gather_output else self.kv_output_size_per_partition
            self.bias_k = Parameter(torch.empty(bias_size, device=self.device, dtype=self.dtype))
            self.bias_v = Parameter(torch.empty(bias_size, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)
    
    def initialize_weight_biases(self):
        # Initialize weight.
        self.master_weight_q = self._init_per_layer_weight(
            self.weight_q, self.output_sizes[0], self.q_output_size_per_partition, 1
        )
        self.master_weight_k = self._init_per_layer_weight(
            self.weight_k, self.output_sizes[1], self.kv_output_size_per_partition, self.kv_size_multiplier
        )
        self.master_weight_v = self._init_per_layer_weight(
            self.weight_v, self.output_sizes[1], self.kv_output_size_per_partition, self.kv_size_multiplier
        )
        if self.use_bias:
            self.master_bias_q = self._init_per_layer_bias(self.bias_q)
            self.master_bias_k = self._init_per_layer_bias(self.bias_k)
            self.master_bias_v = self._init_per_layer_bias(self.bias_v)

    def _init_per_layer_weight(self, weight, output_size, output_size_per_partition, kv_size_multiplier=1):
        master_weight = None
        if weight.device != torch.device("meta"):
            master_weight = _initialize_affine_weight(
                weight,
                output_size,
                self.input_size,
                output_size_per_partition,
                0,
                self._init_weight,
                params_dtype=self.dtype,
                return_master_weight=self.keep_master_weight,
                tp_size_multiplier=kv_size_multiplier,
            )

        return master_weight

    def _init_per_layer_bias(self, bias):
        master_bias = None
        if bias.device != torch.device("meta"):
            bound = 1 / math.sqrt(self.input_size) if fan_in > 0 else 0
            master_bias = Parameter(torch.empty(output_size // kv_size_multiplier, dtype=self.dtype))
            torch.nn.init.uniform_(master_bias, -bound, bound)
            bias.data.copy_(master_bias.repeat(kv_size_multiplier).data)

        if not self.gather_output:
            set_tensor_model_parallel_attributes(bias, True, 0, stride=1, shared_tp=kv_size_multiplier > 1)
        return master_bias if self.keep_master_weight else None

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input)

        # Matrix multiply.
        output_parallel_q, output_parallel_k, output_parallel_v = self._forward_impl(
            input=input_parallel,
            weight_q=self.weight_q,
            weight_k=self.weight_k,
            weight_v=self.weight_v,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
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
        output_q = (output_q + self.bias_q) if self.bias_q is not None else output_q
        output_k = (output_k + self.bias_k) if self.bias_k is not None else output_k
        output_v = (output_v + self.bias_v) if self.bias_v is not None else output_v
        return output_q, output_k, output_v