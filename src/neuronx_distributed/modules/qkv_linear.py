import math
import warnings
from typing import Optional, Tuple, Callable, Any, List
import os

import torch
import torch_xla.core.xla_model as xm
from torch.nn.parameter import Parameter

from neuronx_distributed.parallel_layers.layers import (
    BaseParallelLinear,
    create_local_weight,
)
from neuronx_distributed.parallel_layers.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_replica_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.random import get_xla_rng_tracker
from neuronx_distributed.parallel_layers.utils import (
    cast_if_autocast_enabled,
    divide,
    set_tensor_model_parallel_attributes,
    verify_casted_dtype,
)
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.utils import hardware
from neuronx_distributed.modules.qkv_linear_utils import (
    _qkvlinear_autograd_base_setup_fwd,
    _qkvlinear_autograd_base_setup_bwd,
    _qkvlinear_autograd_bwd_grad_reduce,
    _qkvlinear_autograd_bwd_no_weight_grad,
    _qkvlinear_autograd_bwd_input_grad
)
from torch_neuronx.utils import get_platform_target

logger = get_logger()


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

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride, num_partitions=get_tensor_model_parallel_size(),
    )

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
    if hardware(get_platform_target()) == hardware.TRN2:
        # Replicate single kv head on adjacent ranks (specific to Trn2 currently)
        # For TP64 with kv_replication 8: (K0,K0,K0….. 8 times)(K1,K1,……. 8 times) … for all K heads
        repeated_weight = torch.repeat_interleave(master_weight, tp_size_multiplier, dim=0)
    elif hardware(get_platform_target()) == hardware.TRN1:
        # On Trn1: TP32 with kv_replication 4: (K0,K1,K2,K3)(K0,K1,K2,K3)…. 4 times each nearby ranks holding different K heads
        repeated_weight = master_weight.repeat(tp_size_multiplier, 1)
    else:
        raise Exception("Configure kv weight initialization as per hw architecture")

    repeated_weight = repeated_weight.to(dtype=params_dtype)

    create_local_weight(repeated_weight, partition_dim, per_partition_size, stride, out_weight=weight)
    if return_master_weight:
        return master_weight
    return None


def _linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


def _compute_gradients(
    input: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor, use_bias: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        weight_q: Optional[torch.Tensor],
        weight_k: Optional[torch.Tensor],
        weight_v: Optional[torch.Tensor],
        bias_q: Optional[torch.Tensor],
        bias_k: Optional[torch.Tensor],
        bias_v: Optional[torch.Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        kv_size_multiplier: int,
        weight_qkv: Optional[torch.Tensor] = None,
        bias_qkv: Optional[torch.Tensor] = None,
        fuse_qkv: bool = False,
        output_size_q: Optional[int] = None,
        output_size_kv: Optional[int] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        total_input = _qkvlinear_autograd_base_setup_fwd(
            ctx,
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
            weight_qkv,
            bias_qkv,
            fuse_qkv,
            output_size_q,
            output_size_kv,
            reduce_dtype
        )
        if ctx.fuse_qkv:
            assert weight_qkv is not None and output_size_q is not None and output_size_kv is not None
            output_qkv = _linear_forward(total_input, weight_qkv, bias_qkv)
            # Split the outputs
            output_dimensions = [output_size_q, output_size_kv, output_size_kv]
            output_q, output_k, output_v = torch.split(output_qkv, output_dimensions, dim=-1)
        else:
            assert weight_q is not None and weight_k is not None and weight_v is not None
            output_q = _linear_forward(total_input, weight_q, bias_q)
            output_k = _linear_forward(total_input, weight_k, bias_k)
            output_v = _linear_forward(total_input, weight_v, bias_v)

        return output_q, output_k, output_v

    @staticmethod
    def backward(ctx, grad_output_q, grad_output_k, grad_output_v):
        
        total_input, weight_qkv, weight_q, weight_k, weight_v, grad_output_k, grad_output_v = _qkvlinear_autograd_base_setup_bwd(ctx, grad_output_q, grad_output_k, grad_output_v)

        if ctx.fuse_qkv:
            # Divide grad_output_k and grad_output_v by the kv replication factor
            # because after this step we are going to do an all-reduce over the entire tp group which
            # would cause the K and V duplicate factor to be counted twice.
            grad_output_k_scaled = grad_output_k / ctx.kv_size_multiplier
            grad_output_v_scaled = grad_output_v / ctx.kv_size_multiplier
            grad_output_qkv = torch.cat([grad_output_q, grad_output_k_scaled, grad_output_v_scaled], dim=-1)
            grad_input = grad_output_qkv.matmul(weight_qkv)
        else:
            grad_input_q = grad_output_q.matmul(weight_q)
            grad_input_k = grad_output_k.matmul(weight_k)
            grad_input_v = grad_output_v.matmul(weight_v)
            # Here we need to divide the grad_input_k and grad_input_v by a factor of kv_size_multipler,
            # because after this step we are going to do an all-reduce over the entire tp group which
            # would cause the K and V duplicate factor to be counted twice.
            grad_input = grad_input_q + (grad_input_k + grad_input_v) / ctx.kv_size_multiplier

        original_dtype = grad_input.dtype

        grad_input = _qkvlinear_autograd_bwd_grad_reduce(ctx, grad_input, original_dtype)

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                
                sub_grad_input = _qkvlinear_autograd_bwd_no_weight_grad(ctx, grad_input, original_dtype)

                return (
                    sub_grad_input,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        # Convert the tensor shapes to 2D for execution compatibility
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.sequence_parallel_enabled:
            assert not ctx.async_grad_allreduce
            sub_grad_input = _qkvlinear_autograd_bwd_input_grad(ctx, grad_input, original_dtype)

        if ctx.fuse_qkv:
            # Use grad_output_qkv without scaling by the kv replication factor
            grad_output_qkv_not_scaled = torch.cat([grad_output_q, grad_output_k, grad_output_v], dim=-1)
            grad_weight_qkv, grad_bias_qkv = _compute_gradients(
                total_input, weight_qkv, grad_output_qkv_not_scaled, ctx.use_bias
            )

            if ctx.sequence_parallel_enabled:
                return (
                    sub_grad_input,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grad_weight_qkv,
                    grad_bias_qkv,
                    None,
                    None,
                    None,
                    None,
                )

            return (
                grad_input,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                grad_weight_qkv,
                grad_bias_qkv,
                None,
                None,
                None,
                None,
            )

        else:
            grad_weight_q, grad_bias_q = _compute_gradients(total_input, weight_q, grad_output_q, ctx.use_bias)
            grad_weight_k, grad_bias_k = _compute_gradients(total_input, weight_k, grad_output_k, ctx.use_bias)
            grad_weight_v, grad_bias_v = _compute_gradients(total_input, weight_v, grad_output_v, ctx.use_bias)

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
                None,
                None,
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
            None,
            None,
            None,
            None,
            None,
            None,
        )


def gqa_qkv_linear_with_async_allreduce(
    input: torch.Tensor,
    weight_q: Optional[torch.Tensor],
    weight_k: Optional[torch.Tensor],
    weight_v: Optional[torch.Tensor],
    bias_q: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    kv_size_multiplier: int = 1,
    weight_qkv: Optional[torch.Tensor] = None,
    bias_qkv: Optional[torch.Tensor] = None,
    fuse_qkv: bool = False,
    output_size_q: Optional[int] = None,
    output_size_kv: Optional[int] = None,
    reduce_dtype: torch.dtype = torch.float32,
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
        weight_qkv,
        bias_qkv,
        fuse_qkv,
        output_size_q,
        output_size_kv,
        reduce_dtype,
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
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[..., Any]] = None,
        sequence_parallel_enabled: bool = False,
        keep_master_weight: bool = False,
        kv_size_multiplier: int = 1,
        fuse_qkv: bool = True,
        reduce_dtype: torch.dtype = torch.float32,
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
        assert (
            output_sizes[1] * kv_size_multiplier
        ) % world_size == 0, "kv_output_dim*kv_size_multiplier should be divisible by tp_world_size"
        parallel_state.initialize_kv_group(kv_size_multiplier)
        self.q_output_size_per_partition = divide(output_sizes[0], world_size)
        self.kv_output_size_per_partition = divide(output_sizes[1] * kv_size_multiplier, world_size)
        self.dtype = dtype
        self.keep_master_weight = keep_master_weight
        self.device = device
        self.use_bias = bias
        self.fuse_qkv = fuse_qkv
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
        self.reduce_dtype = reduce_dtype
        self._forward_impl = gqa_qkv_linear_with_async_allreduce

    def _create_weights_biases(self):
        if self.fuse_qkv:
            self.weight_qkv = Parameter(
                torch.empty(
                    self.q_output_size_per_partition + 2 * self.kv_output_size_per_partition,
                    self.input_size,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            if self.use_bias:
                bias_size_q = self.output_sizes[0] if self.gather_output else self.q_output_size_per_partition
                bias_size_kv = self.output_sizes[1] if self.gather_output else self.kv_output_size_per_partition
                self.bias_qkv = Parameter(
                    torch.empty(bias_size_q + 2 * bias_size_kv, device=self.device, dtype=self.dtype)
                )
            else:
                self.register_parameter("bias_qkv", None)

            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)
        else:
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
        if self.fuse_qkv:
            # Split weight_qkv in to components for init
            dimensions = [
                self.q_output_size_per_partition,
                self.kv_output_size_per_partition,
                self.kv_output_size_per_partition,
            ]
            weight_q, weight_k, weight_v = torch.split(self.weight_qkv, dimensions, dim=0)
        else:
            weight_q = self.weight_q
            weight_k = self.weight_k
            weight_v = self.weight_v
        self.master_weight_q = self._init_per_layer_weight(
            weight_q, self.output_sizes[0], self.q_output_size_per_partition, 1
        )
        self.master_weight_k = self._init_per_layer_weight(
            weight_k, self.output_sizes[1], self.kv_output_size_per_partition, self.kv_size_multiplier
        )
        self.master_weight_v = self._init_per_layer_weight(
            weight_v, self.output_sizes[1], self.kv_output_size_per_partition, self.kv_size_multiplier
        )
        if self.fuse_qkv:
            # Concat and update self.weight_qkv
            with torch.no_grad():
                self.weight_qkv = torch.nn.Parameter(torch.cat([weight_q, weight_k, weight_v], dim=0))
        else:
            self.weight_q = weight_q
            self.weight_k = weight_k
            self.weight_v = weight_v
        if self.use_bias:
            if self.fuse_qkv:
                bias_size_q = self.output_sizes[0] if self.gather_output else self.q_output_size_per_partition
                bias_size_kv = self.output_sizes[1] if self.gather_output else self.kv_output_size_per_partition
                dimensions = [bias_size_q, bias_size_kv, bias_size_kv]
                bias_q, bias_k, bias_v = torch.split(self.bias_qkv, dimensions, dim=0)
            else:
                bias_q = self.bias_q
                bias_k = self.bias_k
                bias_v = self.bias_v
            self.master_bias_q = self._init_per_layer_bias(
                bias_q,
                self.output_sizes[0],
                torch.nn.init._calculate_fan_in_and_fan_out(weight_q),
                self.kv_size_multiplier,
            )
            self.master_bias_k = self._init_per_layer_bias(
                bias_k,
                self.output_sizes[1],
                torch.nn.init._calculate_fan_in_and_fan_out(weight_k),
                self.kv_size_multiplier,
            )
            self.master_bias_v = self._init_per_layer_bias(
                bias_v,
                self.output_sizes[1],
                torch.nn.init._calculate_fan_in_and_fan_out(weight_q),
                self.kv_size_multiplier,
            )
            if self.fuse_qkv:
                # Concat and update self.bias_qkv
                with torch.no_grad():
                    self.bias_qkv = torch.nn.Parameter(torch.cat([bias_q, bias_k, bias_v], dim=0))
            else:
                self.bias_q = bias_q
                self.bias_k = bias_k
                self.bias_v = bias_v

        if self.fuse_qkv:
            # Fuse weights and biases
            if self.master_weight_q is not None:
                self.master_weight_qkv = torch.cat(
                    [self.master_weight_q, self.master_weight_k, self.master_weight_v], dim=0
                )
            else:
                self.master_weight_qkv = None
            self.master_weight_q = None
            self.master_weight_k = None
            self.master_weight_v = None
            if self.use_bias:
                if self.master_bias_q is not None:
                    self.master_bias_qkv = torch.cat(
                        [self.master_bias_q, self.master_bias_k, self.master_bias_v], dim=0
                    )
                else:
                    self.master_bias_qkv = None
                self.master_bias_q = None
                self.master_bias_k = None
                self.master_bias_v = None

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

    def _init_per_layer_bias(self, bias, output_size, fan_in, kv_size_multiplier=1):
        master_bias = None
        if bias.device != torch.device("meta"):
            bound = 1 / math.sqrt(self.input_size) if fan_in > 0 else 0
            master_bias = Parameter(torch.empty(output_size // kv_size_multiplier, dtype=self.dtype))
            torch.nn.init.uniform_(master_bias, -bound, bound)
            bias.data.copy_(master_bias.repeat(kv_size_multiplier).data)

        if not self.gather_output:
            set_tensor_model_parallel_attributes(bias, True, 0, stride=1, shared_tp=kv_size_multiplier > 1)
        return master_bias if self.keep_master_weight else None

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        output_q: torch.Tensor
        output_k: torch.Tensor
        output_v: torch.Tensor

        # Matrix multiply.
        if self.fuse_qkv:
            output_parallel_q, output_parallel_k, output_parallel_v = self._forward_impl(
                input=input_parallel,
                weight_q=None,
                weight_k=None,
                weight_v=None,
                bias_q=None,
                bias_k=None,
                bias_v=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                kv_size_multiplier=self.kv_size_multiplier,
                weight_qkv=self.weight_qkv,
                bias_qkv=None,
                fuse_qkv=self.fuse_qkv,
                output_size_q=self.q_output_size_per_partition,
                output_size_kv=self.kv_output_size_per_partition,
                reduce_dtype=self.reduce_dtype,
            )
        else:
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
                weight_qkv=None,
                bias_qkv=None,
                fuse_qkv=self.fuse_qkv,
                output_size_q=self.output_sizes[0] if self.gather_output else self.q_output_size_per_partition,
                output_size_kv=self.output_sizes[1] if self.gather_output else self.kv_output_size_per_partition,
                reduce_dtype=self.reduce_dtype,
            )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output_q = gather_from_tensor_model_parallel_region(output_parallel_q)
            output_k = gather_from_tensor_model_parallel_region(output_parallel_k)
            output_v = gather_from_tensor_model_parallel_region(output_parallel_v)
        else:
            output_q, output_k, output_v = output_parallel_q, output_parallel_k, output_parallel_v

        if self.fuse_qkv:
            if self.bias_qkv is not None:
                bias_size_q = self.output_sizes[0] if self.gather_output else self.q_output_size_per_partition
                bias_size_kv = self.output_sizes[1] if self.gather_output else self.kv_output_size_per_partition
                dimensions = [bias_size_q, bias_size_kv, bias_size_kv]
                bias_q, bias_k, bias_v = torch.split(self.bias_qkv, dimensions, dim=0)
                output_q = output_q + bias_q
                output_k = output_k + bias_k
                output_v = output_v + bias_v
        else:
            output_q = (output_q + self.bias_q) if self.bias_q is not None else output_q
            output_k = (output_k + self.bias_k) if self.bias_k is not None else output_k
            output_v = (output_v + self.bias_v) if self.bias_v is not None else output_v
        return output_q, output_k, output_v
