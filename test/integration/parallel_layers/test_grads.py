import argparse
import traceback
from datetime import datetime

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from commons import print_separator, set_random_seed

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.grads import get_grad_norm
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override
from neuronx_distributed.pipeline.model import NxDPPModel
from neuronx_distributed.utils.model_utils import move_model_to_device

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args


S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


class SingleLinearBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x = self.lin(x)
        x = torch.relu(x)
        x = torch.transpose(x, 0, 1)
        return x


class SingleLayerOutputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([SingleLinearBlock(4, 4) for _ in range(8)])
        self.output_proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        for layer in self.layers:
            x = layer(x)
        x = torch.transpose(x, 0, 1)
        return torch.sum(self.output_proj(x))


def run_test(test_fn, *args, **kwargs) -> None:
    try:
        print_separator(f"test {test_fn.__name__}")
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"testing {test_fn.__name__} with args={args} and kwargs={kwargs}")
        assert not parallel_state.model_parallel_is_initialized()
        test_fn(*args, **kwargs)
        assert parallel_state.model_parallel_is_initialized()

        torch.distributed.barrier()
        if rank == 0:
            print("test passed")
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise
    finally:
        parallel_state.destroy_model_parallel()


def test_single_layer_output_module() -> None:
    seed = 1234
    set_random_seed(seed)
    pipeline_parallel_size = 4
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=pipeline_parallel_size)
    global_batch_size = 32
    per_model_replica_batch = global_batch_size // parallel_state.get_data_parallel_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    model = SingleLayerOutputModule()
    example_input = torch.randn(global_batch_size, 4, 4)
    out = model(example_input)
    out /= global_batch_size
    out.backward()
    cpu_param_name_to_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    for _, p in model.named_parameters():
        p.grad = None
    current_dp_batch = (
        example_input.narrow(0, dp_rank * per_model_replica_batch, per_model_replica_batch).detach().clone()
    )
    pp_model = NxDPPModel(
        model,
        transformer_layer_cls=SingleLinearBlock,
        auto_partition=True,
        num_microbatches=pipeline_parallel_size,
        output_loss_value_spec=True,
        input_names=["x"],
    )
    _ = pp_model.run_train(x=current_dp_batch)
    for n, p in pp_model.local_named_parameters():
        grad = p.grad.detach().cpu()
        assert torch.allclose(
            grad, cpu_param_name_to_grads[n], rtol=5e-2
        ), f"param {n} grad mismatching! CPU {cpu_param_name_to_grads[n]} PP grad {grad}"


def test_tp_zero1_pp_gradient_clipping(tensor_parallel_size: int, pipeline_parallel_size: int) -> None:
    device = xm.xla_device()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size, pipeline_model_parallel_size=pipeline_parallel_size)
    seed = 1234
    set_random_seed(seed)
    max_norm = 1.0
    in_size, out_size = 128, 128

    tp_parallel_layers = [
        layers.ColumnParallelLinear(in_size, out_size, keep_master_weight=True)
        for _ in range(pipeline_parallel_size)
    ]
    cpu_layers = [torch.nn.Linear(in_size, out_size) for _ in range(pipeline_parallel_size)]
    for i in range(pipeline_parallel_size):
        cpu_layers[i].weight.data.copy_(tp_parallel_layers[i].master_weight)
        cpu_layers[i].bias.data.copy_(tp_parallel_layers[i].bias)

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    move_model_to_device(tp_parallel_layers[pp_rank], device)

    for i in range(pipeline_parallel_size):
        cpu_layers[i].weight.grad = 10 * torch.randn((in_size, out_size))
        cpu_layers[i].bias.grad = 10 * torch.randn((out_size))

    partition_size = in_size // tensor_parallel_size

    tp_parallel_layers[pp_rank].weight.grad = (
        cpu_layers[pp_rank]
        .weight.grad.clone()[
            partition_size
            * parallel_state.get_tensor_model_parallel_rank() : partition_size
            * (parallel_state.get_tensor_model_parallel_rank() + 1)
        ]
        .to(device)
    )
    tp_parallel_layers[pp_rank].bias.grad = cpu_layers[pp_rank].bias.grad.clone().to(device)

    opt = NeuronZero1Optimizer(
        list(tp_parallel_layers[pp_rank].parameters()),
        torch.optim.SGD,
        lr=0.01,
        pin_layout=False,
        sharding_groups=parallel_state.get_data_parallel_replica_groups(),
        grad_norm_groups=parallel_state.get_tensor_model_parallel_replica_groups(),
        max_norm=max_norm,
        grad_clipping=True,
    )

    opt.base_optimizer.zero_grad = lambda set_to_none: set_to_none
    opt.step()
    tp_zero1_grad_norm = opt.grad_norm
    cpu_params = []
    for layer in cpu_layers:
        cpu_params.extend(list(layer.parameters()))
    total_norm = torch.nn.utils.clip_grad_norm_(cpu_params, max_norm)
    xm.mark_step()
    assert np.allclose(
        total_norm.numpy(), tp_zero1_grad_norm.detach().cpu().numpy(), rtol=1e-2, atol=1e-2
    ), "grad_norms don't match before clipping"

    all_parameters = []
    for param_group, sharded_param_group in zip(opt.param_groups, opt.base_optimizer.param_groups):
        for param, shard in zip(param_group["params"], sharded_param_group["params"]):
            if param.grad is not None:
                if hasattr(param, "shared"):
                    shard.shared = param.shared
                if hasattr(param, "tensor_model_parallel"):
                    shard.tensor_model_parallel = param.tensor_model_parallel
                all_parameters.append(shard)

    total_norm = get_grad_norm(
        all_parameters,
        norm_type=2,
        zero1_optimizer=True,
        zero1_optimizer_groups=parallel_state.get_data_parallel_replica_groups(),
    )
    cpu_total_norm = 0.0
    for p in cpu_params:
        param_norm_sq = torch.square(p.grad).sum()
        cpu_total_norm += param_norm_sq
    cpu_total_norm = torch.sqrt(cpu_total_norm)

    assert np.allclose(
        cpu_total_norm.numpy(), total_norm.detach().cpu().numpy(), rtol=1e-2
    ), "grad_norms don't match after clipping"

    assert total_norm.detach().cpu().numpy() <= 1.0, "gradients are not clipped to max_norm"


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    run_test(test_tp_zero1_pp_gradient_clipping, tensor_parallel_size=1, pipeline_parallel_size=1)
    run_test(test_tp_zero1_pp_gradient_clipping, tensor_parallel_size=32, pipeline_parallel_size=1)
    run_test(test_tp_zero1_pp_gradient_clipping, tensor_parallel_size=8, pipeline_parallel_size=1)
    run_test(test_tp_zero1_pp_gradient_clipping, tensor_parallel_size=2, pipeline_parallel_size=4)
    run_test(test_single_layer_output_module)
