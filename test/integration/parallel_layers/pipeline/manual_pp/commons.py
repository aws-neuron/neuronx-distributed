import math
import torch.distributed as dist
import torch
from torch import nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import initialize_model_parallel
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed.pipeline.comm import rmsg
from transformers.activations import ACT2FN

from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF
from functools import partial


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class MLPsWithTP(LlamaMLPHF):
    """simplified llama MLP layer, removed unnecessary HF configs
    for testing and debugging purpose
    """

    def __init__(self, config, tp_size=1):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.tp_size = tp_size

        init_method = partial(_init_normal, config.initializer_range)

        if tp_size > 1:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                stride=1,
                bias=False,
                gather_output=False,
                init_method=init_method,
                keep_master_weight=True,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                init_method=init_method,
                keep_master_weight=True,
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
            )

        self.split_size = self.intermediate_size // tp_size

    def forward(self, x):

        gate_proj = self.gate_proj(x)
        down_proj = self.down_proj(gate_proj)

        return down_proj

    @torch.no_grad()
    def copy_(self, other):
        self.gate_proj.weight.copy_(other.gate_proj.weight)
        self.down_proj.weight.copy_(other.down_proj.weight)

        if self.tp_size > 1:
            self.gate_proj.master_weight.copy_(other.gate_proj.master_weight)
            self.down_proj.master_weight.copy_(other.down_proj.master_weight)


class ToyMLPsTP(nn.Module):
    def __init__(self, config, tp_size=1):
        super().__init__()
        self.layers = [
            WrappedMLP_TP(config, tp_size=tp_size)
            for _ in range(config.num_hidden_layers)
        ]
        self.config = config
        self.tp_size = tp_size

        for idx, layer in enumerate(self.layers):
            self.add_module(f"layer_{idx}", layer)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)

        return x[0]

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def get_layers(self):
        return self.layers

    @torch.no_grad()
    def clone(self):
        cloned = ToyMLPsTP(self.config, tp_size=self.tp_size)
        for idx, layer in enumerate(cloned.layers):
            layer.copy_(self.layers[idx])

        return cloned


class WrappedMLP_TP(MLPsWithTP):
    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        return [super().forward(x)]


def init_distributed_env(tp_size, pp_size):
    if cpu_mode():
        torch.distributed.init_process_group(backend="gloo")
    else:
        torch.distributed.init_process_group(backend="xla")

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )

    torch.distributed.barrier()


class WrappedLinear(torch.nn.Linear):
    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        return [super().forward(x)]


class ToyModel(torch.nn.Module):
    def __init__(self, nlayers, hidden_size=8, weight_sharing=False):
        super(ToyModel, self).__init__()
        self.layers = [WrappedLinear(hidden_size, hidden_size, bias=False) for _ in range(nlayers)]
        
        if weight_sharing:
            # assuming the first and last layer share the weights, and no bias
            self.layers[0].weight = self.layers[-1].weight

        for idx, layer in enumerate(self.layers):
            self.add_module(f"layer_{idx}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]

    def get_layers(
        self,
    ):
        return self.layers

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self


@torch.no_grad()
def copy_model(model, weight_sharing=False):
    nlayers = len(model.get_layers())
    hidden_size = model.get_layers()[0].in_features

    copied = ToyModel(nlayers=nlayers, hidden_size=hidden_size)
    for idx, layer in enumerate(copied.get_layers()):
        layer.weight.data.copy_(model.get_layers()[idx].weight.data)
        if layer.bias is not None:
            layer.bias.data.copy_(model.get_layers()[idx].bias.data)

    if weight_sharing:
        # assuming the first and last layer share the weights, and no bias
        copied.layers[0].weight = copied.layers[-1].weight

    return copied


def forward_backward(model, inputs, labels, loss_fn):
    model.train()
    model.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    return loss, outputs


def create_rand_inputs(mbs, dp, hidden_size, seq_len=1):
    if seq_len > 1:
        rand_inputs = torch.rand((mbs * dp, seq_len, hidden_size))
        rand_labels = torch.randint(0, hidden_size, (mbs * dp, seq_len))
    else:
        rand_inputs = torch.rand((mbs * dp, hidden_size))
        rand_labels = torch.randint(0, hidden_size, (mbs * dp,))
    return rand_inputs, rand_labels


def generate_inputs(n_iter, dp_size, bs, hidden_size, seq_len=1):
    res = []
    for i in range(n_iter):
        test_inputs, test_labels = create_rand_inputs(
            bs, dp_size, hidden_size, seq_len=seq_len
        )
        res.append((test_inputs, test_labels))
    return res


def loss_check(logger, rtol, cpu_single_device_losses, xla_single_device_losses, nxd_pp_losses):
    assert len(cpu_single_device_losses) == len(nxd_pp_losses)
    mismatched_losses = []
    for i in range(len(cpu_single_device_losses)):
        cpu_sd_loss = cpu_single_device_losses[i]
        xla_sd_loss = xla_single_device_losses[i] if not cpu_mode() else None
        pp_loss = nxd_pp_losses[i]

        # By default we assume use rtol 1e-5 follows the default setting in torch.isclose
        # https://pytorch.org/docs/stable/generated/torch.isclose.html
        # right now we don't have systematic understanding what is the right tolerance
        # for loss comparison
        if not math.isclose(cpu_sd_loss, pp_loss, rel_tol=rtol):
            mismatched_losses.append((i, cpu_sd_loss, xla_sd_loss, pp_loss))

    if len(mismatched_losses) > 0:
        for i, cpu_sd_loss, xla_sd_loss, pp_loss in mismatched_losses:
            messages = [f"{i}-th loss does not match:", 
                    f"cpu single device loss {cpu_sd_loss},",
                    f"xla single device loss {xla_sd_loss},",
                    f"nxd pp loss {pp_loss},",
                    f"diff pp to cpu {cpu_sd_loss - pp_loss},",
            ]
            if not cpu_mode():
                messages.append(f"diff xla to cpu {cpu_sd_loss - xla_sd_loss}")

            logger.info(
                rmsg(
                    " ".join(messages)
                )
            )
        raise RuntimeError("Loss mismatch found check previous logs")
    else:
        logger.info(rmsg(f"Losses match with rtol {rtol}"))