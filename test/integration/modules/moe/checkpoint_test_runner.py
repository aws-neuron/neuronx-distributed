import dataclasses

import copy
import json
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch_xla.core.xla_model as xm  # TRN enablement

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut

from neuronx_distributed.parallel_layers import mappings, parallel_state, random
from neuronx_distributed.trainer.checkpoint import save_checkpoint, load_checkpoint

from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from examples.training.mixtral.modeling_mixtral_moe_nxd import MixtralForCausalLM

def override_state(stateful_object):
    state_dict = stateful_object.state_dict()

    stack = [state_dict]
    while len(stack) > 0:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            item.data.normal_(std=0.02)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)

    stateful_object.load_state_dict(state_dict)

def get_converter_args(tp_degree, ep_degree, mixtral_config, cur_dir):
    class Arguments:
        pass
    args = Arguments()
    args.input_dir = cur_dir
    args.output_dir = cur_dir
    args.config = os.path.join(cur_dir, "config.json")
    args.model_key = "model"
    args.tp_size = tp_degree
    args.ep_size = ep_degree
    args.pp_size = 1
    args.virtual_pp_size = 1
    args.n_layers = mixtral_config.num_hidden_layers
    args.coalesce_qkv = True
    args.kv_size_multiplier = 1
    args.load_xser = True
    args.save_xser = True


def assert_same_tensors(obj1, obj2):

    #assert type(obj1) == type(obj2), f"Type mismatch {type(obj1)} vs {type(obj2)}"

    if isinstance(obj1, (list, tuple)):
        for item1, item2 in zip(obj1, obj2):
            assert_same_tensors(item1, item2)
    elif isinstance(obj1, dict):
        for k1, k2 in zip(obj1, obj2):
            assert k1 == k2, f"Key mismatch {k1} vs {k2}"
            assert_same_tensors(obj1[k1], obj2[k2])
    elif isinstance(obj1, torch.Tensor):
        ut.check_tensors(obj1.cpu(), obj2.cpu(), atol=0.0, rtol=0.0)


def display_object(obj, indent=0):
    s = ""
    for _ in range(indent):
        s += " "
    if isinstance(obj, dict):
        print(s + "{")
        for k, v in obj.items():
            print(s + str(k))
            display_object(v, indent+2)
        print(s + "}")
    elif isinstance(obj, list):
        print(s + "[")
        for item in obj:
            display_object(item, indent+2)
        print(s + "]")
    elif isinstance(obj, torch.Tensor):
        print(s + str(list(obj.shape)) + " " + str(obj.device))
    else:
        print(s + str(obj))


def _create_optimizer_states(model, optimizer):
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    if optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]:
        optimizer.optimizer._reduce_gradients()
        optimizer.optimizer.ep_zero_optimizer.base_optimizer.step()
        optimizer.optimizer.non_ep_zero_optimizer.base_optimizer.step()
    else:
        optimizer.step()


def run_checkpoint_test(cfg):
    device = "xla"
    tp_degree = getattr(cfg, "tp_degree", 1)
    ep_degree = getattr(cfg, "ep_degree", 1)
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(cur_dir, "config.json")
    with open(config_path, "r") as f:
        json_config = json.load(f)

    # 1. Initialize original model
    mixtral_config = MixtralConfig(**json_config)
    mixtral_config.pretraining_tp = tp_degree
    mixtral_config.sequence_parallel_enabled = True
    mixtral_config.move_model_to_device = True
    mixtral_config.moe_frequency = 1
    mixtral_config.capacity_factor = 2.0
    ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=0)
    model = MixtralForCausalLM(mixtral_config).to(device)

    optimizer = ut.initialize_neuron_optimizer(model, zero1=cfg.zero1, optimizer="adam")

    _create_optimizer_states(model, optimizer)

    xm.mark_step()
    torch.distributed.barrier()

    # 2. Keep a copy of the original state_dicts
    original_model_state_dict = copy.deepcopy(model.state_dict())
    original_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    # 3. Save a partial checkpoint
    save_checkpoint(
        checkpoint_dir_str=cur_dir,
        tag="model",
        model=model,
        optimizer=optimizer,
        use_xser=True
    )

    # 4. Override states
    override_state(model)
    override_state(optimizer)

    # 5. Load the partial checkpoint
    load_checkpoint(
        path=cur_dir,
        tag="model",
        model=model,
        optimizer=optimizer,
        strict=False,
    )

    xm.mark_step()

    # 6. Verify correctness wrt original state_dict
    assert_same_tensors(original_model_state_dict, model.state_dict())
    if torch.distributed.get_rank() == 0:
        display_object(original_optimizer_state_dict)
        display_object(optimizer.state_dict())
    assert_same_tensors(original_optimizer_state_dict, optimizer.state_dict())
