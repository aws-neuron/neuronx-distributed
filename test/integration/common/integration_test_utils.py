import os
import time
import subprocess

from botocore.utils import IMDSFetcher
import torch
import torch_xla.core.xla_model as xm
import random
import numpy

from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed

from typing import Tuple


def print_separator(message: str):
    torch.distributed.barrier()
    filler_len = (78 - len(message)) // 2
    filler = "-" * filler_len
    string = "\n" + filler + " {} ".format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()


def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


# Setup function - sets RNG seeds and inits model parallel state
def test_init(tensor_model_parallel_size: int, rng_seed: int):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    set_random_seed(rng_seed)
    model_parallel_xla_manual_seed(rng_seed)


# Cleanup function to call after test passes
def test_cleanup():
    parallel_state.destroy_model_parallel()
    torch.distributed.barrier()


PARALLEL_LAYERS = tuple(layers.BaseParallelConv.__subclasses__() + layers.BaseParallelLinear.__subclasses__())


# Given an input tuple, move all the tensors within to the given device
def _detach_and_move_nested_tuple_to_device(
    input_tup: Tuple[torch.tensor, ...], device: torch.device
) -> Tuple[torch.tensor, ...]:
    ret = []
    for item in input_tup:
        if isinstance(item, tuple):
            ret.append(_detach_and_move_nested_tuple_to_device(item, device))
        else:
            ret.append(item.detach().to(device))
    return tuple(ret)


def exercise_single_module_fwd_bwd(
    module: torch.nn.Module, input_tensors: Tuple[torch.tensor, ...], mark_step_between_fwd_bwd: bool
) -> Tuple[torch.tensor, Tuple[torch.tensor, ...]]:
    """Run a single forward + backward step for a given module and return the output and gradients

    Given a Torch module, run a single forward and backward step. Multiplies the output of the forward
    pass by 1.1 to form an artificial "target" and computes the MSE loss.

    This function handles all data movement to/from the XLA device:
        - It expects the provided module and input_tensors to all be on CPU
            - Note: the function will copy all input tensors using .detach(), so the tuple can be re-used
              for multiple invocations of this function
        - It returns the forward pass output of the module and backwards pass gradients already on CPU

    Arguments:
        module: The module to test
        input_tensors: A tuple containing the input tensors to exercise the module with
        mark_step_between_fwd_bwd: Whether to call xm.mark_step() between the forward and backward steps
    """
    device = xm.xla_device()

    input_tensors = _detach_and_move_nested_tuple_to_device(input_tensors, device)
    module.to(device=device)

    output = module(*input_tensors)

    # Helper function for flattening a nested tuple
    # SD UNet blocks return a nested tuple of tensors, need to flatten to cat and compute loss
    # Also flattens each tensor while doing this
    def flatten(input_tup):
        result = []
        for item in input_tup:
            if isinstance(item, tuple):
                result.extend(flatten(item))
            else:
                result.append(torch.flatten(item))
        return tuple(result)

    # Some modules produce an output tuple
    if isinstance(output, tuple):
        output = torch.cat(flatten(output))

    target = torch.multiply(output, 1.1)

    loss = torch.nn.functional.mse_loss(target, output)

    # Used for debug purposes. Sometimes compiler compiles the model differently
    # if FWD and BWD are split up
    if mark_step_between_fwd_bwd:
        xm.mark_step()

    loss.backward()

    grads_on_device = []

    # Internally module.modules() iterates over module._modules, which is a dict of name: str -> module: torch.nn.Module
    # Python guarantees dictionary ordering since 3.7, so as long as the two Modules being tested were constructed in
    # the same way then this is safe for getting the same order of grads
    # Note: substituting layers in a module being tested (e.g. my_module.qkv_linear = ColumnParallelLinear()) is safe
    #       because the layer still has the same name and thus order in the dict
    for child in module.modules():
        # Handle parallel conv layers
        bias = hasattr(child, "bias") and child.bias is not None
        if isinstance(child, PARALLEL_LAYERS):
            if isinstance(child, (layers.InputChannelParallelConv2d, layers.RowParallelLinear)):
                # TODO: Workaround for V1304088281 - AllGather on 1st dim crashes compiler
                dldw = torch.permute(child.weight.grad.data, [1, 0, 2, 3])
                if parallel_state.get_tensor_model_parallel_size() > 1:
                    dldw = xm.all_gather(
                        dldw, groups=parallel_state.get_tensor_model_parallel_replica_groups(), pin_layout=False
                    )
                # Permute back to original shape so that it matches the non-parallel layer's shape
                dldw = torch.permute(dldw, [1, 0, 2, 3])
                if bias:
                    dldb = child.bias.grad.data
            else:
                dldw = child.weight.grad.data
                if parallel_state.get_tensor_model_parallel_size() > 1:
                    dldw = xm.all_gather(
                        dldw,
                        groups=parallel_state.get_tensor_model_parallel_replica_groups(),
                        pin_layout=False,
                    )
                if bias:
                    dldb = child.bias.data
                    # Bias is always sharded for OutputChannelConv2d, but only sharded for ColumnParallelLinear if gather_output is False
                    if parallel_state.get_tensor_model_parallel_size() > 1 and (
                        isinstance(child, layers.OutputChannelParallelConv2d) or (
                        isinstance(child, layers.ColumnParallelLinear) and child.gather_output
                    )):
                        dldb = xm.all_gather(
                            dldb,
                            groups=parallel_state.get_tensor_model_parallel_replica_groups(),
                            pin_layout=False,
                        )

            grads_on_device.append(dldw)
            grads_on_device.append(dldb)
        else:
            if hasattr(child, "weight") and child.weight is not None:
                grads_on_device.append(child.weight.grad.data)
            if hasattr(child, "bias") and child.bias is not None:
                grads_on_device.append(child.bias.grad.data)

    torch.distributed.barrier()
    xm.mark_step()

    grads_on_cpu = tuple(grad.detach().to("cpu") for grad in grads_on_device)
    output_on_cpu = output.detach().to("cpu")

    del device

    return (output_on_cpu, grads_on_cpu)


# assert_close_on_output_tensor allows the user to choose if the output tensors should be compared with
#     torch.testing.assert_close (compares both relative diff and absolute diff, preferred) or using a simple absolute
#     error check (not preferred, only used in cases that fail the assert_close check like
#     V1305356298)
def test_modules(
    test_module: torch.nn.Module,
    control_module: torch.nn.Module,
    input_tensors: Tuple[torch.tensor, ...],
    check_output_tensor: bool = True,
    assert_close_on_output_tensor: bool = True,
    mark_step_between_fwd_bwd: bool = False,
) -> Tuple[bool, bool, bool]:
    """Given two modules, runs a forward + backward step for both using the same input tensors and compares their
    outputs and gradients

    Given two modules (the "test" and "control" modules), runs a single forward and backward step for each.

    This function assumes the test and control modules have identical structures, besides the test module containing
    parallel versions of some layers. This function can automatically compare gradients between parallel and
    non-parallel versions of parallel conv2d and linear layers.
    TODO: expand (generalize?) gradient comparison capabilities

    Multiplies the each module's forward pass output of the forward pass by 1.1 to form an artificial "target"
    and computes the MSE loss.

    This function handles all data movement to/from the XLA device:
        - It expects the provided modules and input_tensors to all be on CPU
            - Note: the function will copy all input tensors using .detach(), so the tuple can be re-used
              for multiple invocations of this function

    Returns a tuple containing three bools indicating:
        - Compilation pass/fail
        - Output tensor check pass/fail
        - Gradient check pass/fail

    Arguments:
        test_module: The module being tested
        control_module: The reference module
        input_tensors: A tuple containing the input tensors to exercise the modules with
        check_output_tensor: Whether to check that the output tensors match. Some modules fail this check
                             but pass the gradient check.
        assert_close_on_output_tensor: Whether to use torch.testing.assert_close when comparing the output tensors. If
                             set to False, uses an absolute value check instead
        mark_step_between_fwd_bwd: Whether to call xm.mark_step() between the forward and backward steps
    """
    xm.rendezvous("start_test_modules")
    test_output, test_grads = exercise_single_module_fwd_bwd(test_module, input_tensors, mark_step_between_fwd_bwd)
    del test_module
    xm.master_print("done exercising test module")

    xm.rendezvous("start_testing_control_module")
    control_output, control_grads = exercise_single_module_fwd_bwd(
        control_module, input_tensors, mark_step_between_fwd_bwd
    )
    del control_module
    xm.master_print("done exercising control module")

    xm.rendezvous("check_outputs")

    output_pass = True
    # TODO: the ability to toggle the output tensor check at all exists because some testcases don't even pass
    #       an absolute value check, but still pass the gradient check - see V1310769999
    if check_output_tensor:
        try:
            # TODO: this is only toggle-able because some testcases fail assert_close on their output
            #       e.g. V1305356298
            if assert_close_on_output_tensor:
                torch.testing.assert_close(
                    test_output, control_output, atol=1e-5, rtol=0.01
                ), "Control and test outputs from fwd pass did not match!"
            else:
                error = test_output.sub(control_output).abs().max()
                limit = 1e-3
                output_pass = error < limit
                assert error < limit, f"Expected absolute error in output < {limit}, but got {error}"
        except Exception as e:
            xm.master_print(e)
            output_pass = False

    # Assert on gradients
    grads_pass = True
    assert len(test_grads) == len(
        control_grads
    ), f"Expected to find same number of gradients in test and control modules, but test has {len(test_grads)} and control has {len(control_grads)}"
    for test_grad, control_grad in zip(test_grads, control_grads):
        try:
            torch.testing.assert_close(
                test_grad, control_grad, atol=1e-5, rtol=0.01
            ), "Control and test gradients did not match!"
        except Exception as e:
            xm.master_print(e)
            grads_pass = False

    # If we got here compilation passed
    return (True, output_pass, grads_pass)

def get_dir_size(dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def download_from_s3(s3_file, local_file):
    # track download time and size
    start_time = time.time()

    # download weights from s3
    sync_cmd = ["aws", "s3", "cp", s3_file, local_file, "--only-show-errors"]
    try:
        print(f"Downloading from s3 <{s3_file}> to local <{local_file}>")
        result = subprocess.run(sync_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"{result.stderr}")

    except Exception as e:
        raise Exception(f"Weight download from s3 failed: {e}")

    # calculate download statistics
    total_time = time.time() - start_time
    downloaded_bytes = os.path.getsize(local_file)
    if downloaded_bytes == 0:
        raise Exception("Weight download from s3 failed: downloaded ckpt file is empty")
    print(f"Download duration: {round(total_time, 2)}s")
    print(f"Download size: {round(downloaded_bytes / 1000000000, 2)}GB")
    print(f"Download speed: {round(downloaded_bytes * 8 / 1000000000 / total_time, 2)}Gbps")
