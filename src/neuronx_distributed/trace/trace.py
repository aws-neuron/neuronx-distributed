import concurrent.futures
import multiprocessing
import os
import pathlib
from typing import Any, Callable, List, Optional, Union

import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.utils.utils import get_free_tcp_ports

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import is_pjrt_device

NXD_SKIP_RENDEZVOUS = "NXD_SKIP_RENDEZVOUS"


class ParallelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


class TensorParallelNeuronModel(ParallelModel):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.load = False
        self.tp_degree = len(models)
        self.executor = None  # Initialized with the first forward call

    def _load(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.tp_degree)
        futures = [
            self.executor.submit(
                torch.ops.neuron._load_collectives_neuron,
                model.model,
                i,
                1,
                i,
                self.tp_degree,
            )
            for i, model in enumerate(self.models)
        ]
        for future in concurrent.futures.as_completed(futures):
            # Here we wait for result to make sure all the processes have finished loading
            # models
            future.result()

        # We now move the state params to device
        for i, model in enumerate(self.models):
            torch_neuronx.move_trace_to_device(model, i)
        self.load = True

    def forward(self, *tensors):
        if not self.load:
            self._load()
        results = []
        futures = [self.executor.submit(model, *tensors) for model in self.models]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
        # Here we are making the assumption that we are operating in SPMD mode.
        # We can extend this to return all results.
        return results[0]

    def __del__(self):
        if self.executor:
            self.executor.shutdown()


def _trace(
    rank: int,
    func: Callable,
    example_inputs: Any,
    mp_q: multiprocessing.Queue,
    states=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    inline_weights_to_neff: bool = True,
    tp_degree: int = 1,
    max_parallel_compilations: int = None,
) -> None:
    os.environ["RANK"] = str(rank)
    if is_pjrt_device():
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    # refer to this github issue for context: https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    if compiler_workdir is None:
        compiler_workdir = f"/tmp/trace_compiler_workdir_{rank}"
    else:
        compiler_workdir = f"{compiler_workdir}_{rank}"

    for tp_rank in range(tp_degree):
        if rank == tp_rank:
            # Set flag to stop parallel_layes.load() from waiting on all
            # processes to finish checkpoint loading.
            model, input_output_alias = _get_model_shard(func)

            neff_filename, metaneff, flattener, packer, weights = torch_neuronx.xla_impl.trace._trace(
                model,
                example_inputs,
                states,
                input_output_alias,
                compiler_workdir,
                compiler_args,
                inline_weights_to_neff,
            )
            mp_q.put((neff_filename, metaneff, flattener, packer, example_inputs, input_output_alias, weights, rank))
        if max_parallel_compilations is not None and (tp_rank + 1) % max_parallel_compilations == 0:
            xm.rendezvous(f"compilation-step-{tp_rank + 1}")
    xm.rendezvous("compilation-done")


def parallel_model_trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    states=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    inline_weights_to_neff: bool = True,
    tp_degree: int = 1,
    max_parallel_compilations: int = None,
) -> ParallelModel:
    """
    Trace a distributed module/function to produce a compiled Neuron ScriptModule.

    This uses torch-xla to extract the computation graph. The input `func` should
    return a module that can be moved to the XLA device.

    The resulting module wraps all the individually traced models

    Args:
        func: A function which returns a torch module or computation
        example_inputs: An example set of inputs which will be passed to the
            `torch_module` during tracing.
        states: External state parameters which is required of the `func`
        compiler_workdir: The directory to save any compiler outputs to.
        compiler_args: Additional compiler arguments.
        options: Additional trace `Options`.
        tp_degree: Tensor parallel sharding degree

    Returns:
        A wrapper Module which wraps individual HLO computation which is a
        fused neuron::foward operation.
    """

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    mp_q = manager.Queue()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2022"
    os.environ["TPU_NUM_DEVICES"] = str(tp_degree)
    os.environ["NEURONCORE_NUM_DEVICES"] = str(tp_degree)  # for pjrt
    os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:{}".format(get_free_tcp_ports()[0])
    os.environ["WORLD_SIZE"] = str(tp_degree)
    prev_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    xmp.spawn(
        _trace,
        args=(
            func,
            example_inputs,
            mp_q,
            states,
            compiler_workdir,
            compiler_args,
            inline_weights_to_neff,
            tp_degree,
            max_parallel_compilations if max_parallel_compilations != None else tp_degree,
        ),
        start_method="spawn",
        nprocs=tp_degree,
    )
    models = [None] * tp_degree
    while not mp_q.empty():
        neff_filename, metaneff, flattener, packer, example_inputs, input_output_alias, weights, rank = mp_q.get()
        models[rank] = torch_neuronx.xla_impl.trace.create_neuron_model(
            neff_filename, metaneff, flattener, packer, example_inputs, input_output_alias, weights
        )
    torch.multiprocessing.set_sharing_strategy(prev_sharing_strategy)
    return TensorParallelNeuronModel(models)


def parallel_model_save(model: ParallelModel, save_dir: str) -> None:
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(model.models):
        torch.jit.save(model, f"{save_dir}/tp_{i}.pt")


def parallel_model_load(model_dir: str) -> ParallelModel:
    models = []
    with torch_neuronx.contexts.disable_nrt_load():
        for file_name in os.listdir(model_dir):
            models.append(torch.jit.load(f"{model_dir}/{file_name}"))
    return TensorParallelNeuronModel(models)


def _get_model_shard(func):
    if NXD_SKIP_RENDEZVOUS in os.environ:
        raise ValueError(
            "NXD_SKIP_RENDEZVOUS is a reserved environment variable. Its should not be set outside parallel_model_trace"
        )

    # Turn on skip rendevous flag
    os.environ[NXD_SKIP_RENDEZVOUS] = "1"
    try:
        return func()
    finally:
        del os.environ[NXD_SKIP_RENDEZVOUS]
