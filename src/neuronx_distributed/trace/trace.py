import os
import concurrent.futures
import multiprocessing
import pathlib
import torch
import torch_neuronx
from torch_neuronx.xla_impl.options import Options
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
from typing import List, Any, Union, Callable, Optional, Dict, Iterable

from neuronx_distributed.parallel_layers import layers, parallel_state


class ParallelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


class TensorParallelNeuronModel(ParallelModel):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.load = False
        self.tp_degree = len(models)
    
    def _load(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    torch.ops.neuron._load_collectives_neuron, model.model, i, 1, i, self.tp_degree)
                for i, model in enumerate(self.models)]
            for future in concurrent.futures.as_completed(futures):
                # Here we wait for result to make sure all the processes have finished loading
                # models
                future.result()
        self.load = True
    
    def forward(self, *tensors):
        if not self.load:
            self._load()
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(model, *tensors) for model in self.models]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        # Here we are making the assumption that we are operating in SPMD mode.
        # We can extend this to return all results.
        return results[0]   


def _trace(
    rank: int,
    func: Callable,
    example_inputs: Any,
    mp_q: multiprocessing.Queue,
    states=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    options: Union[Iterable[Options], Options] = None) -> None:
    os.environ['RANK'] = str(rank)
    torch.distributed.init_process_group('xla')
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=2)
    model = func()
    neff_filename, metaneff, flattener, packer = torch_neuronx.xla_impl.trace._trace(
        model, example_inputs, states, f"{compiler_workdir}_{rank}", compiler_args, options)
    mp_q.put((neff_filename, metaneff, flattener, packer))


def parallel_model_trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    states=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    options: Union[Iterable[Options], Options] = None,
    tp_degree: int = 1,
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

    mp_q = multiprocessing.Queue()
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2022"
    os.environ["TPU_NUM_DEVICES"] = str(tp_degree)
    os.environ['WORLD_SIZE'] = str(tp_degree)
    compiler_workdir = "/tmp/compiler_workdir" if compiler_workdir is None else compiler_workdir
    xmp.spawn(
        _trace,
        args=(func, example_inputs, mp_q, states, compiler_workdir, compiler_args, options),
        start_method="fork"
    )
    models = [torch_neuronx.xla_impl.trace.create_neuron_model(*mp_q.get(), example_inputs) for _ in range(tp_degree)]
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