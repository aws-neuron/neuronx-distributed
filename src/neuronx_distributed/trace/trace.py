import concurrent.futures
import filecmp
import logging
import multiprocessing
import os
import pathlib
from typing import Any, Callable, List, Optional, Union

import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_neuronx import BucketModelConfig
from torch_neuronx.xla_impl.bucket_trace import create_bucket_model
from torch_neuronx.xla_impl.trace import generate_hlo, hlo_compile, setup_compiler_dirs
from torch_xla.utils.utils import get_free_tcp_ports

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

logger = logging.getLogger("Neuron")

NXD_SKIP_RENDEZVOUS = "NXD_SKIP_RENDEZVOUS"


class ParallelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


class TensorParallelNeuronModel(ParallelModel):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.load = False
        self.tp_degree = len(models)
        self.executor = None  # Initialized with the first forward call

    def _load(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.tp_degree)
        models = []
        if hasattr(self.models[0], "model"):
            models = [(i, model.model) for i, model in enumerate(self.models)]
        else:
            bucket_degree = len(self.models[0].bucket_model_executor.models)
            for bucket_rank in range(bucket_degree):
                for tp_rank in range(self.tp_degree):
                    models.append((tp_rank, self.models[tp_rank].bucket_model_executor.models[bucket_rank]))
        futures = [
            self.executor.submit(
                torch.ops.neuron._load_collectives_neuron,
                model,
                i,
                1,
                i,
                self.tp_degree,
            )
            for i, model in models
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


def collect_tp_neuron_models(models, mp_q, bucket_config=None, tp_degree=1):
    while not mp_q.empty():
        neff_filename, metaneff, flattener, packer, example_inputs, input_output_alias, weights, tp_rank, _ = mp_q.get()
        models[tp_rank] = torch_neuronx.xla_impl.trace.create_neuron_model(
            neff_filename, metaneff, flattener, packer, example_inputs, input_output_alias, weights
        )


def collect_tp_bucket_neuron_models(models, mp_q, bucket_config=None, tp_degree=1):
    shared_packer = [None] * tp_degree
    shared_flattener = [None] * tp_degree
    shared_weights = [None] * tp_degree
    input_output_aliases = [[None] * bucket_config.bucket_degree] * tp_degree

    while not mp_q.empty():
        (
            neff_filename,
            metaneff,
            flattener,
            packer,
            _,
            input_output_alias,
            weights,
            tp_rank,
            bucket_rank,
        ) = mp_q.get()

        with open(neff_filename, "rb") as handle:
            neff = handle.read()

        models[tp_rank][bucket_rank] = (neff, metaneff)
        shared_packer[tp_rank] = packer
        shared_flattener[tp_rank] = flattener
        shared_weights[tp_rank] = weights
        input_output_aliases[tp_rank][bucket_rank] = input_output_alias

    for tp_rank in range(tp_degree):
        models[tp_rank] = create_bucket_model(
            models[tp_rank],
            bucket_config,
            shared_flattener[tp_rank],
            shared_packer[tp_rank],
            shared_weights[tp_rank],
            input_output_aliases[tp_rank],
        )


def generate_ranked_folder(tp_rank, bucket_rank, bucket_degree):
    return f"_tp{tp_rank}" + (f"_bk{bucket_rank}" if bucket_degree > 1 else "")


def _compile_model_shard(
    tp_rank,
    rank_hlo,
    compiler_workdir,
    compiler_args,
    inline_weights_to_neff,
    bucket_rank,
    bucket_degree,
):
    rank_folder = generate_ranked_folder(tp_rank, bucket_rank, bucket_degree)
    ranked_compiler_workdir = os.path.join(compiler_workdir, rank_folder)

    neff_filename = None
    logging.debug(f"Current TP Rank: {tp_rank} | SPMD Target Rank: 0")
    if tp_rank == 0:
        logger.debug(f"SPMD Target Rank 0 Compilation Started")
        neff_filename = hlo_compile(
            rank_hlo,
            ranked_compiler_workdir,
            compiler_args,
        )
    else:
        zero_rank_folder = generate_ranked_folder(0, bucket_rank, bucket_degree)
        shard_dir = os.path.join(compiler_workdir, zero_rank_folder)
        shard_hlo = os.path.join(shard_dir, "model/graph.hlo")
        # verify that hlo for current tp rank is the same as the one being compiled
        if (filecmp.cmp(rank_hlo, shard_hlo)) and (not inline_weights_to_neff):
            neff_filename = os.path.join(shard_dir, "graph.neff")
        else:
            if not inline_weights_to_neff:
                logger.warning(
                    f"TP Rank {tp_rank} HLO differs from SPMD Target TP Rank 0 HLO, and will therefore need an extra compilation. This potentially indicates unoptimal use of Neuronx-Distributed Parallel Layers."
                )
            # perform parallel compilation if hlo is different from target shard
            neff_filename = hlo_compile(
                rank_hlo,
                ranked_compiler_workdir,
                compiler_args,
            )

    return neff_filename


def _trace(
    rank: int,
    func: Callable,
    example_inputs: Any,
    mp_q: multiprocessing.Queue,
    states=None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    inline_weights_to_neff: bool = True,
    bucket_config: Optional[BucketModelConfig] = None,
    tp_degree: int = 1,
    max_parallel_compilations: int = None,
) -> None:
    os.environ["RANK"] = str(rank)
    if requires_init_pg_override():
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    # refer to this github issue for context: https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    if compiler_workdir is None:
        compiler_workdir = "/tmp/trace_compiler_workdir/"

    bucket_degree = bucket_config.bucket_degree if bucket_config else 1
    if not bucket_config:
        # bucket rank is 1
        example_inputs = [example_inputs]

    for tp_rank in range(tp_degree):
        artifacts_collection = []
        if rank == tp_rank:
            for bucket_rank in range(bucket_degree):
                # Set flag to stop parallel_layes.load() from waiting on all
                # processes to finish checkpoint loading.
                func_kwargs = bucket_config.get_func_kwargs_for_bucket_rank(bucket_rank) if bucket_config else {}
                model, input_output_alias = _get_model_shard(func, func_kwargs)

                (hlo, constant_parameter_tensors, flattener, packer, metaneff, weights) = generate_hlo(
                    model,
                    example_inputs[bucket_rank],
                    states=states,
                    input_output_aliases=input_output_alias,
                    inline_weights_to_neff=inline_weights_to_neff,
                )
                rank_folder = generate_ranked_folder(tp_rank, bucket_rank, bucket_degree)
                rank_hlo = setup_compiler_dirs(
                    hlo, os.path.join(compiler_workdir, rank_folder), constant_parameter_tensors, inline_weights_to_neff
                )

                artifacts_collection.append((rank_hlo, input_output_alias, flattener, packer, metaneff, weights))

        if tp_rank == 0:
            xm.rendezvous("tp-rank-0-hlos-generated-and-saved")

        if rank == tp_rank:
            for bucket_rank, artifacts in enumerate(artifacts_collection):
                (rank_hlo, input_output_alias, flattener, packer, metaneff, weights) = artifacts
                neff_filename = _compile_model_shard(
                    tp_rank,
                    rank_hlo,
                    compiler_workdir,
                    compiler_args,
                    inline_weights_to_neff,
                    bucket_rank,
                    bucket_degree,
                )
                mp_q.put(
                    (
                        neff_filename,
                        metaneff,
                        flattener,
                        packer,
                        example_inputs[bucket_rank],
                        input_output_alias,
                        weights,
                        tp_rank,
                        bucket_rank,
                    )
                )

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
    bucket_config: Optional[BucketModelConfig] = None,
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
        inline_weights_to_neff: If False, it separates the weights from the neff,
            which allows for the possiblity of weight replacement. The default is True.
        bucket_config: To enable bucketing, pass in a BucketModelConfig object. The default
            is None, meaning no bucketing.
        tp_degree: Tensor parallel sharding degree
        max_parallel_compilations: If specified, this function will only trace these number
            of models in parallel, which can be necessary to prevent OOMs while tracing. The default
            is None, which means the number of parallel compilations is equal to the `tp_degree`.

    Returns:
        A wrapper Module which wraps individual HLO computation which is a
        fused neuron::foward operation.
    """

    if bucket_config is not None and inline_weights_to_neff:
        raise ValueError(
            "Bucketing is not supported when inline_weights_to_neff=True. Set inline_weights_to_neff=False, if using the bucketing feature."
        )

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

    if bucket_config:
        bucket_config.store_example_inputs(example_inputs)

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
            bucket_config,
            tp_degree,
            max_parallel_compilations if max_parallel_compilations != None else tp_degree,
        ),
        start_method="spawn",
        nprocs=tp_degree,
    )
    collector_func = collect_tp_neuron_models if not bucket_config else collect_tp_bucket_neuron_models
    models = [None if not bucket_config else [None] * bucket_config.bucket_degree] * tp_degree

    # models will be collected by the collector func depending on if bucketing is enabled
    collector_func(models, mp_q, bucket_config, tp_degree)

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


def _get_model_shard(func, func_kwargs=None):
    if func_kwargs is None:
        func_kwargs = {}
    if NXD_SKIP_RENDEZVOUS in os.environ:
        raise ValueError(
            "NXD_SKIP_RENDEZVOUS is a reserved environment variable. Its should not be set outside parallel_model_trace"
        )

    # Turn on skip rendevous flag
    os.environ[NXD_SKIP_RENDEZVOUS] = "1"
    try:
        return func(**func_kwargs)
    finally:
        del os.environ[NXD_SKIP_RENDEZVOUS]
