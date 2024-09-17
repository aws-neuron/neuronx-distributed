import concurrent.futures
import gc
import logging
import multiprocessing
import os
import pathlib
import shutil
from collections import defaultdict
from typing import Any, Callable, List, Optional, Union, Tuple
from typing import cast


import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_neuronx import BucketModelConfig
from torch_neuronx.xla_impl.bucket_trace import create_bucket_model
from torch_neuronx.xla_impl.torchscript import replace_weights
from torch_xla.utils.utils import get_free_tcp_ports

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.checkpointing import NXD_SKIP_RENDEZVOUS
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.utils import (
    divide,
    is_torch_version_greater_than_2,
    requires_init_pg_override,
)
from neuronx_distributed.quantization.quantization_layers import (
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronx_distributed.utils.model_utils import init_on_device

logger = logging.getLogger("Neuron")

# Varible to specify the types of Moduels that are currently Sharded
__SUPPORTED_SHARDED_MODULES = (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    QuantizedRowParallel,
    QuantizedColumnParallel,
)


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
        if self.executor is not None:
            futures = [self.executor.submit(model, *tensors) for model in self.models]
        else:
            raise RuntimeError("executor is None although it has to be properly initialized")
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
    max_parallel_compilations: Optional[int] = None,
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
        if rank == tp_rank:
            for bucket_rank in range(bucket_degree):
                # Set flag to stop parallel_layes.load() from waiting on all
                # processes to finish checkpoint loading.
                func_kwargs = bucket_config.get_func_kwargs_for_bucket_rank(bucket_rank) if bucket_config else {}
                model, input_output_alias = _get_model_shard(func, func_kwargs)
                rank_folder = generate_ranked_folder(tp_rank, bucket_rank, bucket_degree)

                neff_filename, metaneff, flattener, packer, weights = torch_neuronx.xla_impl.trace._trace(
                    model,
                    example_inputs[bucket_rank],
                    states,
                    input_output_alias,
                    os.path.join(compiler_workdir, rank_folder),
                    compiler_args,
                    inline_weights_to_neff,
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
    max_parallel_compilations: Optional[int] = None,
    spmd_mode: bool = False,
    checkpoint_loader_callable: Optional[Callable] = None,
    force_custom_init_on_device: bool = False,
    serialization_path: Optional[str] = None,
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
        spmd_mode: If True, it compiles a single rank. This compiled model is then loaded with
            the rank specific weights to generate the other ranks.
        checkpoint_loader_callable: A callable method to load the model's checkpoint.  When using
            spmd_mode, checkpoint_loader_callable is a required argument.
        force_custom_init_on_device: Bool to indicate whether to force use custom init_on_device functionality
            NOTE: If you are trying to use it for Quantized api, make sure this bool is set to True
        serialization_path: A path to store the serialized traced model if provided. Currently only works
            for SPMD mode.
    Returns:
        A wrapper Module which wraps individual HLO computation which is a
        fused neuron::forward operation.
    """

    if bucket_config is not None and inline_weights_to_neff:
        raise ValueError(
            "Bucketing is not supported when inline_weights_to_neff=True. Set inline_weights_to_neff=False, if using the bucketing feature."
        )

    if spmd_mode is True and inline_weights_to_neff:
        raise ValueError(
            "Spmd mode is not supported when inline_weights_to_neff=True. Set inline_weights_to_neff=False, if using the spmd mode."
        )

    if spmd_mode is True and checkpoint_loader_callable is None:
        raise ValueError(
            "Argument checkpoint_loader_callable not provided. When using spmd mode you do not need to load the weights in func, instead pass checkpoint_loader_callable to parallel_model_trace"
        )
    if spmd_mode is False and serialization_path is not None:
        raise ValueError(
            "Currently serializing traced model during tracing is only supported for SPMD mode. Please either turn on SPMD mode or set serialization_path as None."
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

    if spmd_mode:
        models = _spmd_trace(
            func,
            example_inputs,
            cast(Callable, checkpoint_loader_callable),
            tp_degree,
            states,
            compiler_workdir,
            compiler_args,
            bucket_config,
            force_custom_init_on_device=force_custom_init_on_device,
            serialization_path=serialization_path,
        )
    else:
        logging.warn(
            "Using non SPMD mode. Set spmd_mode=True if the worlkload is SPMD for a faster trace. Tracing in non SPMD mode for large models can run into OOM errors as we compile all ranks"
        )
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
                max_parallel_compilations if max_parallel_compilations is not None else tp_degree,
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


def _save_traced_model(model, save_dir, rank):
    torch.jit.save(model, f"{save_dir}/tp_{rank}.pt")


def parallel_model_save(model: ParallelModel, save_dir: str) -> None:
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(model.models):
        _save_traced_model(model, save_dir, i)


def find_unique_dtypes(model):
    state_dict = model.state_dict()
    dtype_map: defaultdict = defaultdict(int)
    for _, value in state_dict.items():
        dtype_map[value.dtype] += 1
    return dict(dtype_map)


def _load_script_modules(model_dir: str) -> Tuple[List[Any], List[str]]:
    models = []
    with torch_neuronx.contexts.disable_nrt_load():
        model_rank_files = sorted(
            [pth for pth in os.listdir(model_dir)], key=lambda x: int(x.replace(".pt", "").replace("tp_", ""))
        )
        for file_name in model_rank_files:
            models.append(torch.jit.load(f"{model_dir}/{file_name}"))
    return models, model_rank_files


def parallel_model_load(model_dir: str) -> ParallelModel:
    models, model_rank_files = _load_script_modules(model_dir)

    for count, file_name in enumerate(model_rank_files):
        path = os.path.relpath(os.path.join(model_dir, file_name), os.path.dirname(os.path.dirname(model_dir)))
        logging.debug(f"{path}: {find_unique_dtypes(models[count])}")

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


def _spmd_trace(
    func,
    example_inputs: Any,
    checkpoint_loader_callable: Callable,
    tp_degree,
    states,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    bucket_config: Optional[BucketModelConfig] = None,
    force_custom_init_on_device: bool = False,
    serialization_path: Optional[str] = None,
):
    """
    Xla trace a signle rank and compile it with neuronx-cc.
    The single rank model has it's weight replaced to build the model for all tp ranks.
    """
    # This does not validate if the HLOs are same across all ranks.
    _validate_traceable(func, tp_degree, force_custom_init_on_device=force_custom_init_on_device)

    xmp.spawn(
        _single_rank_trace,
        args=(
            func,
            example_inputs,
            states,
            compiler_workdir,
            compiler_args,
            bucket_config,
            tp_degree,
        ),
        start_method="spawn",
        nprocs=tp_degree,
    )

    if serialization_path is None:
        model = _load_weight_into_model(
            tp_degree, func, compiler_workdir, checkpoint_loader_callable, force_custom_init_on_device
        )
    else:
        model = _load_weight_and_serialize(
            tp_degree,
            func,
            compiler_workdir,
            checkpoint_loader_callable,
            serialization_path,
            force_custom_init_on_device,
        )
    return model


def _load_weight_into_model(tp_degree, func, compiler_workdir, checkpoint_loader_callable, force_custom_init_on_device):
    """
    Load the weight for each sub-model while keeping them in memory, and then return it

    This works faster on small models which doesn't need much memory
    """
    models = []
    checkpoint = checkpoint_loader_callable()
    for rank in range(0, tp_degree):
        model = _load_weights(rank, func, compiler_workdir, checkpoint, tp_degree, force_custom_init_on_device)
        models.append(model)
        gc.collect()
    return models


def _load_weight_and_serialize(
    tp_degree, func, compiler_workdir, checkpoint_loader_callable, serialization_path, force_custom_init_on_device
):
    """
    Load the weight for each sub-model, serialize it and then move to the next sub-model. It
    will load all the sub-models at the end to return the traced model.

    This works better for large models which needs more memory
    """
    checkpoint = checkpoint_loader_callable()

    pathlib.Path(serialization_path).mkdir(parents=True, exist_ok=True)
    for rank in range(0, tp_degree):
        model = _load_weights(rank, func, compiler_workdir, checkpoint, tp_degree, force_custom_init_on_device)
        _save_traced_model(model, serialization_path, rank)

    del checkpoint, model
    gc.collect()
    models, _ = _load_script_modules(serialization_path)
    return models


def _validate_traceable(func: Callable, tp_degree: int, force_custom_init_on_device: bool = False):
    """
    Perform model architecture level validation if the model is
    SPMD traceable.
    """
    _mock_parallel_state(tp_degree, rank=0)
    with init_on_device(torch.device("meta"), force_custom_init_on_device=force_custom_init_on_device):
        model, _ = func()

    assert isinstance(
        model, torch.nn.Module
    ), "The first return value of func is expected to be of type torch.nn.Module"

    def _validate_children(module: torch.nn.Module):
        if module is None:
            return

        # Sharding across vocab dimension requires rank level constants for intput masking.
        if isinstance(module, ParallelEmbedding):
            if not module.shard_across_embedding:
                raise ValueError(
                    "Sharding across vocab dimension in ParallelEmbedding is not supported when tracing with spmd_mode=True"
                )

        for child in module.children():
            if child is not None:
                _validate_children(child)

    _validate_children(model)


def _single_rank_trace(rank, func, example_inputs, states, compiler_workdir, compiler_args, bucket_config, tp_degree):
    os.environ["RANK"] = str(rank)
    if requires_init_pg_override():
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    # refer to this github issue for context: https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    if rank == 0:
        if compiler_workdir is None:
            compiler_workdir = "/tmp/trace_compiler_workdir/"
        if pathlib.Path(compiler_workdir).exists():
            shutil.rmtree(compiler_workdir)

        bucket_degree = bucket_config.bucket_degree if bucket_config else 1
        if not bucket_config:
            # bucket rank is 1
            example_inputs = [example_inputs]
        weights = None
        flattener = None
        packer = None
        input_output_aliases = []
        models = []
        neff_filename = None
        for bucket_rank in range(bucket_degree):
            # Set flag to stop parallel_layes.load() from waiting on all
            # processes to finish checkpoint loading.
            func_kwargs = bucket_config.get_func_kwargs_for_bucket_rank(bucket_rank) if bucket_config else {}
            model, input_output_alias = _get_model_shard(func, func_kwargs)
            input_output_aliases.append(input_output_alias)
            rank_folder = generate_ranked_folder(rank, bucket_rank, bucket_degree)

            neff_filename, metaneff, flattener, packer, weights = torch_neuronx.xla_impl.trace._trace(
                model,
                example_inputs[bucket_rank],
                states,
                input_output_alias,
                os.path.join(compiler_workdir, rank_folder),
                compiler_args,
                False,
            )

            with open(neff_filename, "rb") as handle:
                neff = handle.read()

            models.append((neff, metaneff))

        if bucket_degree == 1:
            traced_model = torch_neuronx.xla_impl.trace.create_neuron_model(
                neff_filename, models[0][1], flattener, packer, example_inputs[0], input_output_aliases[0], weights
            )
        else:
            traced_model = create_bucket_model(models, bucket_config, flattener, packer, weights, input_output_aliases)
        pathlib.Path(compiler_workdir).mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced_model, os.path.join(compiler_workdir, "tp_0.pt"))
    xm.rendezvous("done-strict-tracing")


def _load_weights(
    rank, func, compiler_workdir, checkpoint_source, tp_degree, force_custom_init_on_device: bool = False
):
    """
    Replaces the rank specific weights into the compiled
    torchscript model.
    """

    checkpoint = checkpoint_source.copy()
    _mock_parallel_state(tp_degree, rank)
    with init_on_device(torch.device("meta"), force_custom_init_on_device=force_custom_init_on_device):
        model, _ = func()

    get_sharded_checkpoint(checkpoint, model, rank, tp_degree)

    with torch_neuronx.contexts.disable_nrt_load():
        rank_0_path = os.path.join(compiler_workdir, "tp_0.pt")
        traced_model = torch.jit.load(rank_0_path)
        replace_weights(traced_model, checkpoint)
        return traced_model


def get_sharded_checkpoint(checkpoint, model, rank, tp_degree):
    invoke_preshard_hook(model, checkpoint, "")

    dtype = None
    if hasattr(model, "config") and hasattr(model.config, "torch_dtype"):
        dtype = model.config.torch_dtype

    # Shards the checkpoint to the right weight for the rank
    shard_children(model, checkpoint, "", dtype, rank, tp_degree)


def create_local_weight_qkv(rank, world_size, full_weight, partition_dim, q_len, kv_len, out_weight=None):
    # Shard q,k,v weights separately and then fuse them for each rank
    q_weight, k_weight, v_weight = torch.split(full_weight, [q_len, kv_len, kv_len], dim=partition_dim)
    q_weight_list = torch.split(q_weight, divide(q_len, world_size), dim=partition_dim)[rank::world_size]
    k_weight_list = torch.split(k_weight, divide(kv_len, world_size), dim=partition_dim)[rank::world_size]
    v_weight_list = torch.split(v_weight, divide(kv_len, world_size), dim=partition_dim)[rank::world_size]

    with torch.no_grad():
        return torch.cat((
            torch.cat(q_weight_list, dim=partition_dim),
            torch.cat(k_weight_list, dim=partition_dim),
            torch.cat(v_weight_list, dim=partition_dim),
        ), dim=partition_dim, out=out_weight)

def create_local_weight(rank, world_size, full_weight, partition_dim, per_partition_size, stride, out_weight=None):
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(full_weight, per_partition_per_stride_size, dim=partition_dim)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        return torch.cat(my_weight_list, dim=partition_dim, out=out_weight)


def _mock_parallel_state(tp_degree: int, rank: int):
    """
    Set correct values in parallel_state to let Neuron Models
    load on CPU. This is done to load Neuron models outside
    torch distributed process group.
    """

    class Mock:
        def __init__(self, world_size):
            self.world_size = world_size

        def size(self):
            return self.world_size

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = Mock(tp_degree)
    parallel_state._DATA_PARALLEL_GROUP = Mock(1)
    parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = rank
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = Mock(1)


def invoke_preshard_hook(module, checkpoint, prefix):
    """
    Preshard hooks are hooks to manipulate checkpoints
    Checkpoint manipulation for GQA replication is one usecase.
    """
    if module is None:
        return

    # This is temporary until we formailze the preshard_hook in src
    if hasattr(module, "preshard_hook"):
        module.preshard_hook(checkpoint, prefix + "weight")
        return

    for name, child in module._modules.items():
        if child is not None:
            invoke_preshard_hook(child, checkpoint, prefix + name + ".")


def shard_children(module, checkpoint, prefix, dtype, rank, tp_degree):
    """
    Checkpoint weights are sharded based on rank and tp_degree
    """

    if module is None:
        return

    for name, child in module._modules.items():
        if child is not None:
            shard_children(child, checkpoint, prefix + name + ".", dtype, rank, tp_degree)

    if not isinstance(module, __SUPPORTED_SHARDED_MODULES):
        return

    for module_parameter_name, module_parameter in module.named_parameters():
        parameter_name = prefix + module_parameter_name

        # If a few cases, the module parameter name might not appear exactly in the state dict
        # This is true especially for pytorch quantized models. In that case add the attribute,
        # get_tensor_from_state_dict, for that parameter
        if hasattr(module_parameter, "get_tensor_from_state_dict"):
            tensor = module_parameter.get_tensor_from_state_dict(prefix, checkpoint)
        else:
            if parameter_name not in checkpoint:
                return
            if dtype and checkpoint[parameter_name].dtype != dtype:
                checkpoint[parameter_name] = checkpoint[parameter_name].to(dtype)
            tensor = checkpoint[parameter_name]

        if hasattr(module_parameter, "tensor_model_parallel") and module_parameter.tensor_model_parallel:
            partition_dim = module_parameter.partition_dim
            stride = module_parameter.partition_stride

            per_partition_size = tensor.shape[partition_dim] // tp_degree
            if hasattr(module_parameter, "fused_qkv"):
                query_len = module_parameter.num_attention_heads * module_parameter.head_dim
                kv_len = module_parameter.num_key_value_heads * module_parameter.head_dim
                checkpoint[parameter_name] = create_local_weight_qkv(
                    rank, tp_degree, tensor, partition_dim, query_len, kv_len
                )
            else:
                checkpoint[parameter_name] = create_local_weight(
                    rank, tp_degree, tensor, partition_dim, per_partition_size, stride
                )
        else:
            checkpoint[parameter_name] = tensor
