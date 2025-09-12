import os
import sys
import gc
import torch
import pytest
import unittest
import shutil
import torch_neuronx
from functools import partial
from types import SimpleNamespace
from dataclasses import dataclass
import torch_xla.core.xla_model as xm
from typing import Dict, Any, Optional
from multiprocessing import Process, Queue

from integration.common.integration_test_utils import get_profile_latency
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed_inference.models.config import MoENeuronConfig, InferenceConfig

os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'
os.environ['NEURON_RT_VIRTUAL_CORE_SIZE']='2' 
os.environ['NEURON_RT_NUM_CORES']='64'

torch.manual_seed(0)
NUM_LAYERS_ACCURACY = 1
NUM_LAYERS_PERFORMANCE = 4
SEQ_LENS = [128, 1024]

@dataclass
class TestResult:
    name: str
    model_type: str
    seq_len: int
    test_type: str
    perf_passed: bool
    perf_details: Dict[str, Any]
    error_message: Optional[str] = None

class MoEClass(torch.nn.Module):
    def __init__(self, config: InferenceConfig, num_layers=NUM_LAYERS_ACCURACY, skip_scatter_gather=False):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.skip_scatter_gather = skip_scatter_gather
        self.tensor_parallel_group = get_tp_group(config)
        moe = torch.nn.ModuleList([initialize_moe_module(config).eval() for _ in range(num_layers)])
        self.model = moe

    def forward(self, hidden_states):
        if self.config.neuron_config.sequence_parallel_enabled and not self.skip_scatter_gather:
            hidden_states = mappings._reduce_scatter_along_dim(
                            hidden_states,
                            1,
                            xm.REDUCE_MAX,
                            process_group=get_tp_group(self.config),
                        )

        for layer in self.model:
            hidden_states = layer(hidden_states)[0]
        
        if self.config.neuron_config.sequence_parallel_enabled and not self.skip_scatter_gather:
            hidden_states = mappings.gather_from_sequence_parallel_region(
                            hidden_states,
                            sequence_dimension=1,
                            to_model_parallel=False,
                            process_group=self.tensor_parallel_group,
                        )

        return hidden_states

def _load_module_moe(config, num_layers=NUM_LAYERS_ACCURACY, skip_scatter_gather=False):
    return MoEClass(config, num_layers, skip_scatter_gather).eval()

def generate_weight(a, b, *size, dtype, perf_weight):
    if perf_weight:
        return torch.ones(*size, dtype=dtype)
    return ((b - a) * torch.rand(*size) + a).to(dtype)

def _add_compiler_args(mac_thres):
    compiler_args = " --enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1 --lnc=2"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    compiler_args += f" --internal-hlo2tensorizer-options=' --modular-flow-mac-threshold={str(int(mac_thres))} --verify-hlo=true'"
    return compiler_args

def compile_neuron_moe_model(inference_config, checkpoint, load_module, hidden_states, compiler_workdir, num_layers=NUM_LAYERS_ACCURACY, skip_scatter_gather=False, initialize_weights=False):
    builder = ModelBuilder(
        router=None,
        tp_degree=inference_config.tp_degree,
        checkpoint_loader=lambda: checkpoint,
        compiler_workdir=compiler_workdir,
        logical_nc_config=2,
    )
    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, inference_config, num_layers, skip_scatter_gather),
            input_output_aliases={},
        ),
        example_inputs=[((hidden_states,)),],
        compiler_args=_add_compiler_args(inference_config.mac_thres),
    )
    neuron_model = builder.trace(initialize_model_weights=initialize_weights)
    return neuron_model

def _generate_model_weights(n_routed_experts, hidden_size, intermediate_size, num_layers, dtype=torch.bfloat16):

    perf_weight = num_layers == NUM_LAYERS_PERFORMANCE

    checkpoint = {}
    for i in range(num_layers):
        prefix = f"model.{i}."
        checkpoint.update({
                f"{prefix}expert_mlps.spmd_rank.rank": torch.nn.Parameter(torch.arange(0, 64, dtype=torch.int32), requires_grad=False),
                f"{prefix}router.linear_router.weight": torch.nn.Parameter(generate_weight(-0.05, 0.05,(n_routed_experts, hidden_size),dtype=torch.float32, perf_weight = perf_weight)),
                f"{prefix}expert_mlps.mlp_op.down_proj.weight": torch.nn.Parameter(generate_weight(-0.05, 0.05,(n_routed_experts, intermediate_size, hidden_size),dtype=dtype, perf_weight = perf_weight)),
                f"{prefix}expert_mlps.mlp_op.gate_up_proj.weight": torch.nn.Parameter(generate_weight(-0.05, 0.05,(n_routed_experts, hidden_size, intermediate_size*2),dtype=dtype, perf_weight = perf_weight)),
        })
    return checkpoint

def run_model_execution_with_accuracy(config_no_sp, config_sp, checkpoint, hidden_states):
    """Function to run both models and accuracy check"""
    compiler_workdir = "./test_dir"
    neuron_model = compile_neuron_moe_model(
        config_no_sp, checkpoint, _load_module_moe, hidden_states,
        compiler_workdir, NUM_LAYERS_ACCURACY, initialize_weights=True
    )
    sp_neuron_model = compile_neuron_moe_model(
        config_sp, checkpoint, _load_module_moe, hidden_states,
        compiler_workdir, NUM_LAYERS_ACCURACY, initialize_weights=True
    )

    output_no_sp = neuron_model(hidden_states)
    output_sp = sp_neuron_model(hidden_states)

    print(output_no_sp, output_no_sp.shape)
    print(output_sp, output_sp.shape)

    if os.path.exists(compiler_workdir):
        shutil.rmtree(compiler_workdir)
        print(f"Cleaned up {compiler_workdir}")

    try:
        torch_neuronx.testing.assert_close(output_no_sp, output_sp, atol=1e-2, rtol=1e-2)
        return True, None
    except Exception as e:
        return False, str(e)


def run_neff_generation(config, seq_len, checkpoint, sp_enable, skip_scatter_gather=False):
    """Generate NEFF files for performance testing"""
    sp_type = "SP" if sp_enable else "No_SP"
    model_name_safe = config.model_type.replace(" ", "_")
    compiler_workdir = f"./TEST_COMPILER_DIR_{model_name_safe}_{sp_type}/"

    inference_config = get_moe_config(seq_len, config, sp_enable)
    hidden_states = torch.ones(1, seq_len, config.hidden_size, dtype=torch.bfloat16)

    if skip_scatter_gather: 
        hidden_states = hidden_states[:,:seq_len//64,:]

    _ = compile_neuron_moe_model(
        inference_config, checkpoint, _load_module_moe, hidden_states,
        compiler_workdir, NUM_LAYERS_PERFORMANCE, skip_scatter_gather
    )
    print(f"Generated NEFF for {config.model_type} {sp_type}")

def run_profiling_test(config, seq_len, sp_enable, threshold):
    """Run profiling test using pre-compiled NEFF"""
    sp_type = "SP" if sp_enable else "No_SP"
    model_name_safe = config.model_type.replace(" ", "_")
    compiler_workdir = f"./TEST_COMPILER_DIR_{model_name_safe}_{sp_type}"
    neff_path = f"{compiler_workdir}/main/_tp0_bk0/graph.neff"
    
    try:
        latency = get_profile_latency(neff_path)
        perf_passed = latency is not None and latency <= threshold
        print(f"{config.model_type} {sp_type} latency: {latency:.2f}ms (threshold: {threshold}ms) - {'PASSED' if perf_passed else 'FAILED'}")

        if os.path.exists(compiler_workdir):
            shutil.rmtree(compiler_workdir)
            print(f"Cleaned up {compiler_workdir}")
        
        return TestResult(
            name=f"{config.model_type} {sp_type} at seq_len={seq_len}",
            model_type=config.model_type, seq_len=seq_len, test_type="Performance",
            perf_passed=perf_passed, perf_details={"latency": latency, "threshold": threshold},
            error_message=None if perf_passed else f"Latency {latency}ms exceeds threshold {threshold}ms"
        )
    except Exception as e:
        return TestResult(
            name=f"{config.model_type} {sp_type} at seq_len={seq_len}",
            model_type=config.model_type, seq_len=seq_len, test_type="Performance",
            perf_passed=False, perf_details={}, error_message=str(e)
        )

def get_moe_config(seq_len, config, sp_enable=False):
    dtype = torch.bfloat16
    tp_degree = 64
    
    inference_config = {
        "model_type": config.model_type, "hidden_size": config.hidden_size,
        "hidden_act": config.hidden_act, "num_local_experts": config.num_local_experts,
        "num_experts_per_tok": config.num_experts_per_tok, "intermediate_size": config.intermediate_size,
        "n_shared_experts": config.n_shared_experts, "tp_degree": tp_degree, "dtype": dtype,
        "mac_thres": config.mac_thres
    }
    
    neuron_config = MoENeuronConfig(
        torch_dtype=dtype, 
        tp_degree=tp_degree, 
        seq_len=seq_len,
        router_config={"dtype": torch.float32, "act_fn":"softmax"},
        blockwise_matmul_config=config.blockwise_matmul_config,
        cc_pipeline_tiling_factor=1,
        early_expert_affinity_modulation=config.early_expert_affinity_modulation,
        disable_normalize_top_k_affinities=config.disable_normalize_top_k_affinities,
        sequence_parallel_enabled=sp_enable, 
        logical_nc_config=2,
    )

    return InferenceConfig(neuron_config=neuron_config, **inference_config)

def get_test_configs():
    return [
        {
            "model_type": "llama4 16E", 
            "early_expert_affinity_modulation": True,
            "disable_normalize_top_k_affinities": True, 
            "hidden_size": 5120, 
            "hidden_act":"silu",
            "intermediate_size": 8192, 
            "num_local_experts": 16, 
            "num_experts_per_tok":1,
            "n_shared_experts":0,
            "mac_thres": 10,
            "blockwise_matmul_config":{
                "block_size": 256, 
                "use_block_parallel": True,
                "skip_dma_token": True, 
                "skip_dma_weight": True,
            },
            "latency_threshold": 12.14, 
            "sp_latency_threshold": 10.13,
        },
        {
            "model_type": "llama4 128E", 
            "early_expert_affinity_modulation": True,
            "disable_normalize_top_k_affinities": True, 
            "hidden_size": 5120, 
            "hidden_act":"silu",
            "intermediate_size": 8192, 
            "num_local_experts": 128, 
            "num_experts_per_tok":1,
            "n_shared_experts":0, 
            "mac_thres": 1e15,
            "blockwise_matmul_config":{
                "block_size": 256, 
                "use_block_parallel": True,
                "skip_dma_token": True, 
                "skip_dma_weight": True,
            },
            "latency_threshold": 18.05,     
            "sp_latency_threshold": 16.03,
        },
        {
            "model_type": "DeepSeek", 
            "early_expert_affinity_modulation": False,
            "disable_normalize_top_k_affinities": False, 
            "hidden_size": 7168, 
            "hidden_act":"silu",
            "intermediate_size": 2048, 
            "num_local_experts": 256, 
            "num_experts_per_tok":8,
            "n_shared_experts":0, 
            "mac_thres": 10,
            "blockwise_matmul_config":{
                "block_size": 256, 
                "skip_dma_token": True, 
                "skip_dma_weight": True,
            },
            "latency_threshold": 107.20,
            "sp_latency_threshold": 96.43,
        },
    ]


def test_accuracy():
    configs = get_test_configs()
    failed_tests = []
    dtype = torch.bfloat16
    
    for config in configs:
        config = SimpleNamespace(**config)
        checkpoint = _generate_model_weights(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            n_routed_experts=config.num_local_experts, num_layers=NUM_LAYERS_ACCURACY
        )
        for seq_len in SEQ_LENS:
            if config.model_type == "DeepSeek" and seq_len == 128:
                continue
            
            print(f"\nRunning accuracy tests for {config.model_type} with sequence length {seq_len}\n")
            hidden_states = torch.rand(1, seq_len, config.hidden_size, dtype=dtype)
            
            config_no_sp = get_moe_config(seq_len, config, sp_enable=False)
            config_sp = get_moe_config(seq_len, config, sp_enable=True)

            accuracy_passed, error_message = run_model_execution_with_accuracy(
                config_no_sp, config_sp, checkpoint, hidden_states,
            )
            
            print(f"{config.model_type} with seq_len {seq_len} Accuracy test {'PASSED' if accuracy_passed else 'FAILED'}")
            if not accuracy_passed:
                failed_tests.append(f"{config.model_type} with sequence length {seq_len}: {error_message}")
    
    gc.collect()
    xm.mark_step()
    
    if failed_tests:
        pytest.fail(f"Accuracy tests failed: {'; '.join(failed_tests)}")

def test_neff_generation_for_perf():
    configs = get_test_configs()
    failed_generations = []
    seq_len = 8192
    
    for config in configs:
        config = SimpleNamespace(**config)
        checkpoint = _generate_model_weights(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            n_routed_experts=config.num_local_experts, num_layers=NUM_LAYERS_PERFORMANCE
        )

        print(f"Generating NEFF for {config.model_type}...\n")

        try:
            run_neff_generation(config, seq_len, checkpoint, sp_enable=False)
            run_neff_generation(config, seq_len, checkpoint, sp_enable=True, skip_scatter_gather=True)
        except Exception as e:
            failed_generations.append(f"{config.model_type} at seq_len={seq_len}: {str(e)}")
    
    if failed_generations:
        pytest.fail("NEFF generation failed: " + '\n'.join(failed_generations))

def test_performance():
    configs = get_test_configs()
    perf_results = []
    seq_len = 8192
    
    for config in configs:
        config = SimpleNamespace(**config)
        
        print(f"\nRunning profiling tests for {config.model_type}...")
        perf_result_no_sp = run_profiling_test(config, seq_len, False, config.latency_threshold)
        perf_results.append(perf_result_no_sp)
        print(f"{config.model_type} No_SP latency: {perf_result_no_sp.perf_details.get('latency', 'N/A')} ms")
        
        perf_result_sp = run_profiling_test(config, seq_len, True, config.sp_latency_threshold)
        perf_results.append(perf_result_sp)
        print(f"{config.model_type} SP latency: {perf_result_sp.perf_details.get('latency', 'N/A')} ms")
    
    failed_perf = [r for r in perf_results if not r.perf_passed]
    if failed_perf:
        error_msgs = [f"{r.name}: {r.error_message}" for r in failed_perf]
        pytest.fail(f"Performance tests failed: {'; '.join(error_msgs)}")


def run_test_suite(test_name):
    """Run a specific test suite in a separate process"""
    suite = unittest.TestSuite()
    if test_name == 'accuracy':
        test_accuracy()
    elif test_name == 'neff_generation':
        test_neff_generation_for_perf()
    elif test_name == 'performance':
        test_performance()
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return test_name, result.wasSuccessful()

if __name__ == "__main__":
    process_tests = ['accuracy', 'neff_generation', 'performance']

    results = {}
    for test_name in process_tests:
        print(f"\n=== Running {test_name.replace('_', ' ').title()} Test ===")
        queue = Queue()
        process = Process(target=lambda t=test_name: queue.put(run_test_suite(t)))
        process.start()
        process.join()
        
        name, passed = queue.get()
        results[test_name] = passed
        
        if not passed:
            print(f"\n=== {test_name.replace('_', ' ').title()} test FAILED, stopping execution ===")
            break
        
    print("\nFinal Results:")

    all_tests = process_tests
    for test_name in all_tests:
        status = 'PASSED' if results.get(test_name, False) else 'FAILED' if test_name in results else 'SKIPPED'
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.get(test, False) for test in all_tests if test in results)
    sys.exit(0 if all_passed else 1)
