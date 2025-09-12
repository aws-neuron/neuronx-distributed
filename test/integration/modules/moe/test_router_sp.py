import os
import torch
import unittest
import torch_neuronx
from functools import partial
from types import SimpleNamespace
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed_inference.models.config import MoENeuronConfig, InferenceConfig

torch.manual_seed(0)

# Class to Initialize the ExpertMLPs model for the test
class RouterMoEClass(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.tensor_parallel_group = get_tp_group(config)
        router = RouterTopK(
            num_experts=config.num_experts,
            top_k=config.top_k,
            hidden_size=config.hidden_size,
            dtype=config.neuron_config.router_config.dtype,
            act_fn=config.neuron_config.router_config.act_fn,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            sequence_dimension=1,
        ).eval()
        self.model = router
    
    def gather_from_sequence_parallel_region(self, tensor):
        return mappings.gather_from_sequence_parallel_region(
                    tensor,
                    sequence_dimension=0,
                    to_model_parallel=False,
                    process_group=self.tensor_parallel_group,
                )

    def forward(self, hidden_states):
        if self.config.neuron_config.sequence_parallel_enabled:
            hidden_states = mappings._reduce_scatter_along_dim(
                        hidden_states,
                        1,
                        xm.REDUCE_MAX,
                        process_group=get_tp_group(self.config),
                    )
        router_logits, expert_affinities, expert_index = self.model(hidden_states)
        if self.config.neuron_config.sequence_parallel_enabled:
            router_logits = self.gather_from_sequence_parallel_region(router_logits)
            expert_affinities = self.gather_from_sequence_parallel_region(expert_affinities)
            expert_index = self.gather_from_sequence_parallel_region(expert_index)

        return router_logits, expert_affinities, expert_index
        

def _load_module_router(config):
    return RouterMoEClass(config).eval()

def rand_interval(a, b, *size, dtype):
    return ((b - a) * torch.rand(*size) + a).to(dtype)

def _add_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    return compiler_args

# Generate random weights for the model
def _generate_random_weights(n_routed_experts, hidden_size):
    return {"model.linear_router.weight": torch.nn.Parameter(rand_interval(-0.05, 0.05,(n_routed_experts, hidden_size),dtype=torch.float32))}


# Test class for the ExpertMLPs model
class TestRouterSP(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.dtype = torch.bfloat16
        self.tp_degree = 32

    def get_config(self, sp_enable, seq_len, config):
        inference_config = {
            "hidden_size": config.hidden_size,
            "num_experts": config.num_local_experts,
            "top_k": config.num_experts_per_tok,
            "intermediate_size": config.intermediate_size,
            "dtype": self.dtype,}
        
        neuron_config = MoENeuronConfig(
            torch_dtype=self.dtype,
            tp_degree=self.tp_degree,
            seq_len=seq_len,
            router_config={"dtype": torch.float32, "act_fn":"softmax"},
            sequence_parallel_enabled= sp_enable,
        )

        inference_config = InferenceConfig(
            neuron_config=neuron_config,
            **inference_config,
        )
        return inference_config

    def compile_neuron_router_model(self, inference_config, checkpoint, load_module, hidden_states):
        builder = ModelBuilder(
            router=None,
            tp_degree=self.tp_degree,
            checkpoint_loader=lambda: checkpoint,
            compiler_workdir="./test_compiler_workdir/"
        )
        builder.add(
            key="main",
            model_instance=BaseModelInstance(
                module_cls=partial(load_module, inference_config),
                input_output_aliases={},
            ),
            example_inputs=[((hidden_states,)),],
            compiler_args=_add_compiler_args(),
        )
        neuron_model = builder.trace(initialize_model_weights=True)
        return neuron_model

    def test_moe_topk_router(self):
        configs=[
            {
                # llama4 configs with 16 experts
                "model_type": "llama4 16E",
                "early_expert_affinity_modulation": True,
                "disable_normalize_top_k_affinities": True,
                "hidden_size": 5120,
                "hidden_act":"silu",
                "intermediate_size": 8192,
                "num_local_experts": 16,
                "num_experts_per_tok":1,
            },
            {
                # llama4 configs with 128 experts
                "model_type": "llama4 128E",
                "early_expert_affinity_modulation": True,
                "disable_normalize_top_k_affinities": True,
                "hidden_size": 5120,
                "hidden_act":"silu",
                "intermediate_size": 8192,
                "num_local_experts": 128,
                "num_experts_per_tok":1,
            },
            {
                "model_type": "Qwen3",
                "early_expert_affinity_modulation": False,
                "disable_normalize_top_k_affinities": False,
                "hidden_size": 4096,
                "hidden_act":"silu",
                "intermediate_size": 1536,
                "num_local_experts": 128,
                "num_experts_per_tok":8,
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
            },
            ]

        seq_lens = [32, 128, 8192]
        test_results = []
        
        for seq_len in seq_lens:
            for config in configs:
                test_name = f"seq_len = {seq_len} with configs {config['model_type']}"
                print(f"\nRunning {test_name}\n")
                
                try:
                    config = SimpleNamespace(**config)
                    hidden_states = torch.rand(1, seq_len, config.hidden_size, dtype=self.dtype)
                    
                    # For all other configs or if model path is not provided, use random weights
                    checkpoint = _generate_random_weights(
                        hidden_size=config.hidden_size,
                        n_routed_experts=config.num_local_experts
                    )
                    print("\nRunning Neuron Model without SP.....")
                    neuron_model = self.compile_neuron_router_model(
                        self.get_config(False, seq_len, config), 
                        checkpoint, 
                        _load_module_router,
                        hidden_states
                    )

                    r_logits, e_affinities, e_index = neuron_model(hidden_states)

                    print("\nRunning Neuron Model in SP.....")
                    neuron_model_sp = self.compile_neuron_router_model(
                        self.get_config(True, seq_len, config), 
                        checkpoint, 
                        _load_module_router, 
                        hidden_states
                    )

                    r_logits_sp, e_affinities_sp, e_index_sp = neuron_model_sp(hidden_states)

                    print(f"\n router logits: {r_logits, r_logits.shape} \n \n router logits with SP: {r_logits_sp, r_logits_sp.shape} \n")
                    torch_neuronx.testing.assert_close(r_logits, r_logits_sp, atol=1e-5, rtol=1e-5)

                    print(f"\n expert affinities: {e_affinities, e_affinities.shape} \n \n expert affinities with SP: {e_affinities_sp, e_affinities_sp.shape} \n")
                    torch_neuronx.testing.assert_close(e_affinities, e_affinities_sp, atol=1e-4, rtol=1e-4)

                    print(f"\n expert index: {e_index, e_index.shape} \n \n expert index with SP: {e_index_sp, e_index_sp.shape} \n")
                    for i in range(len(e_index)):
                        if not torch.equal(e_index[i], e_index_sp[i]):
                            raise AssertionError(f"Expert indices don't match at position [{i}]: {e_index[i]} vs {e_index_sp[i]}")
                    
                    test_results.append((test_name, True, None))
                    print(f"PASSED: {test_name}")
                    
                except Exception as e:
                    test_results.append((test_name, False, str(e)))
                    print(f"FAILED: {test_name} - {str(e)}")
        
        # Summary and final failure check
        passed_tests = [r for r in test_results if r[1]]
        failed_tests = [r for r in test_results if not r[1]]
        
        print("\nTEST SUMMARY:")
        print(f"Passed: {len(passed_tests)}/{len(test_results)}")
        print(f"Failed: {len(failed_tests)}/{len(test_results)}")
        
        if failed_tests:
            print("\nFailed tests:")
            for test_name, _, error in failed_tests:
                print(f"  - {test_name}: {error}")
            self.fail(f"{len(failed_tests)} out of {len(test_results)} tests failed")

if __name__ == "__main__":
    unittest.main()