import copy
from functools import partial
import os
import torch
import tempfile
import unittest
import torch_neuronx
import sys

from neuronx_distributed import parallel_layers
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.utils.model_utils import get_platform_lnc, LogicalNCConfig

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut
from test_quantized_experts import ExpertMoEClass, quantize_fp8_per_channel
from unit_test.utils.test_helper import init_cpu_env, destroy_cpu_env
from integration.common.integration_test_utils import download_from_s3

os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'
os.environ['NEURON_RT_VIRTUAL_CORE_SIZE']='2'
os.environ['NEURON_RT_NUM_CORES']='64'

os.environ['XLA_HLO_DEBUG'] = '1'
os.environ['XLA_IR_DEBUG'] = '1'

os.environ["UNSAFE_FP8FNCAST"]="1"
os.environ["XLA_HANDLE_SPECIAL_SCALAR"]="1"

torch.manual_seed(0)

def set_tp_degree():
    return 64 if get_platform_lnc() == 2 else 32

def init_parallel_cpu_golden():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["NXD_CPU_MODE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
    parallel_layers.parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    parallel_layers.parallel_state.initialize_token_shuffle_group(1)

class ExpertMoEClassNoReduce(ExpertMoEClass):
    def forward(self, hidden_states, expert_affinities, expert_index, seq_len):
        return self.module(hidden_states, expert_affinities, expert_index, seq_len)

def _load_module_moe(cfg: ut.ExptCfg):
    return ut.initialize_neuron_model(cfg, move_to_device=False).eval()

def compile_cpu_moe_model(checkpoint, load_module, test_config):
    cpu_model = load_module(test_config)
    cpu_model.load_state_dict(checkpoint)
    return cpu_model

def compile_neuron_moe_model(
        sample_inputs,
        checkpoint,
        load_module,
        test_config,
        tp_degree=set_tp_degree(),
):
    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config=get_platform_lnc(),
        debug=False,
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, test_config),
            input_output_aliases={},
        ),
        example_inputs=[(sample_inputs,)],
        compiler_args=_add_compiler_args(),
    )

    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model

def _add_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    compiler_args += " --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3'"
    return compiler_args

def _load_llama4_fp8_ckpt(ckpt_path: str):
    ckpt = {}
    layer_weights = torch.load(ckpt_path)
    for weight_name, weight in layer_weights.items():
        new_weight_name = weight_name.replace("language_model.model.layers.1.feed_forward.router.", "moe.router.linear_router.")
        new_weight_name = new_weight_name.replace("language_model.model.layers.1.feed_forward.experts.", "moe.expert_mlps.mlp_op.")
        new_weight_name = new_weight_name.replace("language_model.model.layers.1.feed_forward.shared_expert.", "moe.shared_experts.")
        new_weight_name = new_weight_name.replace("language_model.model.layers.1.", "")
        ckpt[new_weight_name] = weight
    return ckpt

def _dequantize_fp8_ckpt(ckpt):
    dequant_ckpt = copy.copy(ckpt)
    for weight_name in list(dequant_ckpt.keys()):
        if ".expert_mlps." in weight_name and "scale" not in weight_name:
            weight = dequant_ckpt[weight_name]
            scale_name = weight_name.replace("weight", "scale")
            scale = dequant_ckpt[scale_name]
            weight = (weight.to(torch.float32) * scale).to(torch.bfloat16)
            dequant_ckpt[weight_name] = weight
            del dequant_ckpt[scale_name]
    return dequant_ckpt

def _generate_random_fp8_weights(n_routed_experts, intermediate_size, hidden_size):
    down_weight, down_scale = quantize_fp8_per_channel(n_routed_experts, intermediate_size, hidden_size)
    fuse_weight, fuse_scale = quantize_fp8_per_channel(n_routed_experts, hidden_size, intermediate_size * 2)

    checkpoint = {
        "post_attention_layernorm.weight": torch.rand(hidden_size, dtype=torch.float32) * 2 - 1,
        "moe.router.linear_router.weight": torch.rand(n_routed_experts, hidden_size, dtype=torch.float32) * 2 - 1,
        "moe.expert_mlps.mlp_op.down_proj.weight" : down_weight,
        "moe.expert_mlps.mlp_op.gate_up_proj.weight": fuse_weight,
        "moe.expert_mlps.mlp_op.down_proj.scale": down_scale,
        "moe.expert_mlps.mlp_op.gate_up_proj.scale": fuse_scale,
        "moe.shared_experts.down_proj.weight": torch.rand(hidden_size, intermediate_size, dtype=torch.bfloat16) * 2 - 1,
        "moe.shared_experts.gate_proj.weight": torch.rand(intermediate_size, hidden_size, dtype=torch.bfloat16) * 2 - 1,
        "moe.shared_experts.up_proj.weight": torch.rand(intermediate_size, hidden_size, dtype=torch.bfloat16) * 2 - 1,
    }

    return checkpoint

def _generate_test_configs():
    llama4_m_configs = dict(
        hidden_size=5120,
        intermediate_size=8192,
        num_experts=128,
        top_k=1,
        num_shared_experts=1,
        early_expert_affinity_modulation=True,
        capacity_factor=None,
        glu_mlp=True,
        test_mode="inference",
        implementation="llama4",
    )

    test_configs = [
        ut.ExptCfg(
            batch_size=1,
            seq_len=1,
            dtype=torch.bfloat16,
            **llama4_m_configs,
            moe_fused_tkg_enabled=True,
        ),
        ut.ExptCfg(
            batch_size=4,
            seq_len=1,
            dtype=torch.bfloat16,
            **llama4_m_configs,
            moe_fused_tkg_enabled=True,
        ),
    ]

    return test_configs

# Test class for the ExpertMLPs model
class TestRoutedExpertsFP8(unittest.TestCase):
    CKPT_PATH: str = ""

    def load_ckpt(self, cfg):
        if self.CKPT_PATH:
            ckpt = _load_llama4_fp8_ckpt(self.CKPT_PATH)
        else:
            ckpt = _generate_random_fp8_weights(
                cfg.num_experts,
                cfg.intermediate_size,
                cfg.hidden_size,
            )
        dequant_ckpt = _dequantize_fp8_ckpt(ckpt)
        return ckpt, dequant_ckpt

    def check_tensors(self, expected_output, actual_output):
        expected_output = expected_output.detach().cpu()
        actual_output = actual_output.detach().cpu()

        print("\nComparing outputs.....")
        print(f"\nCPU Output: {expected_output} \n \n Neuron Output: {actual_output}\n")
        success = False
        for rtol in [1e-5, 1e-4, 1e-3, 1e-2, 2e-2]:
            try:
                torch_neuronx.testing.assert_close(expected_output, actual_output, rtol=rtol)
                print(f"Output rtol < {rtol}, tensors are close")
                success = True
                break
            except AssertionError:
                print(f"Output rtol > {rtol}")
        if success is False:
            self.fail("Tensors not close")

    def test_moe_quantized_flat_compiler_against_cpu(self):
        """
        Test device correctness parallel
        """
        if get_platform_lnc() != LogicalNCConfig.LNC_2:
            pass

        for test_config in _generate_test_configs():
            print(f"Running test config:  {test_config}")
            seq_len = test_config.seq_len
            hidden_size = test_config.hidden_size
            dtype = test_config.dtype
            batch_size = test_config.batch_size
            sample_inputs = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype)
            # Load weights (use real Llama4 weights if s3_ckpt_path provided, otherwise generate random weights)
            ckpt, dequant_ckpt = self.load_ckpt(test_config)
            # Initialize and execute CPU model
            init_cpu_env("nxd")
            parallel_layers.parallel_state.initialize_token_shuffle_group(1)
            test_config.quantized = False
            cpu_model = compile_cpu_moe_model(dequant_ckpt, _load_module_moe, test_config)
            expected_output = cpu_model(sample_inputs)[0]
            destroy_cpu_env()
            # Initialize and execute Neuron model
            neuron_config = copy.copy(test_config)
            neuron_config.moe_fused_tkg_kernel_enabled = False
            neuron_config.router_topk_kernel_enabled = False
            neuron_config.expert_mlp_kernel_enabled = False
            neuron_config.shared_mlp_kernel_enabled = False
            neuron_config.quantized = True
            neuron_model = compile_neuron_moe_model(
                sample_inputs,
                ckpt,
                _load_module_moe,
                neuron_config,
            )
            kernel_output = neuron_model(sample_inputs)[0]
            # Compare outputs
            self.check_tensors(expected_output, kernel_output)
            del cpu_model, neuron_model, ckpt, dequant_ckpt

    def test_moe_quantized_kernel_against_cpu(self):
        """
        Test device correctness parallel
        """
        if get_platform_lnc() != LogicalNCConfig.LNC_2:
            pass

        for test_config in _generate_test_configs():
            print(f"Running test config:  {test_config}")
            seq_len = test_config.seq_len
            hidden_size = test_config.hidden_size
            dtype = test_config.dtype
            batch_size = test_config.batch_size
            sample_inputs = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype)
            # Load weights (use real Llama4 weights if s3_ckpt_path provided, otherwise generate random weights)
            ckpt, dequant_ckpt = self.load_ckpt(test_config)
            # Initialize and execute CPU model
            init_cpu_env("nxd")
            parallel_layers.parallel_state.initialize_token_shuffle_group(1)
            test_config.quantized = False
            cpu_model = compile_cpu_moe_model(dequant_ckpt, _load_module_moe, test_config)
            expected_output = cpu_model(sample_inputs)[0].detach()
            destroy_cpu_env()
            # Initialize and execute Neuron model
            neuron_config = copy.copy(test_config)
            neuron_config.moe_fused_tkg_kernel_enabled = True
            neuron_config.quantized = True
            neuron_model = compile_neuron_moe_model(
                sample_inputs,
                ckpt,
                _load_module_moe,
                neuron_config,
            )
            kernel_output = neuron_model(sample_inputs)[0].detach().cpu()
            # Compare outputs
            self.check_tensors(expected_output, kernel_output)
            del cpu_model, neuron_model, ckpt, dequant_ckpt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        S3_CKPT_PATH = sys.argv.pop()
        temp_dir = tempfile.TemporaryDirectory()
        temp_ckpt_path = os.path.join(temp_dir.name, "llama4_1L_MoE_FP8.pt")
        print(f"Downloading Llama4 ckpt from {S3_CKPT_PATH} to temp dir: {temp_ckpt_path}")
        download_from_s3(S3_CKPT_PATH, temp_ckpt_path)
        TestRoutedExpertsFP8.CKPT_PATH = temp_ckpt_path
    unittest.main()
