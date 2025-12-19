import os

import math
import pytest

import numpy as np
import torch
import torch_neuronx
from torch_xla.core import xla_model
import ml_dtypes

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki._private.private_api as nki_private
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC

from neuronx_distributed.experimental.quantization.microscaling.expert_mlps_mx import (
    topk,
    expert_affinity_mask,
    all_expert_mlps_act_mxfp8_w_mxfp4,
)
from neuronx_distributed.quantization.microscaling.transform_weights import _pad_tensor


RUN_TRN3_TESTS = "1" in os.environ.get("RUN_TRN3_TESTS", "")

WEIGHTS_DIR_ENV_VAR = "_GPT_OSS_MOE_WEIGHTS_DIR"
GOLDENS_DIR_ENV_VAR = "_GPT_OSS_MOE_GOLDENS_DIR"

WEIGHT_SUBDIR_MAP = {
    torch.bfloat16: "bf16",
    torch.uint16: "mxfp4",
}

GPT_OSS_HEAD_DIM = 2880
GPT_OSS_HEAD_DIM_PADDED = 3072

TORCH_NKI_DTYPE_MAP = {
    torch.bfloat16: nl.bfloat16,
    torch.float32: nl.float32,
}

GPT_OSS_INTERMEDIATE_DIM = 2880


class TestAllExpertMLPsKernelRealWeights:
    

    def setup_method(self):
        # Set seed
        torch.manual_seed(0)

        # Assume that the env vars are set and that weights/goldens are downloaded
        self.weights_dir = os.environ.get(WEIGHTS_DIR_ENV_VAR, None)
        self.goldens_dir = os.environ.get(GOLDENS_DIR_ENV_VAR, None)

        if self.weights_dir is None:
            raise ValueError("Weights directory not found!")

        if self.goldens_dir is None:
            raise ValueError("Goldens directory not found!")

        # Capture and set env vars
        self.platform_target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
        self.neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS")
        os.environ["NEURON_CC_FLAGS"] = "--target=trn3"
        self.enable_mla_arch = os.environ.get("ENABLE_MLA_ARCH")
        os.environ["ENABLE_MLA_ARCH"] = "core_v4"

        self.xla_ir_debug = os.environ.get("XLA_IR_DEBUG")
        os.environ["XLA_IR_DEBUG"] = "1"
        self.xla_hlo_debug = os.environ.get("XLA_HLO_DEBUG")
        os.environ["XLA_HLO_DEBUG"] = "1"

        self.ocp_enabled = os.environ.get("NEURON_RT_ENABLE_OCP")
        os.environ["NEURON_RT_ENABLE_OCP"] = "1"
        self.ocp_saturation_enabled = os.environ.get("NEURON_RT_ENABLE_OCP_SATURATION")
        os.environ["NEURON_RT_ENABLE_OCP_SATURATION"] = "1"

        # Uncomment for verbose PJRT/NRT logs during debug
        # self.tf_log_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        # self.tf_vmodule = os.environ.get("TF_CPP_VMODULE")
        # os.environ["TF_CPP_VMODULE"] = "neuronpjrt=1"
        # self.nrt_log_level = os.environ.get("NEURON_RT_LOG_LEVEL")
        # os.environ["NEURON_RT_LOG_LEVEL"] = "DEBUG"


    def teardown_method(self):
        # Reset all env vars
        if self.platform_target is None:
            os.environ.pop("NEURON_PLATFORM_TARGET_OVERRIDE", None)
        else:
            os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = self.platform_target
        if self.neuron_cc_flags is None:
            os.environ.pop("NEURON_CC_FLAGS", None)
        else:
            os.environ["NEURON_CC_FLAGS"] = self.neuron_cc_flags
        if self.enable_mla_arch is None:
            os.environ.pop("ENABLE_MLA_ARCH", None)
        else:
            os.environ["ENABLE_MLA_ARCH"] = self.enable_mla_arch
        if self.xla_ir_debug is None:
            os.environ.pop("XLA_IR_DEBUG", None)
        else:
            os.environ["XLA_IR_DEBUG"] = self.xla_ir_debug
        if self.xla_hlo_debug is None:
            os.environ.pop("XLA_HLO_DEBUG", None)
        else:
            os.environ["XLA_HLO_DEBUG"] = self.xla_hlo_debug
        if self.ocp_enabled is None:
            os.environ.pop("NEURON_RT_ENABLE_OCP", None)
        else:
            os.environ["NEURON_RT_ENABLE_OCP"] = self.ocp_enabled
        if self.ocp_saturation_enabled is None:
            os.environ.pop("NEURON_RT_ENABLE_OCP_SATURATION", None)
        else:
            os.environ["NEURON_RT_ENABLE_OCP_SATURATION"] = self.ocp_saturation_enabled
        
        # Always unset LNC config + VNC settings
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)

        # Uncomment for verbose PJRT/NRT logs during debug
        # if self.tf_log_level is None:
        #     os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)
        # else:
        #     os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.tf_log_level
        # if self.tf_vmodule is None:
        #     os.environ.pop("TF_CPP_VMODULE", None)
        # else:
        #     os.environ["TF_CPP_VMODULE"] = self.tf_vmodule
        # if self.nrt_log_level is None:
        #     os.environ.pop("NEURON_RT_LOG_LEVEL", None)
        # else:
        #     os.environ["NEURON_RT_LOG_LEVEL"] = self.nrt_log_level


    def load_weights(self, layer, dtype):
        # descaled weights
        if dtype == torch.bfloat16:
            W_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_gate_proj_weight.pt"))
            W_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_up_proj_weight.pt"))
            W_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_down_proj_weight.pt"))
            return W_gate, W_up, W_down
        
        # MXFP4 weights
        elif dtype == torch.uint16:
            W_gate_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_gate_proj_weight.pt"))
            W_up_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_up_proj_weight.pt"))
            W_down_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_down_proj_weight.pt"))
            scale_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_gate_proj_scales.pt"))
            scale_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_up_proj_scales.pt"))
            scale_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_down_proj_scales.pt"))
            return W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down
        
        else:
            raise ValueError(f"Invalid weight dtype: {torch.uint16}")


    def load_biases(self, layer, dtype):
        bias_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_gate_proj_bias.pt"))
        bias_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_up_proj_bias.pt"))
        bias_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_down_proj_bias.pt"))
        return bias_gate, bias_up, bias_down


    def load_goldens(self, layer, token, T):
        hidden_states = torch.load(os.path.join(self.goldens_dir, f"hidden_states_input_pre_rmsnorm_layer_{layer}_token_{token}.pt"))
        rmsnorm_out = torch.load(os.path.join(self.goldens_dir, f"rmsnorm_out_layer_{layer}_token_{token}.pt"))
        router_logits = torch.load(os.path.join(self.goldens_dir, f"router_logits_layer_{layer}_token_{token}.pt"))
        moe_out = torch.load(os.path.join(self.goldens_dir, f"moe_out_layer_0_token_{token}.pt"))

        return hidden_states[:T, :], rmsnorm_out[:T, :], router_logits[:T, :], moe_out[:T, :]


    def prep_torch_inputs_topk(
        self, router_logits, E_L, W_gate_uint16, W_up_uint16, W_down_uint16, 
        scale_gate, scale_up, scale_down, bias_gate, bias_up, bias_down,
    ):
        # Prep torch inputs - select subset of experts as input to torch. We do this for speed, because it takes longer to compute all E during a test.
        _, expert_indices = topk(router_logits)
        topk_expert_indices = expert_indices[0, 0:E_L]

        W_gate_uint16_topk = W_gate_uint16.view(torch.int16)[topk_expert_indices, ...].view(torch.uint16)
        W_up_uint16_topk = W_up_uint16.view(torch.int16)[topk_expert_indices, ...].view(torch.uint16)
        W_down_uint16_topk = W_down_uint16.view(torch.int16)[topk_expert_indices, ...].view(torch.uint16)
        scale_gate_topk = scale_gate[topk_expert_indices, ...]
        scale_up_topk = scale_up[topk_expert_indices, ...]
        scale_down_topk = scale_down[topk_expert_indices, ...]
        bias_gate_topk = bias_gate[topk_expert_indices, ...]
        bias_up_topk = bias_up[topk_expert_indices, ...]
        bias_down_topk = bias_down[topk_expert_indices, ...]

        return (
            topk_expert_indices, W_gate_uint16_topk, W_up_uint16_topk, W_down_uint16_topk, 
            scale_gate_topk, scale_up_topk, scale_down_topk, bias_gate_topk, bias_up_topk, bias_down_topk
        )
    

    def prep_nki_inputs_shuffled(
        self, T, E, H, I, tp_degree, rmsnorm_out_padded, W_gate_uint16, W_up_uint16, W_down_uint16,  # noqa: E741
        scale_gate, scale_up, scale_down, bias_gate, bias_up, bias_down, expert_affinities_masked,
        hidden_act_bias=0.0, squeeze_tp_dim=False,
    ):
        """
        Input checkpoint has already been bitcast to fp4_x4/uint16 + padded to H=I=3072. W, bias are padded with 0s, scale is padded with 127s.
        """

        # Tiling + constants
        pmax = nl.tile_size.pmax
        q_height, q_width = 8, 4

        # H dim: assuming we always have H divisible by 512
        H128 = 128
        n_H512_tile = H // (pmax * q_width)
        q_blocks_per_H_tile = 512 // (q_width * q_height)

        # I dim: supports tiles smaller than 512
        I_TP = I // tp_degree
        _, r_I512_tile = divmod(I_TP, pmax * q_width)
        assert r_I512_tile % (q_height * q_width) == 0, "Tile size must be evenly divisible by MX block size"
        n_I512_tile = math.ceil(I_TP / 512)  # always have at least 1 I/512 tile for reshapes
        q_blocks_per_I_tile = 16 if r_I512_tile == 0 else r_I512_tile // (q_height * q_width)  # 16 if I is divisible by 512, else we have remainder/32 blocks per I tile
        down_proj_I_tile = q_blocks_per_I_tile * q_height

        # T dim:
        # TODO: make these assertions less strict as we add support for more-varied T
        assert T % 32 == 0
        T32_tile = 32
        n_T32_tile = T // T32_tile

        # RMSNorm
        # [T, H] -> 
        # [T/32, 32_T, H/512, 16_H, 8_H, 4_H]
        rmsnorm_out_padded_nki = rmsnorm_out_padded\
            .reshape(n_T32_tile, T32_tile, n_H512_tile, q_blocks_per_H_tile, q_height, q_width)
        # [T/32, 32_T, H/512, 16_H, 8_H, 4_H] ->
        # [32_T, 4_H, T/32,  H/512, 16_H, 8_H] ->
        # [32_T * 4_H, T/32, H/512, 16_H * 8_H]
        _n_T32_tile, _T32_tile, _n_H512_tile, _q_blocks_per_H_tile, _q_height, _q_width = list(range(rmsnorm_out_padded_nki.ndim))
        rmsnorm_out_padded_nki = rmsnorm_out_padded_nki\
            .permute(_T32_tile, _q_width, _n_T32_tile, _n_H512_tile, _q_blocks_per_H_tile, _q_height)\
            .reshape(T32_tile * q_width, n_T32_tile, n_H512_tile, q_blocks_per_H_tile * q_height)

        # Gate/up proj weights (note that 4_H dim is always last because W is already fp4_x4)
        # [E, I, H/4] -> 
        # [E, H/4, I] -> 
        # [E, H/512, q_blocks_per_H_tile (16_H), q_height (8_H), tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)]
        W_gate_nki = W_gate_uint16.transpose(1, 2).reshape(E, n_H512_tile, q_blocks_per_H_tile, q_height, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        W_up_nki = W_up_uint16.transpose(1, 2).reshape(E, n_H512_tile, q_blocks_per_H_tile, q_height, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        # [E, H/512, q_blocks_per_H_tile (16_H), q_height (8_H), tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)] -> 
        # [E, q_blocks_per_H_tile (16_H), q_height (8_H), H/512, tp_degree (I), ⌈I/512⌉, q_width (4_I), q_blocks_per_I_tile (12_I or 16_I), q_height (8_I)] -> 
        # [E, 128_H, H/512, tp_degree (I), I_TP]
        _E, _n_H512_tile, _q_blocks_per_H_tile, _q_height_H, _tp_degree, _n_I512_tile, _q_blocks_per_I_tile, _q_height_I, _q_width_I = list(range(W_gate_nki.ndim))
        W_gate_nki = W_gate_nki\
            .permute(_E, _q_blocks_per_H_tile, _q_height_H, _n_H512_tile, _tp_degree, _n_I512_tile, _q_width_I, _q_blocks_per_I_tile, _q_height_I)\
            .reshape(E, H128, n_H512_tile, tp_degree, I_TP)
        W_up_nki = W_up_nki\
            .permute(_E, _q_blocks_per_H_tile, _q_height_H, _n_H512_tile, _tp_degree, _n_I512_tile, _q_width_I, _q_blocks_per_I_tile, _q_height_I)\
            .reshape(E, H128, n_H512_tile, tp_degree, I_TP)
        
        # [E, 128_H, 2, H/512, tp_degree (I), I_TP]
        W_gate_up_nki = torch.stack([W_gate_nki, W_up_nki], dim=2).contiguous()

        assert torch.equal(W_gate_up_nki[:, :, 0, :, :, :], W_gate_nki[...])
        assert torch.equal(W_gate_up_nki[:, :, 1, :, :, :], W_up_nki[...])

        # Gate/up proj scales
        # [E, I, H/32] ->
        # [E, H/32, I] -> 
        # [E, H/512, q_blocks_per_H_tile (16_H), tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)]
        scale_gate_nki = scale_gate.transpose(1, 2).reshape(E, n_H512_tile, q_blocks_per_H_tile, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        scale_up_nki = scale_up.transpose(1, 2).reshape(E, n_H512_tile, q_blocks_per_H_tile, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        # [E, H/512, q_blocks_per_H_tile (16_H), tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)] ->
        # [E, q_blocks_per_H_tile (16_H), H/512, tp_degree (I), ⌈I/512⌉, q_width (4_I), q_blocks_per_I_tile (12_I or 16_I), q_height (8_I)] ->
        # [E, 16_H, H/512, tp_degree (I), I_TP]
        _E, _n_H512_tile, _q_blocks_per_H_tile, _tp_degree, _n_I512_tile, _q_blocks_per_I_tile, _q_height_I, _q_width_I = list(range(scale_gate_nki.ndim))
        scale_gate_nki = scale_gate_nki\
            .permute(_E, _q_blocks_per_H_tile, _n_H512_tile, _tp_degree, _n_I512_tile, _q_width_I, _q_blocks_per_I_tile, _q_height_I)\
            .reshape(E, q_blocks_per_H_tile, n_H512_tile, tp_degree, I_TP)
        scale_up_nki = scale_up_nki\
            .permute(_E, _q_blocks_per_H_tile, _n_H512_tile, _tp_degree, _n_I512_tile, _q_width_I, _q_blocks_per_I_tile, _q_height_I)\
            .reshape(E, q_blocks_per_H_tile, n_H512_tile, tp_degree, I_TP)

        # [E, 16_H, 2, H/512, tp_degree (I), I_TP]
        scale_gate_up_nki = torch.stack([scale_gate_nki, scale_up_nki], dim=2).contiguous()

        assert torch.equal(scale_gate_up_nki[:, :, 0, :, :, :], scale_gate_nki[...])
        assert torch.equal(scale_gate_up_nki[:, :, 1, :, :, :], scale_up_nki[...])

        # Gate/up proj bias
        # IMPORTANT: Add 1 to up proj bias so that we don't have to do it online
        # NOTE: we upcast to fp32 to do the bias add, before downcasting back to bf16. This improves atol by 2.5x and rtol by 6x when looking at the expert MLPs output from one layer.
        # We incur a small logit error from += hidden_act_bias but it is much smaller in fp32 than bf16
        bias_up_nki = bias_up.clone().to(torch.float32)
        bias_up_nki[:, :GPT_OSS_INTERMEDIATE_DIM] += hidden_act_bias
        bias_up_nki = bias_up_nki.to(torch.bfloat16)
        assert bias_up_nki.dtype == torch.bfloat16
        assert torch.all(bias_up_nki[:, GPT_OSS_INTERMEDIATE_DIM:] == 0)

        # [E, I] -> 
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)]
        bias_gate_nki = bias_gate.reshape(E, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        bias_up_nki = bias_up_nki.reshape(E, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, q_width)
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), q_width (4_I)] ->
        # [E, tp_degree (I), q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), ⌈I/512⌉, q_width (4_I)] ->
        # [E, tp_degree (I), down_proj_I_tile (96_I or 128_I), ⌈I/512⌉, q_width (4_I)]
        _E, _tp_degree, _n_I512_tile, _q_blocks_per_I_tile, _q_height_I, _q_width_I = list(range(bias_gate_nki.ndim))
        bias_gate_nki = bias_gate_nki\
            .permute(_E, _tp_degree, _q_blocks_per_I_tile, _q_height_I, _n_I512_tile, _q_width_I)\
            .reshape(E, tp_degree, down_proj_I_tile, n_I512_tile, q_width)
        bias_up_nki = bias_up_nki\
            .permute(_E, _tp_degree, _q_blocks_per_I_tile, _q_height_I, _n_I512_tile, _q_width_I)\
            .reshape(E, tp_degree, down_proj_I_tile, n_I512_tile, q_width)

        # [E, tp_degree (I), down_proj_I_tile (96_I or 128_I), 2, ⌈I/512⌉, q_width (4_I)]
        bias_gate_up_nki = torch.stack([bias_gate_nki, bias_up_nki], dim=3).contiguous()

        assert torch.equal(bias_gate_up_nki[:, :, :, 0, :, :], bias_gate_nki[...])
        assert torch.equal(bias_gate_up_nki[:, :, :, 1, :, :], bias_up_nki[...])

        # Down proj weight (note that 4_I dim is always last because W is already fp4_x4)
        # [E, H, I/4] -> 
        # [E, I/4, H] -> 
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), H]
        W_down_nki = W_down_uint16.transpose(1, 2).reshape(E, tp_degree, n_I512_tile, q_blocks_per_I_tile, q_height, H)
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), H] -> 
        # [E, tp_degree (I), q_blocks_per_I_tile (12_I or 16_I), q_height (8_I), ⌈I/512⌉, H] ->
        # [E, tp_degree (I), down_proj_I_tile (96_I or 128_I), ⌈I/512⌉, H]
        _E, _tp_degree, _n_I512_tile, _q_blocks_per_I_tile, _q_height_I, _H = list(range(W_down_nki.ndim))
        W_down_nki = W_down_nki\
            .permute(_E, _tp_degree, _q_blocks_per_I_tile, _q_height_I, _n_I512_tile, _H)\
            .reshape(E, tp_degree, down_proj_I_tile, n_I512_tile, H)

        # Down proj scales
        # [E, H, I/32] ->
        # [E, I/32, H] ->
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), H]
        scale_down_nki = scale_down.transpose(1, 2).reshape(E, tp_degree, n_I512_tile, q_blocks_per_I_tile, H)
        # [E, tp_degree (I), ⌈I/512⌉, q_blocks_per_I_tile (12_I or 16_I), H] ->
        # [E, tp_degree (I), q_blocks_per_I_tile (12_I or 16_I), ⌈I/512⌉, H]
        _E, _tp_degree, _n_I512_tile, _q_blocks_per_I_tile, _H = list(range(scale_down_nki.ndim))
        scale_down_nki = scale_down_nki.permute(_E, _tp_degree, _q_blocks_per_I_tile, _n_I512_tile, _H)

        # Down proj bias: divide by TP degree, no shape changes
        # [E, H]
        bias_down_nki = torch.div(bias_down, float(tp_degree))

        # Expert affinities: tile on T dim
        # [T, E] ->
        # [⌈T/128], min(T, 128), E] ->
        # [min(T, 128), ⌈T/128], E]
        expert_affinities_masked = expert_affinities_masked\
            .reshape(math.ceil(T / 128), min(T, 128), E)\
            .permute(1, 0, 2)

        # Get rid of extra tp_degree dim when squeeze_tp_dim && tp==1
        if squeeze_tp_dim and tp_degree == 1:
            W_gate_up_nki = W_gate_up_nki.squeeze(4)
            scale_gate_up_nki = scale_gate_up_nki.squeeze(4)
            bias_gate_up_nki = bias_gate_up_nki.squeeze(1)
            W_down_nki = W_down_nki.squeeze(1)
            scale_down_nki = scale_down_nki.squeeze(1)

        return rmsnorm_out_padded_nki, W_gate_up_nki, scale_gate_up_nki, bias_gate_up_nki, W_down_nki, scale_down_nki, bias_down_nki, expert_affinities_masked


    # TODO: add layers 1, 2, 3
    @pytest.mark.parametrize("layer", [0])
    @pytest.mark.parametrize("token", [0])
    @pytest.mark.parametrize(
        "T,accumulation_dtype,atol,rtol", [
            # NOTE: current goldens use the same input broadcast on T dim, so testing accuracy with T=128 gives same error as T=512.
            # NOTE: some experts have better tolerances than others, we are using the worst atol/rtol from across all 4 experts for these tests
            # fp32 PSUM
            pytest.param(64, torch.float32, 1.98974609375e-2, 1.361083984375e-2, id="T64_fp32_psum"),
            pytest.param(128, torch.float32, 1.98974609375e-2, 1.361083984375e-2, id="T128_fp32_psum"),
            pytest.param(256, torch.float32, 1.98974609375e-2, 1.361083984375e-2, id="T256_fp32_psum"),
            pytest.param(512, torch.float32, 1.98974609375e-2, 1.361083984375e-2, id="T512_fp32_psum"),
            
            # bf16 PSUM
            pytest.param(64, torch.bfloat16, 1.3671875e-2, 1.7578125e-2, id="T64_bf16_psum"),
            pytest.param(128, torch.bfloat16, 1.3671875e-2, 1.7578125e-2, id="T128_bf16_psum"),
            pytest.param(256, torch.bfloat16, 1.3671875e-2, 1.7578125e-2, id="T256_bf16_psum"),
            pytest.param(512, torch.bfloat16, 1.3671875e-2, 1.7578125e-25, id="T512_bf16_psum"),
        ]
    )
    @pytest.mark.parametrize("E_L_idx", [0, 1, 2, 3])
    @pytest.mark.parametrize("lnc,limit,hidden_act_bias,alpha,output_dtype", [(2, 7.0, 1.0, 1.702, torch.bfloat16)])
    @pytest.mark.trn3
    @pytest.mark.skipif(not RUN_TRN3_TESTS, reason="Skip test when not run on trn3 hardware!")
    def test_all_experts_mxfp4_real_weights_device_vs_cpu_single_expert_lnc2(self, layer, token, T, E_L_idx, lnc, limit, hidden_act_bias, alpha, accumulation_dtype, output_dtype, atol, rtol):
        """
        Test matchingness between CPU implementation of MoE MX and NKI implementation.
        """
        
        # Needs to be imported after trn3 env vars are set
        from neuronx_distributed.kernels.all_expert_mlps_mx_shard_on_I import all_expert_mlps_mx_shard_on_I_nki_kernel

        # Add LNC=2 env vars if LNC=2; these get cleaned up by teardown_method() post-test.
        if lnc == 2:
            os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
            os.environ["NEURON_CC_FLAGS"] = f"{os.environ['NEURON_CC_FLAGS']} --lnc=2"

        # Load inputs and router logits
        _, rmsnorm_out, router_logits, _ = self.load_goldens(layer, token, T)
        rmsnorm_out_padded = _pad_tensor(rmsnorm_out, (T, GPT_OSS_HEAD_DIM_PADDED))

        # Load MXFP4 weights (bias loaded from MX dir to account for any padding)
        W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down = self.load_weights(layer, dtype=torch.uint16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.uint16)

        # Prep torch inputs
        E_L = 4  # select all k experts that are activated
        topk_expert_indices, W_gate_uint16_topk, W_up_uint16_topk, W_down_uint16_topk, scale_gate_topk, scale_up_topk, scale_down_topk, bias_gate_topk, bias_up_topk, bias_down_topk = \
            self.prep_torch_inputs_topk(router_logits, E_L, W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down, bias_gate, bias_up, bias_down)
        
        # Compute expert affinities mask
        expert_affinities_masked = expert_affinity_mask(router_logits, expert_index=topk_expert_indices)

        # Prep NKI inputs
        # TODO: remove tp_degree hardcode to explicitly test TP>1
        E, H, I, tp_degree = E_L, 3072, 3072, 1  # noqa: E741
        rmsnorm_out_padded_nki, W_gate_up_topk_nki, scale_gate_up_topk_nki, bias_gate_up_topk_nki, W_down_topk_nki, scale_down_topk_nki, bias_down_topk_nki, expert_affinities_masked_nki = \
            self.prep_nki_inputs_shuffled(T, E, H, I, tp_degree, rmsnorm_out_padded, W_gate_uint16_topk, W_up_uint16_topk, W_down_uint16_topk, \
                scale_gate_topk, scale_up_topk, scale_down_topk, bias_gate_topk, bias_up_topk, bias_down_topk, expert_affinities_masked, hidden_act_bias=hidden_act_bias, squeeze_tp_dim=True)

        print(f"Computing expert: {topk_expert_indices[E_L_idx]}")

        # Call reference implementation
        result_cpu = all_expert_mlps_act_mxfp8_w_mxfp4(
            norm_out=rmsnorm_out_padded,
            expert_index=topk_expert_indices[E_L_idx:E_L_idx+1],
            W_gate=W_gate_uint16_topk[E_L_idx:E_L_idx+1, ...], 
            W_up=W_up_uint16_topk[E_L_idx:E_L_idx+1, ...], 
            W_down=W_down_uint16_topk[E_L_idx:E_L_idx+1, ...], 
            scale_gate=scale_gate_topk[E_L_idx:E_L_idx+1, ...], 
            scale_up=scale_up_topk[E_L_idx:E_L_idx+1, ...], 
            scale_down=scale_down_topk[E_L_idx:E_L_idx+1, ...],
            bias_gate=bias_gate_topk[E_L_idx:E_L_idx+1, ...],
            bias_up=bias_up_topk[E_L_idx:E_L_idx+1, ...],
            bias_down=bias_down_topk[E_L_idx:E_L_idx+1, ...],
            expert_affinities_masked=expert_affinities_masked[..., E_L_idx:E_L_idx+1],
            matmul_accumulation_dtype=accumulation_dtype,
            matmul_output_dtype=output_dtype,
        )

        # Call NKI implementation
        simulation = False  # flip flag for debugging
        print(f"Running kernel in mode: {simulation=} with {lnc=}")
        if simulation:
            result_nki = nki.simulate_kernel(
                all_expert_mlps_mx_shard_on_I_nki_kernel[VNC(lnc)],
                input=rmsnorm_out_padded_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                gate_up_weights=W_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].numpy().view(nki_private.float4_e2m1fn_x4),
                down_weights=W_down_topk_nki[E_L_idx:E_L_idx+1, ...].numpy().view(nki_private.float4_e2m1fn_x4),
                gate_up_weights_scale=scale_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].numpy(),
                down_weights_scale=scale_down_topk_nki[E_L_idx:E_L_idx+1, ...].numpy(),
                gate_up_weights_bias=bias_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                down_weights_bias=bias_down_topk_nki[E_L_idx:E_L_idx+1, ...].view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                expert_affinities_masked=expert_affinities_masked_nki[..., E_L_idx:E_L_idx+1].view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                hidden_act_scale_factor=alpha,
                gate_clamp_upper_limit=limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=limit+hidden_act_bias,
                up_clamp_lower_limit=-limit+hidden_act_bias,
            )
            result_nki = torch.from_numpy(result_nki.view(np.uint16)).view(torch.bfloat16)
        else:
            device = xla_model.xla_device()
            result_nki = all_expert_mlps_mx_shard_on_I_nki_kernel[VNC(lnc)](
                input=rmsnorm_out_padded_nki.to(device),
                gate_up_weights=W_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                down_weights=W_down_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                gate_up_weights_scale=scale_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                down_weights_scale=scale_down_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                gate_up_weights_bias=bias_gate_up_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                down_weights_bias=bias_down_topk_nki[E_L_idx:E_L_idx+1, ...].to(device),
                expert_affinities_masked=expert_affinities_masked_nki[..., E_L_idx:E_L_idx+1].to(device),
                hidden_act_scale_factor=alpha,
                gate_clamp_upper_limit=limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=limit+hidden_act_bias,
                up_clamp_lower_limit=-limit+hidden_act_bias,
                psum_accumulation_dtype=TORCH_NKI_DTYPE_MAP[accumulation_dtype],
                activation_compute_dtype=TORCH_NKI_DTYPE_MAP[output_dtype],
            )
            result_nki = result_nki.cpu()

        # Ensure that padding is correct
        assert torch.all(result_nki[:, GPT_OSS_HEAD_DIM:] == 0)

        # Unpad cpu and NKI impl for comparison against golden
        result_cpu_unpadded = result_cpu[:, :GPT_OSS_HEAD_DIM]
        result_nki_unpadded = result_nki[:, :GPT_OSS_HEAD_DIM]

        # Validate accuracy
        print(f"{result_cpu_unpadded=}")
        print(f"{result_nki_unpadded=}")

        torch_neuronx.testing.assert_close(result_nki_unpadded, result_cpu_unpadded, atol=atol, rtol=rtol)
        print("Test Passes!")


    # TODO: add layers 1, 2, 3
    @pytest.mark.parametrize("layer", [0])
    @pytest.mark.parametrize("token", [0])
    @pytest.mark.parametrize(
        "T,accumulation_dtype,atol,rtol", [
            # NOTE: current goldens use the same input broadcast on T dim, so testing accuracy with T=128 gives same error as T=512.
            # fp32 PSUM
            # NOTE: was able to previously able to achieve atol of 9.3e-2 and 3.9e-2 here with < 2% rtol.
            pytest.param(64, torch.float32, 1.25e-1, 2.5146484375e-2, id="T64_fp32_psum"),
            pytest.param(128, torch.float32, 1.25e-1, 2.5146484375e-2, id="T128_fp32_psum"),
            pytest.param(256, torch.float32, 1.25e-1, 2.5146484375e-2, id="T256_fp32_psum"),
            pytest.param(512, torch.float32, 1.25e-1, 2.5146484375e-2, id="T512_fp32_psum"),
            
            # bf16 PSUM
            pytest.param(64, torch.bfloat16, 5.2734375e-2, 1.06201171875e-2, id="T64_bf16_psum"),
            pytest.param(128, torch.bfloat16, 5.2734375e-2, 1.06201171875e-2, id="T128_bf16_psum"),
            pytest.param(256, torch.bfloat16, 5.2734375e-2, 1.06201171875e-2, id="T256_bf16_psum"),
            pytest.param(512, torch.bfloat16, 5.2734375e-2, 1.06201171875e-2, id="T512_bf16_psum"),
        ]
    )
    @pytest.mark.parametrize("lnc,limit,hidden_act_bias,output_dtype", [(2, 7.0, 1.0, torch.bfloat16)])
    @pytest.mark.parametrize("loop_over_experts", [True])
    @pytest.mark.trn3
    @pytest.mark.skipif(not RUN_TRN3_TESTS, reason="Skip test when not run on trn3 hardware!")
    def test_all_experts_mxfp4_real_weights_device_vs_golden_and_cpu_lnc2(self, layer, token, T, lnc, limit, hidden_act_bias, accumulation_dtype, output_dtype, loop_over_experts, atol, rtol):
        """
        Test matchingness between CPU implementation of MoE MX and NKI implementation.

        NOTE: uses only 4x activated experts to speed up test time. We only need to test with k=4 because these 4 experts are used for all tokens in the current goldens.

        TODO: add option to call all experts with masking
        """
        
        # Skip running non-activated experts for these goldens
        E_L = 4

        # Needs to be imported after trn3 env vars are set
        from neuronx_distributed.kernels.all_expert_mlps_mx_shard_on_I import all_expert_mlps_mx_shard_on_I_nki_kernel

        # Load inputs and router logits
        _, rmsnorm_out, router_logits, golden_output = self.load_goldens(layer, token, T)
        rmsnorm_out_padded = _pad_tensor(rmsnorm_out, (T, GPT_OSS_HEAD_DIM_PADDED))

        # Load MXFP4 weights (bias loaded from MX dir to account for any padding)
        W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down = self.load_weights(layer, dtype=torch.uint16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.uint16)

        # Select topk experts
        topk_expert_indices, W_gate_uint16_topk, W_up_uint16_topk, W_down_uint16_topk, scale_gate_topk, scale_up_topk, scale_down_topk, bias_gate_topk, bias_up_topk, bias_down_topk = \
            self.prep_torch_inputs_topk(router_logits, E_L, W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down, bias_gate, bias_up, bias_down)
        
        # Compute expert affinities mask
        expert_affinities_masked = expert_affinity_mask(router_logits, expert_index=topk_expert_indices)

        # Prep NKI inputs
        # TODO: remove tp_degree hardcode to explicitly test TP>1
        E, H, I, tp_degree = E_L, 3072, 3072, 1  # noqa: E741
        rmsnorm_out_padded_nki, W_gate_up_topk_nki, scale_gate_up_topk_nki, bias_gate_up_topk_nki, W_down_topk_nki, scale_down_topk_nki, bias_down_topk_nki, expert_affinities_masked_nki = \
            self.prep_nki_inputs_shuffled(T, E, H, I, tp_degree, rmsnorm_out_padded, W_gate_uint16_topk, W_up_uint16_topk, W_down_uint16_topk, \
                scale_gate_topk, scale_up_topk, scale_down_topk, bias_gate_topk, bias_up_topk, bias_down_topk, expert_affinities_masked, hidden_act_bias=hidden_act_bias, squeeze_tp_dim=True)

        print(f"Using {topk_expert_indices=}")

        # Call NKI implementation
        simulation = False  # flip flag for debugging
        print(f"Running kernel in mode: {simulation=} with {lnc=}")
        if simulation:
            print("Simulating kernel")
            out_nki = nki.simulate_kernel(
                all_expert_mlps_mx_shard_on_I_nki_kernel[VNC(lnc)],
                input=rmsnorm_out_padded_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                gate_up_weights=W_gate_up_topk_nki.numpy().view(nki_private.float4_e2m1fn_x4),
                down_weights=W_down_topk_nki.numpy().view(nki_private.float4_e2m1fn_x4),
                gate_up_weights_scale=scale_gate_up_topk_nki.numpy(),
                down_weights_scale=scale_down_topk_nki.numpy(),
                gate_up_weights_bias=bias_gate_up_topk_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                down_weights_bias=bias_down_topk_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                expert_affinities_masked=expert_affinities_masked_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
                gate_clamp_upper_limit=limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=limit+hidden_act_bias,
                up_clamp_lower_limit=-limit+hidden_act_bias,
            )
            out_nki = torch.from_numpy(out_nki.transpose(1, 0, 2, 3).reshape(T, 3072).view(np.uint16)).view(torch.bfloat16)
        
        elif loop_over_experts:
            print(f"Running kernel on device with {loop_over_experts=}")
            out_nki = torch.zeros(size=(T, H), dtype=torch.bfloat16)
            for E in range(E_L):
                print(f"Computing expert: {topk_expert_indices[E]}")
                device = xla_model.xla_device()
                out_nki_one_expert = all_expert_mlps_mx_shard_on_I_nki_kernel[VNC(lnc)](
                    input=rmsnorm_out_padded_nki.to(device),
                    gate_up_weights=W_gate_up_topk_nki[E:E+1, ...].to(device),
                    down_weights=W_down_topk_nki[E:E+1, ...].to(device),
                    gate_up_weights_scale=scale_gate_up_topk_nki[E:E+1, ...].to(device),
                    down_weights_scale=scale_down_topk_nki[E:E+1, ...].to(device),
                    gate_up_weights_bias=bias_gate_up_topk_nki[E:E+1, ...].to(device),
                    down_weights_bias=bias_down_topk_nki[E:E+1, ...].to(device),
                    expert_affinities_masked=expert_affinities_masked_nki[..., E:E+1].to(device),
                    gate_clamp_upper_limit=limit,
                    gate_clamp_lower_limit=None,
                    up_clamp_upper_limit=limit+hidden_act_bias,
                    up_clamp_lower_limit=-limit+hidden_act_bias,
                    psum_accumulation_dtype=TORCH_NKI_DTYPE_MAP[accumulation_dtype],
                    activation_compute_dtype=TORCH_NKI_DTYPE_MAP[output_dtype],
                )

                out_nki_one_expert = out_nki_one_expert.cpu()
                out_nki += out_nki_one_expert
        else:
            print(f"Running kernel on device with {loop_over_experts=}")
            device = xla_model.xla_device()
            out_nki = all_expert_mlps_mx_shard_on_I_nki_kernel[VNC(lnc)](
                input=rmsnorm_out_padded_nki.to(device),
                gate_up_weights=W_gate_up_topk_nki.to(device),
                down_weights=W_down_topk_nki.to(device),
                gate_up_weights_scale=scale_gate_up_topk_nki.to(device),
                down_weights_scale=scale_down_topk_nki.to(device),
                gate_up_weights_bias=bias_gate_up_topk_nki.to(device),
                down_weights_bias=bias_down_topk_nki.to(device),
                expert_affinities_masked=expert_affinities_masked_nki.to(device),
                gate_clamp_upper_limit=limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=limit+hidden_act_bias,
                up_clamp_lower_limit=-limit+hidden_act_bias,
                psum_accumulation_dtype=TORCH_NKI_DTYPE_MAP[accumulation_dtype],
                activation_compute_dtype=TORCH_NKI_DTYPE_MAP[output_dtype],
            )

            out_nki = out_nki.cpu()

        # Ensure that padding is correct
        assert torch.all(out_nki[:, GPT_OSS_HEAD_DIM:] == 0)

        # Unpad cpu and NKI impl for comparison against golden
        result_nki_unpadded = out_nki[:, :GPT_OSS_HEAD_DIM]

        # Validate accuracy
        print(f"{result_nki_unpadded=}")
        print(f"{golden_output=}")

        torch_neuronx.testing.assert_close(result_nki_unpadded, golden_output, atol=atol, rtol=rtol)
        print("Test Passes!")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])