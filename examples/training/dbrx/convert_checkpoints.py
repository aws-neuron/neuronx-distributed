import argparse
import json
import torch

from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase


class CheckpointConverterDbrx(CheckpointConverterBase):
    # Define the mapping between standard config names and DBRX-specific names
    attribute_map = {
        'num_attention_heads': 'n_heads',
        'hidden_size': 'd_model',
        'num_hidden_layers': 'n_layers',
        'max_position_embeddings': 'max_seq_len',
        'num_key_value_heads': 'attn_config.kv_n_heads',
    }
    # ExpertFusedColumnParallelLinear
    gate_up_proj_partition_dim = 2
    # ExpertFusedRowParallelLinear
    down_proj_partition_dim = 1
    # Pattern of layer name
    layer_name = "blocks"
    layer_name_pattern = r"^(transformer\.blocks\.\d+)"

    def pre_process_full_state_before_tp_conversion(self, state_dict, args):
        with open(args.config, "r") as f:
            config = json.load(f)

        # Basic config values
        hidden_size = self._get_config_value(config, "hidden_size")
        num_layers = self._get_config_value(config, "num_hidden_layers")
        ffn_hidden_size = self._get_config_value(config["ffn_config"], "ffn_hidden_size")

        with torch.no_grad():
            for i in range(num_layers):
                # --------------------------------
                # A) Q/K/V
                # --------------------------------
                qkv_name = f"transformer.blocks.{i}.norm_attn_norm.attn.Wqkv.weight"
                qkv_weight = state_dict.pop(qkv_name)  # [3 * hidden_size, hidden_size], usually
                q_size = hidden_size
                kv_size = (qkv_weight.size(0) - q_size) // 2

                q_weight = qkv_weight[:q_size]
                k_weight = qkv_weight[q_size : q_size + kv_size]
                v_weight = qkv_weight[q_size + kv_size :]

                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.q_proj.weight"] = q_weight
                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.k_proj.weight"] = k_weight
                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.v_proj.weight"] = v_weight

                # Out proj
                out_name = f"transformer.blocks.{i}.norm_attn_norm.attn.out_proj.weight"
                out_weight = state_dict.pop(out_name)
                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.o_proj.weight"] = out_weight

                # --------------------------------
                # B) Router rename
                # --------------------------------
                router_src = f"transformer.blocks.{i}.ffn.router.layer.weight"
                router_dst = f"transformer.blocks.{i}.ffn.router.linear_router.weight"
                router_weight = state_dict.pop(router_src)
                state_dict[router_dst] = router_weight

                # --------------------------------
                # C) GATE / UP / DOWN
                # --------------------------------
                w1 = state_dict.pop(f"transformer.blocks.{i}.ffn.experts.mlp.w1")
                v1 = state_dict.pop(f"transformer.blocks.{i}.ffn.experts.mlp.v1")
                w2 = state_dict.pop(f"transformer.blocks.{i}.ffn.experts.mlp.w2")

                w1 = w1.reshape(-1, ffn_hidden_size, hidden_size).permute(0, 2, 1)
                v1 = v1.reshape(-1, ffn_hidden_size, hidden_size).permute(0, 2, 1)
                w2 = w2.reshape(-1, ffn_hidden_size, hidden_size)

                # Finally, store them
                state_dict[f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.gate_proj.weight"] = w1.detach().clone()
                state_dict[f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.up_proj.weight"] = v1.detach().clone()
                state_dict[f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.down_proj.weight"] = w2.detach().clone()

        return state_dict

    def post_process_full_state_after_tp_conversion(self, state_dict, args):
        with open(args.config, "r") as f:
            config = json.load(f)
        num_layers = self._get_config_value(config, "num_hidden_layers")
        hidden_size = self._get_config_value(config, "hidden_size")

        with torch.no_grad():
            for i in range(num_layers):
                # q, k, v
                q_weight = state_dict.pop(f"transformer.blocks.{i}.norm_attn_norm.attn.q_proj.weight")
                k_weight = state_dict.pop(f"transformer.blocks.{i}.norm_attn_norm.attn.k_proj.weight")
                v_weight = state_dict.pop(f"transformer.blocks.{i}.norm_attn_norm.attn.v_proj.weight")

                merged_qkv = torch.cat([q_weight, k_weight, v_weight], dim=0)
                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.Wqkv.weight"] = merged_qkv

                o_weight = state_dict.pop(f"transformer.blocks.{i}.norm_attn_norm.attn.o_proj.weight")
                state_dict[f"transformer.blocks.{i}.norm_attn_norm.attn.out_proj.weight"] = o_weight

                # gate, up, proj
                gate_weight = state_dict.pop(f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.gate_proj.weight")
                up_weight = state_dict.pop(f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.up_proj.weight")
                down_weight = state_dict.pop(f"transformer.blocks.{i}.ffn.expert_mlps.mlp_op.down_proj.weight")

                gate_weight = gate_weight.permute(0, 2, 1).reshape(-1, hidden_size)
                up_weight = up_weight.permute(0, 2, 1).reshape(-1, hidden_size)
                down_weight = down_weight.reshape(-1, hidden_size)

                state_dict[f"transformer.blocks.{i}.ffn.experts.mlp.w1"] = gate_weight.detach().clone()
                state_dict[f"transformer.blocks.{i}.ffn.experts.mlp.v1"] = up_weight.detach().clone()
                state_dict[f"transformer.blocks.{i}.ffn.experts.mlp.w2"] = down_weight.detach().clone()

                # router
                router_weight = state_dict.pop(f"transformer.blocks.{i}.ffn.router.linear_router.weight")
                state_dict[f"transformer.blocks.{i}.ffn.router.layer.weight"] = router_weight

        return state_dict


if __name__ == "__main__":
    checkpoint_converter = CheckpointConverterDbrx()
    parser = checkpoint_converter.get_arg_parser()
    args, _ = parser.parse_known_args()
    checkpoint_converter.run(args)
