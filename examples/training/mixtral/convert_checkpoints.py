import json

import torch
from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase


class CheckpointConverterMixtral(CheckpointConverterBase):
    # ExpertFusedColumnParallelLinear
    gate_up_proj_partition_dim = 2
    # ExpertFusedRowParallelLinear
    down_proj_partition_dim = 1

    def pre_process_full_state_before_tp_conversion(self, state_dict, args):
        """Stack the MLP weights across experts as expected by the MoE module."""

        with open(args.config, "r") as f:
            config = json.load(f)

        for i in range(config["num_hidden_layers"]):
            router_weight = state_dict.pop(f"model.layers.{i}.block_sparse_moe.gate.weight")

            gate_proj_per_expert = []
            up_proj_per_expert = []
            down_proj_per_expert = []
            for j in range(config["num_local_experts"]):
                gate_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"))
                down_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"))
                up_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"))
            gate_proj = torch.stack(gate_proj_per_expert)
            up_proj = torch.stack(up_proj_per_expert)
            down_proj = torch.stack(down_proj_per_expert)

            state_dict[f"model.layers.{i}.mlp.router.linear_router.weight"] = router_weight
            state_dict[f"model.layers.{i}.mlp.expert_mlps.mlp_op.gate_proj.weight"] = gate_proj
            state_dict[f"model.layers.{i}.mlp.expert_mlps.mlp_op.up_proj.weight"] = up_proj
            state_dict[f"model.layers.{i}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        return state_dict

    def post_process_full_state_after_tp_conversion(self, state_dict, args):
        """Split the MLP weights across experts."""

        with open(args.config, "r") as f:
            config = json.load(f)

        for i in range(config["num_hidden_layers"]):
            router_weight = state_dict.pop(f"model.layers.{i}.mlp.router.linear_router.weight")
            gate_proj = state_dict.pop(f"model.layers.{i}.mlp.expert_mlps.mlp_op.gate_proj.weight")
            up_proj = state_dict.pop(f"model.layers.{i}.mlp.expert_mlps.mlp_op.up_proj.weight")
            down_proj = state_dict.pop(f"model.layers.{i}.mlp.expert_mlps.mlp_op.down_proj.weight")

            gate_proj_per_expert = torch.unbind(gate_proj)
            up_proj_per_expert = torch.unbind(up_proj)
            down_proj_per_expert = torch.unbind(down_proj)
            for j in range(config["num_local_experts"]):
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = gate_proj_per_expert[j]
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = down_proj_per_expert[j]
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = up_proj_per_expert[j]
            state_dict[f"model.layers.{i}.block_sparse_moe.gate.weight"] = router_weight

        return state_dict


if __name__ == "__main__":
    checkpoint_converter = CheckpointConverterMixtral()
    parser = checkpoint_converter.get_arg_parser()
    args, _ = parser.parse_known_args()
    checkpoint_converter.run(args)
