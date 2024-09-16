import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

import torch

from neuronx_distributed.pipeline.trace import get_concrete_args
from neuronx_distributed.pipeline.model import NxDPPModel


class NxDModule(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(num_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        return None


def get_model_nxd(num_layers):
    model = NxDPPModel(module=NxDModule(num_layers), transformer_layer_cls=torch.nn.Linear, tracer_cls="torch")
    return model


def run_concrete_args(*args, **kwargs):
    model = get_model_nxd(4)
    return get_concrete_args(model.original_torch_module, None, args, kwargs)


class TestSignatureAnalysis(unittest.TestCase):
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    @patch("torch.distributed.get_rank")
    def test_signature_analyse(self, rank_mock, state_mock):
        concrete_args = run_concrete_args(1, 2, "a", use_cache=True, output_hidden_states=True, return_dict=True)
        expected_concrete_args = [
            "past_key_values",
            "inputs_embeds",
            "labels",
            "output_attentions",
            "cache_position",
        ]
        assert list(concrete_args.keys()) == expected_concrete_args


if __name__ == "__main__":
    unittest.main()
