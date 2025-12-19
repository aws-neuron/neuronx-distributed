# Standard Library
import unittest
import json
import os

from neuronx_distributed.trace.model_builder import _check_weight_name_regex

class WeightConfig:
    def __init__(self, num_lora_weights=5):
        self.num_lora_weights = num_lora_weights

        self.lora_weights = [f"lora_weight_{idx}" for idx in range(self.num_lora_weights)]
        self.weights = [f"weight_{idx}" for idx in range(self.num_lora_weights)] + self.lora_weights
        self.num_weights = len(self.weights)
        self.weight_name_to_idx = {name : idx for idx, name in zip(range(self.num_weights), self.weights)}


class TestLoraWeights(unittest.TestCase):
    def test_lora_weights_regex(self):
        weights = {r".*lora.*"}
        config = WeightConfig()
        
        weight_names_to_skip = _check_weight_name_regex(weights, config.weight_name_to_idx)
        assert weight_names_to_skip == set(config.lora_weights)


if __name__ == "__main__":
    unittest.main()