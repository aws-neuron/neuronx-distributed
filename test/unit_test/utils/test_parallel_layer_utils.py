# Standard Library
import unittest
# Third Party
import torch

from neuronx_distributed.parallel_layers.utils import indices_split_along_dim


def get_indices_split_test_cases():
    test_config1 = {
        "input": torch.rand(1, 128, 256),
        "dim": 1,
        "rank": 0,
        "num_partitions": 32,
        "expected_value": torch.arange(0, 4),
    }
    test_config2 = {
        "input": torch.rand(1, 256, 1024),
        "dim": 2,
        "rank": 2,
        "num_partitions": 32,
        "expected_value": torch.arange(64, 96),
    }
    test_config3 = {
        "input": torch.rand(1, 64, 512),
        "dim": 2,
        "rank": 31,
        "num_partitions": 64,
        "expected_value": torch.arange(248, 256),
    }

    test_configs = []
    test_configs.append(test_config1)
    test_configs.append(test_config2)
    test_configs.append(test_config3)
    return test_configs


class TestParallelLayerUtils(unittest.TestCase):
    torch.manual_seed(42)
    def test_indices_split_utils(self):
        for test_config in get_indices_split_test_cases():
            input = test_config["input"]
            dim = test_config["dim"]
            rank = test_config["rank"]
            num_partitions = test_config["num_partitions"]
            expected_value = test_config["expected_value"]
            indices = indices_split_along_dim(input, dim, rank, num_partitions)
            torch.testing.assert_close(indices, expected_value)


if __name__ == "__main__":
    unittest.main(verbosity=3)
