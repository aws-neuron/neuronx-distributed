# Standard Library
import unittest

# Third Party
import torch
from transformers import PretrainedConfig

from neuronx_distributed.utils.sampling import Sampler


class TestSampling(unittest.TestCase):
    def test_greedy_sampling(self):
        config = PretrainedConfig()
        config.do_sample = True
        config.on_device_sampling = False
        config.top_k = 1
        config.num_beams = 1
        sampler = Sampler(config)
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sampled = sampler.sample(x)
        assert torch.equal(sampled, torch.tensor([0, 1, 2]))

    def test_multinomial_sampling(self):
        """
        To test multinomial sampling, we fix the seed to 5 and compare
        to previously collected token ids (goldens for) [9, 71]
        """
        config = PretrainedConfig()
        config.do_sample = True
        config.on_device_sampling = False
        config.top_k = 3
        config.num_beams = 1
        sampler = Sampler(config)
        neg_inf = -float("inf")
        torch.random.manual_seed(5)
        x = torch.rand((2, 100))
        sampled = sampler.sample(x)
        assert torch.equal(sampled, torch.tensor([9, 71]))


if __name__ == "__main__":
    unittest.main()
