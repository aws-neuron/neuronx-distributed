"""
Tests for padding-related functionality in parallel configurations.

Usage:
    python -m unittest test_padding.py -v
"""

# Standard Library
import os
import unittest

# Third Party
import torch

from neuronx_distributed.parallel_layers.pad import generate_padding_mask
from neuronx_distributed.utils.utils import hardware


class GeneratePaddingMaskTest(unittest.TestCase):
    """
    Test the attention head padding mask for a given attention config, TP degree,
    TP rank, and the hardware type.
    """

    def test_mask_generation_48_heads_8_kv_heads_trn1(self):
        # On TRN1, KV heads are replicated in groups.
        # TP32 with 48 query heads, 8 kv heads and kv_replicator=4:
        # Query heads get padded from 48 to 64.
        # KV heads layout: (K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7)
        # We need to mask out the query heads associated with the last group
        # of (K0,K1,...K7).

        num_heads = 48
        num_heads_with_pad = 64
        num_kv_heads = 8
        tp_degree = 32
        num_heads_per_tp_rank = num_heads_with_pad // tp_degree

        expected_mask = torch.ones((tp_degree, num_heads_per_tp_rank), dtype=torch.bool)
        expected_mask[24:] = False

        for tp_rank in range(tp_degree):
            padding_mask = generate_padding_mask(
                num_heads=num_heads,
                num_heads_with_pad=num_heads_with_pad,
                num_kv_heads=num_kv_heads,
                tp_degree=tp_degree,
                tp_rank=tp_rank,
                hardware_type=hardware.TRN1,
            )
            assert (padding_mask == expected_mask[tp_rank]).all()

    def test_mask_generation_200_heads_8_kv_heads_trn1(self):
        # On TRN1, KV heads are replicated in groups.
        # TP32 with 200 query heads, 8 kv heads and kv_replicator=4:
        # Query heads get padded from 200 to 224.
        # KV heads layout: (K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7),(K0,K1,...,K7)
        # We need to mask out the 3 of the 7 query heads in each rank associated with
        # the last replicate KV head group.

        num_heads = 200
        num_heads_with_pad = 224
        num_kv_heads = 8
        tp_degree = 32
        num_heads_per_tp_rank = num_heads_with_pad // tp_degree

        expected_mask = torch.ones((tp_degree, num_heads_per_tp_rank), dtype=torch.bool)
        expected_mask[24:, 4:] = False

        for tp_rank in range(tp_degree):
            padding_mask = generate_padding_mask(
                num_heads=num_heads,
                num_heads_with_pad=num_heads_with_pad,
                num_kv_heads=num_kv_heads,
                tp_degree=tp_degree,
                tp_rank=tp_rank,
                hardware_type=hardware.TRN1,
            )
            assert (padding_mask == expected_mask[tp_rank]).all()

    def test_mask_generation_trn2(self):
        # On TRN2, a KV head is replicated on adjacent ranks.
        # For example, TP32 with 48 query heads, 8 kv heads and kv_replicator=4:
        # KV heads layout: (K0,K0,K0,K0),(K1,K1,K1,K1),...,(K7,K7,K7,K7)
        # We need to mask out the query heads associated with the last replicate
        # kv head in each 4 replicates.
        num_heads = 48
        num_heads_with_pad = 64
        num_kv_heads = 8
        tp_degree = 32
        num_heads_per_tp_rank = num_heads_with_pad // tp_degree

        expected_mask = torch.ones((tp_degree, num_heads_per_tp_rank), dtype=torch.bool)
        expected_mask[3::4] = False

        for tp_rank in range(tp_degree):
            padding_mask = generate_padding_mask(
                num_heads=num_heads,
                num_heads_with_pad=num_heads_with_pad,
                num_kv_heads=num_kv_heads,
                tp_degree=tp_degree,
                tp_rank=tp_rank,
                hardware_type=hardware.TRN2,
            )
            assert (padding_mask == expected_mask[tp_rank]).all()

    def test_mask_generation_200_heads_8_kv_heads_trn2(self):
        # On TRN2, KV heads are replicated in groups.
        # TP32 with 200 query heads, 8 kv heads and kv_replicator=4:
        # Query heads get padded from 200 to 224.
        # KV heads layout: (K0,K0,K0,K0),(K1,K1,K1,K1),...,(K7,K7,K7,K7)
        # We need to mask out the 3 of the 7 query heads associated with the last replicate
        # kv head in each 4 replicates.

        num_heads = 200
        num_heads_with_pad = 224
        num_kv_heads = 8
        tp_degree = 32
        num_heads_per_tp_rank = num_heads_with_pad // tp_degree

        expected_mask = torch.ones((tp_degree, num_heads_per_tp_rank), dtype=torch.bool)
        expected_mask[3::4, 4:] = False

        for tp_rank in range(tp_degree):
            padding_mask = generate_padding_mask(
                num_heads=num_heads,
                num_heads_with_pad=num_heads_with_pad,
                num_kv_heads=num_kv_heads,
                tp_degree=tp_degree,
                tp_rank=tp_rank,
                hardware_type=hardware.TRN2,
            )
            assert (padding_mask == expected_mask[tp_rank]).all()

    def test_mask_generation_invalid_hardware(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected hardware_type"):
            generate_padding_mask(
                num_heads=48,
                num_heads_with_pad=64,
                num_kv_heads=8,
                tp_degree=32,
                tp_rank=0,
                hardware_type="trn999",
            )

    def test_mask_generation_tp_degree_less_than_kv_heads(self):
        # Test case where tp_degree < num_kv_heads..
        # This would cause kv_replicator to be 0.
        
        num_heads = 24
        num_heads_with_pad = 32
        num_kv_heads = 8
        tp_degree = 4
        num_heads_per_tp_rank = num_heads_with_pad // tp_degree

        expected_mask = torch.ones((tp_degree, num_heads_per_tp_rank), dtype=torch.bool)
        expected_mask[:, 6:] = False

        for tp_rank in range(tp_degree):
            padding_mask = generate_padding_mask(
                num_heads=num_heads,
                num_heads_with_pad=num_heads_with_pad,
                num_kv_heads=num_kv_heads,
                tp_degree=tp_degree,
                tp_rank=tp_rank,
                hardware_type=hardware.TRN2,
            )
            assert (padding_mask == expected_mask[tp_rank]).all()

if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
