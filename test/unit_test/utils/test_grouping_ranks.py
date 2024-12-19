import unittest

from neuronx_distributed.parallel_layers.parallel_state import arrange_kv_groups


class TestGroupingRanks(unittest.TestCase):

    def test_grouping_tp32_group_by_4(self):
        # sequential
        EXPECT_GROUPS = [[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]]
        inputs1 = {"num_tensor_model_parallel_groups": 1, "tensor_model_parallel_size": 32, "kv_shared_group_size": 4, "sequential_ranks_in_group": True}
        kv_groups = arrange_kv_groups(**inputs1)
        assert kv_groups == EXPECT_GROUPS

        # interleaved
        EXPECT_GROUPS = [[0, 8, 16, 24],
                         [1, 9, 17, 25],
                         [2, 10, 18, 26],
                         [3, 11, 19, 27],
                         [4, 12, 20, 28],
                         [5, 13, 21, 29],
                         [6, 14, 22, 30],
                         [7, 15, 23, 31]]
        inputs2 = {"num_tensor_model_parallel_groups": 1, "tensor_model_parallel_size": 32, "kv_shared_group_size": 4}
        kv_groups = arrange_kv_groups(**inputs2)
        assert kv_groups == EXPECT_GROUPS

    def test_grouping_tp32_group_by_8(self):
        # sequential
        EXPECT_GROUPS = [[0, 1, 2, 3, 4, 5, 6, 7],
                         [8, 9, 10, 11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20, 21, 22, 23],
                         [24, 25, 26, 27, 28, 29, 30, 31]]
        inputs3 = {"num_tensor_model_parallel_groups": 1, "tensor_model_parallel_size": 32, "kv_shared_group_size": 8, "sequential_ranks_in_group": True}
        kv_groups = arrange_kv_groups(**inputs3)
        assert kv_groups == EXPECT_GROUPS

        # interleaved
        EXPECT_GROUPS = [[0, 4, 8, 12, 16, 20, 24, 28],
                         [1, 5, 9, 13, 17, 21, 25, 29],
                         [2, 6, 10, 14, 18, 22, 26, 30],
                         [3, 7, 11, 15, 19, 23, 27, 31]]
        inputs4 = {"num_tensor_model_parallel_groups": 1, "tensor_model_parallel_size": 32, "kv_shared_group_size": 8}
        kv_groups = arrange_kv_groups(**inputs4)
        assert kv_groups == EXPECT_GROUPS

    def test_grouping_tp32x2_group_by_4(self):
        # sequential
        EXPECT_GROUPS = [[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31],
                         [32, 33, 34, 35],
                         [36, 37, 38, 39],
                         [40, 41, 42, 43],
                         [44, 45, 46, 47],
                         [48, 49, 50, 51],
                         [52, 53, 54, 55],
                         [56, 57, 58, 59],
                         [60, 61, 62, 63]]
        inputs5 = {"num_tensor_model_parallel_groups": 2, "tensor_model_parallel_size": 32, "kv_shared_group_size": 4, "sequential_ranks_in_group": True}
        kv_groups = arrange_kv_groups(**inputs5)
        assert kv_groups == EXPECT_GROUPS

        # interleaved
        EXPECT_GROUPS = [[0, 8, 16, 24],
                         [1, 9, 17, 25],
                         [2, 10, 18, 26],
                         [3, 11, 19, 27],
                         [4, 12, 20, 28],
                         [5, 13, 21, 29],
                         [6, 14, 22, 30],
                         [7, 15, 23, 31],
                         [32, 40, 48, 56],
                         [33, 41, 49, 57],
                         [34, 42, 50, 58],
                         [35, 43, 51, 59],
                         [36, 44, 52, 60],
                         [37, 45, 53, 61],
                         [38, 46, 54, 62],
                         [39, 47, 55, 63]]
        inputs6 = {"num_tensor_model_parallel_groups": 2, "tensor_model_parallel_size": 32, "kv_shared_group_size": 4}
        kv_groups = arrange_kv_groups(**inputs6)
        assert kv_groups == EXPECT_GROUPS

    def test_grouping_tp32x2_group_by_8(self):
        # sequential
        EXPECT_GROUPS = [[0, 1, 2, 3, 4, 5, 6, 7],
                         [8, 9, 10, 11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20, 21, 22, 23],
                         [24, 25, 26, 27, 28, 29, 30, 31],
                         [32, 33, 34, 35, 36, 37, 38, 39],
                         [40, 41, 42, 43, 44, 45, 46, 47],
                         [48, 49, 50, 51, 52, 53, 54, 55],
                         [56, 57, 58, 59, 60, 61, 62, 63]]
        inputs7 = {"num_tensor_model_parallel_groups": 2, "tensor_model_parallel_size": 32, "kv_shared_group_size": 8, "sequential_ranks_in_group": True}
        kv_groups = arrange_kv_groups(**inputs7)
        assert kv_groups == EXPECT_GROUPS

        # interleaved
        EXPECT_GROUPS = [[0, 4, 8, 12, 16, 20, 24, 28],
                         [1, 5, 9, 13, 17, 21, 25, 29],
                         [2, 6, 10, 14, 18, 22, 26, 30],
                         [3, 7, 11, 15, 19, 23, 27, 31],
                         [32, 36, 40, 44, 48, 52, 56, 60],
                         [33, 37, 41, 45, 49, 53, 57, 61],
                         [34, 38, 42, 46, 50, 54, 58, 62],
                         [35, 39, 43, 47, 51, 55, 59, 63]]
        inputs8 = {"num_tensor_model_parallel_groups": 2, "tensor_model_parallel_size": 32, "kv_shared_group_size": 8}
        kv_groups = arrange_kv_groups(**inputs8)
        assert kv_groups == EXPECT_GROUPS


if __name__ == "__main__":
    unittest.main()
