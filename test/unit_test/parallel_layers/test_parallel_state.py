import unittest
from unittest.mock import patch

from neuronx_distributed.parallel_layers.parallel_state import (
    get_logic_chosen, 
    PG_Group_Logic, 
    ascending_ring_PG_group, 
    ParallelGroups, 
    ascending_descending_ring_PG_group, 
    get_pipeline_model_parallel_rank, 
    get_data_parallel_rank
)
from neuronx_distributed.utils.utils import hardware
import torch

MODULE = "neuronx_distributed.parallel_layers.parallel_state"


class ParallelGroupTest(unittest.TestCase):
 
    def test_get_logic_chosen(self):
        
        inputs1 = {"lnc_size":1, "hardware_type": hardware.TRN1, "tp": 8}
        inputs2 = {"lnc_size":2, "hardware_type": hardware.TRN1, "tp": 8}

        inputs3 = {"lnc_size":1, "hardware_type": hardware.TRN2, "tp": 8}
        inputs4 = {"lnc_size":1, "hardware_type": hardware.TRN2, "tp": 64}

        inputs5 = {"lnc_size":2, "hardware_type": hardware.TRN2, "tp": 8}
        inputs6 = {"lnc_size":2, "hardware_type": hardware.TRN2, "tp": 32}

        res = get_logic_chosen(**inputs1)
        assert res == PG_Group_Logic.LOGIC1

        res = get_logic_chosen(**inputs2)
        assert res == PG_Group_Logic.LOGIC1

        res = get_logic_chosen(**inputs3)
        assert res == PG_Group_Logic.LOGIC1

        res = get_logic_chosen(**inputs4)
        assert res == PG_Group_Logic.LOGIC2

        with patch('torch.distributed.get_world_size', return_value=64):
            res = get_logic_chosen(**inputs5)
            assert res == PG_Group_Logic.LOGIC1

            res = get_logic_chosen(**inputs6)
            assert res == PG_Group_Logic.LOGIC2

        with patch('torch.distributed.get_world_size', return_value=32):
            res = get_logic_chosen(**inputs6)
            assert res == PG_Group_Logic.LOGIC1

    def test_ascending_ring_pg_group_creation(self):
        ground_truth_64 = ParallelGroups(tp_groups=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]], 
            dp_groups=[
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16],
            [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], 
            [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], 
            [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63]], 
            pp_groups=[
            [0, 32], [1, 33], [2, 34], [3, 35], [4, 36], [5, 37], [6, 38], [7, 39], [8, 40], [9, 41], [10, 42],
            [11, 43], [12, 44], [13, 45], [14, 46], [15, 47], [16, 48], [17, 49], [18, 50], [19, 51], [20, 52], 
            [21, 53], [22, 54], [23, 55], [24, 56], [25, 57], [26, 58], [27, 59], [28, 60], [29, 61], [30, 62], 
            [31, 63]], 
            ep_model_groups=[
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
            [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], 
            [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], 
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63]], 
            ep_data_groups=[
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], 
            [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], 
            [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], 
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63]]
            ) # noqa: F841

        ground_truth_128 = ParallelGroups(tp_groups=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
            [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
            80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], 
            [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]], 
            dp_groups=[
            [0, 32], [1, 33], [2, 34], [3, 35], [4, 36], [5, 37], [6, 38], [7, 39], [8, 40], 
            [9, 41], [10, 42], [11, 43], [12, 44], [13, 45], [14, 46], [15, 47], [16, 48], [17, 49],
            [18, 50], [19, 51], [20, 52], [21, 53], [22, 54], [23, 55], [24, 56], [25, 57], [26, 58],
            [27, 59], [28, 60], [29, 61], [30, 62], [31, 63], [64, 96], [65, 97], [66, 98], [67, 99], 
            [68, 100], [69, 101], [70, 102], [71, 103], [72, 104], [73, 105], [74, 106], [75, 107],
            [76, 108], [77, 109], [78, 110], [79, 111], [80, 112], [81, 113], [82, 114], [83, 115], 
            [84, 116], [85, 117], [86, 118], [87, 119], [88, 120], [89, 121], [90, 122], [91, 123], 
            [92, 124], [93, 125], [94, 126], [95, 127]], 
            pp_groups=[
            [0, 64], [1, 65], [2, 66], [3, 67], [4, 68], [5, 69], [6, 70], [7, 71], [8, 72], 
            [9, 73], [10, 74], [11, 75], [12, 76], [13, 77], [14, 78], [15, 79], [16, 80], 
            [17, 81], [18, 82], [19, 83], [20, 84], [21, 85], [22, 86], [23, 87], [24, 88], 
            [25, 89], [26, 90], [27, 91], [28, 92], [29, 93], [30, 94], [31, 95], [32, 96], 
            [33, 97], [34, 98], [35, 99], [36, 100], [37, 101], [38, 102], [39, 103], [40, 104],
            [41, 105], [42, 106], [43, 107], [44, 108], [45, 109], [46, 110], [47, 111], 
            [48, 112], [49, 113], [50, 114], [51, 115], [52, 116], [53, 117], [54, 118], 
            [55, 119], [56, 120], [57, 121], [58, 122], [59, 123], [60, 124], [61, 125], 
            [62, 126], [63, 127]], 
            ep_model_groups=[
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], 
            [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], 
            [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38],
            [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51],
            [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64],
            [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77],
            [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], 
            [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], 
            [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115],
            [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127]], 
            ep_data_groups=[
            [0, 32], [1, 33], [2, 34], [3, 35], [4, 36], [5, 37], [6, 38], [7, 39],
            [8, 40], [9, 41], [10, 42], [11, 43], [12, 44], [13, 45], [14, 46], [15, 47],
            [16, 48], [17, 49], [18, 50], [19, 51], [20, 52], [21, 53], [22, 54], [23, 55],
            [24, 56], [25, 57], [26, 58], [27, 59], [28, 60], [29, 61], [30, 62], [31, 63],
            [64, 96], [65, 97], [66, 98], [67, 99], [68, 100], [69, 101], [70, 102], [71, 103],
            [72, 104], [73, 105], [74, 106], [75, 107], [76, 108], [77, 109], [78, 110], [79, 111],
            [80, 112], [81, 113], [82, 114], [83, 115], [84, 116], [85, 117], [86, 118], [87, 119],
            [88, 120], [89, 121], [90, 122], [91, 123], [92, 124], [93, 125], [94, 126], [95, 127]]
            )

        for world_size in [64,128]:
            tensor_model_parallel_size = 32
            pipeline_model_parallel_size = 2
            expert_model_parallel_size = 1
            data_parallel_size = world_size//(tensor_model_parallel_size*pipeline_model_parallel_size)
            cluster_ranks = torch.arange(0, world_size)
            expert_data_parallel_size: int = world_size // (
                tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size
            )
            cluster_ranks_exp = cluster_ranks.reshape(
                [
                    pipeline_model_parallel_size,
                    expert_data_parallel_size,
                    expert_model_parallel_size,
                    tensor_model_parallel_size,  # important: contiguous parallelism dimension
                ]
            )
            cluster_ranks_nonexp = cluster_ranks.reshape(
                [
                    pipeline_model_parallel_size,
                    data_parallel_size,
                    tensor_model_parallel_size,  # important: contiguous parallelism dimension
                ]
            )
            res = ascending_ring_PG_group(lnc_size=1, cluster_ranks_nonexp= cluster_ranks_nonexp,
                                        cluster_ranks_exp=cluster_ranks_exp, tp=tensor_model_parallel_size, 
                                        dp=data_parallel_size, pp=pipeline_model_parallel_size, 
                                        ep_model_degree=expert_model_parallel_size, ep_data_degree=expert_data_parallel_size)
            assert res==locals()[f'ground_truth_{world_size}']
    
    def test_ascending_descending_ring_pg_group_creation(self):
        ground_truth_64 = ParallelGroups(
            tp_groups=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            ],
            dp_groups=[
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
                [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63],
                [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
                [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47]
            ],
            pp_groups=[
                [0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23],
                [8, 24], [9, 25], [10, 26], [11, 27], [12, 28], [13, 29], [14, 30], [15, 31],
                [48, 32], [49, 33], [50, 34], [51, 35], [52, 36], [53, 37], [54, 38], [55, 39],
                [56, 40], [57, 41], [58, 42], [59, 43], [60, 44], [61, 45], [62, 46], [63, 47]
            ],
            ep_model_groups=[
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
                [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
                [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],
                [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63]
            ],
            ep_data_groups=[
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
                [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
                [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],
                [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63]
            ]
        )
        ground_truth_128 = ParallelGroups(
            tp_groups=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
            ],
            dp_groups=[
                [0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23],
                [8, 24], [9, 25], [10, 26], [11, 27], [12, 28], [13, 29], [14, 30], [15, 31],
                [48, 32], [49, 33], [50, 34], [51, 35], [52, 36], [53, 37], [54, 38], [55, 39],
                [56, 40], [57, 41], [58, 42], [59, 43], [60, 44], [61, 45], [62, 46], [63, 47],
                [64, 80], [65, 81], [66, 82], [67, 83], [68, 84], [69, 85], [70, 86], [71, 87],
                [72, 88], [73, 89], [74, 90], [75, 91], [76, 92], [77, 93], [78, 94], [79, 95],
                [112, 96], [113, 97], [114, 98], [115, 99], [116, 100], [117, 101], [118, 102], [119, 103],
                [120, 104], [121, 105], [122, 106], [123, 107], [124, 108], [125, 109], [126, 110], [127, 111]
            ],
            pp_groups=[
                [0, 64], [16, 80], [1, 65], [17, 81], [2, 66], [18, 82], [3, 67], [19, 83],
                [4, 68], [20, 84], [5, 69], [21, 85], [6, 70], [22, 86], [7, 71], [23, 87],
                [8, 72], [24, 88], [9, 73], [25, 89], [10, 74], [26, 90], [11, 75], [27, 91],
                [12, 76], [28, 92], [13, 77], [29, 93], [14, 78], [30, 94], [15, 79], [31, 95],
                [48, 112], [32, 96], [49, 113], [33, 97], [50, 114], [34, 98], [51, 115], [35, 99],
                [52, 116], [36, 100], [53, 117], [37, 101], [54, 118], [38, 102], [55, 119], [39, 103],
                [56, 120], [40, 104], [57, 121], [41, 105], [58, 122], [42, 106], [59, 123], [43, 107],
                [60, 124], [44, 108], [61, 125], [45, 109], [62, 126], [46, 110], [63, 127], [47, 111]
            ],
            ep_model_groups=[
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
                [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
                [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],
                [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63],
                [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79],
                [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95],
                [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111],
                [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127]
            ],
            ep_data_groups=[
                [0, 32], [1, 33], [2, 34], [3, 35], [4, 36], [5, 37], [6, 38], [7, 39],
                [8, 40], [9, 41], [10, 42], [11, 43], [12, 44], [13, 45], [14, 46], [15, 47],
                [16, 48], [17, 49], [18, 50], [19, 51], [20, 52], [21, 53], [22, 54], [23, 55],
                [24, 56], [25, 57], [26, 58], [27, 59], [28, 60], [29, 61], [30, 62], [31, 63],
                [64, 96], [65, 97], [66, 98], [67, 99], [68, 100], [69, 101], [70, 102], [71, 103],
                [72, 104], [73, 105], [74, 106], [75, 107], [76, 108], [77, 109], [78, 110], [79, 111],
                [80, 112], [81, 113], [82, 114], [83, 115], [84, 116], [85, 117], [86, 118], [87, 119],
                [88, 120], [89, 121], [90, 122], [91, 123], [92, 124], [93, 125], [94, 126], [95, 127]
            ]
        )

        ground_truth_256 = ParallelGroups(
            tp_groups=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191],
                [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175],
                [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255],
                [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239]
            ],
            dp_groups=[
                [0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23],
                [8, 24], [9, 25], [10, 26], [11, 27], [12, 28], [13, 29], [14, 30], [15, 31],
                [48, 32], [49, 33], [50, 34], [51, 35], [52, 36], [53, 37], [54, 38], [55, 39],
                [56, 40], [57, 41], [58, 42], [59, 43], [60, 44], [61, 45], [62, 46], [63, 47],
                [64, 80], [65, 81], [66, 82], [67, 83], [68, 84], [69, 85], [70, 86], [71, 87],
                [72, 88], [73, 89], [74, 90], [75, 91], [76, 92], [77, 93], [78, 94], [79, 95],
                [112, 96], [113, 97], [114, 98], [115, 99], [116, 100], [117, 101], [118, 102], [119, 103],
                [120, 104], [121, 105], [122, 106], [123, 107], [124, 108], [125, 109], [126, 110], [127, 111],
                [128, 144], [129, 145], [130, 146], [131, 147], [132, 148], [133, 149], [134, 150], [135, 151],
                [136, 152], [137, 153], [138, 154], [139, 155], [140, 156], [141, 157], [142, 158], [143, 159],
                [176, 160], [177, 161], [178, 162], [179, 163], [180, 164], [181, 165], [182, 166], [183, 167],
                [184, 168], [185, 169], [186, 170], [187, 171], [188, 172], [189, 173], [190, 174], [191, 175],
                [192, 208], [193, 209], [194, 210], [195, 211], [196, 212], [197, 213], [198, 214], [199, 215],
                [200, 216], [201, 217], [202, 218], [203, 219], [204, 220], [205, 221], [206, 222], [207, 223],
                [240, 224], [241, 225], [242, 226], [243, 227], [244, 228], [245, 229], [246, 230], [247, 231],
                [248, 232], [249, 233], [250, 234], [251, 235], [252, 236], [253, 237], [254, 238], [255, 239]
            ],
            pp_groups=[
                [0, 64, 128, 192], [16, 80, 144, 208], [1, 65, 129, 193], [17, 81, 145, 209],
                [2, 66, 130, 194], [18, 82, 146, 210], [3, 67, 131, 195], [19, 83, 147, 211],
                [4, 68, 132, 196], [20, 84, 148, 212], [5, 69, 133, 197], [21, 85, 149, 213],
                [6, 70, 134, 198], [22, 86, 150, 214], [7, 71, 135, 199], [23, 87, 151, 215],
                [8, 72, 136, 200], [24, 88, 152, 216], [9, 73, 137, 201], [25, 89, 153, 217],
                [10, 74, 138, 202], [26, 90, 154, 218], [11, 75, 139, 203], [27, 91, 155, 219],
                [12, 76, 140, 204], [28, 92, 156, 220], [13, 77, 141, 205], [29, 93, 157, 221],
                [14, 78, 142, 206], [30, 94, 158, 222], [15, 79, 143, 207], [31, 95, 159, 223],
                [48, 112, 176, 240], [32, 96, 160, 224], [49, 113, 177, 241], [33, 97, 161, 225],
                [50, 114, 178, 242], [34, 98, 162, 226], [51, 115, 179, 243], [35, 99, 163, 227],
                [52, 116, 180, 244], [36, 100, 164, 228], [53, 117, 181, 245], [37, 101, 165, 229],
                [54, 118, 182, 246], [38, 102, 166, 230], [55, 119, 183, 247], [39, 103, 167, 231],
                [56, 120, 184, 248], [40, 104, 168, 232], [57, 121, 185, 249], [41, 105, 169, 233],
                [58, 122, 186, 250], [42, 106, 170, 234], [59, 123, 187, 251], [43, 107, 171, 235],
                [60, 124, 188, 252], [44, 108, 172, 236], [61, 125, 189, 253], [45, 109, 173, 237],
                [62, 126, 190, 254], [46, 110, 174, 238], [63, 127, 191, 255], [47, 111, 175, 239]
            ],
            ep_model_groups=[
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
            [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
            [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63],
            [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79],
            [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95],
            [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111],
            [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127],
            [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143],
            [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159],
            [160], [161], [162], [163], [164], [165], [166], [167], [168], [169], [170], [171], [172], [173], [174], [175],
            [176], [177], [178], [179], [180], [181], [182], [183], [184], [185], [186], [187], [188], [189], [190], [191],
            [192], [193], [194], [195], [196], [197], [198], [199], [200], [201], [202], [203], [204], [205], [206], [207],
            [208], [209], [210], [211], [212], [213], [214], [215], [216], [217], [218], [219], [220], [221], [222], [223],
            [224], [225], [226], [227], [228], [229], [230], [231], [232], [233], [234], [235], [236], [237], [238], [239],
            [240], [241], [242], [243], [244], [245], [246], [247], [248], [249], [250], [251], [252], [253], [254], [255]
        ],
        ep_data_groups=[
            [0, 32], [1, 33], [2, 34], [3, 35], [4, 36], [5, 37], [6, 38], [7, 39],
            [8, 40], [9, 41], [10, 42], [11, 43], [12, 44], [13, 45], [14, 46], [15, 47],
            [16, 48], [17, 49], [18, 50], [19, 51], [20, 52], [21, 53], [22, 54], [23, 55],
            [24, 56], [25, 57], [26, 58], [27, 59], [28, 60], [29, 61], [30, 62], [31, 63],
            [64, 96], [65, 97], [66, 98], [67, 99], [68, 100], [69, 101], [70, 102], [71, 103],
            [72, 104], [73, 105], [74, 106], [75, 107], [76, 108], [77, 109], [78, 110], [79, 111],
            [80, 112], [81, 113], [82, 114], [83, 115], [84, 116], [85, 117], [86, 118], [87, 119],
            [88, 120], [89, 121], [90, 122], [91, 123], [92, 124], [93, 125], [94, 126], [95, 127],
            [128, 160], [129, 161], [130, 162], [131, 163], [132, 164], [133, 165], [134, 166], [135, 167],
            [136, 168], [137, 169], [138, 170], [139, 171], [140, 172], [141, 173], [142, 174], [143, 175],
            [144, 176], [145, 177], [146, 178], [147, 179], [148, 180], [149, 181], [150, 182], [151, 183],
            [152, 184], [153, 185], [154, 186], [155, 187], [156, 188], [157, 189], [158, 190], [159, 191],
            [192, 224], [193, 225], [194, 226], [195, 227], [196, 228], [197, 229], [198, 230], [199, 231],
            [200, 232], [201, 233], [202, 234], [203, 235], [204, 236], [205, 237], [206, 238], [207, 239],
            [208, 240], [209, 241], [210, 242], [211, 243], [212, 244], [213, 245], [214, 246], [215, 247],
            [216, 248], [217, 249], [218, 250], [219, 251], [220, 252], [221, 253], [222, 254], [223, 255]
        ]
        )

        # 1 node and 2 node of trn2, for first world_size of 64 we test tp32, pp2, 
        # for world size 128, we have tp32, pp2, dp2
        # for 256 world size thats our 4 node final config. TP32, PP4, DP2
        for world_size in [64,128,256]: 
            with patch('torch.distributed.get_world_size', return_value=world_size):
                tensor_model_parallel_size = 32
                pipeline_model_parallel_size = 2 if world_size<256 else 4
                expert_model_parallel_size = 1
                data_parallel_size = world_size//(tensor_model_parallel_size*pipeline_model_parallel_size)
                cluster_ranks = torch.arange(0, world_size)
                expert_data_parallel_size: int = world_size // (
                    tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size
                )
                cluster_ranks_exp = cluster_ranks.reshape(
                    [
                        pipeline_model_parallel_size,
                        expert_data_parallel_size,
                        expert_model_parallel_size,
                        tensor_model_parallel_size,  # important: contiguous parallelism dimension
                    ]
                )
                cluster_ranks_nonexp = cluster_ranks.reshape(
                    [
                        pipeline_model_parallel_size,
                        data_parallel_size,
                        tensor_model_parallel_size,  # important: contiguous parallelism dimension
                    ]
                )
                res = ascending_descending_ring_PG_group(lnc_size=1, cluster_ranks_nonexp= cluster_ranks_nonexp,
                                            cluster_ranks_exp=cluster_ranks_exp, tp=tensor_model_parallel_size, 
                                            dp=data_parallel_size, pp=pipeline_model_parallel_size, 
                                            ep_model_degree=expert_model_parallel_size, ep_data_degree=expert_data_parallel_size)                
                assert res==locals()[f'ground_truth_{world_size}']
    
    def test_pp_rank_in_group(self):
        with patch(f"{MODULE}._REPLICA_LOGIC", PG_Group_Logic.LOGIC2):
            with patch(f"{MODULE}._PIPELINE_MODEL_PARALLEL_GROUP_SPMD",[[0,16],[48,32],[63,47,31,15]]):
                ground_truth_results_trn2 = [0,1,0,1,2,3]
                results = []
                for i,cur_rank in enumerate([48,32,63,47,31,15]):
                    with patch('torch.distributed.get_rank',return_value=cur_rank):           
                        results.append(get_pipeline_model_parallel_rank())
                assert results == ground_truth_results_trn2

    def test_dp_rank_in_group(self):
        with patch(f"{MODULE}._REPLICA_LOGIC", PG_Group_Logic.LOGIC2):
            with patch(f"{MODULE}._DATA_PARALLEL_GROUP_SPMD",[[0,16],[48,32],[63,47,31,15]]):
                ground_truth_results_trn2 = [0,1,0,1,2,3]
                results = []
                for i,cur_rank in enumerate([48,32,63,47,31,15]):
                    with patch('torch.distributed.get_rank',return_value=cur_rank):           
                        results.append(get_data_parallel_rank())
                assert results == ground_truth_results_trn2

if __name__ == "__main__":
    unittest.main()