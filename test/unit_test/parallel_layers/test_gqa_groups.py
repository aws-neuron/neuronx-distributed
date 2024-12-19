import unittest

from neuronx_distributed.parallel_layers.parallel_state import arrange_kv_groups
from neuronx_distributed.utils.utils import hardware


class GQAParallelGroupTest(unittest.TestCase):
 
    def test_gqa_pg_logic(self):
        """

        On Trn1:
            For TP32 with kv_replication 4: (K0,K1,K2,K3)(K0,K1,K2,K3)…. 4 times and each nearby ranks holding different K heads
            actual kv_replication_groups: [
                [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27],
                [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]
            ]
        On Trn2: We replicate single kv head on adjacent ranks (specific to Trn2 currently)
            For TP32 with kv_replication 4: (K0,K0,K0….. 4 times)(K1,K1,……. 4 times) … for all K heads
            actual kv_replication_groups: [
                [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]
            ]
        
        """   
        # trn1
        inputs1 = {"num_tensor_model_parallel_groups": 1, "tensor_model_parallel_size": 32,"kv_shared_group_size": 4, "hardware_type": hardware.TRN1}
        
        # trn2
        inputs2 = {"num_tensor_model_parallel_groups":1 ,"tensor_model_parallel_size": 128, "kv_shared_group_size": 16, "hardware_type": hardware.TRN2}
        inputs3 = {"num_tensor_model_parallel_groups":1 ,"tensor_model_parallel_size": 64, "kv_shared_group_size": 8, "hardware_type": hardware.TRN2}
        inputs4 = {"num_tensor_model_parallel_groups":1 ,"tensor_model_parallel_size": 32, "kv_shared_group_size": 4, "hardware_type": hardware.TRN2}
        inputs5 = {"num_tensor_model_parallel_groups":1 ,"tensor_model_parallel_size": 16, "kv_shared_group_size": 2, "hardware_type": hardware.TRN2}


        res = arrange_kv_groups(**inputs1)
        expected1 = [[i + j*8 for j in range(4)] for i in range(8)]
        assert res == expected1

        res = arrange_kv_groups(**inputs2)
        expected2 = [
            list(range(0, 16)), list(range(16, 32)), list(range(32, 48)), list(range(48, 64)),
            list(range(64, 80)), list(range(80, 96)), list(range(96, 112)), list(range(112, 128))
        ]
        assert res == expected2

        res = arrange_kv_groups(**inputs3)
        expected3 = [
            list(range(0, 8)), list(range(8, 16)), list(range(16, 24)), list(range(24, 32)),
            list(range(32, 40)), list(range(40, 48)), list(range(48, 56)), list(range(56, 64))
        ]
        assert res == expected3

        res = arrange_kv_groups(**inputs4)
        expected4 = [list(range(i, i+4)) for i in range(0, 32, 4)]
        assert res == expected4

        res = arrange_kv_groups(**inputs5)
        expected5 = [list(range(i, i+2)) for i in range(0, 16, 2)]
        assert res == expected5
      
if __name__ == "__main__":
    unittest.main()