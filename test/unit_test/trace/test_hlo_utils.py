import os
import unittest

import neuronx_distributed.trace.hlo_utils as hlo_utils

class TestHloUtils(unittest.TestCase):

    def setUp(self):
        self.parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test_data")
        self.hlo_txt_path = os.path.join(self.parent_dir, "dummy_hlo_nki_kernel.txt.hlo")
        self.hlo_module = hlo_utils.read_hlo(self.hlo_txt_path)
        self.id_to_computation = {cpt.id: cpt for cpt in self.hlo_module.computations}
        self.entry_computation = self.id_to_computation[self.hlo_module.entry_computation_id]
        self.idx_to_weight_name = {3: "dummy->weight_name->weight"}

    def test_get_nki_kernel_weight_names(self):
        nki_kernel_weight_names = hlo_utils.get_nki_kernel_weight_names(
            self.entry_computation, self.idx_to_weight_name, set(), self.id_to_computation
        )
        self.assertTrue(self.idx_to_weight_name[3] in nki_kernel_weight_names)

    def test_get_nki_kernel_weight_names_no_weight_names(self):
        idx_to_weight_name = dict()
        nki_kernel_weight_names = hlo_utils.get_nki_kernel_weight_names(
            self.entry_computation, idx_to_weight_name, set(), self.id_to_computation
        )
        self.assertEqual(nki_kernel_weight_names, set())

    def test_get_nki_kernel_weight_names_no_kernel_calls(self):
        hlo_txt_path = os.path.join(self.parent_dir, "dummy_hlo_no_kernel_calls.txt.hlo")
        hlo_module = hlo_utils.read_hlo(hlo_txt_path)
        id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
        entry_computation = id_to_computation[hlo_module.entry_computation_id]
        nki_kernel_weight_names = hlo_utils.get_nki_kernel_weight_names(
            entry_computation, self.idx_to_weight_name, set(), id_to_computation
        )
        self.assertEqual(nki_kernel_weight_names, set())


if __name__ == "__main__":
    unittest.main()
