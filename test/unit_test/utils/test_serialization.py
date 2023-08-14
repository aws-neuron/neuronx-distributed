# Standard Library
import unittest

# Third Party
import torch
from torch import device

from neuronx_distributed.utils.serialization import SerializationManager, TensorMeta


class TestSerialization(unittest.TestCase):
    def test_with_class_type(self):
        class A:
            def __init__(self, x):
                self.x = x

            def increment(self):
                self.x += 1

        data = A(3.2)
        s = SerializationManager()
        serialized, tx_list, tensor_meta = s.serialize(data)
        obj = s.deserialize(serialized, tx_list)
        self.assertTrue(data == obj)
        self.assertTrue(len(tx_list) == 0)
        self.assertTrue(len(tensor_meta) == 0)

    def test_with_tensor(self):
        data = torch.ones(2, 3)
        s = SerializationManager()
        serialized, tx_list, tensor_meta = s.serialize(data)
        expected_tensor_meta = [
            TensorMeta(tensor_index=0, dtype=torch.float32, shape=torch.Size([2, 3]), requires_grad=False, device=device(type='cpu'))
        ]
        obj = s.deserialize(serialized, tx_list)
        self.assertTrue(torch.equal(obj, data))
        self.assertTrue(tensor_meta == expected_tensor_meta)
        self.assertTrue(torch.equal(tx_list[0], torch.ones(2, 3)))

    def test_with_mixed_type(self):
        class MyClass:
            pass
        cls_type = MyClass()
        data = {"a": 1, "b": [torch.ones([2, 4]), torch.ones([2, 4]), (1, 2)], 
                "c" : (1, 2, torch.tensor(1.0)), "d": torch.zeros([2, 4]), "f": cls_type}
        s = SerializationManager()
        serialized, tx_list, tensor_meta = s.serialize(data)
        expected_tensor_meta = [
            TensorMeta(tensor_index=0, dtype=torch.float32, shape=torch.Size([2, 4]), requires_grad=False, device=device(type='cpu')), 
            TensorMeta(tensor_index=1, dtype=torch.float32, shape=torch.Size([2, 4]), requires_grad=False, device=device(type='cpu')), 
            TensorMeta(tensor_index=2, dtype=torch.float32, shape=torch.Size([]), requires_grad=False, device=device(type='cpu')), 
            TensorMeta(tensor_index=3, dtype=torch.float32, shape=torch.Size([2, 4]), requires_grad=False, device=device(type='cpu'))]
        expected_tx_list = [torch.ones([2, 4]), torch.ones([2, 4]), torch.tensor(1.0), torch.zeros([2, 4])]
        obj = s.deserialize(serialized, tx_list)
        self.assertTrue(data == obj)
        self.assertTrue(tensor_meta == expected_tensor_meta)
        for i in range(4):
            self.assertTrue(torch.equal(tx_list[i], expected_tx_list[i]))


if __name__ == "__main__":
    unittest.main()