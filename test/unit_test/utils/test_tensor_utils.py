# Standard Library
import itertools
import unittest

# Third Party
from parameterized import parameterized
import torch

from neuronx_distributed.utils.tensor_utils import cumsum

from .. import update_result


def get_cumsum_test_cases():
    testcase_tensor_shapes = [
        (3, 2),
        (2, 3),
        (5, 5),
        (10, 10),
        (100, 2048),
        (2048, 100),
        (2048, 2048),
        (2048, 5000),
        (5000, 2028),
        (10000, 10000),
        (50000, 100),
        (100, 75000),
    ]
    testcase_dtypes = [torch.float32, torch.bfloat16]
    test_cases = itertools.product(testcase_tensor_shapes, testcase_dtypes)
    return test_cases


class TestCumSum(unittest.TestCase):

    @parameterized.expand(get_cumsum_test_cases())
    def test_cumsum(self, tensor_shape, dtype):
        try:
            # Set random seed for reproducibility
            torch.manual_seed(tensor_shape[0] * tensor_shape[1])
            # Generate random 0-1 matrix
            ip = torch.randint(high=2, size=tensor_shape, dtype=dtype)
            op = cumsum(ip)
            op_gt = torch.cumsum(ip, dim=0)
            # Check that outputs match
            torch.testing.assert_close(op, op_gt)
        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main(verbosity=3)
