import os
import unittest

import torch

from neuronx_distributed import parallel_layers
from neuronx_distributed.modules.moe.blockwise import can_use_blockwise_matmul_nki
from neuronx_distributed.utils.model_utils import get_platform_lnc

if not torch.distributed.is_initialized():
     os.environ["MASTER_ADDR"] = "localhost"
     os.environ["MASTER_PORT"] = "12355"
     os.environ["RANK"] = "0"
     os.environ["WORLD_SIZE"] = "1"
     torch.distributed.init_process_group(backend="xla", init_method="env://")
     parallel_layers.parallel_state.initialize_model_parallel(
         tensor_model_parallel_size=1,
         pipeline_model_parallel_size=1,
     )

def _generate_test_configs():
    test_config1 = {"hidden_size": 4096,
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": True,
                    "use_torch_block_wise": False,
                    "device": "xla",
                    "result": True
                    }
    test_config2 = {"hidden_size": 4096,
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": True,
                    "use_torch_block_wise": False,
                    "device": "cpu",
                    "result": False # should return false because of device type
                    }
    test_config3 = {"hidden_size": 4096,
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": True,
                    "use_torch_block_wise": True, # should return false because of using torch blockwise
                    "device": "xla",
                    "result": False
                    }
    test_config4 = {"hidden_size": 4096,
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": False, # should return false because of glu_mlp is false
                    "use_torch_block_wise": False,
                    "device": "xla",
                    "result": False
                    }
    test_config5 = {"hidden_size": 4096,
                    "intermediate_size_tp": 2048,
                    "block_size": 16, # should return false because of unsupported block size
                    "glu_mlp": True,
                    "use_torch_block_wise": False,
                    "device": "xla",
                    "result": False
                    }
    test_config6 = {"hidden_size": 10240, # should return false because of unsupported hidden_size [4096, 8192]
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": True,
                    "use_torch_block_wise": False,
                    "device": "xla",
                    "result": False
                    }

    test_config7 = {"hidden_size": 5122, # should return false because of unsupported hidden_size not divisible by PSUM size
                    "intermediate_size_tp": 2048,
                    "block_size": 512,
                    "glu_mlp": True,
                    "use_torch_block_wise": False,
                    "device": "xla",
                    "result": False
                    }

    test_configs = []
    test_configs.append(test_config1)
    test_configs.append(test_config2)
    test_configs.append(test_config3)
    test_configs.append(test_config4)
    test_configs.append(test_config5)
    test_configs.append(test_config6)
    test_configs.append(test_config7)
    return test_configs

class BlockWiseNkiAvailabilityTest(unittest.TestCase):
    def test_can_use_blockwise_nki(self):
        for test_config in _generate_test_configs():
            hidden_size = test_config["hidden_size"]
            intermediate_size_tp = test_config["intermediate_size_tp"]
            block_size = test_config["block_size"]
            glu_mlp = test_config["glu_mlp"]
            use_torch_block_wise = test_config["use_torch_block_wise"]
            device = torch.device(test_config["device"])
            lnc = test_config.get("logical_nc_config", get_platform_lnc())
            use_block_parallel = test_config.get("use_block_parallel", False)
            res = can_use_blockwise_matmul_nki(hidden_size, intermediate_size_tp, block_size, glu_mlp, use_torch_block_wise, device, lnc, use_block_parallel)
            assert res == test_config["result"]

if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)