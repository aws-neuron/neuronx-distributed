import os

import torch
import torch_xla.utils.serialization as xser

for dp in range(4):
    for tp in range(8):
        file_name = "dp_rank_{:02d}_tp_rank_{:02d}_pp_rank_00.pt".format(dp, tp)
        ckpt0 = xser.load(os.path.join("test_data", "optim", file_name))
        ckpt1 = xser.load(os.path.join("zero1_sharder", "optim", file_name))
        torch.testing.assert_close(ckpt0, ckpt1, rtol=0, atol=0)
