import os
import time

import torch


class Logger:
    def __init__(self, args, should_print=False):
        if should_print and args.tb_dir != "":
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = args.tb_dir
            import shutil

            exist = os.path.exists(tb_dir)
            if exist:
                shutil.rmtree(tb_dir)
            self.writer = SummaryWriter(os.path.join(
                tb_dir,
                f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                f"_lr{args.lr}"
                f"_bs{args.num_microbatches}"
                f"_acc{args.train_batch_size}"
                f"_warmup{args.warmup_steps}"
                f"_max{args.max_steps}"
            ))
        else:
            self.writer = None
        self.throughputs = []

    def log(self, total_steps, loss, global_norm, current_lr, input_ids, throughput, start):
        end = time.time()
        iteration_time = end - start
        tps = throughput.get_throughput()
        print(
            f"step {total_steps} step_time {iteration_time}s throughput {tps} seq/s step loss {loss.detach().cpu().item()} grad norm {global_norm.item() if global_norm is not None else None}"
        )
        if self.writer is not None:
            self.writer.add_scalar("step loss", loss.item(), total_steps)
            if global_norm is not None:
                self.writer.add_scalar("global_norm", global_norm.item(), total_steps)
            self.writer.add_scalar("lr", current_lr, total_steps)
            self.writer.add_scalar("iteration_time", iteration_time, total_steps)
            self.writer.add_scalar("throughput", tps, total_steps)
            self.writer.add_scalar(
                "input_ids",
                torch.sum(input_ids.detach().cpu()).item(),
                total_steps,
            )
        self.throughputs.append(tps)
        throughput.reset_time()
