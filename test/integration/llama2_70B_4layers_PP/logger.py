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
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None
        
        self.gpu_losses_to_compare = None
        if os.getenv("GOLDEN_EVENT_FILE", None) is not None and os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) is None:
            self.gpu_losses_to_compare = torch.load(os.getenv("GOLDEN_EVENT_FILE"))


    def log(self, total_steps, loss, global_norm, current_lr, input_ids, throughput, start):
        end = time.time()
        iteration_time = end - start
        tps = throughput.get_throughput()
        print(
            f"step {total_steps} step_time {iteration_time}s throughput {tps} seq/s loss {loss.detach().cpu().item()} grad norm {global_norm.item() if global_norm is not None else None}"
        )
        if self.writer is not None:
            self.writer.add_scalar("loss", loss.item(), total_steps)
            if global_norm is not None:
                self.writer.add_scalar(
                    "global_norm", global_norm.item(), total_steps
                )
            self.writer.add_scalar("lr", current_lr, total_steps)
            self.writer.add_scalar("iteration_time", iteration_time, total_steps)
            self.writer.add_scalar("throughput", tps, total_steps)
            self.writer.add_scalar(
                "input_ids",
                torch.sum(input_ids.detach().cpu()).item(),
                total_steps,
            )
        
        if self.gpu_losses_to_compare is not None:
            if total_steps < len(self.gpu_losses_to_compare):
                if not torch.allclose(
                    loss.cpu().float(), self.gpu_losses_to_compare[total_steps].float(), rtol=2.3e-1
                ):
                    raise RuntimeError(
                        f"Loss mismtach with golden, Trn {loss.item()} GPU {self.gpu_losses_to_compare[total_steps].item()}"
                    )