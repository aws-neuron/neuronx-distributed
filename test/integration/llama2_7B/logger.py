import inspect
import os
import sys
import time

import numpy as np
import requests
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def load_events(event_file):
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()

    data = {}
    for tag in tags["scalars"]:
        data[tag] = accumulator.Scalars(tag)
    return data


class Logger:
    def __init__(self, args, world_size, model_dtype):
        xla = "torch_xla" in sys.modules
        self.throughputs = []
        dtype_short = model_dtype.replace("torch.", "")
        self.tb = SummaryWriter(
            os.path.join(
                args.output_dir,
                f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                f"_{dtype_short}"
                f"_w{world_size}"
                f"_lr{args.lr}"
                f"_bs{args.batch_size}"
                f"_acc{args.grad_accum_usteps}"
                f"_warmup{args.warmup_steps}"
                f"_max{args.max_steps}"
                f"_xla{xla}"
                f"_{self.get_instance_type()}",
            )
        )
        self.tb.add_text("script", "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)

        self.golden_steploss = []
        event_file = os.getenv("GOLDEN_EVENT_FILE") if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) else None
        if event_file is not None:
            data = load_events(event_file)
            for step in data["step loss"]:
                self.golden_steploss.append(step.value)
        else:
            golden = "golden_steploss.txt"
            if os.path.exists(golden):
                with open(golden, "r") as f:
                    self.golden_steploss = [float(i) for i in f]
                print(f"Read {len(self.golden_steploss)} golden step loss values from {golden}")

    def get_instance_type(self):
        try:
            token = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )
            data = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token.text},
            )
            return data.text
        except Exception:
            return os.environ.get("HOSTNAME", "unknown")

    def log(self, epoch, step, step_loss, learning_rate, throughput, grad_norm=None):
        time_now = time.asctime()
        grad_norm_msg = f"grad-norm : {grad_norm}" if grad_norm else ""
        print(
            f"LOG {time_now} - ({epoch}, {step}) step_loss : {step_loss:.4f} "
            f"learning_rate : {learning_rate:.2e} throughput : {throughput:.2f} seq/s "
            f"{grad_norm_msg}",
            flush=True,
        )
        self.tb.add_scalar("step loss", step_loss, step)
        self.tb.add_scalar("learning rate", learning_rate, step)
        self.tb.add_scalar("throughput", throughput, step)
        if grad_norm:
            self.tb.add_scalar("grad-norm", grad_norm, step)
        self.throughputs.append(throughput)

        # Comparing Loss to Golden
        if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            step_0start = step - 1
            if step_0start < len(self.golden_steploss) and step_0start >= 0:
                np.testing.assert_allclose(step_loss, self.golden_steploss[step_0start], rtol=1.5e-1)
