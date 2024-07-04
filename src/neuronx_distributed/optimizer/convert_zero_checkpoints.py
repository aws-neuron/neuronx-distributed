import argparse
import concurrent.futures
import copy
import os
import re
import shutil
import time

import torch
import torch.nn.functional as F
import torch_xla.utils.serialization as xser


def is_full(args):
    opt_path = os.path.join(args.input_dir, "optim")
    for p in os.listdir(opt_path):
        if p.startswith("full"):
            return True
    return False


def is_xser(args):
    opt_path = os.path.join(args.input_dir, "optim")
    for p in os.listdir(opt_path):
        p = os.path.join(opt_path, p)
        if os.path.isdir(p):
            if p.endswith(".tensors"):
                return True
    return False


def get_parallel_info(args):
    dp_size = 0
    tp_size = 0
    pp_size = 0
    for p in os.listdir(os.path.join(args.input_dir, "optim")):
        if "full" in p:
            tp, pp = re.findall(r"\d+", p)
            dp_size = -1
        else:
            dp, tp, pp = re.findall(r"\d+", p)
        if dp_size != -1 and int(dp) > dp_size:
            dp_size = int(dp)
        if int(tp) > tp_size:
            tp_size = int(tp)
        if int(pp) > pp_size:
            pp_size = int(pp)
    dp_size += 1
    tp_size += 1
    pp_size += 1
    return dp_size, tp_size, pp_size


def merge_optim_dp_checkpoints(args, tp_rank, pp_rank):
    partial_ckpts = []
    for dp_rank in range(args.dp_size):
        file_name = "dp_rank_{:02d}_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(dp_rank, tp_rank, pp_rank)
        if args.is_xser:
            partial_ckpts.append(xser.load(os.path.join(args.input_dir, "optim", file_name)))
        else:
            partial_ckpts.append(torch.load(os.path.join(args.input_dir, "optim", file_name)))

    merged_ckpt = copy.deepcopy(partial_ckpts[0])
    del merged_ckpt["base_state"]

    def _merge(values, shape):
        if isinstance(values[0], torch.Tensor):
            # concat
            result = torch.cat(values)
            # unpad
            if result.shape != shape:
                result = result[: shape[0]]
            assert list(result.shape) == shape
            return result
        elif isinstance(values[0], dict):
            result = {}
            for k in values[0].keys():
                if k == "step":
                    result[k] = values[0][k]
                else:
                    vs = [v[k] for v in values]
                    result[k] = _merge(vs, shape)
            return result
        elif isinstance(values[0], set):
            raise ValueError
        elif isinstance(values[0], (list, tuple)):
            raise ValueError
        else:
            return values[0]

    merged_base_state = {}
    for k, s in merged_ckpt["shape_info"].items():
        values = []
        for c in partial_ckpts:
            values.append(c["base_state"][k])
        merged_base_state[k] = _merge(values, s)
    merged_ckpt["base_state"] = merged_base_state

    return merged_ckpt


def split_and_save_ckpts(args, merged_ckpt, tp_rank, pp_rank):
    def _split(value, idx):
        if isinstance(value, torch.Tensor):
            # pad
            if value.size(0) % args.new_dp_size != 0:
                pad_size = args.new_dp_size - value.size(0) % args.new_dp_size
                value = F.pad(value, [0, 0] * (value.dim() - 1) + [0, pad_size])
            # split
            return value.chunk(args.new_dp_size)[idx].clone()
        elif isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k == "step":
                    result[k] = v
                else:
                    result[k] = _split(v, idx)
            return result
        elif isinstance(value, set):
            raise ValueError
        elif isinstance(value, (list, tuple)):
            result = []
            for v in value:
                result.append(_split(v, idx))
            return result
        else:
            return value

    def _run(merged_ckpt, dp_rank):
        splited_ckpt = _split(merged_ckpt, dp_rank)
        file_name = "dp_rank_{:02d}_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(dp_rank, tp_rank, pp_rank)
        if args.is_xser:
            xser.save(splited_ckpt, os.path.join(args.output_dir, "optim", file_name), master_only=False)
        else:
            torch.save(splited_ckpt, os.path.join(args.output_dir, "optim", file_name))

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        fs = []
        for dp_rank in range(args.new_dp_size):
            fs.append(executor.submit(_run, merged_ckpt, dp_rank))
        for f in fs:
            res = f.result()
            if res is not None:
                print(res)


def _sharded_to_full_task(args, tp_rank, pp_rank):
    print("Converting tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))
    full_ckpt = merge_optim_dp_checkpoints(args, tp_rank, pp_rank)
    file_name = "full_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(tp_rank, pp_rank)
    if args.is_xser:
        xser.save(full_ckpt, os.path.join(args.output_dir, "optim", file_name), master_only=False)
    else:
        torch.save(full_ckpt, os.path.join(args.output_dir, "optim", file_name))
    print("Converted to tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))


def _full_to_sharded_task(args, tp_rank, pp_rank):
    print("Converting tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))
    file_name = "full_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(tp_rank, pp_rank)
    if args.is_xser:
        full_ckpt = xser.load(os.path.join(args.input_dir, "optim", file_name))
    else:
        full_ckpt = torch.load(os.path.join(args.input_dir, "optim", file_name))
    split_and_save_ckpts(args, full_ckpt, tp_rank, pp_rank)
    print("Converted to tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))


def _sharded_to_sharded_task(args, tp_rank, pp_rank):
    print("Converting tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))
    full_ckpt = merge_optim_dp_checkpoints(args, tp_rank, pp_rank)
    split_and_save_ckpts(args, full_ckpt, tp_rank, pp_rank)
    print("Converted to tp-{:02d}/pp-{:02d}".format(tp_rank, pp_rank))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input optim states")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted optim states")
    parser.add_argument("--num_workers", type=int, default=1, help="Num of works to process checkpoints")
    parser.add_argument("--dp_size", type=int, default=None, help="New dp size to be used for sharding")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--convert_to_full", action="store_true", help="Convert optim states to full")
    group.add_argument("--convert_to_sharded", action="store_true", help="Convert optim states to sharded")

    args, _ = parser.parse_known_args()

    # get some aux infos
    args.is_xser = is_xser(args)
    args.new_dp_size = args.dp_size
    dp_size, tp_size, pp_size = get_parallel_info(args)
    args.dp_size = dp_size
    args.tp_size = tp_size
    args.pp_size = pp_size

    shutil.rmtree(os.path.join(args.output_dir, "optim"), ignore_errors=True)
    os.makedirs(os.path.join(args.output_dir, "optim"), exist_ok=True)

    task = None
    if args.convert_to_full:
        if is_full(args):
            raise ValueError("Invalid inputs: convert full optim states to full optim states")
        else:
            task = _sharded_to_full_task
    elif args.convert_to_sharded:
        if is_full(args):
            task = _full_to_sharded_task
        else:
            task = _sharded_to_sharded_task

    print("Task {} started.".format(task.__name__))
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for tp_rank in range(args.tp_size):
            for pp_rank in range(args.pp_size):
                futures.append(executor.submit(task, args, tp_rank, pp_rank))
        for f in futures:
            res = f.result()
            if res is not None:
                print(res)
    print("Task {} done.".format(task.__name__))
    print("Time spent: {}".format(time.time() - start))


if __name__ == "__main__":
    main()
