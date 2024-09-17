import argparse
import os
import sys
from json import dump, loads


def parse_common_options(logdir=None, num_cores=None, num_workers=0, opts=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--logdir", type=str, default=logdir)
    parser.add_argument("--num_cores", type=int, default=num_cores)
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--metrics_debug", action="store_true")
    parser.add_argument("--async_closures", action="store_true")
    if opts:
        for name, aopts in opts:
            parser.add_argument(name, **aopts)
    args, leftovers = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + leftovers
    # Setup import folders.
    xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path.append(os.path.join(os.path.dirname(xla_folder), "test"))
    return args


FLAGS = parse_common_options()
