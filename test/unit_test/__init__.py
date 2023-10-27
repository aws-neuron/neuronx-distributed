import os
import sys
import argparse
from json import loads, dump

def parse_common_options(logdir=None,
                         num_cores=None,
                         num_workers=0,
                         opts=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--logdir', type=str, default=logdir)
  parser.add_argument('--num_cores', type=int, default=num_cores)
  parser.add_argument('--num_workers', type=int, default=num_workers)
  parser.add_argument('--metrics_debug', action='store_true')
  parser.add_argument('--async_closures', action='store_true')
  parser.add_argument('--test_json', required=False, help='input json listing the test spec for network to compile')
  parser.add_argument('--s3_dir', required=False, help='location to upload all test artifacts')
  parser.add_argument('--s3_bucket', default='neuron-canary-nn-models')
  if opts:
    for name, aopts in opts:
      parser.add_argument(name, **aopts)
  args, leftovers = parser.parse_known_args()
  sys.argv = [sys.argv[0]] + leftovers
  # Setup import folders.
  xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
  return args

def update_result(results):
    data[test_name].update(results)
    os.system(f'rm {FLAGS.test_json}')
    with open(FLAGS.test_json, 'w+') as file:
        dump(data, file)

FLAGS = parse_common_options()
with open(FLAGS.test_json) as file:
    data = loads(file.read())
test_name = next(iter(data))
update_result({"inference_success": 1})

