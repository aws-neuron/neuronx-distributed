#!/bin/bash

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export NXD_DIR=$(dirname $(dirname $(dirname $(dirname $SCRIPT_DIR))))
export UNIT_TEST_DIR=$NXD_DIR/test/unit_test/modules/moe

export PYTHONPATH=$NXD_DIR/src:$NXD_DIR:$UNIT_TEST_DIR:$PYTHONPATH
echo $PYTHONPATH

python3 $SCRIPT_DIR/test_moe_spmd.py