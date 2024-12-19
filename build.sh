#! /bin/bash
set -e

: ${DIST_DIR:=build}

python3.8 -m pip install ruff
# removing cache fails in ToD
python3.8 -m ruff check --no-cache --line-length=120 --ignore=F401,E203
# exit when asked to run `ruff` only
if [[ "$1" == "ruff" ]]
then
  exit 0
fi

# Run static code analysis
python3.8 -m pip install mypy
# Install type bindings
python3.8 -m pip install types-requests boto3-stubs[s3] types-PyYAML
# removing cache fails in ToD
python3.8 -m mypy --no-incremental --cache-dir=/dev/null
# exit when asked to run `mypy` only
if [[ "$1" == "mypy" ]]
then
  exit 0
fi



# Build wheel
python3.8 setup.py bdist_wheel --dist-dir ${DIST_DIR}
