#! /bin/bash
set -e

: ${DIST_DIR:=build}

python -m pip install ruff
# removing cache fails in ToD
python -m ruff check --no-cache --line-length=120 --ignore=F401,E203
# exit when asked to run `ruff` only
if [[ "$1" == "ruff" ]]
then
  exit 0
fi

# Run static code analysis
python -m pip install mypy
# Install type bindings
python -m pip install types-requests boto3-stubs[s3] types-PyYAML
# Dependencies for s3 checkpoint storage
python -m pip install tenacity
# removing cache fails in ToD
python -m mypy --no-incremental --cache-dir=/dev/null
# exit when asked to run `mypy` only
if [[ "$1" == "mypy" ]]
then
  exit 0
fi



# Build wheel
python setup.py bdist_wheel --dist-dir ${DIST_DIR}
