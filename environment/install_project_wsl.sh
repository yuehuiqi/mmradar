#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <conda-env-name> <project-directory>" >&2
  exit 2
fi

ENV_NAME="$1"
PROJECT_DIR="$2"
PREFIX="/home/yuehui/miniforge3/envs/${ENV_NAME}"

export PATH="${PREFIX}/bin:${PATH}"
export CUDA_HOME="${PREFIX}"
export CPATH="${PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${PREFIX}/targets/x86_64-linux/lib:${PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${PREFIX}/targets/x86_64-linux/lib:${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS="${MAX_JOBS:-4}"

cd "/mnt/e/Scholar/mmradarDetect/${PROJECT_DIR}"
python -m pip install -e . --no-build-isolation
