#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <conda-env-name> <project-directory>" >&2
  echo "Example: $0 CenterPoint CenterPoint" >&2
  exit 2
fi

ENV_NAME="$1"
PROJECT_DIR="$2"
REPO_ROOT="/mnt/e/Scholar/mmradarDetect"
CONDA_PREFIX="/home/yuehui/miniforge3/envs/${ENV_NAME}"
PYTHON="${CONDA_PREFIX}/bin/python"
PROJECT_ROOT="${REPO_ROOT}/${PROJECT_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  exit 3
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1

compile_op() {
  local rel="$1"
  echo "==> [${ENV_NAME}] compiling ${PROJECT_DIR}/${rel}"
  cd "${PROJECT_ROOT}/${rel}"
  rm -rf build
  "${PYTHON}" setup.py build_ext --inplace
}

case "${PROJECT_DIR}" in
  CenterPoint)
    compile_op det3d/ops/iou3d_nms
    compile_op det3d/ops/dcn
    ;;
  PillarNet-LTS)
    compile_op det3d/ops/iou3d_nms
    compile_op det3d/ops/pillar_ops
    compile_op det3d/ops/pillar_ops-ba
    compile_op det3d/ops/roiaware_pool3d
    ;;
  *)
    echo "Unsupported Det3D project: ${PROJECT_DIR}" >&2
    exit 4
    ;;
esac

echo "==> [${ENV_NAME}] ${PROJECT_DIR} Det3D CUDA ops compiled"
