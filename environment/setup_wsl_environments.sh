#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
CONDA="${MINIFORGE_ROOT}/bin/conda"
MINIFORGE_VERSION="26.3.2-3"
MINIFORGE_SHA256="848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c"

ENV_NAMES=(
  CenterPoint
  DSVT
  InterFusion
  PointPillar
  PFANet
  PillarNetLTS
  VoxelNeXt
)

install_miniforge() {
  if [[ -x "${CONDA}" ]]; then
    return
  fi

  local installer="/tmp/Miniforge3-Linux-x86_64.sh"
  local url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh"

  curl -fL --retry 3 --output "${installer}" "${url}"
  echo "${MINIFORGE_SHA256}  ${installer}" | sha256sum --check --strict
  bash "${installer}" -b -p "${MINIFORGE_ROOT}"
}

environment_exists() {
  "${CONDA}" env list --json | "${MINIFORGE_ROOT}/bin/python" -c \
    'import json, os, sys; name=sys.argv[1]; data=json.load(sys.stdin); sys.exit(0 if any(os.path.basename(p)==name for p in data["envs"]) else 1)' \
    "$1"
}

install_miniforge
"${CONDA}" config --set auto_activate_base false
"${CONDA}" init bash >/dev/null

if ! environment_exists CenterPoint; then
  "${CONDA}" create --yes --name CenterPoint \
    python=3.10 pip=25.1 setuptools=69.5.1 wheel ninja cmake packaging
fi

"${CONDA}" install --yes --name CenterPoint --channel nvidia/label/cuda-12.8.1 \
  cuda-toolkit=12.8

SEED_PYTHON="${MINIFORGE_ROOT}/envs/CenterPoint/bin/python"
PIP_NETWORK_ARGS=(--timeout 180 --retries 10 --resume-retries 10)

"${SEED_PYTHON}" -m pip install "${PIP_NETWORK_ARGS[@]}" \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

"${SEED_PYTHON}" -m pip install "${PIP_NETWORK_ARGS[@]}" \
  numpy==1.23.5 \
  llvmlite==0.40.1 \
  numba==0.57.1 \
  scipy==1.10.1 \
  matplotlib==3.7.5 \
  opencv-python==4.8.1.78 \
  opencv-contrib-python==4.8.1.78 \
  Pillow==9.5.0 \
  pandas==2.0.3 \
  scikit-image==0.21.0 \
  scikit-learn==1.3.2 \
  Shapely==2.0.7 \
  protobuf==4.25.8 \
  tensorboardX \
  easydict \
  PyYAML \
  tqdm \
  pyquaternion \
  fire \
  pybind11 \
  terminaltables \
  addict \
  imagecorruptions \
  objgraph \
  cachetools \
  descartes \
  requests \
  SharedArray \
  cython \
  einops \
  timm==0.9.16 \
  yapf

for env_name in "${ENV_NAMES[@]:1}"; do
  if ! environment_exists "${env_name}"; then
    "${CONDA}" create --yes --name "${env_name}" --clone CenterPoint
  fi
done

for env_name in "${ENV_NAMES[@]}"; do
  env_python="${MINIFORGE_ROOT}/envs/${env_name}/bin/python"
  "${env_python}" - <<'PY'
import torch
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability()
assert (major, minor) == (12, 0), (major, minor)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (major, minor))
PY
done

"${CONDA}" env list
