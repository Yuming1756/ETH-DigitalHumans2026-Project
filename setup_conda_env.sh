#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_conda_env_dir.sh /work/scratch/ankgupta/env_dir [python_version]
ENV_PREFIX="${1:-/tmp/env-egohumans-smplx}"
PY_VER="${2:-3.10}"
CUDA_TAG="cu120"   # target CUDA runtime tag for PyTorch
PYTORCH_CONSTRAINT="torch>=2.0.0"
PIP_PKGS=(pyyaml smplx trimesh plyfile numpy scipy opencv-python)

# choose package tool: prefer mamba if present
PKG_TOOL="conda"
if command -v mamba >/dev/null 2>&1; then
  PKG_TOOL="mamba"
fi

echo "Using package manager: ${PKG_TOOL}"
echo "Creating/using env at prefix: ${ENV_PREFIX}"
echo "Python version: ${PY_VER}"

# Create env directory if not exists (conda will populate)
if ! conda env list --json 2>/dev/null | jq -r '.envs[]' 2>/dev/null | grep -qx "${ENV_PREFIX}"; then
  echo "Creating conda env at prefix ${ENV_PREFIX}..."
  ${PKG_TOOL} create -y --prefix "${ENV_PREFIX}" python="${PY_VER}"
else
  echo "Conda env already exists at ${ENV_PREFIX} (skipping creation)."
fi

# helper to run python in the prefix env
run_in_env() {
  conda run --prefix "${ENV_PREFIX}" python - <<'PYCODE'
import sys, json
print("ok")
PYCODE
}

# Ensure pip tooling in env is up to date
echo "Upgrading pip/setuptools/wheel in env..."
conda run --prefix "${ENV_PREFIX}" bash -lc "python -m pip install --upgrade pip setuptools wheel"

# Check torch presence and CUDA version
echo "Checking PyTorch in env..."
TORCH_STATUS="missing"
if conda run --prefix "${ENV_PREFIX}" python -c "import importlib.util,sys; print(importlib.util.find_spec('torch') is not None)" 2>/dev/null | grep -q True; then
  TORCH_CUDA_VER=$(conda run --prefix "${ENV_PREFIX}" python -c "import torch,sys; print(torch.version.cuda if hasattr(torch,'version') else '')" 2>/dev/null || echo "")
  if [ "${TORCH_CUDA_VER}" = "12.0" ] || [ "${TORCH_CUDA_VER}" = "12.0.0" ] || [ "${TORCH_CUDA_VER}" = "12" ]; then
    echo "Found torch in env with CUDA ${TORCH_CUDA_VER} — skipping PyTorch install."
    TORCH_STATUS="ok"
  else
    echo "Found torch but CUDA version is '${TORCH_CUDA_VER}' (wanted 12.0). Will (re)install matching wheel."
    TORCH_STATUS="reinstall"
  fi
else
  echo "PyTorch not found in env. Will install."
  TORCH_STATUS="install"
fi

install_pytorch_pip() {
  echo "Attempting pip install of PyTorch (CUDA ${CUDA_TAG})..."
  conda run --prefix "${ENV_PREFIX}" bash -lc \
    "pip install pyyaml wcedit"
  conda run --prefix "${ENV_PREFIX}" bash -lc \
    "pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130"
}

install_pytorch_conda() {
  echo "Falling back to conda install for PyTorch (CUDA cudatoolkit=12.0)..."
  conda run --prefix "${ENV_PREFIX}" bash -lc \
    "${PKG_TOOL} install -y pytorch torchvision torchaudio cudatoolkit=12.0 -c pytorch -c nvidia"
}

if [ "${TORCH_STATUS}" != "ok" ]; then
  echo "Installing PyTorch (${PYTORCH_CONSTRAINT}) for CUDA 12.0 inside env..."
  set +e
  install_pytorch_pip
  PIP_RC=$?
  set -e
  # if [ "${PIP_RC}" -ne 0 ]; then
  #   echo "pip install failed (rc=${PIP_RC}). Collecting debug info and trying conda fallback..."
  #   conda run --prefix "${ENV_PREFIX}" bash -lc "python -V; python -m pip --version; python -m pip debug --verbose || true; uname -a || true"
  #   install_pytorch_conda
  # fi
fi

# Check and install pip packages if missing
echo "Checking pip packages..."
MISSING_PIP=()
for pkg in "${PIP_PKGS[@]}"; do
  if conda run --prefix "${ENV_PREFIX}" python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${pkg}') else 1)"; then
    echo "Package ${pkg} present in env."
  else
    MISSING_PIP+=("${pkg}")
  fi
done

if [ ${#MISSING_PIP[@]} -gt 0 ]; then
  echo "Installing missing pip packages into env: ${MISSING_PIP[*]}"
  conda run --prefix "${ENV_PREFIX}" bash -lc "python -m pip install ${MISSING_PIP[*]}"
else
  echo "All pip packages present."
fi

echo "Done. Environment available at: ${ENV_PREFIX}"
echo "Activate with: conda activate ${ENV_PREFIX}"
echo "Verify CUDA/PyTorch: conda run --prefix ${ENV_PREFIX} python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)\""
echo "Verify smplx: conda run --prefix ${ENV_PREFIX} python -c \"import smplx; print('smplx ok', getattr(smplx, '__version__', 'n/a'))\""
