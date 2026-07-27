#!/usr/bin/env bash

# GPU pipeline environment bootstrap script for Ubuntu + NVIDIA CUDA 12.x.
# This script creates an isolated Python 3.11 virtual environment,
# installs all required dependencies, verifies GPU visibility, and exits.

set -euo pipefail

BASE_DIR="/run/media/pranam/Laksh 320GB"
PIPELINE_DIR="$BASE_DIR/PIPELINE"
PYTHON_BIN="python3.11"

# The hard drive is exFAT, which does not support symlinks.
# Python venv creates a lib64->lib symlink that exFAT rejects, so
# we place the venv on the Linux filesystem instead.
VENV_DIR="$HOME/.venvs/pipeline"

# Ensure the expected Python version exists on the host.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3.11 was not found. Run: sudo pacman -S python311" >&2
  exit 1
fi

# Create directories used by the project if they are missing.
mkdir -p "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR/results"

# Build a fresh virtual environment in the exact requested location.
"$PYTHON_BIN" -m venv "$VENV_DIR"

# Activate the environment for package installation and validation.
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Keep installer tooling current to reduce wheel/build compatibility issues.
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch and Torchaudio with CUDA-enabled wheels.
# RTX 4050 (driver 595) supports CUDA 12.4+ — use cu124 index.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch torchaudio

# Install CUDA 12 focused NVIDIA/RAPIDS dependencies.
# nvidia-dali-cuda120 and cudf-cu12 are hosted through NVIDIA indexing.
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  nvidia-dali-cuda120 cudf-cu12 cupy-cuda12x

# Install remaining CPU-side and plotting/data dependencies.
python -m pip install opencv-python pandas matplotlib openpyxl

# Validate package imports and confirm a visible CUDA device.
python - <<'PY'
import sys

import cupy
import cudf
import nvidia.dali
import torch
import torchaudio

print("Imported: cupy, cudf, nvidia.dali, torch, torchaudio")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
print(f"Torch CUDA device count: {torch.cuda.device_count()}")
print(f"CuPy CUDA device count: {cupy.cuda.runtime.getDeviceCount()}")

if not torch.cuda.is_available() or cupy.cuda.runtime.getDeviceCount() < 1:
    print("ERROR: GPU was not detected by one or more libraries.", file=sys.stderr)
    sys.exit(1)
PY

# Optional hardware-level confirmation for user visibility.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "Setup complete"
