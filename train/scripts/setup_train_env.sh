#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="fishbowl-t"
PYTHON_VERSION="3.11"

echo "[train] Creating clean conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "[train] Installing base Python tooling"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "[train] Installing project dependencies"
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

cat <<EOF

[train] Done.
Activate with:
  conda activate ${ENV_NAME}

Use this env for:
  python main.py

If you need a CUDA-specific PyTorch build, install it inside this env before running training.
Example:
  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
EOF
