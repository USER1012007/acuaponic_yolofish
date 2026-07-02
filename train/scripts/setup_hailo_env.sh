#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="fishbowl-hailo"
PYTHON_VERSION="3.10"
DFC_WHL="hailo/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"
HAILORT_WHL="hailo/hailort-4.24.0-cp311-cp311-linux_x86_64.whl"
MODEL_ZOO_DIR="hailo/hailo_model_zoo"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "[hailo] ERROR: Hailo Dataflow Compiler must be installed on Linux x86_64, not $(uname -m)." >&2
  exit 1
fi

if [[ ! -f "${DFC_WHL}" ]]; then
  echo "[hailo] ERROR: Missing ${DFC_WHL}" >&2
  exit 1
fi

echo "[hailo] Creating clean conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "[hailo] Installing base Python tooling"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda install -n "${ENV_NAME}" -y -c conda-forge libglib glib

echo "[hailo] Installing Hailo Dataflow Compiler wheel"
conda run -n "${ENV_NAME}" python -m pip install "${DFC_WHL}"
conda run -n "${ENV_NAME}" python -m pip check

if [[ -f "${HAILORT_WHL}" ]]; then
  cat <<EOF

[hailo] NOTE:
  Found ${HAILORT_WHL}, but it is cp311 and this compile environment uses Python ${PYTHON_VERSION}.
  HailoRT is not required for model.onnx -> model.hef compilation, so it is intentionally not installed here.
EOF
fi

echo "[hailo] Checking local Hailo Model Zoo compatibility"
if [[ -f "${MODEL_ZOO_DIR}/setup.py" ]]; then
  if grep -q 'CUR_DFC_VERSION = CUR_MZ_VERSION = "5.3.0"' "${MODEL_ZOO_DIR}/setup.py"; then
    cat <<EOF

[hailo] WARNING:
  ${MODEL_ZOO_DIR} is Model Zoo 5.3.0, but your DFC wheel is 3.34.0.
  Do NOT install this Model Zoo for Hailo-8.

  For Hailo-8, use a Hailo Model Zoo v2.x release/branch compatible with DFC 3.x.
  If this repo has tags, use the actual release tag, not a docs/update branch:
    git -C ${MODEL_ZOO_DIR} checkout v2.19.0

  Then rerun:
    conda activate ${ENV_NAME}
    pip install --no-build-isolation -e ${MODEL_ZOO_DIR}
    pip check

  Until then, the 'hailomz' command will not be available.
EOF
  else
    echo "[hailo] Installing local Hailo Model Zoo"
    conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation -e "${MODEL_ZOO_DIR}"
  fi
else
  echo "[hailo] WARNING: ${MODEL_ZOO_DIR} not found. 'hailomz' will not be available."
fi

echo "[hailo] Verifying installed commands"
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
print("python", sys.version)
try:
    import hailo_sdk_client
    print("hailo_sdk_client OK")
except Exception as exc:
    print("hailo_sdk_client ERROR:", exc)
PY

cat <<EOF

[hailo] Done.
Activate with:
  conda activate ${ENV_NAME}

Expected:
  hailo --help      # from Dataflow Compiler
  hailomz --help    # only after compatible Hailo Model Zoo v2.x is installed

Use this env only for:
  model.onnx -> model.hef

Do not install the local Model Zoo 5.3.0 with DFC 3.34.0 for Hailo-8.
EOF
