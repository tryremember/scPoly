#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
PIP_REQUIREMENTS="${SCRIPT_DIR}/requirements-scpoly-pip.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f "${PIP_REQUIREMENTS}" ]]; then
  echo "Requirements file not found: ${PIP_REQUIREMENTS}" >&2
  exit 1
fi

ENV_NAME="$(awk -F': *' '/^name:/ {print $2}' "${ENV_FILE}")"

if [[ -z "${ENV_NAME}" ]]; then
  echo "Could not determine environment name from ${ENV_FILE}" >&2
  exit 1
fi

echo "Setting up conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists." >&2
  echo "Please remove it first or choose a different environment name in env_setup/environment.yml." >&2
  exit 1
fi

echo "Creating environment ${ENV_NAME} from env_setup/environment.yml..."
conda env create -f "${ENV_FILE}"

echo "Installing pip dependencies into ${ENV_NAME}..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${PIP_REQUIREMENTS}"

echo "Validating DeepSpeed Python dependencies..."
conda run -n "${ENV_NAME}" python -c "import pkg_resources; import torch.utils.cpp_extension; import deepspeed"

echo "Applying patched DeepSpeed moe module..."
conda run -n "${ENV_NAME}" bash "${SCRIPT_DIR}/install_patched_moe.sh"

echo
echo "scPoly environment setup is complete."
echo "Activate it with:"
echo "conda activate ${ENV_NAME}"
