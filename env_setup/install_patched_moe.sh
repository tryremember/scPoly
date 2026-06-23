#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCHED_MOE_DIR="${PROJECT_ROOT}/third_party/deepspeed_moe/moe"

if [[ ! -d "${PATCHED_MOE_DIR}" ]]; then
  echo "Patched moe directory not found: ${PATCHED_MOE_DIR}" >&2
  exit 1
fi

DEEPSPEED_DIR="$(python -c 'import importlib.util, sys; spec = importlib.util.find_spec("deepspeed"); print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else ""); sys.exit(0 if spec and spec.submodule_search_locations else 1)')"
TARGET_MOE_DIR="${DEEPSPEED_DIR}/moe"
BACKUP_DIR="${DEEPSPEED_DIR}/moe.bak.$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "${DEEPSPEED_DIR}" ]]; then
  echo "DeepSpeed package directory not found." >&2
  exit 1
fi

if [[ ! -d "${TARGET_MOE_DIR}" ]]; then
  echo "Original DeepSpeed moe directory not found: ${TARGET_MOE_DIR}" >&2
  exit 1
fi

echo "DeepSpeed package found at: ${DEEPSPEED_DIR}"
echo "Backing up original moe to: ${BACKUP_DIR}"
mv "${TARGET_MOE_DIR}" "${BACKUP_DIR}"

echo "Installing patched moe from: ${PATCHED_MOE_DIR}"
cp -r "${PATCHED_MOE_DIR}" "${TARGET_MOE_DIR}"

find "${TARGET_MOE_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${TARGET_MOE_DIR}" -type f -name "*.pyc" -delete

echo "Patched DeepSpeed moe installed successfully."
echo "If needed, restore the original version with:"
echo "mv \"${BACKUP_DIR}\" \"${TARGET_MOE_DIR}\""
