#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:"
  echo "  source /workspace/scripts/activate_cxr_foundation_env.sh"
  exit 1
fi

VENV_PATH="${CXR_FOUNDATION_VENV:-/workspace/.venv_cxr_foundation}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "CXR Foundation virtualenv not found at ${VENV_PATH}" >&2
  echo "Create it with:" >&2
  echo "  python -m venv ${VENV_PATH}" >&2
  echo "  ${VENV_PATH}/bin/python -m pip install --upgrade pip" >&2
  echo "  ${VENV_PATH}/bin/python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt" >&2
  return 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

EXTRA_NVIDIA_LIBS="$(
python - <<'PY'
from pathlib import Path
import sysconfig

purelib = Path(sysconfig.get_paths()["purelib"])
nvidia_root = purelib / "nvidia"
paths = [str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir()]
print(":".join(paths))
PY
)"

if [[ -n "${EXTRA_NVIDIA_LIBS}" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${EXTRA_NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${EXTRA_NVIDIA_LIBS}"
  fi
fi

echo "Activated CXR Foundation environment: ${VENV_PATH}"
if [[ -n "${EXTRA_NVIDIA_LIBS}" ]]; then
  echo "Prepended TensorFlow GPU runtime libraries from site-packages/nvidia/*/lib"
fi
