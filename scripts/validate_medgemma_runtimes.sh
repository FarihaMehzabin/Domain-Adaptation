#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p outputs/runtime_validation

SYSTEM_PY="python3"
FA2_PY="/workspace/.venv-fa2/bin/python"
LONG_PY="/workspace/.venv-longgen/bin/python"

run_validation() {
  local label="$1"
  local python_bin="$2"
  shift 2
  "$python_bin" /workspace/medgemma4b.py validate-runtime \
    --runtime-name "$label" \
    --prompt-mode compact \
    --output-json "/workspace/outputs/runtime_validation/${label}.json" \
    "$@"
}

run_validation eager "$SYSTEM_PY" --attn-implementation eager || true

if [[ -x "$FA2_PY" ]]; then
  run_validation flash_attention_2 "$FA2_PY" --attn-implementation flash_attention_2 --skip-budget-guard || true
else
  echo "Skipping flash_attention_2 validation because $FA2_PY is not available."
fi

if [[ -x "$LONG_PY" ]]; then
  run_validation longgen "$LONG_PY" --attn-implementation eager --skip-budget-guard || true
else
  echo "Skipping longgen validation because $LONG_PY is not available."
fi

python3 - <<'PY'
import json
from pathlib import Path
base = Path('/workspace/outputs/runtime_validation')
selected = None
for name in ['eager', 'flash_attention_2', 'longgen']:
    path = base / f'{name}.json'
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('passed'):
        selected = payload
        break
if selected is None:
    print('No runtime passed validation.')
else:
    out = base / 'selected_runtime.json'
    out.write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    print(f"Selected runtime: {selected['runtime_name']}")
    print(out)
PY
