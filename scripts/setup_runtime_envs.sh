#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
FLASH_ENV="/workspace/.venv-fa2"
LONG_ENV="/workspace/.venv-longgen"

setup_flash_attn() {
  python3 -m venv "$FLASH_ENV" --system-site-packages
  "$FLASH_ENV/bin/pip" install --upgrade pip setuptools wheel ninja packaging
  "$FLASH_ENV/bin/pip" install flash-attn --no-build-isolation
}

setup_longgen() {
  python3 -m venv "$LONG_ENV"
  "$LONG_ENV/bin/pip" install --upgrade pip setuptools wheel
  "$LONG_ENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  "$LONG_ENV/bin/pip" install "transformers>=5.3.0" accelerate sentencepiece pillow
}

case "$TARGET" in
  flash-attn)
    setup_flash_attn
    ;;
  longgen)
    setup_longgen
    ;;
  all)
    setup_flash_attn
    setup_longgen
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    exit 1
    ;;
esac
