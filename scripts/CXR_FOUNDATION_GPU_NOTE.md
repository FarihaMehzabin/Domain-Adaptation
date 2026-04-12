# CXR Foundation GPU Note

Date: 2026-04-12

## Symptom

After a fresh `git pull`, running `14_generate_cxr_foundation_embeddings.py` or `18_benchmark_cxr_foundation_batch_sizes.py` sometimes starts on CPU even though the machine has a GPU.

## What is happening

The CXR Foundation branch uses TensorFlow, not PyTorch. TensorFlow uses the GPU runtime from the Python environment that launched the script.

In our setup, the mixed main environment can drift into this state:

- `tensorflow==2.17.1`
- TensorFlow build expects:
  - `CUDA 12.3`
  - `cuDNN 8`
- but the environment can end up with PyTorch-side NVIDIA wheels such as:
  - `nvidia-cudnn-cu12==9.1.0.70`

When that happens, TensorFlow prints:

- `Cannot dlopen some GPU libraries`
- `Skipping registering GPU devices...`

and the CXR Foundation exporter silently falls back to CPU.

## Why this shows up after a fresh pull

A repo pull does not restore your shell environment.

Two things must both be correct for TensorFlow GPU runs:

1. The Python environment must have the TensorFlow-compatible NVIDIA runtime packages.
2. `LD_LIBRARY_PATH` must include the `site-packages/nvidia/*/lib` directories from that same environment.

If you launch the exporter from the default mixed environment, or from a fresh shell that has not restored those library paths, TensorFlow sees no usable GPU.

## Current repo fix

The repo now avoids silent CPU fallback in three ways:

1. `requirements_cxr_foundation.txt` uses `tensorflow[and-cuda]==2.17.1`
   - This pulls the TensorFlow-compatible NVIDIA runtime versions, including:
     - `nvidia-cudnn-cu12==8.9.7.29`
     - `nvidia-cuda-runtime-cu12==12.3.101`
     - `nvidia-cublas-cu12==12.3.4.1`
2. `activate_cxr_foundation_env.sh` activates the dedicated CXR Foundation virtualenv and prepends its `site-packages/nvidia/*/lib` directories to `LD_LIBRARY_PATH`.
3. `cxr_foundation_common.py` now fails fast if TensorFlow detects no GPU.
   - This prevents accidentally launching a long export on CPU without noticing.
   - You can still bypass this intentionally with:
     - `export CXR_FOUNDATION_ALLOW_CPU=1`

## Future-safe workflow

Use a dedicated TensorFlow environment for the CXR Foundation branch.

```bash
python -m venv /workspace/.venv_cxr_foundation
/workspace/.venv_cxr_foundation/bin/python -m pip install --upgrade pip
/workspace/.venv_cxr_foundation/bin/python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt
source /workspace/scripts/activate_cxr_foundation_env.sh
```

Then run the exporter or benchmark from that shell.

Examples:

```bash
source /workspace/scripts/activate_cxr_foundation_env.sh
python /workspace/scripts/18_benchmark_cxr_foundation_batch_sizes.py --help
python /workspace/scripts/14_generate_cxr_foundation_embeddings.py --help
```

## Practical takeaway

If CXR Foundation suddenly starts using CPU after a pull, the problem is not the repo code itself. The problem is that TensorFlow is running from the wrong environment, or from an environment whose GPU runtime packages and library paths are not aligned with TensorFlow `2.17.1`.
