#!/usr/bin/env bash

# Start the OpenPI policy server with the system CUDA toolchain selected
# explicitly.  This avoids JAX/XLA selecting a different ptxas executable.

set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.8}"

if [[ ! -x "${CUDA_ROOT}/bin/ptxas" ]]; then
  echo "ERROR: ptxas was not found at ${CUDA_ROOT}/bin/ptxas" >&2
  echo "Set CUDA_ROOT to the installed CUDA toolkit directory." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver is not available to this process." >&2
  echo "Run nvidia-smi and fix the driver/device access before starting a GPU server." >&2
  exit 1
fi

# ptxas is used by XLA during model compilation.  JAX itself supplies the
# CUDA runtime libraries through the locked Python dependencies, so do not
# prepend CUDA lib64 to LD_LIBRARY_PATH here.
export PATH="${CUDA_ROOT}/bin:${PATH}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CUDA_ROOT}"
unset JAX_PLATFORMS

exec uv run scripts/serve_policy.py "$@"
