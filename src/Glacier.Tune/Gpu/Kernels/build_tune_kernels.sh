#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
    for cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin; do
        if [ -x "$cuda_dir/nvcc" ]; then
            export PATH="$cuda_dir:$PATH"
            break
        fi
    done
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found in PATH or standard CUDA directories." >&2
    exit 1
fi

echo "[INFO] Compiling tune_bwd.cu to tune_bwd.cubin..."
nvcc -fatbin -O3 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_75,code=compute_75 \
  -o tune_bwd.cubin tune_bwd.cu

echo "[SUCCESS] Successfully built tune_bwd.cubin ($(wc -c < tune_bwd.cubin) bytes)."
