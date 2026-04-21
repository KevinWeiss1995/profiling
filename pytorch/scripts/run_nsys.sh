#!/usr/bin/env bash
# Run a single scenario under nsys with recommended flags.
#
# Usage:
#   scripts/run_nsys.sh <scenario> [extra CLI args passed through]
#
# Example:
#   scripts/run_nsys.sh baseline
#   scripts/run_nsys.sh amp --steps 20
#
# Output goes to ${OUT_DIR:-out}/<scenario>.nsys-rep .

set -euo pipefail

SCENARIO="${1:?missing scenario name (e.g. baseline|large-batch|amp|compiled)}"
shift || true

OUT_DIR="${OUT_DIR:-out}"
mkdir -p "${OUT_DIR}"

# Pick a Python. Honors $PYTHON if set; otherwise prefers a local .venv
# (handy under apptainer where a bare `python` may resolve to the container's
# system interpreter instead of your venv).
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi
fi

# Apptainer + NGC PyTorch on aarch64: Triton can't locate libcuda.so at
# JIT-compile time because `--nv` bind-mounts the real driver but Triton's
# probe doesn't search /usr/local/cuda/compat/lib by default. Point it there
# if present. Harmless outside the container (directory just won't exist).
if [[ -e /usr/local/cuda/compat/lib/libcuda.so.1 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
fi

# --capture-range=cudaProfilerApi + Python-side cudaProfilerStart/Stop means
# the recorded window is exactly the profiled steps (not warmup, not startup).
# Keeps the trace small and the timeline clean.
#
# --trace list covers:
#   cuda    : kernels, memcpy, memset, sync
#   nvtx    : our forward/backward/optimizer ranges
#   osrt    : OS runtime (pthreads, sleeps) - makes CPU-side gaps visible
#   cudnn   : cuDNN API ranges for SDPA paths
#   cublas  : GEMM dispatch
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output "${OUT_DIR}/${SCENARIO}" \
    "${PYTHON}" -m profiling_demo \
        --scenario "${SCENARIO}" \
        --profiler nsys \
        --out-dir "${OUT_DIR}" \
        "$@"

echo
echo "wrote ${OUT_DIR}/${SCENARIO}.nsys-rep"
echo "open with: nsys-ui ${OUT_DIR}/${SCENARIO}.nsys-rep"
echo "    stats: nsys stats ${OUT_DIR}/${SCENARIO}.nsys-rep"
