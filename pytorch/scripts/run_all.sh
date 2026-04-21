#!/usr/bin/env bash
# Run all four scenarios back-to-back and print a comparison table.
#
# Default path uses torch.profiler (works on any CUDA box, no Nsight needed).
# Set PROFILER=nsys to instead produce nsys-rep files (one per scenario).
#
# Usage:
#   scripts/run_all.sh                  # torch.profiler
#   PROFILER=nsys scripts/run_all.sh    # nsys
#   PROFILER=none scripts/run_all.sh    # just time them, no profiler overhead

set -euo pipefail

OUT_DIR="${OUT_DIR:-out}"
PROFILER="${PROFILER:-torch}"
SCENARIOS=(baseline large-batch amp compiled)
STEPS="${STEPS:-10}"
WARMUP="${WARMUP:-5}"

mkdir -p "${OUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for scn in "${SCENARIOS[@]}"; do
    echo
    echo "======================================================================"
    echo "  scenario: ${scn}  (profiler=${PROFILER})"
    echo "======================================================================"
    if [[ "${PROFILER}" == "nsys" ]]; then
        OUT_DIR="${OUT_DIR}" \
            bash "${SCRIPT_DIR}/run_nsys.sh" "${scn}" \
                --warmup "${WARMUP}" --steps "${STEPS}"
    else
        python -m profiling_demo \
            --scenario "${scn}" \
            --profiler "${PROFILER}" \
            --out-dir "${OUT_DIR}" \
            --warmup "${WARMUP}" \
            --steps "${STEPS}"
    fi
done

echo
echo "======================================================================"
echo "  summary"
echo "======================================================================"
python - <<'PY'
import json, os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", "out"))
rows = []
for scn in ["baseline", "large-batch", "amp", "compiled"]:
    p = out_dir / f"{scn}.report.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    tp = r.get("torch_profiler") or {}
    rows.append((
        r["scenario"],
        r["batch_size"],
        r["autocast"],
        "yes" if r["compile"] else "no",
        r["median_step_ms"],
        r["peak_memory_mib"],
        tp.get("total_cuda_kernels", "-"),
    ))

cols = ["scenario", "bs", "dtype", "compiled", "step ms", "peak MiB", "kernels"]
widths = [max(len(str(x)) for x in [c] + [r[i] for r in rows]) for i, c in enumerate(cols)]
fmt = "  ".join(f"{{:<{w}}}" for w in widths)
print(fmt.format(*cols))
print(fmt.format(*("-" * w for w in widths)))
for r in rows:
    cells = [
        r[0], r[1], r[2], r[3],
        f"{r[4]:.2f}" if isinstance(r[4], float) else r[4],
        f"{r[5]:.0f}" if isinstance(r[5], float) else r[5],
        r[6],
    ]
    print(fmt.format(*cells))
PY
