"""CLI entry point.

Typical usage::

    # With nsys: bounded capture, ~30MB trace
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o out/baseline -f true \
        python -m profiling_demo --scenario baseline --profiler nsys

    # Without Nsight: still get a kernel summary + chrome trace
    python -m profiling_demo --scenario amp --profiler torch --out-dir out

    # Just run, no profiling (sanity check / timing)
    python -m profiling_demo --scenario compiled --profiler none --steps 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import __version__
from .data import make_iterator
from .model import parameter_count
from .profile_utils import (
    TorchProfilerSession,
    cuda_profiler_session,
    cuda_synchronize,
    device_name,
    in_nsys,
    nvtx_range,
    peak_memory_mib,
    reset_peak_memory,
)
from .scenarios import build_for_scenario, get_scenario, list_scenarios, override
from .step import train_step


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="profiling-demo",
        description="PyTorch training-step profiling demo (GH200-oriented).",
    )
    p.add_argument(
        "--scenario",
        required=True,
        choices=list_scenarios(),
        help="Which scenario to run.",
    )
    p.add_argument(
        "--profiler",
        default="none",
        choices=["none", "nsys", "torch"],
        help="none: just time it. nsys: bound cudaProfilerStart/Stop "
        "for use with nsys. torch: use torch.profiler and emit a chrome trace.",
    )
    p.add_argument("--warmup", type=int, default=3, help="Unprofiled warmup steps.")
    p.add_argument("--steps", type=int, default=10, help="Profiled steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--device", default="cuda")

    # Scenario overrides
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--slow-host-data", action="store_true", default=None)

    # Precision knobs (mostly for experimentation; the baseline uses TF32)
    p.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 matmul. Default is on (matches modern PyTorch defaults).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _set_determinism(seed: int, tf32: bool) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; aborting.", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    _set_determinism(args.seed, tf32=not args.no_tf32)

    scn = get_scenario(args.scenario)
    scn = override(
        scn,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        slow_host_data=args.slow_host_data,
    )

    print(f"== profiling-demo {__version__} ==")
    print(f"device      : {device_name()}")
    print(f"scenario    : {scn.name}  ({scn.description})")
    print(f"model       : d={scn.model.d_model} L={scn.model.n_layers} "
          f"heads={scn.model.n_heads} seq={scn.model.seq_len} vocab={scn.model.vocab_size}")
    print(f"batch_size  : {scn.batch_size}")
    print(f"autocast    : {scn.autocast_dtype}")
    print(f"compile     : {scn.compile} (mode={scn.compile_mode})")
    print(f"warmup/steps: {args.warmup}/{args.steps}")
    print(f"profiler    : {args.profiler}{' (in-nsys)' if in_nsys() else ''}")
    print()

    model, optimizer, autocast_ctx = build_for_scenario(scn, device)
    params = parameter_count(model)
    print(f"params      : {params/1e6:.1f}M")

    data_iter = make_iterator(scn.data_config(), device, seed=args.seed)

    # ------------------------------------------------------------------
    # Warmup (unprofiled). Crucial for torch.compile + cudagraphs; without
    # this the first "profiled" step would dominate the trace with compile
    # + graph-capture noise.
    # ------------------------------------------------------------------
    print("warming up...", flush=True)
    for i in range(args.warmup):
        x, y = next(data_iter)
        train_step(model, x, y, optimizer, autocast_ctx, step_idx=-1 - i)
    cuda_synchronize()

    # ------------------------------------------------------------------
    # Profiled region.
    # ------------------------------------------------------------------
    reset_peak_memory()
    step_times_ms: list[float] = []

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_sess: TorchProfilerSession | None = None
    if args.profiler == "torch":
        torch_sess = TorchProfilerSession(
            out_dir=out_dir,
            tag=scn.name,
            wait=0,
            warmup=0,
            active=args.steps,
            repeat=1,
        )

    # nsys window: bound cudaProfilerStart/Stop around the loop. Safe to call
    # regardless of whether we're actually under nsys - a no-op otherwise.
    nsys_enabled = args.profiler == "nsys"

    print(f"profiling {args.steps} steps...", flush=True)

    def _profiled_loop() -> None:
        for step_idx in range(args.steps):
            x, y = next(data_iter)
            # Per-step timing via CUDA events so we don't force a global sync.
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if start is not None:
                start.record()
            with nvtx_range(f"iter/{step_idx}"):
                train_step(model, x, y, optimizer, autocast_ctx, step_idx=step_idx)
            if end is not None:
                end.record()
            if torch_sess is not None:
                torch_sess.step()
            if start is not None and end is not None:
                end.synchronize()
                step_times_ms.append(start.elapsed_time(end))

    wall_start = time.perf_counter()
    if torch_sess is not None:
        with torch_sess:
            _profiled_loop()
    else:
        with cuda_profiler_session(enabled=nsys_enabled):
            _profiled_loop()
    cuda_synchronize()
    wall_s = time.perf_counter() - wall_start

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    median_ms = statistics.median(step_times_ms) if step_times_ms else float("nan")
    p10_ms = (
        statistics.quantiles(step_times_ms, n=10)[0]
        if len(step_times_ms) >= 10
        else min(step_times_ms, default=float("nan"))
    )
    peak_mib = peak_memory_mib()

    report: dict[str, object] = {
        "scenario": scn.name,
        "device": device_name(),
        "params_m": params / 1e6,
        "batch_size": scn.batch_size,
        "seq_len": scn.model.seq_len,
        "autocast": str(scn.autocast_dtype) if scn.autocast_dtype else "fp32",
        "compile": scn.compile,
        "warmup": args.warmup,
        "steps": args.steps,
        "median_step_ms": median_ms,
        "p10_step_ms": p10_ms,
        "peak_memory_mib": peak_mib,
        "wall_seconds": wall_s,
    }
    if torch_sess is not None and torch_sess.result is not None:
        report["torch_profiler"] = {
            "total_cuda_kernels": torch_sess.result.total_cuda_kernels,
            "total_cuda_time_us": torch_sess.result.total_cuda_time_us,
            "top_kernels": [
                {"name": n, "count": c, "cuda_us": us}
                for n, c, us in torch_sess.result.top_kernels[:5]
            ],
            "trace": str(torch_sess.result.trace_path),
        }

    print()
    print(f"median step : {median_ms:7.2f} ms")
    print(f"p10 step    : {p10_ms:7.2f} ms")
    print(f"peak memory : {peak_mib:7.1f} MiB")
    print(f"wall time   : {wall_s:7.2f} s")
    if "torch_profiler" in report:
        tp = report["torch_profiler"]
        print(f"cuda kernels: {tp['total_cuda_kernels']} launches over "
              f"{tp['total_cuda_time_us']:.0f} us")
        print("top kernels by cuda time:")
        for row in tp["top_kernels"]:
            print(f"  {row['cuda_us']:9.0f} us  x{row['count']:5d}  {row['name']}")

    # Machine-readable summary sibling (one per scenario).
    summary_path = out_dir / f"{scn.name}.report.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
