"""Profiling helpers: NVTX ranges, bounded cudaProfiler window, torch.profiler."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


# ---------------------------------------------------------------------------
# NVTX
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push/pop an NVTX range. No-op off-CUDA so tests / dev boxes still run."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Bounded cudaProfiler window
# ---------------------------------------------------------------------------
#
# Pair this with ``nsys profile --capture-range=cudaProfilerApi
# --capture-range-end=stop`` so the recorded timeline covers only the profiled
# steps (not warmup, not process startup). Traces stay small and the part you
# care about lights up immediately when opened.

@contextlib.contextmanager
def cuda_profiler_session(enabled: bool = True) -> Iterator[None]:
    if not enabled or not torch.cuda.is_available():
        yield
        return
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()


# ---------------------------------------------------------------------------
# torch.profiler
# ---------------------------------------------------------------------------

@dataclass
class TorchProfilerResult:
    trace_path: Path
    summary_path: Path
    total_cuda_kernels: int
    total_cuda_time_us: float
    top_kernels: list[tuple[str, int, float]]  # (name, count, total cuda us)


class TorchProfilerSession:
    """Thin wrapper around ``torch.profiler.profile`` that emits a chrome
    trace plus a tiny JSON summary (kernel count, top-k kernels by CUDA time).

    Usage::

        with TorchProfilerSession(out_dir, tag="baseline") as sess:
            for step in ...:
                sess.step()   # call once per profiled step

        result = sess.result   # populated on __exit__
    """

    def __init__(
        self,
        out_dir: Path,
        tag: str,
        wait: int = 0,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        self._activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            self._activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof: torch.profiler.profile | None = None
        self.result: TorchProfilerResult | None = None

    def __enter__(self) -> "TorchProfilerSession":
        self._prof = torch.profiler.profile(
            activities=self._activities,
            schedule=self._schedule,
            record_shapes=False,
            with_stack=False,
            with_flops=False,
        )
        self._prof.__enter__()
        return self

    def step(self) -> None:
        assert self._prof is not None
        self._prof.step()

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._prof is not None
        # Flush any pending window.
        trace_path = self.out_dir / f"{self.tag}.trace.json"
        summary_path = self.out_dir / f"{self.tag}.summary.json"
        try:
            self._prof.__exit__(exc_type, exc, tb)
            self._prof.export_chrome_trace(str(trace_path))
            self.result = _summarize_profile(self._prof, trace_path, summary_path)
        finally:
            self._prof = None


def _summarize_profile(
    prof: torch.profiler.profile, trace_path: Path, summary_path: Path
) -> TorchProfilerResult:
    key_averages = prof.key_averages()
    total_cuda_kernels = 0
    total_cuda_time_us = 0.0
    kernel_rows: list[tuple[str, int, float]] = []
    for evt in key_averages:
        # A "device" event here is a CUDA kernel (or memcpy/memset on CUDA).
        dev_time = getattr(evt, "device_time_total", None)
        if dev_time is None:  # torch < 2.4 fallback
            dev_time = getattr(evt, "cuda_time_total", 0.0)
        if dev_time and dev_time > 0:
            total_cuda_kernels += evt.count
            total_cuda_time_us += float(dev_time)
            kernel_rows.append((evt.key, int(evt.count), float(dev_time)))

    kernel_rows.sort(key=lambda r: r[2], reverse=True)
    top = kernel_rows[:10]

    summary = {
        "total_cuda_kernels": total_cuda_kernels,
        "total_cuda_time_us": total_cuda_time_us,
        "top_kernels": [
            {"name": name, "count": count, "cuda_us": cuda_us}
            for name, count, cuda_us in top
        ],
        "trace_path": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return TorchProfilerResult(
        trace_path=trace_path,
        summary_path=summary_path,
        total_cuda_kernels=total_cuda_kernels,
        total_cuda_time_us=total_cuda_time_us,
        top_kernels=top,
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def device_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def in_nsys() -> bool:
    """Heuristic: are we running under ``nsys profile``?"""
    return any(k.startswith("NSYS_") for k in os.environ)
