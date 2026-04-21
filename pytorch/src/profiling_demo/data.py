"""Synthetic data generators.

Two paths:

- Device batches (default): random tokens generated directly on the GPU. No
  DataLoader, no H2D copies. Keeps most scenarios focused on the GPU story.
- Host batches (``slow_host_data=True``): generated on CPU, pinned, copied
  to device. Optionally sleeps first to simulate a slow collate / dataloader
  worker. Used by the ``starved`` scenario to produce the classic
  "GPU idle between steps" timeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class DataConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    slow_host_data: bool = False
    slow_host_sleep_s: float = 0.0


def _make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, generator=generator
    )
    # Next-token prediction target: shift and pad last position with random.
    y = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, generator=generator
    )
    return x, y


def device_batch_iterator(
    cfg: DataConfig, device: torch.device, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield fresh random batches entirely on-device.

    We re-roll each step so the profiler sees a representative cross-section
    of memory traffic rather than a single cached allocation.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    while True:
        yield _make_batch(cfg.batch_size, cfg.seq_len, cfg.vocab_size, device, gen)


def host_batch_iterator(
    cfg: DataConfig, device: torch.device, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Host-side batches copied to device each step.

    Optionally sleeps before the copy to simulate a slow collate. The sleep
    and the copy are both wrapped in NVTX ranges so they're visible on the
    nsys timeline as ``dataloader_sleep`` and ``h2d_copy``.
    """
    from .profile_utils import nvtx_range  # local import avoids cycle at import time

    gen = torch.Generator(device="cpu").manual_seed(seed)
    pinned = torch.cuda.is_available()
    while True:
        if cfg.slow_host_sleep_s > 0:
            with nvtx_range("dataloader_sleep"):
                time.sleep(cfg.slow_host_sleep_s)
        with nvtx_range("host_collate"):
            x = torch.randint(
                0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), generator=gen
            )
            y = torch.randint(
                0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), generator=gen
            )
            if pinned:
                x = x.pin_memory()
                y = y.pin_memory()
        with nvtx_range("h2d_copy"):
            yield (
                x.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )


def make_iterator(
    cfg: DataConfig, device: torch.device, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if cfg.slow_host_data:
        return host_batch_iterator(cfg, device, seed=seed)
    return device_batch_iterator(cfg, device, seed=seed)
