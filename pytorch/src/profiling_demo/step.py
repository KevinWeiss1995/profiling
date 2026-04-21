"""NVTX-annotated training step.

One call produces a very readable nsys timeline: each phase gets its own
NVTX row, and the host-side launch stream is easy to correlate with the
device-side kernel stream.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profile_utils import nvtx_range


def make_autocast_ctx(dtype: torch.dtype | None) -> Callable[[], contextlib.AbstractContextManager]:
    """Return a zero-arg factory producing an autocast context (or nullcontext)."""
    if dtype is None:
        return contextlib.nullcontext
    def _factory() -> contextlib.AbstractContextManager:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return _factory


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    autocast_ctx: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
    step_idx: int = 0,
) -> torch.Tensor:
    """Run one training step, annotated for nsys.

    Returns the (detached) loss. Caller is responsible for any sync / logging.
    """
    with nvtx_range(f"step/{step_idx}"):
        with nvtx_range("forward"), autocast_ctx():
            logits = model(x)
            # Flatten once, outside the autocast for the loss (safer for fp32
            # numerics; cross_entropy in bf16 is fine but this matches common
            # recipes).
        with nvtx_range("loss"):
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                y.view(-1),
            )
        with nvtx_range("backward"):
            loss.backward()
        with nvtx_range("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return loss.detach()
