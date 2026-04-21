"""Scenario registry.

Each scenario is a ``Scenario`` dataclass that tweaks a common baseline. The
CLI stays tiny: one flag picks a scenario, everything else is derived.

Teaching goal per scenario:

- ``baseline``    - dense stream of short kernels, visible launch gaps.
- ``large-batch`` - same kernels, each longer; launch overhead amortized.
- ``amp``         - GEMMs switch to bf16 Tensor Core variants; backward shrinks.
- ``starved``     - slow host dataloader leaves the GPU idle between steps
                    (the single most common real-world bottleneck).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import torch
import torch.nn as nn

from .data import DataConfig
from .model import ModelConfig, build_model
from .step import make_autocast_ctx


@dataclass
class Scenario:
    name: str
    description: str
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 8
    learning_rate: float = 3e-4
    autocast_dtype: torch.dtype | None = None
    compile: bool = False
    compile_mode: str = "default"
    # Data-pipeline knobs. slow_host_data routes batches through CPU + H2D
    # copy; slow_host_sleep_s adds an artificial delay before each copy to
    # simulate a slow collate / dataloader worker.
    slow_host_data: bool = False
    slow_host_sleep_s: float = 0.0

    def data_config(self) -> DataConfig:
        return DataConfig(
            batch_size=self.batch_size,
            seq_len=self.model.seq_len,
            vocab_size=self.model.vocab_size,
            slow_host_data=self.slow_host_data,
            slow_host_sleep_s=self.slow_host_sleep_s,
        )


_BASELINE_MODEL = ModelConfig(
    vocab_size=32_000,
    d_model=1024,
    n_heads=16,
    n_layers=6,
    mlp_mult=4,
    seq_len=512,
    dropout=0.0,
)


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="fp32, bs=8. The 'what does a training step look like?' picture.",
        model=_BASELINE_MODEL,
        batch_size=8,
    ),
    "large-batch": Scenario(
        name="large-batch",
        description="fp32, bs=64. Amortizes launch overhead; improves SM occupancy.",
        model=_BASELINE_MODEL,
        batch_size=64,
    ),
    "amp": Scenario(
        name="amp",
        description="bf16 autocast, bs=8. Tensor Cores + ~halved memory bandwidth.",
        model=_BASELINE_MODEL,
        batch_size=8,
        autocast_dtype=torch.bfloat16,
    ),
    "starved": Scenario(
        name="starved",
        description=(
            "fp32, bs=8, with a slow CPU dataloader. GPU sits idle between "
            "steps - the classic 'dataloader bottleneck' shape."
        ),
        model=_BASELINE_MODEL,
        batch_size=8,
        slow_host_data=True,
        # 20 ms is visibly larger than a baseline step (~19 ms on GH200),
        # so the GPU-idle gaps are obvious in the timeline. Tunable via
        # --slow-host-sleep-s on the CLI for playing with the ratio.
        slow_host_sleep_s=0.020,
    ),
}


def list_scenarios() -> list[str]:
    return list(SCENARIOS.keys())


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {list_scenarios()}")
    return SCENARIOS[name]


def build_for_scenario(
    scn: Scenario, device: torch.device
) -> tuple[nn.Module, torch.optim.Optimizer, Callable]:
    """Construct (possibly compiled) model, optimizer, and autocast factory."""
    model = build_model(scn.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=scn.learning_rate, fused=True)

    compiled: nn.Module = model
    if scn.compile:
        # ``reduce-overhead`` triggers CUDA graphs under the hood, which is
        # exactly the effect we want to showcase on the nsys timeline.
        compiled = torch.compile(model, mode=scn.compile_mode)  # type: ignore[assignment]

    autocast_ctx = make_autocast_ctx(scn.autocast_dtype)
    return compiled, optimizer, autocast_ctx


def override(scn: Scenario, **kwargs) -> Scenario:
    """Apply CLI-side overrides (e.g. --batch-size, --slow-host-data) to a scenario."""
    model_overrides = {}
    for key in ("seq_len", "d_model", "n_layers", "n_heads"):
        if key in kwargs and kwargs[key] is not None:
            model_overrides[key] = kwargs.pop(key)
    if model_overrides:
        scn = replace(scn, model=replace(scn.model, **model_overrides))
    clean = {k: v for k, v in kwargs.items() if v is not None}
    return replace(scn, **clean)
