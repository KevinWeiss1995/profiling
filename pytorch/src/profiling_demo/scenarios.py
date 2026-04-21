"""Scenario registry.

Each scenario is a ``Scenario`` dataclass that tweaks a common baseline. The
CLI stays tiny: one flag picks a scenario, everything else is derived.

Teaching goal per scenario:

- ``baseline``    - dense stream of short kernels, visible launch gaps.
- ``large-batch`` - same kernels, each longer; launch overhead amortized.
- ``amp``         - GEMMs switch to bf16 Tensor Core variants; backward shrinks.
- ``compiled``    - far fewer, longer (Triton-fused) kernels + cudaGraphLaunch.
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
    # Advanced knobs (not used by default scenarios but available for
    # experimentation via the CLI).
    slow_host_data: bool = False

    def data_config(self) -> DataConfig:
        return DataConfig(
            batch_size=self.batch_size,
            seq_len=self.model.seq_len,
            vocab_size=self.model.vocab_size,
            slow_host_data=self.slow_host_data,
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
    "compiled": Scenario(
        name="compiled",
        description="torch.compile(mode='reduce-overhead'), bs=8. Kernel fusion + CUDA graphs.",
        model=_BASELINE_MODEL,
        batch_size=8,
        compile=True,
        compile_mode="reduce-overhead",
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
