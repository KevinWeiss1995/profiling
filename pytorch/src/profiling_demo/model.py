"""Small, matmul-heavy transformer block stack.

Sized so fp32 fits comfortably on a single Hopper GPU while still giving
bf16 / torch.compile something meaningful to optimize. Uses pre-LN and the
fused ``F.scaled_dot_product_attention`` so the backward pass stays realistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 6
    mlp_mult: int = 4
    seq_len: int = 512
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # (b, h, t, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_mult
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """Decoder-only transformer returning per-token logits."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying cuts params and makes the backward pass a bit meatier.
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def build_model(cfg: ModelConfig, device: torch.device) -> TinyTransformer:
    model = TinyTransformer(cfg).to(device)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
