"""Model zoo for demos and benchmarks: a transformer "big brain" policy built from identical blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Pre-norm transformer block (attention over a short token sequence + MLP)."""

    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, dtype=torch.float16):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, dtype=dtype)
        self.qkv = nn.Linear(d, 3 * d, bias=False, dtype=dtype)
        self.proj = nn.Linear(d, d, bias=False, dtype=dtype)
        self.n2 = nn.LayerNorm(d, dtype=dtype)
        self.fc1 = nn.Linear(d, mlp_ratio * d, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(mlp_ratio * d, d, bias=False, dtype=dtype)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


def iter_blocks(depth: int, d: int, heads: int, dtype=torch.float16, seed: int = 0):
    """Build blocks one at a time (deterministic per block), so a huge model never sits in RAM twice."""
    for i in range(depth):
        torch.manual_seed(seed * 100_003 + i)
        b = Block(d, heads, dtype=dtype)
        with torch.no_grad():
            for p in b.parameters():
                if p.dim() == 2:
                    p.normal_(0, 0.02 / (2 * depth) ** 0.5)  # small init keeps a deep random stack finite
        yield b


def make_blocks(depth: int, d: int, heads: int, dtype=torch.float16, seed: int = 0) -> list[nn.Module]:
    return list(iter_blocks(depth, d, heads, dtype, seed))


def params_of(depth: int, d: int, mlp_ratio: int = 4) -> int:
    return depth * ((3 + 1 + 2 * mlp_ratio) * d * d + 4 * d)


PRESETS = {
    # name: (depth, d, heads)  -> parameter count
    "tiny": (4, 256, 4),          # ~3M, sanity checks
    "1b": (24, 1856, 16),         # ~1.0B, 2.0 GB fp16
    "2b": (32, 2304, 18),         # ~2.0B, 4.1 GB fp16: does not fit a 4 GB card
    "3b": (40, 2560, 20),         # ~3.1B, 6.3 GB fp16
}
