"""
Tensor-parallel layout of Inkling-2 for any number of ranks (three single-GPU DGX Spark nodes in particular).

Inkling-2-Small has 48 query heads in 8 GQA groups (6 query heads per KV head). The original engine required the KV heads
to divide evenly over the ranks, which rules out 3 ranks. Here attention is split by whole KV groups instead: rank r
holds KV heads [kv0, kv1) and the query heads of those groups, [6 * kv0, 6 * kv1). Every rank keeps the uniform group
size the attention kernels are written for (G = 6); only the number of heads per rank varies (3 / 3 / 2 KV heads, 18 /
18 / 12 query heads on 3 ranks). The output projection's partial sums are added over the ranks exactly as before, so
the math does not depend on how the heads are distributed.

The routed experts are split evenly (expert parallel, 384 / 3 = 128 per rank) and the vocabulary into equal contiguous
row ranges (201024 = 3 * 67008).

Override the KV-group split with I2_KV_SPLIT, e.g. I2_KV_SPLIT=3,3,2 (one count per rank, summing to the KV heads).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import torch


def split_counts(n: int, world: int) -> list[int]:
    """n items over `world` ranks as evenly as possible; the first ranks take the remainder"""
    assert world >= 1 and n >= world, f"cannot split {n} items over {world} ranks"
    q, r = divmod(n, world)
    return [q + (1 if i < r else 0) for i in range(world)]


def kv_counts(kv_heads: int, world: int) -> list[int]:
    env = os.environ.get("I2_KV_SPLIT", "").strip()
    if env:
        counts = [int(x) for x in env.split(",")]
        assert len(counts) == world, f"I2_KV_SPLIT={env} has {len(counts)} entries for {world} ranks"
        assert sum(counts) == kv_heads and min(counts) >= 1, f"I2_KV_SPLIT={env} must be >= 1 per rank and sum to {kv_heads}"
        return counts
    return split_counts(kv_heads, world)


@dataclass(frozen = True)
class HeadShard:
    """The attention heads one rank holds: KV heads [kv0, kv1) and query heads [q0, q1) (whole GQA groups)"""
    rank: int
    world: int
    heads: int          # total query heads
    kv_heads: int       # total KV heads
    kv0: int
    kv1: int

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads

    @property
    def q0(self) -> int:
        return self.kv0 * self.group

    @property
    def q1(self) -> int:
        return self.kv1 * self.group

    @property
    def hkl(self) -> int:
        return self.kv1 - self.kv0

    @property
    def hl(self) -> int:
        return self.q1 - self.q0

    def q_rows(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """this rank's slice of a tensor that stacks one equal block per query head along `dim`"""
        return head_slice(t, self.heads, self.q0, self.q1, dim)

    def kv_rows(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """this rank's slice of a tensor that stacks one equal block per KV head along `dim`"""
        return head_slice(t, self.kv_heads, self.kv0, self.kv1, dim)


def head_shard(heads: int, kv_heads: int, rank: int, world: int) -> HeadShard:
    assert heads % kv_heads == 0, f"{heads} query heads do not form whole groups over {kv_heads} KV heads"
    counts = kv_counts(kv_heads, world)
    kv0 = sum(counts[:rank])
    return HeadShard(rank, world, heads, kv_heads, kv0, kv0 + counts[rank])


def head_slice(t: torch.Tensor, total: int, h0: int, h1: int, dim: int = 0) -> torch.Tensor:
    """slice [h0, h1) of `total` equal blocks of `t` along `dim`, contiguous"""
    n = t.shape[dim]
    assert n % total == 0, f"dim {dim} of {tuple(t.shape)} is not {total} equal blocks"
    u = n // total
    return t.narrow(dim, h0 * u, (h1 - h0) * u).contiguous()


def even_slice(t: torch.Tensor, rank: int, world: int, dim: int = 0) -> torch.Tensor:
    """rank's equal contiguous slice along `dim` (dimension must divide evenly)"""
    n = t.shape[dim]
    assert n % world == 0, f"dim {dim} of {tuple(t.shape)} does not split evenly over {world} ranks"
    u = n // world
    return t.narrow(dim, rank * u, u).contiguous()


def local_device_index() -> int:
    """CUDA device of this process: LOCAL_RANK under torchrun (one GPU per node on DGX Spark), else RANK"""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
