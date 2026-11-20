"""
Prefix caching support for the Inkling-2 engine: snapshots of the per-slot state that is not paged.

Inkling-2 keeps two kinds of sequence state:
  * paged KV of the 8 global-attention layers (and of the draft's global layer): positions map to pages through the
    slot's page table, so a prefix can be shared by mapping the same pages (copy-on-write for the page that holds the
    prefix boundary, since the next request writes the rest of that page);
  * per-slot rings that are overwritten as a sequence advances: the sliding-window KV of the 40 local layers (positions
    modulo ~1.5K), the short-convolution input rings of every layer (positions modulo 16) and the draft's local-layer KV.
    These cannot be shared in place. A snapshot copies them at a prefix boundary; restoring it into another slot puts
    that slot exactly where the prefix ended.

Every rank holds its own shard of all of this, so snapshot / restore / page copies run on every rank (the server
broadcasts them). The pool is a set of preallocated tensors with a leading snapshot dimension; save and load are plain
device copies (~70 MB per snapshot per rank with 3 KV heads, 8-bit local KV).
"""
from __future__ import annotations
import torch


class SlotSnapshots:
    def __init__(self, eng, draft, n: int):
        self.n = n
        self.pairs = []                      # (live tensor with a leading slot dim, pool tensor with a leading snapshot dim)
        live = []
        for i, t4 in eng.kv.items():
            if not eng.cfg.is_global(i):
                live += [t for t in t4 if t is not None]
        for i, rs in eng.rings.items():
            live += list(rs)
        if draft is not None:
            for i in draft.LOCAL:
                live += [t for t in draft.kv[i] if t is not None]
        for t in live:
            assert t.shape[0] == eng.max_slots, f"per-slot tensor with leading dim {t.shape[0]} != {eng.max_slots}"
            self.pairs.append((t, torch.empty((n,) + tuple(t.shape[1:]), dtype = t.dtype, device = t.device)))

    @staticmethod
    def bytes_per_snapshot(eng, draft) -> int:
        tot = 0
        for i, t4 in eng.kv.items():
            if not eng.cfg.is_global(i):
                tot += sum(t[0].numel() * t.element_size() for t in t4 if t is not None)
        for i, rs in eng.rings.items():
            tot += sum(t[0].numel() * t.element_size() for t in rs)
        if draft is not None:
            for i in draft.LOCAL:
                tot += sum(t[0].numel() * t.element_size() for t in draft.kv[i] if t is not None)
        return tot

    @torch.no_grad()
    def save(self, slot: int, sid: int):
        for t, p in self.pairs:
            p[sid].copy_(t[slot])

    @torch.no_grad()
    def load(self, sid: int, slot: int):
        for t, p in self.pairs:
            t[slot].copy_(p[sid])


@torch.no_grad()
def copy_page(eng, draft, src: int, dst: int):
    """Copy one KV page (all global layers of the target and the draft's global layer) from src to dst"""
    for i, t4 in eng.kv.items():
        if eng.cfg.is_global(i):
            for t in t4:
                if t is not None:
                    t[dst].copy_(t[src])
    if draft is not None:
        for i in range(len(draft.kv)):
            if i not in draft.LOCAL:
                for t in draft.kv[i]:
                    t[dst].copy_(t[src])
                if draft.kvs[i] is not None:
                    for t in draft.kvs[i]:
                        t[dst].copy_(t[src])
