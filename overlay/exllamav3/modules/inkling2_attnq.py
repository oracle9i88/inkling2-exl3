"""
EXL3-quantized attention projections of Inkling-2 (written by spark/quant/quantize_attention.py), sliced for tensor
parallelism.

Tensors in the checkpoint (the whole model, for any tensor-parallel split by whole KV groups):
  sliding layers  attn.wqgkv   rows [q (H*128) | gate (H*128) | k (Hk*128) | v (Hk*128)], input: the attention norm
  global layers   attn.wqg     rows [q | gate], input: the q norm;  attn.wk, attn.wv: the k and v norms
  all layers      attn.wo      input columns: the H*128 gated attention outputs
EXL3 keeps the trellis as [in / 16, out / 16, 16 * bits] tiles and applies its Hadamard transforms in 128-blocks on
both sides, so whole 128-blocks of output rows (or of input columns) of a tensor quantized as a whole are a quantized
tensor on their own: a rank takes its heads' blocks. wr (16 rows a head), attn_sink_proj (1 a head) and the global
layers' k / v conv projections stay bf16 (their per-rank slices are not whole blocks).
"""
from __future__ import annotations
import os
import torch
from .quant.exl3 import LinearEXL3

ENABLED = os.environ.get("I2_ATTN_Q", "1") == "1"        # 0: run the bf16 projections even if quantized ones exist


def has_quantized_attention(stc, prefix: str, is_global: bool) -> bool:
    first = prefix + ("wqg" if is_global else "wqgkv") + ".trellis"
    return ENABLED and stc.has_tensor(first) and stc.has_tensor(prefix + "wo.trellis")


def exl3_select(stc, key: str, device, rows = None, cols = None, config = None) -> LinearEXL3:
    """LinearEXL3 of `key` restricted to the output row ranges `rows` or the input column ranges `cols`
    ([(start, end), ...], multiples of 128), concatenated in the given order"""
    g = lambda s: stc.get_tensor(f"{key}.{s}", device, no_defer = True) if stc.has_tensor(f"{key}.{s}") else None
    trellis, suh, svh, mcg, mul1 = g("trellis"), g("suh"), g("svh"), g("mcg"), g("mul1")
    if rows is not None:
        assert all(r0 % 128 == 0 and r1 % 128 == 0 and r0 < r1 <= svh.shape[0] for r0, r1 in rows), (key, rows)
        trellis = torch.cat([trellis[:, r0 // 16:r1 // 16] for r0, r1 in rows], 1).contiguous()
        svh = torch.cat([svh[r0:r1] for r0, r1 in rows]).contiguous()
    if cols is not None:
        assert all(c0 % 128 == 0 and c1 % 128 == 0 and c0 < c1 <= suh.shape[0] for c0, c1 in cols), (key, cols)
        trellis = torch.cat([trellis[c0 // 16:c1 // 16] for c0, c1 in cols], 0).contiguous()
        suh = torch.cat([suh[c0:c1] for c0, c1 in cols]).contiguous()
    assert trellis.shape[0] * 16 == suh.shape[0] and trellis.shape[1] * 16 == svh.shape[0], (key, tuple(trellis.shape))
    return LinearEXL3(config, suh.shape[0], svh.shape[0], suh = suh, svh = svh, trellis = trellis, mcg = mcg, mul1 = mul1,
                      out_dtype = torch.half, key = key)


def rank_ranges(shard, heads: int, kv_heads: int, head_dim: int = 128):
    """rows of this rank: query heads [q0, q1) and KV heads [kv0, kv1) as (start, end) in a [q | gate | k | v] tensor"""
    hq, hk = heads * head_dim, kv_heads * head_dim
    q = (shard.q0 * head_dim, shard.q1 * head_dim)
    kv = (shard.kv0 * head_dim, shard.kv1 * head_dim)
    off = lambda r, o: (r[0] + o, r[1] + o)
    return {"q": q, "gate": off(q, hq), "k": off(kv, 2 * hq), "v": off(kv, 2 * hq + hk), "kv": kv}


def run(lin: LinearEXL3, x: torch.Tensor) -> torch.Tensor:
    """x [T, in] (any float dtype) -> [T, out] fp16"""
    return lin.forward(x.to(torch.half).contiguous(), {}, torch.half)


# ---- the head (model.llm.unembed_q: the vocabulary padded with zero rows to a multiple of 256, so every rank of a
# 1-, 2- or 3-way split holds whole 128-row blocks; logits of the padding are cut by unpadded_vocab as before)

HEAD_ENABLED = os.environ.get("I2_HEAD_Q", "1") == "1"   # 0: the bf16 head even if a quantized one exists
HEAD_KEY = "model.llm.unembed_q"


def has_quantized_head(stc) -> bool:
    return HEAD_ENABLED and stc.has_tensor(HEAD_KEY + ".trellis")


def head_rows(stc, rank: int, world: int):
    vp = stc.get_tensor(HEAD_KEY + ".svh", "cpu", no_defer = True).shape[0]
    assert vp % (128 * world) == 0, (vp, world)
    n = vp // world
    return rank * n, (rank + 1) * n


def run_f32(lin: LinearEXL3, x: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """x [T, in] -> [T, out] fp32 logits from the small-batch kernels (fp32 output), at most `chunk` rows a call"""
    xh = x.to(torch.half).contiguous()
    if xh.shape[0] <= chunk:
        return lin.forward(xh, {}, torch.float)
    return torch.cat([lin.forward(xh[s:s + chunk].contiguous(), {}, torch.float) for s in range(0, xh.shape[0], chunk)], 0)
