"""
Fused DSpark Markov-head step for the Inkling-2 drafter (same bf16 rounding points as I2Draft.draft's torch loop):

    raw    = ctx_i + [state | w1[prev]] @ Wr^T                 (fp32)
    gate   = bf16(sigmoid(bf16(raw_g)));  cand = bf16(tanh(bf16(raw_c)))
    state  = bf16(bf16(gate * state) + bf16(bf16(1 - gate) * cand));  latent = bf16(tanh(bf16(raw_o)))
    bias   = bf16(fold_b + latent @ fold_w^T);  z = bf16(tanh(bf16(base_i + bias) / 30) * 30)
    prev   = argmax over the full vocabulary (per-shard partial argmax, one packed (value, index) all-gather)

Kernels: _cell (one program per 32 state channels, all rows), _fold (one program per 256 vocab columns: GEMM + softcap +
tile argmax), _reduce (per row: tile partials -> one (value, global index) pair).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
import torch.distributed as dist


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _cell_kernel(state_ptr, prev_ptr, w1_ptr, wr_ptr, swr, ctx_ptr, state_out_ptr, latent_ptr, B,
                 BP: tl.constexpr, JT: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    jt = tl.program_id(0)
    offs_b = tl.arange(0, BP)
    mb = offs_b < B
    offs_j = jt * JT + tl.arange(0, JT)
    prev = tl.load(prev_ptr + offs_b, mask=mb, other=0)
    acc_g = tl.zeros([BP, JT], dtype=tl.float32)
    acc_c = tl.zeros([BP, JT], dtype=tl.float32)
    acc_o = tl.zeros([BP, JT], dtype=tl.float32)
    for k0 in tl.static_range(0, 2 * S, BK):
        offs_k = k0 + tl.arange(0, BK)
        if k0 < S:
            xin = tl.load(state_ptr + offs_b[:, None] * S + offs_k[None, :], mask=mb[:, None], other=0.0)
        else:
            xin = tl.load(w1_ptr + prev[:, None].to(tl.int64) * S + (offs_k[None, :] - S), mask=mb[:, None], other=0.0)
        wg = tl.load(wr_ptr + offs_j[:, None] * swr + offs_k[None, :])
        wc = tl.load(wr_ptr + (S + offs_j)[:, None] * swr + offs_k[None, :])
        wo = tl.load(wr_ptr + (2 * S + offs_j)[:, None] * swr + offs_k[None, :])
        acc_g += tl.dot(xin, tl.trans(wg))
        acc_c += tl.dot(xin, tl.trans(wc))
        acc_o += tl.dot(xin, tl.trans(wo))
    m2 = mb[:, None]
    cp = ctx_ptr + offs_b[:, None] * (3 * S) + offs_j[None, :]
    rg = _bf(acc_g + tl.load(cp, mask=m2, other=0.0))
    rc = _bf(acc_c + tl.load(cp + S, mask=m2, other=0.0))
    ro = _bf(acc_o + tl.load(cp + 2 * S, mask=m2, other=0.0))
    gate = _bf(tl.sigmoid(rg))
    cand = _bf(tl.extra.cuda.libdevice.tanh(rc))
    st = tl.load(state_ptr + offs_b[:, None] * S + offs_j[None, :], mask=m2, other=0.0).to(tl.float32)
    keep = _bf(gate * st)
    upd = _bf(_bf(1.0 - gate) * cand)
    tl.store(state_out_ptr + offs_b[:, None] * S + offs_j[None, :], (keep + upd).to(tl.bfloat16), mask=m2)
    tl.store(latent_ptr + offs_b[:, None] * S + offs_j[None, :], tl.extra.cuda.libdevice.tanh(ro).to(tl.bfloat16), mask=m2)


@triton.jit
def _fold_kernel(latent_ptr, fw_ptr, fb_ptr, base_ptr, sbase, pv_ptr, pi_ptr, B, Vs, valid, NT,
                 BP: tl.constexpr, BN: tl.constexpr, S: tl.constexpr):
    nt = tl.program_id(0)
    offs_b = tl.arange(0, BP)
    mb = offs_b < B
    offs_n = nt * BN + tl.arange(0, BN)
    mn = offs_n < Vs
    offs_k = tl.arange(0, S)
    lat = tl.load(latent_ptr + offs_b[:, None] * S + offs_k[None, :], mask=mb[:, None], other=0.0)
    fw = tl.load(fw_ptr + offs_n[:, None].to(tl.int64) * S + offs_k[None, :], mask=mn[:, None], other=0.0)
    acc = tl.dot(lat, tl.trans(fw))
    fb = tl.load(fb_ptr + offs_n, mask=mn, other=0.0).to(tl.float32)
    bias = _bf(acc + fb[None, :])
    bs = tl.load(base_ptr + offs_b[:, None] * sbase + offs_n[None, :], mask=mb[:, None] & mn[None, :], other=0.0).to(tl.float32)
    y = _bf(bs + bias) / 30.0
    z = _bf(tl.extra.cuda.libdevice.tanh(y) * 30.0)
    z = tl.where((offs_n < valid)[None, :], z, -float("inf"))
    v = tl.max(z, 1)
    i = tl.argmax(z, 1)
    tl.store(pv_ptr + offs_b * NT + nt, v, mask=mb)
    tl.store(pi_ptr + offs_b * NT + nt, (nt * BN + i).to(tl.int32), mask=mb)


@triton.jit
def _reduce_kernel(pv_ptr, pi_ptr, out_ptr, NT, base_idx, NTP: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, NTP)
    m = offs < NT
    v = tl.load(pv_ptr + b * NT + offs, mask=m, other=-float("inf"))
    i = tl.load(pi_ptr + b * NT + offs, mask=m, other=2147483647)
    best = tl.max(v, 0)
    # first (lowest) vocab index among the tile maxima
    bi = tl.min(tl.where(v == best, i, 2147483647), 0)
    tl.store(out_ptr + b * 2, best)
    tl.store(out_ptr + b * 2 + 1, (bi + base_idx).to(tl.float32))


class MarkovStep:
    def __init__(self, draft, B):
        self.d = draft
        self.B = B
        self.BP = max(16, triton.next_power_of_2(B))
        eng = draft.eng
        self.Vs = eng.head.rows_n
        self.valid = min(self.Vs, eng.cfg.unpadded_vocab - eng.rank * self.Vs)
        self.BN = 128
        self.NT = triton.cdiv(self.Vs, self.BN)
        self.rank, self.world = eng.rank, eng.world

    def __call__(self, state, prev, ctx_i, base_i):
        """state [B, 256] bf16, prev [B] int64, ctx_i [B, 768] fp32, base_i [B, Vs] bf16 view (row stride arbitrary).
        Returns (new state, next tokens [B] int64)"""
        d, B, dev = self.d, self.B, state.device
        S = 256
        Wr = d.rnn_w
        new_state = torch.empty_like(state)
        latent = torch.empty_like(state)
        _cell_kernel[(S // 32,)](state, prev, d.w1, Wr, Wr.stride(0), ctx_i, new_state, latent, B,
                                 BP=self.BP, JT=32, BK=64, S=S, num_warps=4)
        pv = torch.empty((B, self.NT), dtype=torch.float32, device=dev)
        pi = torch.empty((B, self.NT), dtype=torch.int32, device=dev)
        _fold_kernel[(self.NT,)](latent, d.fold_w, d.fold_b, base_i, base_i.stride(0), pv, pi, B, self.Vs, self.valid, self.NT,
                                 BP=self.BP, BN=self.BN, S=S, num_warps=8)
        loc = torch.empty((B, 2), dtype=torch.float32, device=dev)
        _reduce_kernel[(B,)](pv, pi, loc, self.NT, self.rank * self.Vs, NTP=triton.next_power_of_2(self.NT), num_warps=4)
        if self.world == 1:
            return new_state, loc[:, 1].long()
        allv = torch.empty((self.world, B, 2), dtype=torch.float32, device=dev)
        dist.all_gather_into_tensor(allv, loc)
        best = allv[:, :, 0].argmax(0)
        tok = allv[:, :, 1].gather(0, best[None])[0].long()
        return new_state, tok
