"""
Inkling-2 DSpark draft (mtp.safetensors): an anchor token plus 7 mask slots run through a 5-layer dense transformer that
attends to K/V projected from DFlare-mixed target features, followed by a recurrent (rnn_hidden) Markov head that
corrects the base logits token by token. Numerics follow vllm.models.inkling2.dspark (the reference's draft).
Tensor parallel over attention heads / MLP width, like the target engine.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from safetensors import safe_open

from .inkling2_parallel import even_slice

BLOCK = 8                 # anchor + 7 masks
RINGD = 1024 + 8 + 56     # local-layer draft KV ring (positions modulo RINGD)


def rms(x, w, eps = 1e-6):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return (y * w.float()).to(torch.bfloat16) if w is not None else y.to(torch.bfloat16)


def rope_neox(x, pos, theta = 1e6):
    """x: [R, H, 128] bf16, pos: [R] -> bf16 (fp32 math)"""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device = x.device, dtype = torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)


@triton.jit
def _dattn_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, page_table_ptr, pt_stride, slot_ptr, pa_ptr, pm_ptr, pl_ptr, po_ptr,
                  G: tl.constexpr, HKL: tl.constexpr, DH: tl.constexpr, BN: tl.constexpr, NSPLIT: tl.constexpr,
                  IS_GLOBAL: tl.constexpr, RING: tl.constexpr, PS: tl.constexpr, WINDOW: tl.constexpr, BLK: tl.constexpr,
                  ROWS: tl.constexpr, Q8: tl.constexpr):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    slot = tl.load(slot_ptr + b)
    pa = tl.load(pa_ptr + b)              # position of the block's first row (anchor)
    HL: tl.constexpr = G * HKL
    offs_d = tl.arange(0, DH)
    r = tl.arange(0, ROWS)
    qi = r // G
    g = r % G
    rm = r < BLK * G
    hq = hk * G + g
    q = tl.load(q_ptr + ((b * BLK + qi) * HL + hq)[:, None] * DH + offs_d[None, :], mask = rm[:, None], other = 0.0)
    if IS_GLOBAL:
        lo = 0
    else:
        lo = tl.maximum(0, pa - WINDOW)
    hi = pa + BLK - 1
    n = hi - lo + 1
    nblk = (n + BN - 1) // BN
    bps = (nblk + NSPLIT - 1) // NSPLIT
    b0 = sp * bps
    b1 = tl.minimum(b0 + bps, nblk)
    m_i = tl.full([ROWS], -1e30, dtype = tl.float32)
    l_i = tl.zeros([ROWS], dtype = tl.float32)
    acc = tl.zeros([ROWS, DH], dtype = tl.float32)
    for blk in range(b0, b1):
        j = lo + blk * BN + tl.arange(0, BN)
        valid = j <= hi
        if IS_GLOBAL:
            pg = tl.load(page_table_ptr + slot * pt_stride + j // PS, mask = valid, other = 0)
            rows = pg.to(tl.int64) * PS + j % PS
        else:
            rows = slot.to(tl.int64) * RING + j % RING
        k = tl.load(kc_ptr + (rows * HKL + hk)[:, None] * DH + offs_d[None, :], mask = valid[:, None], other = 0.0)
        if Q8:
            ksc = tl.load(ks_ptr + rows * HKL + hk, mask = valid, other = 0.0).to(tl.float32)
            k = (k.to(tl.float32) * ksc[:, None]).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * 0.08838834764831845
        s = tl.where(valid[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc_ptr + (rows * HKL + hk)[:, None] * DH + offs_d[None, :], mask = valid[:, None], other = 0.0)
        if Q8:
            vsc = tl.load(vs_ptr + rows * HKL + hk, mask = valid, other = 0.0).to(tl.float32)
            v = (v.to(tl.float32) * vsc[:, None]).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    base = ((b * BLK + qi) * HL + hq) * NSPLIT + sp
    tl.store(pm_ptr + base, m_i, mask = rm)
    tl.store(pl_ptr + base, l_i, mask = rm)
    tl.store(po_ptr + base[:, None] * DH + offs_d[None, :], acc, mask = rm[:, None])


@triton.jit
def _dattn_combine(pm_ptr, pl_ptr, po_ptr, o_ptr, DH: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)          # (b * BLK + qi) * HL + hq
    offs_s = tl.arange(0, NSPLIT)
    offs_d = tl.arange(0, DH)
    base = row * NSPLIT
    m = tl.load(pm_ptr + base + offs_s)
    l = tl.load(pl_ptr + base + offs_s)
    mt = tl.max(m, 0)
    w = tl.exp(m - mt)
    den = tl.sum(l * w, 0)
    o = tl.load(po_ptr + (base + offs_s)[:, None] * DH + offs_d[None, :])
    o = tl.sum(o * w[:, None], 0) / den
    tl.store(o_ptr + row * DH + offs_d, o.to(tl.bfloat16))


class I2Draft:
    LOCAL = (0, 1, 3, 4)

    def __init__(self, path, eng, device, rank, world, max_slots, num_pages, page_size):
        self.eng, self.dev, self.rank, self.world = eng, device, rank, world
        self.cfg = eng.cfg
        self.max_slots, self.ps = max_slots, page_size
        D = 4608
        # attention by whole KV groups like the target (same split, see inkling2_parallel.py); MLP width split evenly
        self.shard = eng.shard
        self.Hl, self.Hkl = self.shard.hl, self.shard.hkl
        self.G = self.Hl // self.Hkl
        f = safe_open(path, "pt")
        g = lambda k: f.get_tensor(k).to(device)
        qs, ks = self.shard.q_rows, self.shard.kv_rows
        sh = lambda t, dim = 0: even_slice(t, rank, world, dim)
        self.L = []
        for i in range(5):
            p = f"layers.{i}."
            L = {}
            L["ln_in"], L["ln_post"], L["dflare"] = g(p + "input_layernorm.weight"), g(p + "post_attention_layernorm.weight"), g(p + "dflare_norm.weight")
            L["qn"], L["kn"], L["knc"] = g(p + "self_attn.q_norm.weight"), g(p + "self_attn.k_norm.weight"), g(p + "self_attn.k_norm_ctx.weight")
            a = p + "self_attn."
            L["wqkv"] = torch.cat([qs(g(a + "q_proj.weight")), ks(g(a + "k_proj.weight")), ks(g(a + "v_proj.weight"))], 0).contiguous()
            L["wkv_ctx"] = torch.cat([ks(g(a + "k_proj_ctx.weight")), ks(g(a + "v_proj_ctx.weight"))], 0).contiguous()
            L["wo"] = qs(g(a + "o_proj.weight"), 1)
            L["wgu"] = torch.cat([sh(g(p + "mlp.gate_proj.weight")), sh(g(p + "mlp.up_proj.weight"))], 0).contiguous()
            L["wd"] = sh(g(p + "mlp.down_proj.weight"), 1)
            self.L.append(L)
        # FP8 weight-only (e4m3, per-output-row scale) for the block-forward projections (M = 8 x batch rows <= 128)
        self.fp8 = __import__("os").environ.get("I2_DRAFT_FP8", "0") == "1"
        if self.fp8:
            from .inkling2_gemm import quant_fp8_rows
            for L in self.L:
                for k in ("wqkv", "wo", "wgu", "wd"):
                    L[k + "_q"], L[k + "_s"] = quant_fp8_rows(L[k])
                    del L[k]
            torch.cuda.empty_cache()
        self.markov_fused = __import__("os").environ.get("I2_MARKOV_FUSED", "1") == "1"
        self._mstep = {}
        self.norm = g("norm.weight")
        self.mask_embed = g("mask_embed")
        self.fuse_alpha = torch.softmax(g("fuse_weight").float(), -1)          # [6, 8]
        mh = "markov_head."
        self.m_dflare = g(mh + "dflare_norm.weight")
        self.w1 = g(mh + "markov_w1.weight")                                  # [V, 256]
        self.rnn_w, self.rnn_b = g(mh + "rnn_proj.weight"), g(mh + "rnn_proj.bias").float()
        ho_w, ho_b = g(mh + "hidden_out.weight"), g(mh + "hidden_out.bias")
        # fold hidden_out into this rank's lm_head rows (fp64, rounded once)
        U = eng.head.dense_rows()                        # bf16 rows, or the EXL3 head (v3) reconstructed once here
        self.fold_w = torch.empty((U.shape[0], 256), dtype = torch.bfloat16, device = device)
        self.fold_b = torch.empty((U.shape[0],), dtype = torch.bfloat16, device = device)
        for s in range(0, U.shape[0], 4096):
            hd = U[s:s + 4096].double()
            self.fold_w[s:s + 4096] = (hd @ ho_w.double()).bfloat16()
            self.fold_b[s:s + 4096] = (hd @ ho_b.double()).bfloat16()
        # draft KV caches (bf16): local layers ring per slot, global layer paged (shares the target's page table)
        # global layer (2): int8 codes + fp16 per-(token, head) scale, as the target's 8-bit KV
        self.kv = [None] * 5; self.kvs = [None] * 5
        self.alloc(num_pages)

    def alloc(self, num_pages):
        dev = self.dev
        for i in range(5):
            if i in self.LOCAL:
                if self.kv[i] is None:
                    shape = (self.max_slots, RINGD, self.Hkl, 128)
                    self.kv[i] = (torch.zeros(shape, dtype = torch.bfloat16, device = dev), torch.zeros(shape, dtype = torch.bfloat16, device = dev))
            else:
                shape = (num_pages, self.ps, self.Hkl, 128)
                self.kv[i] = (torch.zeros(shape, dtype = torch.int8, device = dev), torch.zeros(shape, dtype = torch.int8, device = dev))
                self.kvs[i] = (torch.zeros(shape[:-1], dtype = torch.half, device = dev), torch.zeros(shape[:-1], dtype = torch.half, device = dev))

    @staticmethod
    def global_bytes_per_token(hkl):
        return 2 * hkl * (128 + 2)

    def _write(self, i, rows, k, v):
        """k, v: [R, Hkl, 128] bf16 -> cache rows"""
        kc, vc = self.kv[i]
        if i in self.LOCAL:
            kc.view(-1, self.Hkl, 128)[rows] = k
            vc.view(-1, self.Hkl, 128)[rows] = v
            return
        ks, vs = self.kvs[i]
        for src, dst, sdst in ((k, kc, ks), (v, vc, vs)):
            f = src.float()
            sc = (f.abs().amax(-1) / 127.0).clamp_min(1e-8)
            q = torch.round(f / sc[..., None]).clamp_(-128, 127).to(torch.int8)
            dst.view(-1, self.Hkl, 128)[rows] = q
            sdst.view(-1, self.Hkl)[rows] = sc.half()

    # ---- helpers
    def mix(self, feats, row, w):
        """feats [R, 8, D] bf16 -> rmsnorm(sum alpha_t f_t) * w  (fp32 math, one rounding)"""
        a = self.fuse_alpha[row]
        m = (feats.float() * a[None, :, None]).sum(1)
        y = m * torch.rsqrt(m.square().mean(-1, keepdim = True) + 1e-6)
        return (y * w.float()).to(torch.bfloat16)

    def _cache_rows(self, i, slot, pos):
        if i in self.LOCAL:
            return slot.long() * RINGD + (pos % RINGD).long()
        pt = self.eng.page_table
        return pt[slot.long(), (pos // self.ps).long()].long() * self.ps + (pos % self.ps).long()

    def store_context(self, feats, slot, pos):
        """feats [R, 8, D], slot/pos [R] int32: write the context K/V (all 5 layers) of the target features"""
        R = feats.shape[0]
        full = feats, slot, pos
        for i, L in enumerate(self.L):
            feats, slot, pos = full
            R = feats.shape[0]
            if i in self.LOCAL and R > RINGD:        # long single-sequence chunk: only the last RINGD rows can be read
                feats, slot, pos = feats[-RINGD:], slot[-RINGD:], pos[-RINGD:]
            R = feats.shape[0]
            ctx = self.mix(feats, i, L["dflare"])
            kv = torch.mm(ctx, L["wkv_ctx"].t())
            k, v = kv[:, : self.Hkl * 128], kv[:, self.Hkl * 128:]
            k = rms(k.reshape(-1, 128), L["knc"]).view(R, self.Hkl, 128)
            k = rope_neox(k, pos)
            rows = self._cache_rows(i, slot, pos)
            self._write(i, rows, k, v.reshape(R, self.Hkl, 128))

    def attention(self, i, q, slot, pa, nsplit):
        B = slot.shape[0]
        HL = self.Hl
        kc, vc = self.kv[i]
        ks, vs = self.kvs[i] if self.kvs[i] is not None else (kc, vc)
        pm = torch.empty((B * BLOCK * HL, nsplit), dtype = torch.float32, device = q.device)
        pl = torch.empty_like(pm)
        po = torch.empty((B * BLOCK * HL, nsplit, 128), dtype = torch.float32, device = q.device)
        o = torch.empty((B * BLOCK, HL * 128), dtype = torch.bfloat16, device = q.device)
        _dattn_kernel[(B, self.Hkl, nsplit)](
            q, kc, vc, ks, vs, self.eng.page_table, self.eng.page_table.stride(0), slot, pa, pm, pl, po,
            G = self.G, HKL = self.Hkl, DH = 128, BN = 64, NSPLIT = nsplit, IS_GLOBAL = i not in self.LOCAL,
            RING = RINGD, PS = self.ps, WINDOW = 1024, BLK = BLOCK, ROWS = 64, Q8 = i not in self.LOCAL, num_warps = 4)
        _dattn_combine[(B * BLOCK * HL,)](pm, pl, po, o, DH = 128, NSPLIT = nsplit)
        return o

    def _mm(self, x, L, k):
        if self.fp8:
            from .inkling2_gemm import skinny_mm
            return skinny_mm(x, L[k + "_q"], L[k + "_s"], out_dtype = torch.bfloat16)
        return torch.mm(x, L[k].t())

    def block_forward(self, anchor_tok, slot, pa, nsplit_g = 4, nsplit_l = 4):
        """anchor_tok [B] long, slot [B] int32, pa [B] int32 (anchor positions). Returns hidden [B*8, D] bf16"""
        B = slot.shape[0]
        D = 4608
        eng = self.eng
        x = self.mask_embed.expand(B, BLOCK, D).clone()
        x[:, 0] = eng.embed_tokens(anchor_tok)
        x = x.view(B * BLOCK, D)
        pos = (pa[:, None] + torch.arange(BLOCK, device = x.device, dtype = torch.int32)[None]).reshape(-1)
        slot_r = slot.repeat_interleave(BLOCK)
        res = x
        xn = rms(x, self.L[0]["ln_in"])
        for i, L in enumerate(self.L):
            qkv = self._mm(xn, L, "wqkv")
            q = qkv[:, : self.Hl * 128].reshape(-1, 128)
            k = qkv[:, self.Hl * 128: (self.Hl + self.Hkl) * 128].reshape(-1, 128)
            v = qkv[:, (self.Hl + self.Hkl) * 128:]
            q = rope_neox(rms(q, L["qn"]).view(-1, self.Hl, 128), pos).reshape(B * BLOCK, self.Hl * 128)
            k = rope_neox(rms(k, L["kn"]).view(-1, self.Hkl, 128), pos)
            rows = self._cache_rows(i, slot_r, pos)
            self._write(i, rows, k, v.reshape(-1, self.Hkl, 128))
            o = self.attention(i, q, slot, pa, nsplit_l if i in self.LOCAL else nsplit_g)
            o = self._mm(o, L, "wo")
            eng.allreduce(o)
            res = (res.float() + o.float()).to(torch.bfloat16)
            xn = rms(res, L["ln_post"])
            gu = self._mm(xn, L, "wgu")
            g, u = gu.float().chunk(2, -1)
            act = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
            y = self._mm(act, L, "wd")
            eng.allreduce(y)
            res = (res.float() + y.float()).to(torch.bfloat16)
            xn = rms(res, self.L[i + 1]["ln_in"] if i + 1 < 5 else self.norm)
        return xn

    def draft(self, anchor_tok, slot, pa, f_feats, nsplit_g = 4, nsplit_l = 4, K = 7):
        """Draft K (<= 7) tokens for B sequences. f_feats: [B, 8, D] target features of the last committed row.
        Returns tokens [B, K] (int64). The block forward always runs the full 8-row block (same numerics for any K)"""
        B = slot.shape[0]
        eng = self.eng
        hid = self.block_forward(anchor_tok, slot, pa, nsplit_g, nsplit_l).view(B, BLOCK, -1)[:, 1:K + 1]   # [B, K, D]
        hid = hid.reshape(B * K, -1)
        base = eng.head_mm_bf16(hid)                                                                         # bf16 [B*K, V/w]
        f_rnn = self.mix(f_feats, 5, self.m_dflare)                                                          # [B, D]
        Wc = self.rnn_w[:, 512:]
        inp = torch.cat([hid.view(B, K, -1), f_rnn[:, None].expand(B, K, -1)], -1).transpose(0, 1).reshape(K * B, -1)
        ctx = torch.addmm(self.rnn_b, inp, Wc.t(), out_dtype = torch.float32).view(K, B, -1)
        Wr = self.rnn_w[:, :512]
        state = torch.zeros((B, 256), dtype = torch.bfloat16, device = hid.device)
        prev = anchor_tok
        base = base.view(B, K, -1)
        out = []
        if self.markov_fused:
            from .inkling2_markov import MarkovStep
            ms = self._mstep.get(B)
            if ms is None:
                ms = self._mstep[B] = MarkovStep(self, B)
            for i in range(K):
                state, prev = ms(state, prev, ctx[i], base[:, i])
                out.append(prev)
            return torch.stack(out, 1)
        r = lambda t: t.bfloat16().float()
        for i in range(K):
            inputs = torch.cat([state, self.w1[prev]], -1)
            raw = torch.addmm(ctx[i], inputs, Wr.t(), out_dtype = torch.float32)
            gate_raw, cand_raw, out_raw = r(raw).chunk(3, -1)
            gate = r(torch.sigmoid(gate_raw))
            cand = r(torch.tanh(cand_raw))
            keep = r(gate * state.float())
            upd = r(r(1.0 - gate) * cand)
            state = (keep + upd).to(torch.bfloat16)
            latent = torch.tanh(out_raw).to(torch.bfloat16)
            bias = torch.addmm(self.fold_b, latent, self.fold_w.t())
            z = ((base[:, i] + bias).float() / 30.0).tanh() * 30.0
            prev = eng.global_argmax(z.bfloat16().float())
            out.append(prev)
        return torch.stack(out, 1)
