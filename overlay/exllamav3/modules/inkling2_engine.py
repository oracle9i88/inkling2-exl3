"""
Rank-local batched Inkling-2 engine: state (KV pages, SWA rings, conv rings), layer math over a token batch, tensor
parallel attention (heads) + expert parallel MoE. One instance per GPU; ranks synchronize through torch.distributed
(NCCL) all-reduces. See inkling2.py for the reference math and inkling2_ops.py for the kernels.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import inkling2_ops as ops
from . import inkling2_attnq as AQ
from .inkling2_gemm import skinny_mm, quant_fp8_rows
from ..architecture.inkling2 import Inkling2Model, Inkling2Layer

RING_EXTRA = 64            # SWA ring headroom for speculative tokens beyond the window
# The SWA ring must also hold every row of one multi-row call (prefill chunk) beyond the window: all rows' K/V are written
# before any row's attention reads, so with window + 64 a 512-row chunk overwrote entries its own early rows still read
# (KV16 KLD 0.01623 at chunk 512 vs 0.01295 at chunk 64). max_rows = the largest rows-per-sequence of any call.
MAX_ROWS_DEFAULT = 512


class Batch:
    """Token batch descriptor (all int32 device tensors of length T; rows of a sequence contiguous, ascending)"""

    def __init__(self, slot, pos, nprev, multi: bool, nsplit_g: int = 1, nsplit_l: int = 4, nafter = None, seq_rows = None):
        self.slot, self.pos, self.nprev = slot, pos, nprev
        self.seq_rows = seq_rows        # every sequence has exactly this many rows (speculative verify), or None
        self.single = nafter is None
        if nafter is None:      # single sequence in the batch
            nafter = (slot.shape[0] - 1 - nprev).to(torch.int32)
        self.nafter = nafter
        self.T = slot.shape[0]
        self.multi = multi              # more than one row per sequence (prefill chunk / speculative verify)
        self.nsplit_g, self.nsplit_l = nsplit_g, nsplit_l


class I2Engine:
    def __init__(self, model: Inkling2Model, device, rank: int, world: int, max_slots: int, num_pages: int,
                 page_size: int = 64, bits_g: int = 4, bits_l: int = 8, max_pages_per_seq: int = 8192, embed_on_rank: int = 0,
                 kv_rot: bool = False, kv_ng: int = 1, kv_q16: bool = False, max_rows: int = MAX_ROWS_DEFAULT):
        """kv_rot / kv_ng / kv_q16 (global-layer KV only): randomized-Hadamard rotation of K/V/Q along head_dim before
        quantizing, scale groups per 128-vector (fp16 scales), 16-level symmetric int4 grid"""
        self.model, self.cfg = model, model.config.i2
        self.device, self.rank, self.world = device, rank, world
        self.max_slots, self.num_pages, self.ps = max_slots, num_pages, page_size
        self.bits_g, self.bits_l = bits_g, bits_l
        self.kv_rot = ops.make_kv_rotation(device) if kv_rot else None
        self.kv_ng, self.kv_q16 = kv_ng, kv_q16
        self.layers = model.layers
        self.embed = model.embed
        # vocab-parallel embedding: this rank keeps rows [rank*n, (rank+1)*n); lookups are summed over ranks (exact: one
        # nonzero contribution per token). Frees (world-1)/world of the 1.85 GB bf16 table per GPU
        V = model.embed.weight.shape[0]
        assert V % world == 0
        self.embed_n = V // world
        if world > 1:
            w = model.embed.weight[rank * self.embed_n:(rank + 1) * self.embed_n].clone()
            model.embed.weight = None
            torch.cuda.empty_cache()
            self.embed_w = w
        else:
            self.embed_w = model.embed.weight
        self.head = model.head
        c = self.cfg
        # heads of this rank: whole KV groups, possibly uneven over the ranks (3 / 3 / 2 KV heads on 3 ranks)
        self.shard = self.layers[0].shard
        assert all(L.shard == self.shard for L in self.layers)
        self.Hl, self.Hkl = self.shard.hl, self.shard.hkl
        self.G = self.Hl // self.Hkl
        assert self.G * self.Hkl == self.Hl and self.G == c.heads // c.kv_heads
        self.ring = c.window + max(RING_EXTRA, max_rows)
        self.max_rows = max_rows
        dev = device
        D = c.hidden
        self.blocks = None
        dhp = lambda bits: 64 if bits == 4 else 128
        ct = lambda bits: torch.bfloat16 if bits == 16 else torch.int8 if bits == 8 else torch.uint8
        self.kv = {}
        for L in self.layers:
            i = L.idx
            if c.is_global(i):
                shape = (num_pages, page_size, self.Hkl, dhp(bits_g))
                bits = bits_g
            else:
                shape = (max_slots, self.ring, self.Hkl, dhp(bits_l))
                bits = bits_l
            kq = torch.zeros(shape, dtype = ct(bits), device = dev)
            vq = torch.zeros(shape, dtype = ct(bits), device = dev)
            ks = vs = None
            if bits != 16:
                sshape = shape[:-1] + ((kv_ng,) if c.is_global(i) and kv_ng > 1 else ())
                ks = torch.zeros(sshape, dtype = torch.half, device = dev)
                vs = torch.zeros(sshape, dtype = torch.half, device = dev)
            self.kv[i] = (kq, vq, ks, vs)
        z = lambda ch: torch.zeros((max_slots, ops.RC, ch), dtype = torch.bfloat16, device = dev)
        self.rings = {L.idx: (z(self.Hkl * 128), z(self.Hkl * 128), z(D), z(D)) for L in self.layers}
        self.page_table = torch.zeros((max_slots, max_pages_per_seq), dtype = torch.int32, device = dev)
        self.group = None
        self.moe_chunk_max = int(__import__('os').environ.get('I2_MOE_CHUNK_MAX', 192))
        # dtype of the two per-layer all-reduces (attention o_proj partials, routed-expert partials): the sums are
        # rounded to bf16 by the consuming sconv anyway; fp16 halves the PCIe bytes vs fp32 with 3 more mantissa bits than bf16
        self.ar_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[__import__('os').environ.get('I2_AR_DTYPE', 'fp16')]
        self.skinny = __import__('os').environ.get('I2_SKINNY', '1') == '1'      # Triton skinny GEMM for T <= 64 projections
        self.rel_fused = __import__('os').environ.get('I2_REL_FUSED', '1') == '1'   # relative bias inside the attention kernel
        # long prefill, global layers: the keys more than rel_extent behind a chunk's first row go to a kernel that loads
        # each key tile once for many rows (ops._attn_far_kernel). I2_FAR = "near splits,far splits,rows*heads per far
        # program" (0: off), used for single-sequence calls of at least I2_FAR_MIN rows
        fs = __import__('os').environ.get('I2_FAR', '2,6,64')
        self.far = tuple(int(x) for x in fs.split(',')) if fs not in ('', '0') else None
        if self.far is not None:
            ns = self.far[0] + self.far[1]
            assert ns & (ns - 1) == 0 and self.far[2] & (self.far[2] - 1) == 0, f"I2_FAR {fs}: splits must sum to a power of 2"
        self.far_min = int(__import__('os').environ.get('I2_FAR_MIN', 256))
        # the same for speculative verify (Batch.seq_rows = R rows per sequence, one far program per sequence):
        # I2_FAR_V = "near splits,far splits" (0: off)
        fs = __import__('os').environ.get('I2_FAR_V', '2,30')
        self.far_v = tuple(int(x) for x in fs.split(',')) if fs not in ('', '0') else None
        if self.far_v is not None:
            ns = self.far_v[0] + self.far_v[1]
            assert ns & (ns - 1) == 0, f"I2_FAR_V {fs}: splits must sum to a power of 2"
        self.b12x = None          # optional inkling2_b12x.B12xMoE (b12x fused Trellis MoE for the prepared batch sizes)
        self.moe_group = int(__import__('os').environ.get('I2_MOE_GROUP', 64))   # rows per fused coop MoE call (P0, validated on v3; 8 = old path)
        _env = __import__('os').environ
        # FP8 (e4m3, per-row scale) copies of the attention projections for decode-sized calls (<= 64 rows, skinny GEMM).
        # On DGX Spark a one-token step is bound by reading these BF16 weights (~3.5 GB per rank); prefill keeps BF16.
        self.attn_fp8 = _env.get('I2_ATTN_FP8', '0') == '1'
        if self.attn_fp8:
            for L in self.layers:
                if getattr(L, "attn_q", False):                 # EXL3 projections (v3) have no bf16 copy to cast
                    continue
                names = ("wq_cat", "wk_cat", "wv_cat", "wo") if L.is_global else ("wqkv", "wo")
                L.fp8 = {n: quant_fp8_rows(getattr(L, n)) for n in names}
            torch.cuda.empty_cache()
        # decode-sized all-reduces as one exchange (send the partial to every peer, add the world's partials in rank
        # order in fp32: identical on every rank) instead of NCCL's ring (2 * (world - 1) hops)
        self.p2p = _env.get('I2_P2P_AR', '0') == '1' and world > 1
        self.p2p_max_rows = int(_env.get('I2_P2P_MAX_ROWS', 64))
        self.peers = [r for r in range(world) if r != rank]

    # ---- memory accounting
    def kv_bytes(self):
        tot = 0
        for t in self.kv.values():
            tot += sum(x.numel() * x.element_size() for x in t if x is not None)
        return tot

    def mm(self, x, w, w8 = None):
        """x [T, K] @ w[N, K]^T in bf16 (skinny Triton GEMM for decode/verify-sized T; w8 = its FP8 copy, if any)"""
        if self.skinny and x.shape[0] <= 64:
            if w8 is not None:
                return skinny_mm(x, w8[0], scale = w8[1], out_dtype = torch.bfloat16)
            return skinny_mm(x, w, out_dtype = torch.bfloat16)
        return torch.mm(x, w.t())

    def w8(self, L, name):
        return L.fp8[name] if self.attn_fp8 else None

    def attn_proj_q(self, L, xq, xk = None, xv = None):
        """EXL3 projections (v3) assembled into the layouts of the fused bf16 matrices. Sliding layers (xk None):
        qb = [q | rel | gate | sink | k | v]; global layers: qb = [q | rel | gate | sink], kb = [k | kproj],
        vb = [v | vproj]. rel and sink come from the bf16 wrs, the conv projections from kproj / vproj."""
        Hl, Hkl = self.Hl, self.Hkl
        T, hq, hr = xq.shape[0], Hl * 128, Hl * 16
        y = AQ.run(L.lin_qgkv if xk is None else L.lin_qg, xq)                  # fp16 [T, 2 hq (+ 2 Hkl * 128)]
        rs = self.mm(xq, L.wrs)                                                 # bf16 [T, Hl * 17]
        n = 2 * hq + hr + Hl
        qb = torch.empty((T, n + (2 * Hkl * 128 if xk is None else 0)), dtype = torch.bfloat16, device = xq.device)
        qb[:, :hq] = y[:, :hq]
        qb[:, hq:hq + hr] = rs[:, :hr]
        qb[:, hq + hr:2 * hq + hr] = y[:, hq:2 * hq]
        qb[:, 2 * hq + hr:n] = rs[:, hr:]
        if xk is None:
            qb[:, n:] = y[:, 2 * hq:]
            return qb
        kb = torch.cat([AQ.run(L.lin_k, xk).to(torch.bfloat16), self.mm(xk, L.kproj)], 1)
        vb = torch.cat([AQ.run(L.lin_v, xv).to(torch.bfloat16), self.mm(xv, L.vproj)], 1)
        return qb, kb, vb

    # ---- collectives
    _NO_AR = __import__('os').environ.get('I2_NO_AR', '0') == '1'      # timing diagnostics only: outputs are wrong

    def allreduce(self, x):
        if self.world > 1 and not self._NO_AR:
            if self.p2p and x.shape[0] <= self.p2p_max_rows:
                buf = torch.empty((self.world,) + tuple(x.shape), dtype = x.dtype, device = x.device)
                buf[self.rank].copy_(x)
                ops = []
                for p in self.peers:
                    ops.append(dist.P2POp(dist.isend, x, p))
                    ops.append(dist.P2POp(dist.irecv, buf[p], p))
                for w in dist.batch_isend_irecv(ops):
                    w.wait()
                x.copy_(buf.float().sum(0))
            else:
                dist.all_reduce(x)
        return x

    # ---- embedding / head
    def embed_tokens(self, ids):
        c = self.cfg
        if self.world == 1:
            e = self.embed_w[ids]
        else:
            loc = ids - self.rank * self.embed_n
            ok = (loc >= 0) & (loc < self.embed_n)
            e = self.embed_w[loc.clamp(0, self.embed_n - 1)] * ok[:, None].to(self.embed_w.dtype)
            dist.all_reduce(e)
        return ops_rms_norm(e, self.embed.norm, c.eps)

    def head_hidden(self, h, blocks):
        c = self.cfg
        return ops.attn_res_norm(blocks, c.layers // c.block_layers, h, self.head.out_query[None], self.head.norm, c.eps)[0]

    def head_mm(self, x):
        """x [T, D] -> this rank's raw logits [T, V/world] fp32: the EXL3 head of v3 (padded vocabulary), or bf16"""
        if self.head.lin_head is not None:
            return AQ.run_f32(self.head.lin_head, x)
        return torch.mm(x, self.head.unembed.t(), out_dtype = torch.float32)

    def head_mm_bf16(self, x):
        """the drafter's base logits [T, V/world] bf16 from the same head"""
        if self.head.lin_head is not None:
            return AQ.run(self.head.lin_head, x).to(torch.bfloat16)
        return torch.mm(x, self.head.unembed.t())

    def head_logits(self, h, blocks):
        """h: [T, D] bf16, blocks: [8, T, D]. Returns this rank's softcapped logits shard [T, V/world] fp32"""
        c = self.cfg
        x = self.head_hidden(h, blocks)
        z = self.head_mm(x)
        return torch.tanh(z / c.final_cap) * c.final_cap

    def global_argmax(self, z):
        """argmax over the full (unpadded) vocabulary of vocab-sharded logits z [T, V/world]"""
        c = self.cfg
        n = z.shape[1]
        valid = min(n, c.unpadded_vocab - self.rank * n)
        if valid < n:
            z = z.clone()
            z[:, valid:] = float("-inf")
        v, i = z.max(-1)
        i = i + self.rank * n
        if self.world == 1:
            return i
        vs = torch.empty((self.world,) + v.shape, dtype = v.dtype, device = v.device)
        is_ = torch.empty((self.world,) + v.shape, dtype = i.dtype, device = v.device)
        dist.all_gather_into_tensor(vs, v.contiguous())
        dist.all_gather_into_tensor(is_, i.contiguous())
        best = vs.argmax(0)
        return is_.gather(0, best[None])[0]

    # ---- one decoder layer
    def layer(self, L: Inkling2Layer, h, blocks, B: Batch):
        c = self.cfg
        i, T = L.idx, h.shape[0]
        Hl, Hkl, G = self.Hl, self.Hkl, self.G
        nb = c.num_blocks_before(i)
        boundary = c.is_boundary(i)
        kq, vq, ks, vs = self.kv[i]
        kring, vring, aring, mring = self.rings[i]
        bits = self.bits_g if L.is_global else self.bits_l
        q_out = torch.empty((T, Hl, 128), dtype = torch.bfloat16, device = h.device)
        rel_out = torch.empty((T, Hl, 16), dtype = torch.bfloat16, device = h.device)
        write_ring = not B.multi

        # attention input(s) and projections
        if L.is_global:
            xs = ops.attn_res_norm(blocks, nb, h, L.attn_query, L.qkv_norms, c.eps)        # [3, T, D]
            if L.attn_q:
                qb, kb, vb = self.attn_proj_q(L, xs[0], xs[1], xs[2])
            else:
                qb = self.mm(xs[0], L.wq_cat, self.w8(L, "wq_cat"))
                kb = self.mm(xs[1], L.wk_cat, self.w8(L, "wk_cat"))
                vb = self.mm(xs[2], L.wv_cat, self.w8(L, "wv_cat"))
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            ops.qkv_post(qb, kb, vb, L.kc_static, L.vc_static, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                         q_out, rel_out, kq, vq, ks, vs, self.page_table, c, True, bits, rel_off, Hkl * 128, Hkl * 128,
                         self.ring, self.ps, write_ring, Hkl, G, rot = self.kv_rot, ng = self.kv_ng, q16 = self.kv_q16)
            kraw, vraw = kb, vb
            Nq = qb
        else:
            if i == 0:
                x = ops.attn_res_norm(blocks, 0, h, torch.zeros((1, c.hidden), device = h.device), L.attn_norm, c.eps)
            else:
                x = ops.attn_res_norm(blocks, nb, h, L.attn_query[None], L.attn_norm, c.eps)
            qb = self.attn_proj_q(L, x[0]) if L.attn_q else self.mm(x[0], L.wqkv, self.w8(L, "wqkv"))
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            koff = sink_off + Hl
            kbv, vbv = qb[:, koff:], qb[:, koff + Hkl * 128:]
            ops.qkv_post(qb, kbv, vbv, L.kc_w, L.vc_w, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                         q_out, rel_out, kq, vq, ks, vs, None, c, False, bits, rel_off, 0, 0, self.ring, self.ps,
                         write_ring, Hkl, G)
            kraw, vraw = kbv, vbv
            Nq = qb
        if B.multi:
            ops.ring_write(kraw[:, :Hkl * 128], kring, B.slot, B.pos, B.nafter)
            ops.ring_write(vraw[:, :Hkl * 128], vring, B.slot, B.pos, B.nafter)
        gate = Nq[:, gate_off:gate_off + Hl * 128]
        sink = Nq[:, sink_off:sink_off + Hl]
        if self.rel_fused:
            R = None
            if not hasattr(L, "rel_proj_f32"):
                L.rel_proj_f32 = L.rel_proj.float().contiguous()
            relarg = (rel_out, L.rel_proj_f32)
        else:
            R = torch.mm(rel_out.view(T * Hl, 16).float(), L.rel_proj.float()).view(T, Hl, c.rel_extent)
            relarg = None
        o = torch.empty((T, Hl * 128), dtype = torch.bfloat16, device = h.device)
        nsplit = B.nsplit_g if L.is_global else B.nsplit_l
        gq = dict(rot = self.kv_rot, ng = self.kv_ng, q16 = self.kv_q16) if L.is_global else {}
        ns = nsplit
        if L.is_global and self.far is not None and B.multi and B.single and T >= self.far_min:
            nsplit, gq["ns_far"], gq["far_m"] = self.far
            ns = nsplit + gq["ns_far"]
        elif L.is_global and self.far_v is not None and B.multi and B.seq_rows and B.seq_rows * G <= 128:
            nsplit, gq["ns_far"] = self.far_v
            gq["far_m"] = max(16, 1 << (B.seq_rows * G - 1).bit_length())
            gq["far_seq_rows"] = B.seq_rows
            ns = nsplit + gq["ns_far"]
        pm = torch.empty((T, Hl, ns), dtype = torch.float32, device = h.device)
        pl = torch.empty((T, Hl, ns), dtype = torch.float32, device = h.device)
        po = torch.empty((T, Hl, ns, 128), dtype = torch.float32, device = h.device)
        ops.attention(q_out, R, sink, gate, kq, vq, ks, vs, self.page_table if L.is_global else None, B.slot, B.pos,
                      (pm, pl, po), o, c, L.is_global, bits, self.ring, self.ps, nsplit, Hkl, G, rel = relarg, **gq)
        if L.attn_q:
            oa = AQ.run(L.lin_o, o)
            if oa.dtype != self.ar_dtype:
                oa = oa.to(self.ar_dtype)
        elif self.skinny and T <= 64:
            if self.attn_fp8:
                oa = skinny_mm(o, L.fp8["wo"][0], scale = L.fp8["wo"][1], out_dtype = self.ar_dtype)
            else:
                oa = skinny_mm(o, L.wo, out_dtype = self.ar_dtype)          # fp32 accumulation, one rounding to the AR dtype
        else:
            oa = torch.mm(o, L.wo.t(), out_dtype = torch.float32)
            if self.ar_dtype != torch.float32:
                oa = oa.to(self.ar_dtype)
        self.allreduce(oa)

        # post-attention conv + residual, attention residual for the MLP input
        resid = torch.empty((T, c.hidden), dtype = torch.float32, device = h.device)
        ops.sconv(oa, L.attn_sconv, aring, B.slot, B.pos, B.nprev, 0, h = h, resid = resid, boundary = boundary,
                  write_ring = write_ring)
        if B.multi:
            ops.ring_write(oa, aring, B.slot, B.pos, B.nafter)
        if boundary:
            blocks[nb].copy_(h)
        nb_m = nb + 1 if boundary else nb
        xm = ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], L.mlp_norm, c.eps, out_f16 = True)[0]

        # routed experts (expert parallel), summed over ranks
        G = self.moe_group
        if self.b12x is not None and self.b12x.supports(T):
            moe = self.b12x.forward(L.idx, L.mlp, xm, out_dtype = self.ar_dtype)
        elif G < T <= self.moe_chunk_max:
            # the fused coop kernels take up to MAX_BSZN (64) rows per call: groups of `moe_group` rows
            moe = torch.cat([L.mlp.forward(xm[s:s + G], {}).view(-1, c.hidden).clone() for s in range(0, T, G)], 0)
        else:
            moe = L.mlp.forward(xm, {})
        moe = moe.view(T, c.hidden)
        if moe.dtype != self.ar_dtype:
            moe = moe.to(self.ar_dtype)
        self.allreduce(moe)
        h_new = torch.empty_like(h)
        ops.sconv(moe, L.mlp_sconv, mring, B.slot, B.pos, B.nprev, 1, resid = resid, out = h_new, write_ring = write_ring)
        if B.multi:
            ops.ring_write(moe, mring, B.slot, B.pos, B.nafter)
        return h_new

    def new_blocks(self, T):
        c = self.cfg
        return torch.zeros((c.layers // c.block_layers, T, c.hidden), dtype = torch.bfloat16, device = self.device)

    TAP_LAYERS = (5, 11, 17, 23, 29, 35, 41)

    def forward(self, ids, B: Batch, stop_layer = None, taps = None, return_hidden = False, img = None):
        """ids: [T] token ids (all ranks pass the same ids). Returns this rank's logits shard [T, V/world].
        taps: optional [T, 8, D] bf16 buffer receiving the draft's target features (inputs of layers 5..41, final-norm output)"""
        assert not (B.multi and B.single and B.T > self.max_rows), \
            f"{B.T} rows of one sequence exceed the SWA ring headroom (max_rows {self.max_rows}): chunk the call"
        h = self.embed_tokens(ids)
        if img is not None:
            # image features replace the placeholder rows after the embedding norm, like the reference
            rows, feats = img
            h[rows] = feats.to(h.dtype)
        blocks = self.new_blocks(h.shape[0])
        for L in self.layers:
            if taps is not None and L.idx in self.TAP_LAYERS:
                taps[:, self.TAP_LAYERS.index(L.idx)].copy_(h)
            h = self.layer(L, h, blocks, B)
            if stop_layer is not None and L.idx == stop_layer:
                return h
        x = self.head_hidden(h, blocks)
        if taps is not None:
            taps[:, 7].copy_(x)
        if return_hidden:
            return x
        z = self.head_mm(x)
        return torch.tanh(z / self.cfg.final_cap) * self.cfg.final_cap


def ops_rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return (w.float() * xf).to(torch.bfloat16)
