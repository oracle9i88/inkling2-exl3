"""
Triton kernels of the batched Inkling-2 engine (see inkling2.py for the reference math).

Token batch conventions: T rows; slot[t] = sequence slot, pos[t] = absolute position, nprev[t] = number of earlier
rows of the same sequence in this batch (rows of a sequence are contiguous and ascending). Short-convolution inputs of
earlier positions come from the batch rows themselves (nprev) or from a per-slot raw-input ring buffer (RC entries).
"""
import math
import torch
import triton
import triton.language as tl

RC = 16                # conv input ring length (positions modulo RC)
SQRT5 = math.sqrt(5.0)


# ----------------------------------------------------------------------------------------------------------------------
# Attention residual (+ optional weighted RMS norm)

@triton.jit
def _ar_update(x, q, m, l, acc, D, eps, inv_sqrt_d):
    ms = tl.sum(x * x, 0) / D
    r = tl.rsqrt(ms + eps)
    s = tl.sum(q * x, 0) * r * inv_sqrt_d
    m_new = tl.maximum(m, s)
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new)
    return m_new, l * alpha + p, acc * alpha + (p * r) * x


@triton.jit
def _attn_res_kernel(
    blocks_ptr, extra_ptr, cur_ptr, q_ptr, w_ptr, out_ptr,
    nb, T, D, eps, inv_sqrt_d,
    stride_bn, stride_bt, stride_et, stride_ct, stride_oq, stride_ot,
    NQ: tl.constexpr, HAS_EXTRA: tl.constexpr, HAS_W: tl.constexpr, OUT_F32: tl.constexpr, OUT_F16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    q0 = tl.load(q_ptr + offs, mask = mask, other = 0.0).to(tl.float32)
    m0 = -float("inf"); l0 = 0.0
    a0 = tl.zeros([BLOCK], dtype = tl.float32)
    if NQ >= 2:
        q1 = tl.load(q_ptr + D + offs, mask = mask, other = 0.0).to(tl.float32)
        m1 = -float("inf"); l1 = 0.0
        a1 = tl.zeros([BLOCK], dtype = tl.float32)
    if NQ >= 3:
        q2 = tl.load(q_ptr + 2 * D + offs, mask = mask, other = 0.0).to(tl.float32)
        m2 = -float("inf"); l2 = 0.0
        a2 = tl.zeros([BLOCK], dtype = tl.float32)
    for j in range(0, nb):
        x = tl.load(blocks_ptr + j * stride_bn + t * stride_bt + offs, mask = mask, other = 0.0).to(tl.float32)
        m0, l0, a0 = _ar_update(x, q0, m0, l0, a0, D, eps, inv_sqrt_d)
        if NQ >= 2:
            m1, l1, a1 = _ar_update(x, q1, m1, l1, a1, D, eps, inv_sqrt_d)
        if NQ >= 3:
            m2, l2, a2 = _ar_update(x, q2, m2, l2, a2, D, eps, inv_sqrt_d)
    if HAS_EXTRA:
        x = tl.load(extra_ptr + t * stride_et + offs, mask = mask, other = 0.0).to(tl.float32)
        m0, l0, a0 = _ar_update(x, q0, m0, l0, a0, D, eps, inv_sqrt_d)
        if NQ >= 2:
            m1, l1, a1 = _ar_update(x, q1, m1, l1, a1, D, eps, inv_sqrt_d)
        if NQ >= 3:
            m2, l2, a2 = _ar_update(x, q2, m2, l2, a2, D, eps, inv_sqrt_d)
    x = tl.load(cur_ptr + t * stride_ct + offs, mask = mask, other = 0.0).to(tl.float32)
    m0, l0, a0 = _ar_update(x, q0, m0, l0, a0, D, eps, inv_sqrt_d)
    if NQ >= 2:
        m1, l1, a1 = _ar_update(x, q1, m1, l1, a1, D, eps, inv_sqrt_d)
    if NQ >= 3:
        m2, l2, a2 = _ar_update(x, q2, m2, l2, a2, D, eps, inv_sqrt_d)

    y = a0 / l0
    if HAS_W:
        w = tl.load(w_ptr + offs, mask = mask, other = 0.0).to(tl.float32)
        y = y * tl.rsqrt(tl.sum(y * y, 0) / D + eps) * w
    if OUT_F32:
        tl.store(out_ptr + t * stride_ot + offs, y, mask = mask)
    elif OUT_F16:
        tl.store(out_ptr + t * stride_ot + offs, y.to(tl.bfloat16).to(tl.float16), mask = mask)
    else:
        tl.store(out_ptr + t * stride_ot + offs, y.to(tl.bfloat16), mask = mask)
    if NQ >= 2:
        y = a1 / l1
        if HAS_W:
            w = tl.load(w_ptr + D + offs, mask = mask, other = 0.0).to(tl.float32)
            y = y * tl.rsqrt(tl.sum(y * y, 0) / D + eps) * w
        if OUT_F32:
            tl.store(out_ptr + stride_oq + t * stride_ot + offs, y, mask = mask)
        else:
            tl.store(out_ptr + stride_oq + t * stride_ot + offs, y.to(tl.bfloat16), mask = mask)
    if NQ >= 3:
        y = a2 / l2
        if HAS_W:
            w = tl.load(w_ptr + 2 * D + offs, mask = mask, other = 0.0).to(tl.float32)
            y = y * tl.rsqrt(tl.sum(y * y, 0) / D + eps) * w
        if OUT_F32:
            tl.store(out_ptr + 2 * stride_oq + t * stride_ot + offs, y, mask = mask)
        else:
            tl.store(out_ptr + 2 * stride_oq + t * stride_ot + offs, y.to(tl.bfloat16), mask = mask)


def attn_res_norm(blocks, nb, cur, queries, norm_w, eps, extra = None, out = None, out_f32 = False, out_f16 = False):
    """
    blocks: [NBMAX, T, D] bf16 (first nb used), extra: optional [T, D] bf16 state, cur: [T, D] (bf16/fp32) current state.
    queries: [nq, D] fp32. norm_w: None or [nq, D]/[D] bf16 (for nq = 3 a stacked [3, D]). Returns [nq, T, D].
    """
    T, D = cur.shape
    nq = queries.shape[0]
    if out is None:
        out = torch.empty((nq, T, D), dtype = torch.float32 if out_f32 else torch.float16 if out_f16 else torch.bfloat16, device = cur.device)
    w = norm_w
    _attn_res_kernel[(T,)](
        blocks, extra if extra is not None else cur, cur, queries, w if w is not None else queries, out,
        nb, T, D, eps, D ** -0.5,
        blocks.stride(0), blocks.stride(1), (extra.stride(0) if extra is not None else 0), cur.stride(0),
        out.stride(0), out.stride(1),
        NQ = nq, HAS_EXTRA = extra is not None, HAS_W = w is not None, OUT_F32 = out_f32, OUT_F16 = out_f16,
        BLOCK = 8192, num_warps = 16,
    )
    return out


# ----------------------------------------------------------------------------------------------------------------------
# KV quantization helpers

@triton.jit
def _quant_store(x, base_q, base_s, offs_d, BITS: tl.constexpr, DH: tl.constexpr):
    """x: [DH] fp32 (already rounded to bf16 values). Store codes at base_q (+offs) and the fp16 scale at base_s."""
    if BITS == 16:
        tl.store(base_q + offs_d, x.to(tl.bfloat16))
    else:
        QMAX: tl.constexpr = 127 if BITS == 8 else 7
        amax = tl.max(tl.abs(x), 0)
        sc = tl.maximum(amax / QMAX, 1e-8)
        qf = tl.extra.libdevice.rint(x / sc)
        qf = tl.minimum(tl.maximum(qf, -QMAX - 1), QMAX)
        tl.store(base_s, sc.to(tl.float16))
        if BITS == 8:
            tl.store(base_q + offs_d, qf.to(tl.int8))
        else:
            # pack pairs (2i, 2i + 1): low nibble = even element
            u = (qf + 8).to(tl.int32)
            u2 = tl.reshape(u, [DH // 2, 2])
            lo, hi = tl.split(u2)
            packed = (lo | (hi << 4)).to(tl.uint8)
            tl.store(base_q + tl.arange(0, DH // 2), packed)


# ----------------------------------------------------------------------------------------------------------------------
# Conv input history / ring buffers

@triton.jit
def _hist(buf_ptr, stride_row, t, col0, ring_ptr, ring_slot_stride, ring_pos_stride, slot, pos, lag, nprev, offs):
    """raw conv input `lag` positions before the current one: from the batch row if it is in this batch, else the ring"""
    in_batch = lag <= nprev
    a = tl.load(buf_ptr + (t - lag) * stride_row + col0 + offs, mask = (offs >= 0) & in_batch, other = 0.0).to(tl.bfloat16).to(tl.float32)
    p = pos - lag
    use_ring = (lag > nprev) & (p >= 0)
    b = tl.load(ring_ptr + slot * ring_slot_stride + (p % 16) * ring_pos_stride + col0 + offs,
                mask = (offs >= 0) & use_ring, other = 0.0).to(tl.float32)
    return a + b


@triton.jit
def _ring_write_kernel(src_ptr, stride_src, ring_ptr, ring_slot_stride, ring_pos_stride, slot_ptr, pos_ptr, nafter_ptr, C,
                       BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    m = offs < C
    slot = tl.load(slot_ptr + t)
    pos = tl.load(pos_ptr + t)
    na = tl.load(nafter_ptr + t)
    m = m & (na < 16)
    x = tl.load(src_ptr + t * stride_src + offs, mask = m, other = 0.0)
    tl.store(ring_ptr + slot * ring_slot_stride + (pos % 16) * ring_pos_stride + offs, x.to(ring_ptr.dtype.element_ty), mask = m)


def ring_write(src, ring, slot, pos, nafter):
    """src: [T, C] (row stride arbitrary); ring: [slots, RC, C]; writes src[t] at ring[slot[t], pos[t] % RC] for the last
    RC rows of each sequence in the batch (nafter[t] = rows of the same sequence after row t)"""
    T, C = src.shape
    _ring_write_kernel[(T, triton.cdiv(C, 1024))](src, src.stride(0), ring, ring.stride(0), ring.stride(1), slot, pos, nafter, C, BLOCK = 1024)


# ----------------------------------------------------------------------------------------------------------------------
# QKV post-processing: short conv on K/V, L2 clip, value norm, q/rel scaling, KV cache write

@triton.jit
def _qkv_post_kernel(
    qbuf_ptr, kbuf_ptr, vbuf_ptr,                  # projection outputs: q-part [T, Nq], k-part [T, Nk], v-part [T, Nv]
    kc_ptr, vc_ptr,                                # conv weights: local [Hkl*128, 4] fp32 / global static [4, Hkl*128] fp32
    qscale_ptr, rscale_ptr,                        # [Hl, 128], [Hl, 16] fp32
    kring_ptr, vring_ptr,                          # [slots, RC, Hkl*128] bf16
    slot_ptr, pos_ptr, nprev_ptr,
    q_out_ptr, rel_out_ptr,                        # [T, Hl, 128], [T, Hl, 16] bf16
    kq_ptr, vq_ptr, ks_ptr, vs_ptr,                # KV cache tensors
    page_table_ptr, pt_stride,
    stride_qb, stride_kb, stride_vb,
    rel_off, k_dyn_off, v_dyn_off,
    cache_stride_pos, cache_stride_h, sc_stride_pos,
    qk_clip, rel_clip, eps, log_alpha, log_n_floor, ring_pos_stride,
    G: tl.constexpr, GP: tl.constexpr, DH: tl.constexpr, DR: tl.constexpr, HKL: tl.constexpr,
    IS_GLOBAL: tl.constexpr, BITS: tl.constexpr, RING: tl.constexpr, PS: tl.constexpr, WRITE_RING: tl.constexpr,
    rot_ptr = None, ROT: tl.constexpr = False, NG: tl.constexpr = 1, Q16: tl.constexpr = False,
):
    t = tl.program_id(0)
    hk = tl.program_id(1)
    slot = tl.load(slot_ptr + t)
    pos = tl.load(pos_ptr + t)
    nprev = tl.load(nprev_ptr + t)
    offs_d = tl.arange(0, DH)
    C: tl.constexpr = HKL * DH
    col0 = hk * DH
    ring_slot_stride = 16 * C

    kx = tl.load(kbuf_ptr + t * stride_kb + col0 + offs_d).to(tl.float32)
    vx = tl.load(vbuf_ptr + t * stride_vb + col0 + offs_d).to(tl.float32)
    ky = tl.zeros([DH], dtype = tl.float32)
    vy = tl.zeros([DH], dtype = tl.float32)
    for lag in tl.static_range(0, 4):
        if lag == 0:
            kh = kx
            vh = vx
        else:
            kh = _hist(kbuf_ptr, stride_kb, t, col0, kring_ptr, ring_slot_stride, ring_pos_stride, slot, pos, lag, nprev, offs_d)
            vh = _hist(vbuf_ptr, stride_vb, t, col0, vring_ptr, ring_slot_stride, ring_pos_stride, slot, pos, lag, nprev, offs_d)
        if IS_GLOBAL:
            kdyn = tl.load(kbuf_ptr + t * stride_kb + k_dyn_off + hk * 4 + lag).to(tl.float32)
            vdyn = tl.load(vbuf_ptr + t * stride_vb + v_dyn_off + hk * 4 + lag).to(tl.float32)
            kw = tl.load(kc_ptr + lag * C + col0 + offs_d) + kdyn
            vw = tl.load(vc_ptr + lag * C + col0 + offs_d) + vdyn
        else:
            kw = tl.load(kc_ptr + (col0 + offs_d) * 4 + (3 - lag))
            vw = tl.load(vc_ptr + (col0 + offs_d) * 4 + (3 - lag))
        ky += kw * kh
        vy += vw * vh
    if IS_GLOBAL:
        DEN: tl.constexpr = 3.0
    else:
        DEN: tl.constexpr = 2.23606797749979
    kc = ((ky + kx) / DEN).to(tl.bfloat16).to(tl.float32)
    vc = ((vy + vx) / DEN).to(tl.bfloat16).to(tl.float32)
    if WRITE_RING:
        tl.store(kring_ptr + slot * ring_slot_stride + (pos % 16) * ring_pos_stride + col0 + offs_d, kx.to(tl.bfloat16))
        tl.store(vring_ptr + slot * ring_slot_stride + (pos % 16) * ring_pos_stride + col0 + offs_d, vx.to(tl.bfloat16))

    # K: l2 clip, V: rms norm without gain
    kn = tl.sqrt(tl.sum(kc * kc, 0))
    kk = (kc * tl.minimum(qk_clip / (kn + 1e-6), 1.0)).to(tl.bfloat16).to(tl.float32)
    vv = (vc * tl.rsqrt(tl.sum(vc * vc, 0) / DH + eps)).to(tl.bfloat16).to(tl.float32)
    if ROT:
        # cache-space rotation x M / sqrt(DH), M = diag(signs) * Hadamard (+-1, exact in bf16); fp32 accumulation
        offs_r16 = tl.arange(0, 16)
        Mrot = tl.load(rot_ptr + offs_d[:, None] * DH + offs_d[None, :])
        kv2 = tl.where(offs_r16[:, None] == 0, kk[None, :], tl.where(offs_r16[:, None] == 1, vv[None, :], 0.0))
        kvr = tl.dot(kv2.to(tl.bfloat16), Mrot) * (DH ** -0.5)
        kk = tl.sum(tl.where(offs_r16[:, None] == 0, kvr, 0.0), 0)
        vv = tl.sum(tl.where(offs_r16[:, None] == 1, kvr, 0.0), 0)

    # cache row
    if IS_GLOBAL:
        page = tl.load(page_table_ptr + slot * pt_stride + pos // PS)
        row = page.to(tl.int64) * PS + pos % PS
    else:
        row = slot.to(tl.int64) * RING + pos % RING
    kbase = kq_ptr + row * cache_stride_pos + hk * cache_stride_h
    vbase = vq_ptr + row * cache_stride_pos + hk * cache_stride_h
    if BITS == 16:
        tl.store(kbase + offs_d, kk.to(tl.bfloat16))
        tl.store(vbase + offs_d, vv.to(tl.bfloat16))
    elif NG > 1 or Q16:
        # NG scale groups of DH // NG elements; Q16 (4-bit): 16 symmetric levels (c - 7.5) * amax / 7.5
        GS: tl.constexpr = DH // NG
        if BITS == 8:
            QMAX2: tl.constexpr = 127.0
        elif Q16:
            QMAX2: tl.constexpr = 7.5
        else:
            QMAX2: tl.constexpr = 7.0
        kg = tl.reshape(kk, [NG, GS])
        vg = tl.reshape(vv, [NG, GS])
        ksg = tl.maximum(tl.max(tl.abs(kg), 1) / QMAX2, 1e-8)
        vsg = tl.maximum(tl.max(tl.abs(vg), 1) / QMAX2, 1e-8)
        ksg = ksg.to(tl.float16).to(tl.float32)          # quantize against the stored scale
        vsg = vsg.to(tl.float16).to(tl.float32)
        offs_ng = tl.arange(0, NG)
        tl.store(ks_ptr + row * sc_stride_pos + hk * NG + offs_ng, ksg.to(tl.float16))
        tl.store(vs_ptr + row * sc_stride_pos + hk * NG + offs_ng, vsg.to(tl.float16))
        if Q16 and BITS == 4:
            kqf = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(kg / ksg[:, None] + 7.5), 0.0), 15.0)
            vqf = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(vg / vsg[:, None] + 7.5), 0.0), 15.0)
            kqf = tl.reshape(kqf, [DH]) - 8.0                   # stored as (code - 8) + 8 like the plain path
            vqf = tl.reshape(vqf, [DH]) - 8.0
        else:
            kqf = tl.reshape(tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(kg / ksg[:, None]), -QMAX2 - 1), QMAX2), [DH])
            vqf = tl.reshape(tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(vg / vsg[:, None]), -QMAX2 - 1), QMAX2), [DH])
        if BITS == 8:
            tl.store(kbase + offs_d, kqf.to(tl.int8))
            tl.store(vbase + offs_d, vqf.to(tl.int8))
        else:
            ku = tl.reshape((kqf + 8).to(tl.int32), [DH // 2, 2])
            kl, kh2 = tl.split(ku)
            vu = tl.reshape((vqf + 8).to(tl.int32), [DH // 2, 2])
            vl, vh2 = tl.split(vu)
            offs_p = tl.arange(0, DH // 2)
            tl.store(kbase + offs_p, (kl | (kh2 << 4)).to(tl.uint8))
            tl.store(vbase + offs_p, (vl | (vh2 << 4)).to(tl.uint8))
    else:
        QMAX: tl.constexpr = 127 if BITS == 8 else 7
        ksc = tl.maximum(tl.max(tl.abs(kk), 0) / QMAX, 1e-8)
        vsc = tl.maximum(tl.max(tl.abs(vv), 0) / QMAX, 1e-8)
        kqf = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(kk / ksc), -QMAX - 1), QMAX)
        vqf = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(vv / vsc), -QMAX - 1), QMAX)
        tl.store(ks_ptr + row * sc_stride_pos + hk, ksc.to(tl.float16))
        tl.store(vs_ptr + row * sc_stride_pos + hk, vsc.to(tl.float16))
        if BITS == 8:
            tl.store(kbase + offs_d, kqf.to(tl.int8))
            tl.store(vbase + offs_d, vqf.to(tl.int8))
        else:
            ku = tl.reshape((kqf + 8).to(tl.int32), [DH // 2, 2])
            kl, kh2 = tl.split(ku)
            vu = tl.reshape((vqf + 8).to(tl.int32), [DH // 2, 2])
            vl, vh2 = tl.split(vu)
            offs_p = tl.arange(0, DH // 2)
            tl.store(kbase + offs_p, (kl | (kh2 << 4)).to(tl.uint8))
            tl.store(vbase + offs_p, (vl | (vh2 << 4)).to(tl.uint8))

    # Q and relative-logit queries of the G heads of this kv group: l2 clip with scale (+ log-length scaling on global layers)
    if IS_GLOBAL:
        tau = 1.0 + log_alpha * tl.log(tl.maximum((pos.to(tl.float32) + 1.0) / log_n_floor, 1.0))
    else:
        tau = 1.0
    offs_g = tl.arange(0, GP)
    gm = offs_g < G
    hq = hk * G + offs_g
    qp = qbuf_ptr + t * stride_qb + hq[:, None] * DH + offs_d[None, :]
    qx = tl.load(qp, mask = gm[:, None], other = 0.0).to(tl.float32)
    qs = tl.load(qscale_ptr + hq[:, None] * DH + offs_d[None, :], mask = gm[:, None], other = 0.0)
    qx = qx * qs
    qn = tl.sqrt(tl.sum(qx * qx, 1))
    qo = qx * tl.minimum(qk_clip / (qn + 1e-6), 1.0)[:, None]
    qo = qo.to(tl.bfloat16).to(tl.float32)
    if ROT:
        Mq = tl.load(rot_ptr + offs_d[:, None] * DH + offs_d[None, :])
        qo = tl.dot(qo.to(tl.bfloat16), Mq) * (DH ** -0.5)
    qo = qo * tau
    tl.store(q_out_ptr + (t * (G * HKL) + hq[:, None]) * DH + offs_d[None, :], qo.to(tl.bfloat16), mask = gm[:, None])
    offs_r = tl.arange(0, DR)
    rp = qbuf_ptr + t * stride_qb + rel_off + hq[:, None] * DR + offs_r[None, :]
    rx = tl.load(rp, mask = gm[:, None], other = 0.0).to(tl.float32)
    rs = tl.load(rscale_ptr + hq[:, None] * DR + offs_r[None, :], mask = gm[:, None], other = 0.0)
    rx = rx * rs
    rn = tl.sqrt(tl.sum(rx * rx, 1))
    ro = rx * tl.minimum(rel_clip / (rn + 1e-6), 1.0)[:, None]
    ro = ro.to(tl.bfloat16).to(tl.float32) * tau
    tl.store(rel_out_ptr + (t * (G * HKL) + hq[:, None]) * DR + offs_r[None, :], ro.to(tl.bfloat16), mask = gm[:, None])


def qkv_post(qbuf, kbuf, vbuf, kc, vc, qscale, rscale, kring, vring, slot, pos, nprev, q_out, rel_out,
             kq, vq, ks, vs, page_table, cfg, is_global, bits, rel_off, k_dyn_off, v_dyn_off, ring_size, ps, write_ring,
             hkl, g, rot = None, ng = 1, q16 = False):
    """rot: optional [128, 128] bf16 +-1 matrix M (cache-space rotation x M / sqrt(128) of K, V and Q);
    ng: scale groups per 128-vector (ks/vs [..., Hkl, ng]); q16: 16-level symmetric 4-bit grid"""
    T = qbuf.shape[0]
    _qkv_post_kernel[(T, hkl)](
        qbuf, kbuf, vbuf, kc, vc, qscale, rscale, kring, vring, slot, pos, nprev, q_out, rel_out,
        kq, vq, ks if ks is not None else kq, vs if vs is not None else vq,
        page_table if page_table is not None else slot, page_table.stride(0) if page_table is not None else 0,
        qbuf.stride(0), kbuf.stride(0), vbuf.stride(0),
        rel_off, k_dyn_off, v_dyn_off,
        kq.stride(-3), kq.stride(-2), (_sc_stride_pos(ks, ng) if ks is not None else 0),
        cfg.qk_clip, cfg.rel_clip, cfg.eps, cfg.log_alpha, float(cfg.log_n_floor), kring.stride(1),
        G = g, GP = 16 if rot is not None else 8, DH = 128, DR = 16, HKL = hkl, IS_GLOBAL = is_global, BITS = bits,
        RING = ring_size, PS = ps, WRITE_RING = write_ring,
        rot_ptr = rot if rot is not None else qbuf, ROT = rot is not None, NG = ng, Q16 = q16, num_warps = 4,
    )


def _sc_stride_pos(ks, ng):
    """scale tensor stride per cache row: ks is [..., PS or RING, Hkl] (ng = 1) or [..., PS or RING, Hkl, ng]"""
    return ks.stride(-2) if ng == 1 else ks.stride(-3)


# ----------------------------------------------------------------------------------------------------------------------
# Attention (decode style: one program per (token, kv head, kv split)), relative-logit bias, sinks, KV dequant

@triton.jit
def _load_kv_tile(base_ptr, sc_ptr, rows, hk, cache_stride_pos, cache_stride_h, sc_stride_pos, valid, offs_d,
                  BITS: tl.constexpr, DH: tl.constexpr, NG: tl.constexpr = 1, Q16: tl.constexpr = False):
    """[BN, DH] dequantized tile (bf16 values as fp32)"""
    if BITS != 16 and (NG > 1 or Q16):
        GS: tl.constexpr = DH // NG
        if BITS == 8:
            ptr = base_ptr + rows[:, None] * cache_stride_pos + hk * cache_stride_h + offs_d[None, :]
            c = tl.load(ptr, mask = valid[:, None], other = 0).to(tl.float32)
        else:
            offs_p = tl.arange(0, DH // 2)
            ptr = base_ptr + rows[:, None] * cache_stride_pos + hk * cache_stride_h + offs_p[None, :]
            b = tl.load(ptr, mask = valid[:, None], other = 0).to(tl.int32)
            if Q16:
                lo = ((b & 15).to(tl.float32) - 7.5)
                hi = ((b >> 4).to(tl.float32) - 7.5)
            else:
                lo = ((b & 15) - 8).to(tl.float32)
                hi = ((b >> 4) - 8).to(tl.float32)
            c = tl.reshape(tl.join(lo, hi), [lo.shape[0], DH])
        offs_ng = tl.arange(0, NG)
        sc = tl.load(sc_ptr + rows[:, None] * sc_stride_pos + hk * NG + offs_ng[None, :], mask = valid[:, None], other = 0.0)
        c3 = tl.reshape(c, [c.shape[0], NG, GS]) * sc.to(tl.float32)[:, :, None]
        return tl.reshape(c3, [c.shape[0], DH]).to(tl.bfloat16).to(tl.float32)
    if BITS == 16:
        ptr = base_ptr + rows[:, None] * cache_stride_pos + hk * cache_stride_h + offs_d[None, :]
        return tl.load(ptr, mask = valid[:, None], other = 0.0).to(tl.float32)
    elif BITS == 8:
        ptr = base_ptr + rows[:, None] * cache_stride_pos + hk * cache_stride_h + offs_d[None, :]
        c = tl.load(ptr, mask = valid[:, None], other = 0).to(tl.float32)
        sc = tl.load(sc_ptr + rows * sc_stride_pos + hk, mask = valid, other = 0.0).to(tl.float32)
        return (c * sc[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        offs_p = tl.arange(0, DH // 2)
        ptr = base_ptr + rows[:, None] * cache_stride_pos + hk * cache_stride_h + offs_p[None, :]
        b = tl.load(ptr, mask = valid[:, None], other = 0).to(tl.int32)
        lo = ((b & 15) - 8).to(tl.float32)
        hi = ((b >> 4) - 8).to(tl.float32)
        c = tl.reshape(tl.join(lo, hi), [lo.shape[0], DH])
        sc = tl.load(sc_ptr + rows * sc_stride_pos + hk, mask = valid, other = 0.0).to(tl.float32)
        return (c * sc[:, None]).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _attn_kernel(
    q_ptr, R_ptr, sink_ptr, kq_ptr, vq_ptr, ks_ptr, vs_ptr, page_table_ptr, pt_stride, slot_ptr, pos_ptr,
    pm_ptr, pl_ptr, po_ptr,
    stride_sink_t, cache_stride_pos, cache_stride_h, sc_stride_pos, sm_scale,
    G: tl.constexpr, GP: tl.constexpr, DH: tl.constexpr, HKL: tl.constexpr, BN: tl.constexpr, NSPLIT: tl.constexpr,
    IS_GLOBAL: tl.constexpr, BITS: tl.constexpr, RING: tl.constexpr, PS: tl.constexpr, WINDOW: tl.constexpr,
    EXTENT: tl.constexpr, NG: tl.constexpr = 1, Q16: tl.constexpr = False,
    relq_ptr = None, relp_ptr = None, REL_FUSED: tl.constexpr = False,
    FAR: tl.constexpr = False, NSTRIDE: tl.constexpr = 0, FAR_SEQ_ROWS: tl.constexpr = 0,
):
    # FAR (global layers): this kernel covers only [far_end, pos] of every row; the keys before far_end are
    # _attn_far_kernel's (no bias, no mask there), whose partials sit after this kernel's NSPLIT ones. far_end comes from
    # the first row of the row's sequence: row 0 (one sequence, a prefill chunk) or, with FAR_SEQ_ROWS = R, row
    # t - t % R (speculative verify: R consecutive positions per sequence)
    t = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    slot = tl.load(slot_ptr + t)
    pos = tl.load(pos_ptr + t)
    HL: tl.constexpr = G * HKL
    offs_d = tl.arange(0, DH)
    offs_g = tl.arange(0, GP)
    gm = offs_g < G
    hq = hk * G + offs_g
    q = tl.load(q_ptr + (t * HL + hq[:, None]) * DH + offs_d[None, :], mask = gm[:, None], other = 0.0)
    sink = tl.load(sink_ptr + t * stride_sink_t + hq, mask = gm, other = 0.0).to(tl.float32)
    if IS_GLOBAL:
        lo = 0
        if FAR:
            if FAR_SEQ_ROWS > 0:
                pf = pos - t % FAR_SEQ_ROWS
            else:
                pf = tl.load(pos_ptr)
            lo = tl.maximum(0, pf - EXTENT + 1) // BN * BN
    else:
        lo = tl.maximum(0, pos - WINDOW + 1)
    n = pos - lo + 1
    nblk = (n + BN - 1) // BN
    bps = (nblk + NSPLIT - 1) // NSPLIT
    b0 = sp * bps
    b1 = tl.minimum(b0 + bps, nblk)
    if sp == 0:
        m_i = sink
        l_i = tl.full([GP], 1.0, dtype = tl.float32)
    else:
        m_i = tl.full([GP], -1e30, dtype = tl.float32)
        l_i = tl.zeros([GP], dtype = tl.float32)
    acc = tl.zeros([GP, DH], dtype = tl.float32)
    R_row = R_ptr + (t * HL + hq) * EXTENT
    if REL_FUSED:
        # relative-logit bias computed in-kernel: bias[g, j] = rel_q[t, g, :] . rel_proj[:, pos - j] (fp32, IEEE dot)
        offs_c = tl.arange(0, 16)
        rq = tl.load(relq_ptr + (t * HL + hq[:, None]) * 16 + offs_c[None, :], mask = gm[:, None], other = 0.0).to(tl.float32)
    for b in range(b0, b1):
        j = lo + b * BN + tl.arange(0, BN)
        valid = j <= pos
        if IS_GLOBAL:
            pg = tl.load(page_table_ptr + slot * pt_stride + j // PS, mask = valid, other = 0)
            rows = pg.to(tl.int64) * PS + j % PS
        else:
            rows = slot.to(tl.int64) * RING + j % RING
        k = _load_kv_tile(kq_ptr, ks_ptr, rows, hk, cache_stride_pos, cache_stride_h, sc_stride_pos, valid, offs_d, BITS, DH, NG, Q16)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
        d = pos - j
        if REL_FUSED:
            md = valid & (d < EXTENT)
            pr = tl.load(relp_ptr + offs_c[:, None] * EXTENT + d[None, :], mask = md[None, :], other = 0.0)
            bias = tl.dot(rq, pr, input_precision = "ieee")
        else:
            bias = tl.load(R_row[:, None] + d[None, :], mask = gm[:, None] & (valid & (d < EXTENT))[None, :], other = 0.0)
        s = s + bias
        s = tl.where(valid[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = _load_kv_tile(vq_ptr, vs_ptr, rows, hk, cache_stride_pos, cache_stride_h, sc_stride_pos, valid, offs_d, BITS, DH, NG, Q16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m_i = m_new
    if NSTRIDE > 0:
        pbase = ((t * HL + hq) * NSTRIDE + sp)
    else:
        pbase = ((t * HL + hq) * NSPLIT + sp)
    tl.store(pm_ptr + pbase, m_i, mask = gm)
    tl.store(pl_ptr + pbase, l_i, mask = gm)
    tl.store(po_ptr + pbase[:, None] * DH + offs_d[None, :], acc, mask = gm[:, None])


@triton.jit
def _attn_far_kernel(
    q_ptr, kq_ptr, vq_ptr, ks_ptr, vs_ptr, page_table_ptr, pt_stride, slot_ptr, pos_ptr, pm_ptr, pl_ptr, po_ptr, T,
    cache_stride_pos, cache_stride_h, sc_stride_pos, sm_scale,
    G: tl.constexpr, DH: tl.constexpr, HKL: tl.constexpr, M: tl.constexpr, BMR: tl.constexpr, BN: tl.constexpr,
    NS_FAR: tl.constexpr, SPLIT_OFF: tl.constexpr, NSTRIDE: tl.constexpr, BITS: tl.constexpr, PS: tl.constexpr,
    EXTENT: tl.constexpr, NG: tl.constexpr = 1, Q16: tl.constexpr = False, SEQ_ROWS: tl.constexpr = 0,
):
    """Global layer, one sequence's prefill chunk (rows ascending from pos[0]) or, with SEQ_ROWS = R = BMR, R rows of
    consecutive positions per sequence and one sequence per program (speculative verify): the keys [0, far_end) with
    far_end = (first row's pos - EXTENT + 1) rounded down to BN. They precede every row by at least EXTENT positions, so there is
    no relative bias and no causal mask, and one program serves BMR rows x G heads (packed into M matrix rows, the
    last M - BMR * G unused) with each key tile it loads.
    Writes split SPLIT_OFF + sp of each row's partials (no sink: the per-row kernel's split 0 has it)"""
    rb = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    HL: tl.constexpr = G * HKL
    r = tl.arange(0, M)
    t = rb * BMR + r // G
    hq = hk * G + r % G
    rmask = (r < BMR * G) & (t < T)
    offs_d = tl.arange(0, DH)
    q = tl.load(q_ptr + (t * HL + hq)[:, None] * DH + offs_d[None, :], mask = rmask[:, None], other = 0.0)
    if SEQ_ROWS > 0:
        fr = rb * BMR
    else:
        fr = 0
    slot = tl.load(slot_ptr + fr)
    far_end = tl.maximum(0, tl.load(pos_ptr + fr) - EXTENT + 1) // BN * BN
    nblk = far_end // BN
    bps = (nblk + NS_FAR - 1) // NS_FAR
    b0 = sp * bps
    b1 = tl.minimum(b0 + bps, nblk)
    m_i = tl.full([M], -1e30, dtype = tl.float32)
    l_i = tl.zeros([M], dtype = tl.float32)
    acc = tl.zeros([M, DH], dtype = tl.float32)
    for b in range(b0, b1):
        j = b * BN + tl.arange(0, BN)
        valid = j < far_end
        pg = tl.load(page_table_ptr + slot * pt_stride + j // PS, mask = valid, other = 0)
        rows = pg.to(tl.int64) * PS + j % PS
        k = _load_kv_tile(kq_ptr, ks_ptr, rows, hk, cache_stride_pos, cache_stride_h, sc_stride_pos, valid, offs_d, BITS, DH, NG, Q16)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = _load_kv_tile(vq_ptr, vs_ptr, rows, hk, cache_stride_pos, cache_stride_h, sc_stride_pos, valid, offs_d, BITS, DH, NG, Q16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m_i = m_new
    pbase = (t * HL + hq) * NSTRIDE + SPLIT_OFF + sp
    tl.store(pm_ptr + pbase, m_i, mask = rmask)
    tl.store(pl_ptr + pbase, l_i, mask = rmask)
    tl.store(po_ptr + pbase[:, None] * DH + offs_d[None, :], acc, mask = rmask[:, None])


@triton.jit
def _attn_combine_kernel(pm_ptr, pl_ptr, po_ptr, gate_ptr, o_ptr, stride_gate_t, HL: tl.constexpr, DH: tl.constexpr,
                         NSPLIT: tl.constexpr, rot_ptr = None, ROT: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NSPLIT)
    offs_d = tl.arange(0, DH)
    base = (t * HL + h) * NSPLIT
    m = tl.load(pm_ptr + base + offs_s)
    l = tl.load(pl_ptr + base + offs_s)
    mt = tl.max(m, 0)
    w = tl.exp(m - mt)
    den = tl.sum(l * w, 0)
    o = tl.load(po_ptr + (base + offs_s)[:, None] * DH + offs_d[None, :])
    o = tl.sum(o * w[:, None], 0) / den
    if ROT:
        # back from cache space: o = o' M^T / sqrt(DH)  (o'[i] summed against row i of M)
        Mr = tl.load(rot_ptr + offs_d[:, None] * DH + offs_d[None, :]).to(tl.float32)     # Mr[j, i] = M[j, i]
        o = tl.sum(Mr * o[None, :], 1) * (DH ** -0.5)                                       # o[j] = sum_i o'[i] M[j, i]
    o = o.to(tl.bfloat16).to(tl.float32)
    g = tl.load(gate_ptr + t * stride_gate_t + h * DH + offs_d).to(tl.float32)
    o = o * tl.sigmoid(g)
    tl.store(o_ptr + (t * HL + h) * DH + offs_d, o.to(tl.bfloat16))


FAR_WARPS = int(__import__('os').environ.get('I2_FAR_WARPS', 4))


def attention(q, R, sink, gate, kq, vq, ks, vs, page_table, slot, pos, scratch, o_out, cfg, is_global, bits, ring_size, ps,
              nsplit, hkl, g, bn = 64, rot = None, ng = 1, q16 = False, rel = None, ns_far = 0, far_m = 64, far_seq_rows = 0):
    """ns_far > 0 (global layer, rows = one sequence's consecutive positions from pos[0]): keys more than rel_extent
    behind pos[0] go to _attn_far_kernel (ns_far splits, far_m // g rows per program); scratch must then hold
    nsplit + ns_far partials per row and head, a power of 2. far_seq_rows = R > 0: rows come as R consecutive positions
    per sequence (speculative verify), one far program per sequence (R * g <= far_m)"""
    """rel: optional (rel_q [T, Hl, 16] bf16, rel_proj [16, EXTENT] fp32) -> relative bias computed in-kernel (R unused)"""
    """q: [T, Hl, 128] bf16, R: [T, Hl, 1024] fp32, sink: [T, Hl] view, gate: [T, Hl*128] view. Writes o_out [T, Hl*128] bf16"""
    T = q.shape[0]
    pm, pl, po = scratch
    HL = g * hkl
    far = bool(ns_far) and is_global                    # scratch then holds nsplit + ns_far partials per row and head
    NS = nsplit + (ns_far if far else 0)
    if far:
        bmr = far_seq_rows if far_seq_rows else far_m // g
        assert bmr * g <= far_m
        _attn_far_kernel[(triton.cdiv(T, bmr), hkl, ns_far)](
            q, kq, vq, ks if ks is not None else kq, vs if vs is not None else vq, page_table, page_table.stride(0), slot,
            pos, pm, pl, po, T, kq.stride(-3), kq.stride(-2), (_sc_stride_pos(ks, ng) if ks is not None else 0), 128 ** -0.5,
            G = g, DH = 128, HKL = hkl, M = far_m, BMR = bmr, BN = bn, NS_FAR = ns_far, SPLIT_OFF = nsplit, NSTRIDE = NS,
            BITS = bits, PS = ps, EXTENT = cfg.rel_extent, NG = ng, Q16 = q16, SEQ_ROWS = far_seq_rows, num_warps = FAR_WARPS,
        )
    _attn_kernel[(T, hkl, nsplit)](
        q, R if R is not None else q, sink, kq, vq, ks if ks is not None else kq, vs if vs is not None else vq,
        page_table if page_table is not None else slot, page_table.stride(0) if page_table is not None else 0, slot, pos,
        pm, pl, po,
        sink.stride(0), kq.stride(-3), kq.stride(-2), (_sc_stride_pos(ks, ng) if ks is not None else 0), 128 ** -0.5,
        G = g, GP = 16, DH = 128, HKL = hkl, BN = bn, NSPLIT = nsplit, IS_GLOBAL = is_global, BITS = bits,
        RING = ring_size, PS = ps, WINDOW = cfg.window, EXTENT = cfg.rel_extent, NG = ng, Q16 = q16,
        relq_ptr = rel[0] if rel is not None else q, relp_ptr = rel[1] if rel is not None else q, REL_FUSED = rel is not None,
        FAR = far, NSTRIDE = NS if far else 0, FAR_SEQ_ROWS = far_seq_rows if far else 0, num_warps = 4,
    )
    _attn_combine_kernel[(T, HL)](pm, pl, po, gate, o_out, gate.stride(0), HL = HL, DH = 128, NSPLIT = NS,
                                  rot_ptr = rot if rot is not None else gate, ROT = rot is not None)


def make_kv_rotation(device, seed = 20261114, dh = 128):
    """M = diag(random signs) @ H_dh (Sylvester Hadamard), +-1 entries in bf16; M M^T = dh I"""
    H = torch.ones((1, 1))
    while H.shape[0] < dh:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (dh,), generator = g) * 2 - 1).float()
    return (s[:, None] * H).to(torch.bfloat16).contiguous().to(device)


# ----------------------------------------------------------------------------------------------------------------------
# Output short convolutions (post-attention / post-MLP) fused with the residual update

@triton.jit
def _sconv_kernel(
    x_ptr, stride_x, w_ptr, ring_ptr, ring_slot_stride, ring_pos_stride, slot_ptr, pos_ptr, nprev_ptr,
    h_ptr, stride_h, resid_ptr, stride_r, out_ptr, stride_o, C,
    MODE: tl.constexpr, BOUNDARY: tl.constexpr, WRITE_RING: tl.constexpr, BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    m = offs < C
    slot = tl.load(slot_ptr + t)
    pos = tl.load(pos_ptr + t)
    nprev = tl.load(nprev_ptr + t)
    x = tl.load(x_ptr + t * stride_x + offs, mask = m, other = 0.0).to(tl.bfloat16).to(tl.float32)
    y = tl.zeros([BLOCK], dtype = tl.float32)
    for lag in tl.static_range(0, 4):
        if lag == 0:
            xh = x
        else:
            xh = _hist(x_ptr, stride_x, t, 0, ring_ptr, ring_slot_stride, ring_pos_stride, slot, pos, lag, nprev, offs)
        w = tl.load(w_ptr + offs * 4 + (3 - lag), mask = m, other = 0.0)
        y += w * xh
    c = ((y + x) / 2.23606797749979).to(tl.bfloat16).to(tl.float32)
    if WRITE_RING:
        tl.store(ring_ptr + slot * ring_slot_stride + (pos % 16) * ring_pos_stride + offs, x.to(tl.bfloat16), mask = m)
    if MODE == 0:
        # post-attention: residual stream
        if BOUNDARY:
            r = c
        else:
            r = tl.load(h_ptr + t * stride_h + offs, mask = m, other = 0.0).to(tl.float32) + c
        tl.store(resid_ptr + t * stride_r + offs, r, mask = m)
    else:
        r = tl.load(resid_ptr + t * stride_r + offs, mask = m, other = 0.0) + c
        tl.store(out_ptr + t * stride_o + offs, r.to(tl.bfloat16), mask = m)


def sconv(x, w, ring, slot, pos, nprev, mode, h = None, resid = None, out = None, boundary = False, write_ring = True):
    """mode 0: resid = (h or 0) + conv(x); mode 1: out(bf16) = resid + conv(x). x: [T, C] (fp32/bf16)"""
    T, C = x.shape
    dummy = x
    _sconv_kernel[(T, triton.cdiv(C, 1024))](
        x, x.stride(0), w, ring, ring.stride(0), ring.stride(1), slot, pos, nprev,
        h if h is not None else dummy, h.stride(0) if h is not None else 0,
        resid if resid is not None else dummy, resid.stride(0) if resid is not None else 0,
        out if out is not None else dummy, out.stride(0) if out is not None else 0, C,
        MODE = mode, BOUNDARY = boundary, WRITE_RING = write_ring, BLOCK = 1024, num_warps = 4,
    )
