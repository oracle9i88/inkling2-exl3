"""
Inkling-2 (thinkingmachines/Inkling-2-Small) building blocks for ExLlamaV3.

Numerics mirror the transformers `inkling2` reference (modeling_inkling2.py):

  * decoder layers are driven by block attention residuals: a layer input is a learned softmax-attention over the
    outputs of earlier 6-layer blocks and the running partial block (`attn_res`)
  * no RoPE; position enters only through a data-dependent relative logit bias (`wr_du` -> `rel_logits_proj`) that is
    non-zero for 0 <= distance < 1024; sliding layers (window 1024) and 8 global layers
  * short causal convolutions on K/V (static taps on sliding layers, token-dependent taps on global layers) and after the
    attention / MLP outputs
  * gated attention with sink logits, q/k/rel L2 clipping, value RMS norm without gain
  * 384 routed experts, top-8, sigmoid router with bias (selection only), route_scale 8, softcapped SwiGLU, no shared experts

The routed experts are stored as NVFP4 (layers 0-39) or MXFP8 (layers 40-47) in the source checkpoint. They are exposed as
fp32/bf16 weights through `Inkling2Source`, which dequantizes on the fly. After conversion they are stored as regular EXL3
tensors (one Linear per expert and projection: gate_proj / up_proj / down_proj).

The "state" that is carried between layers is a stack [..., nb + 1, hidden]: nb finished block outputs followed by the
current partial block (hidden state).
"""
from __future__ import annotations
import os, json, math, threading
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from safetensors import safe_open

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_byte_lut_cache = {}


def _byte_lut(device):
    # byte -> (value of low nibble, value of high nibble); low nibble is the first element of the pair
    key = str(device)
    if key not in _byte_lut_cache:
        lut = torch.tensor(FP4_VALUES, dtype = torch.float32)
        b = torch.arange(256)
        _byte_lut_cache[key] = torch.stack([lut[b & 15], lut[b >> 4]], dim = -1).to(device)
    return _byte_lut_cache[key]


def dequant_nvfp4(w: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """
    NVFP4 -> fp32. w: uint8 [..., N, K/2] (two E2M1, low nibble first), scale: float8_e4m3 [..., N, K/16] block scales,
    gscale: fp32 broadcastable per-expert factor (amax / outer_scale_denominator).
    """
    lut = _byte_lut(w.device)
    q = lut[w.long()]                                   # [..., N, K/2, 2]
    q = q.reshape(*w.shape[:-1], w.shape[-1] * 2)
    K = q.shape[-1]
    s = scale.float()
    out = (q.reshape(*q.shape[:-1], K // 16, 16) * s.unsqueeze(-1)).reshape(q.shape)
    return out * gscale.float().reshape(*gscale.shape, *([1] * (out.dim() - gscale.dim())))


def dequant_mxfp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    MXFP8 -> fp32. w: float8_e4m3 [..., N, K], scale: uint8 E8M0 [..., N, K/32] (per row, per 32 input elements):
    value = e4m3 * 2^(s - 127)
    """
    K = w.shape[-1]
    s = torch.exp2(scale.float() - 127.0)
    return (w.float().reshape(*w.shape[:-1], K // 32, 32) * s.unsqueeze(-1)).reshape(w.shape)


class Inkling2Source:
    """Reads tensors of an Inkling-2 checkpoint directory (source or converted) through its safetensors index"""

    def __init__(self, directory: str):
        self.dir = directory
        idx_path = os.path.join(directory, "model.safetensors.index.json")
        with open(idx_path) as f:
            self.map = json.load(f)["weight_map"]
        mtp = os.path.join(directory, "mtp.safetensors")
        self._lock = threading.Lock()
        self._handles = {}

    def has(self, key: str) -> bool:
        return key in self.map

    def _handle(self, fn):
        with self._lock:
            h = self._handles.get(fn)
            if h is None:
                h = safe_open(os.path.join(self.dir, fn), "pt")
                self._handles[fn] = h
            return h

    def get(self, key: str, device = "cpu") -> torch.Tensor:
        with self._lock:
            h = self._handles.get(self.map[key])
            if h is None:
                h = safe_open(os.path.join(self.dir, self.map[key]), "pt")
                self._handles[self.map[key]] = h
            t = h.get_tensor(key)
        return t.to(device) if str(device) != "cpu" else t

    def slice(self, key: str, start: int, stop: int, device = "cpu") -> torch.Tensor:
        """Rows [start, stop) along dim 0"""
        with self._lock:
            h = self._handles.get(self.map[key])
            if h is None:
                h = safe_open(os.path.join(self.dir, self.map[key]), "pt")
                self._handles[self.map[key]] = h
            t = h.get_slice(key)[start:stop]
        return t.to(device) if str(device) != "cpu" else t


# ----------------------------------------------------------------------------------------------------------------------
# Configuration

@dataclass
class Inkling2Cfg:
    hidden: int = 4608
    layers: int = 48
    heads: int = 48
    kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    window: int = 1024
    vocab: int = 201024
    unpadded_vocab: int = 200058
    eps: float = 1e-6
    conv_k: int = 4
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    log_n_floor: int = 128000
    log_alpha: float = 0.1
    block_layers: int = 6
    experts: int = 384
    topk: int = 8
    inter: int = 2304
    route_scale: float = 8.0
    gate_cap: float = 4.0
    up_cap: float = 25.0
    final_cap: float = 30.0
    global_layers: tuple = (5, 11, 17, 23, 29, 35, 41, 47)

    @staticmethod
    def from_dir(directory: str) -> "Inkling2Cfg":
        with open(os.path.join(directory, "config.json")) as f:
            c = json.load(f)["text_config"]
        n = c["num_hidden_layers"]
        local = set(c["local_layer_ids"])
        return Inkling2Cfg(
            hidden = c["hidden_size"], layers = n, heads = c["num_attention_heads"], kv_heads = c["num_key_value_heads"],
            head_dim = c["head_dim"], d_rel = c["d_rel"], rel_extent = c["rel_extent"],
            window = c["sliding_window_size"], vocab = c["vocab_size"], unpadded_vocab = c["unpadded_vocab_size"],
            eps = c["rms_norm_eps"], conv_k = c["sconv_kernel_size"], qk_clip = c["qk_l2_clip_threshold"],
            rel_clip = c["rel_l2_clip_threshold"], log_n_floor = c["log_scaling_n_floor"],
            log_alpha = c["log_scaling_alpha"], block_layers = c["n_layers_per_attn_res_block"],
            experts = c["n_routed_experts"], topk = c["num_experts_per_tok"], inter = c["intermediate_size"],
            route_scale = c["route_scale"], gate_cap = c["mlp_gate_softcap"], up_cap = c["mlp_up_softcap"],
            final_cap = c["final_logit_softcapping"],
            global_layers = tuple(i for i in range(n) if i not in local),
        )

    def is_global(self, i): return i in self.global_layers
    def is_boundary(self, i): return i % self.block_layers == 0
    def num_blocks_before(self, i):
        """Number of finished blocks available to layer i (before it appends its own input at a boundary)"""
        return (i + self.block_layers - 1) // self.block_layers


# ----------------------------------------------------------------------------------------------------------------------
# Elementwise / small ops (dtype generic: `adt` is the activation dtype, bf16 in the reference)

def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Matches Inkling2RMSNorm before the final cast: fp32 compute, weight multiplied in fp32"""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return xf if w is None else w.float() * xf


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None) -> torch.Tensor:
    """x: [..., heads, d] -> fp32 result (caller casts)"""
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.linalg.vector_norm(xf, dim = -1, keepdim = True)
    return xf * (threshold / (norm + 1e-6)).clamp(max = 1.0)


def attn_res(blocks: torch.Tensor | None, hidden: torch.Tensor, query: torch.Tensor, eps: float = 1e-6,
             chunk: int = 4096) -> torch.Tensor:
    """
    Block attention residual. blocks: [nb, S, D] (may be None/empty), hidden: [S, D], query: [D] or [nq, D].
    Returns fp32 [S, D] (query [D]) or [nq, S, D].
    """
    single = query.dim() == 1
    q = query.float().reshape(-1, hidden.shape[-1])            # [nq, D]
    D = hidden.shape[-1]
    S = hidden.shape[0]
    out = torch.empty((q.shape[0], S, D), dtype = torch.float32, device = hidden.device)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        h = hidden[s0:s1].float().unsqueeze(0)
        st = h if blocks is None or blocks.shape[0] == 0 else torch.cat([blocks[:, s0:s1].float(), h], dim = 0)
        st = st * torch.rsqrt(st.square().mean(-1, keepdim = True) + eps)           # [n, s, D]
        scores = torch.einsum("qd,nsd->sqn", q, st) * (D ** -0.5)                     # [s, nq, n]
        p = torch.softmax(scores, dim = -1)
        out[:, s0:s1] = torch.einsum("sqn,nsd->qsd", p, st)
    return out[0] if single else out


def short_conv(x: torch.Tensor, w: torch.Tensor, k: int, adt: torch.dtype, state: torch.Tensor | None = None):
    """
    Causal depthwise conv + residual, /sqrt(k + 1). x: [S, C], w: [C, 1, k].
    state: previous k-1 raw inputs [k-1, C] (incremental decode). Returns adt [S, C], or (out, new_state) when state is given
    """
    xf = x.float()
    S, C = xf.shape
    if state is None:
        y = F.conv1d(xf.t().unsqueeze(0), w.float(), padding = k - 1, groups = C)[0, :, :S].t()
    else:
        ctx = torch.cat([state.float(), xf], 0)
        y = F.conv1d(ctx.t().unsqueeze(0), w.float(), padding = 0, groups = C)[0].t()
    out = ((y + xf) / math.sqrt(k + 1)).to(adt)
    if state is None:
        return out
    return out, torch.cat([state, x.to(state.dtype)], 0)[-(k - 1):]


def dyn_short_conv(x: torch.Tensor, dyn_in: torch.Tensor, static_w: torch.Tensor, w_proj: torch.Tensor,
                   n_heads: int, k: int, adt: torch.dtype, state: torch.Tensor | None = None):
    """Token-dependent causal conv (global layers). x: [S, C]; static_w: [k, C]; w_proj: [n_heads*k, hidden]"""
    S, C = x.shape
    dyn = F.linear(dyn_in, w_proj).float().view(S, n_heads, 1, k)
    weights = static_w.float().t().reshape(n_heads, -1, k) + dyn                        # [S, nh, C/nh, k]
    xf = x.float()
    ctx = F.pad(xf.t(), (k - 1, 0)) if state is None else torch.cat([state.float(), xf], 0).t()   # [C, S + k - 1]
    win = ctx.unfold(-1, k, 1).flip(-1).transpose(0, 1).reshape(S, n_heads, -1, k)    # [S, nh, C/nh, k]; [..., lag]
    y = (win * weights).sum(-1).reshape(S, C)
    out = ((y + xf) / math.sqrt(2 * k + 1)).to(adt)
    if state is None:
        return out
    return out, torch.cat([state, x.to(state.dtype)], 0)[-(k - 1):]


TF32_ATTENTION = True


class _tf32:
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = TF32_ATTENTION
    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rel: torch.Tensor, sinks: torch.Tensor,
                   proj: torch.Tensor, window: int | None, extent: int, scale: float,
                   max_elems: int = 1 << 28, q_off: int = 0) -> torch.Tensor:
    """
    Single-sequence causal attention with relative-position logit bias and sink logits.
      q: [S, H, dh], k/v: [S, Hk, dh]  (activation dtype; computed in fp32 with tf32 matmuls, exact for bf16 inputs)
      rel: [S, H, d_rel], proj: [d_rel, extent]: bias[h, i, j] = (rel[i, h] @ proj)[i - j] for 0 <= i - j < extent
      sinks: [S, H] extra softmax logit per query (zero value)
    Returns [S, H, dh] fp32.
    """
    S, H, dh = q.shape
    M = k.shape[0]                  # keys; query i sits at key index q_off + i (q_off > 0 when k includes cached history)
    Hk = k.shape[1]
    G = H // Hk
    out = torch.empty((S, H, dh), dtype = torch.float32, device = q.device)
    qf, kf, vf = q.float(), k.float(), v.float()
    relf, projf, sinkf = rel.float(), proj.float(), sinks.float()
    with _tf32():
        # query chunk so that H * qc * kv_len stays bounded
        kv_max = M if window is None else min(M, window + 1024)
        qc = max(64, min(2048, (max_elems // (H * max(kv_max, 1))) // 64 * 64))
        for i0 in range(0, S, qc):
            i1 = min(S, i0 + qc)
            a0, a1 = q_off + i0, q_off + i1
            j0 = 0 if window is None else max(0, a0 - window + 1)
            nq, kl = i1 - i0, a1 - j0
            qi = torch.arange(a0, a1, device = q.device)
            kj = torch.arange(j0, a1, device = q.device)
            dist = qi[:, None] - kj[None, :]                                          # [nq, kl]
            vis = dist >= 0
            if window is not None:
                vis &= dist < window
            # scores
            qg = qf[i0:i1].reshape(nq, Hk, G, dh).permute(1, 2, 0, 3).reshape(Hk, G * nq, dh)
            kg = kf[j0:a1].permute(1, 2, 0)                                           # [Hk, dh, kl]
            sc = torch.bmm(qg, kg).reshape(H, nq, kl) * scale
            # relative bias
            R = torch.matmul(relf[i0:i1], projf).permute(1, 0, 2)                      # [H, nq, extent]
            idx = dist.clamp(0, extent - 1)
            bias = R.gather(2, idx.unsqueeze(0).expand(H, -1, -1))
            bias.masked_fill_(((dist < 0) | (dist >= extent)).unsqueeze(0), 0.0)
            sc += bias
            del bias, R
            sc.masked_fill_(~vis.unsqueeze(0), float("-inf"))
            sk = sinkf[i0:i1].t().unsqueeze(-1)                                       # [H, nq, 1]
            m = torch.maximum(sc.amax(-1, keepdim = True), sk)
            sc -= m
            sc.exp_()
            den = sc.sum(-1, keepdim = True) + torch.exp(sk - m)
            sc /= den
            o = torch.bmm(sc.reshape(Hk, G * nq, kl), vf[j0:a1].permute(1, 0, 2))      # [Hk, G*nq, dh]
            out[i0:i1] = o.reshape(Hk, G, nq, dh).permute(2, 0, 1, 3).reshape(nq, H, dh)
            del sc, o
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Non-expert layer weights

class LayerWeights:
    """Non-expert tensors of one decoder layer on a device (bf16 as stored; small tensors are used in fp32)"""

    def __init__(self, src: Inkling2Source, cfg: Inkling2Cfg, i: int, device, adt = torch.bfloat16):
        self.cfg, self.i, self.device = cfg, i, device
        self.is_global = cfg.is_global(i)
        p = f"model.llm.layers.{i}."
        g = lambda k: src.get(k, device)
        g_ = g
        g = lambda k: (lambda t: t.to(adt) if (t.dtype == torch.bfloat16 and t.dim() == 2 and t.shape[-1] >= 4608) else t)(g_(k))
        a = p + "attn."
        self.wq, self.wk, self.wv = g(a + "wq_du.weight"), g(a + "wk_dv.weight"), g(a + "wv_dv.weight")
        self.wr, self.wo = g(a + "wr_du.weight"), g(a + "wo_ud.weight")
        self.wgate, self.wsink = g(a + "gate_attn.weight"), g(a + "attn_sink_proj.weight")
        self.q_scale, self.rel_scale = g(a + "q_l2_clip.scale"), g(a + "rel_l2_clip.scale")
        self.rel_proj = g(a + "rel_logits_proj.proj")
        if self.is_global:
            self.kc_static, self.kc_proj = g(a + "k_sconv.static_w"), g(a + "k_sconv.weight_proj")
            self.vc_static, self.vc_proj = g(a + "v_sconv.static_w"), g(a + "v_sconv.weight_proj")
            self.q_norm, self.k_norm, self.v_norm = g(p + "q_input_norm.weight"), g(p + "k_input_norm.weight"), g(p + "v_input_norm.weight")
        else:
            self.kc_w, self.vc_w = g(a + "k_sconv.weight"), g(a + "v_sconv.weight")
            self.attn_norm = g(p + "attn_norm.weight")
        self.attn_sconv = g(p + "attn_sconv.weight")
        self.mlp_sconv = g(p + "mlp_sconv.weight")
        self.mlp_norm = g(p + "mlp_norm.weight")
        self.router_w = g(p + "mlp.gate.weight")
        self.router_b = g(p + "mlp.gate.bias")
        qk = f"model.llm.attn_res.attn_queries.{i - 1}"
        self.attn_query = g(qk) if i > 0 else None
        self.mlp_query = g(f"model.llm.attn_res.mlp_queries.{i}")


# ----------------------------------------------------------------------------------------------------------------------
# Layer computation

def attn_inputs(W: LayerWeights, hidden: torch.Tensor, blocks: torch.Tensor | None, adt = torch.bfloat16):
    """The normed inputs (xq, xk, xv) [S, D] adt of the attention projections; one shared tensor in sliding layers"""
    cfg, i = W.cfg, W.i
    if W.is_global:
        if i == 0:
            xq = xk = xv = hidden.float()
        else:
            xq, xk, xv = attn_res(blocks, hidden, W.attn_query, cfg.eps)           # each fp32 [S, D]
        xq = rms_norm(xq, W.q_norm, cfg.eps).to(adt)
        xk = rms_norm(xk, W.k_norm, cfg.eps).to(adt)
        xv = rms_norm(xv, W.v_norm, cfg.eps).to(adt)
    else:
        x = hidden.float() if i == 0 else attn_res(blocks, hidden, W.attn_query, cfg.eps)
        xq = xk = xv = rms_norm(x, W.attn_norm, cfg.eps).to(adt)
    return xq, xk, xv


_ATTN_W = {"q": "wq", "k": "wk", "v": "wv", "gate": "wgate", "o": "wo"}


def _proj(W: LayerWeights, name: str, x: torch.Tensor, qlin) -> torch.Tensor:
    if qlin is not None and name in qlin:
        return qlin[name](x)
    return F.linear(x, getattr(W, _ATTN_W[name]))


def layer_attention(W: LayerWeights, hidden: torch.Tensor, blocks: torch.Tensor | None, adt = torch.bfloat16,
                    cache = None, pos0: int = 0, taps: dict | None = None, qlin: dict | None = None):
    """
    Attention half of a decoder layer for one sequence.
    hidden: [S, D] adt, blocks: [nb, S, D] adt or None. Returns the post-attention-conv output [S, D] (adt).
    taps: if a dict, receives the projection inputs xq / xk / xv and o (the gated input of wo).
    qlin: optional replacements {"q" | "k" | "v" | "gate" | "o": callable x -> x @ W^T} (quantized projections).
    """
    cfg, i = W.cfg, W.i
    S = hidden.shape[0]
    H, Hk, dh, dr = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.d_rel
    xq, xk, xv = attn_inputs(W, hidden, blocks, adt)
    if taps is not None:
        taps["xq"], taps["xk"], taps["xv"] = xq, xk, xv

    q = _proj(W, "q", xq, qlin)
    k = _proj(W, "k", xk, qlin)
    v = _proj(W, "v", xv, qlin)
    cs = cache.conv if cache is not None else {}
    if W.is_global:
        k = dyn_short_conv(k, xk, W.kc_static, W.kc_proj, Hk, cfg.conv_k, adt, cs.get("k"))
        v = dyn_short_conv(v, xv, W.vc_static, W.vc_proj, Hk, cfg.conv_k, adt, cs.get("v"))
    else:
        k = short_conv(k, W.kc_w, cfg.conv_k, adt, cs.get("k"))
        v = short_conv(v, W.vc_w, cfg.conv_k, adt, cs.get("v"))
    if cache is not None:
        (k, cs["k"]), (v, cs["v"]) = k, v
    r = F.linear(xq, W.wr)
    gate = _proj(W, "gate", xq, qlin)
    sinks = F.linear(xq, W.wsink)

    q = l2_clip(q.view(S, H, dh), cfg.qk_clip, W.q_scale)
    rel = l2_clip(r.view(S, H, dr), cfg.rel_clip, W.rel_scale)
    if W.is_global:
        pos = torch.arange(pos0, pos0 + S, device = hidden.device, dtype = torch.float32)
        tau = 1.0 + cfg.log_alpha * torch.log(((pos + 1.0) / cfg.log_n_floor).clamp(min = 1.0))
        tau = tau[:, None, None]
        q = (q.to(adt).float() * tau)
        rel = (rel.to(adt).float() * tau)
    q, rel = q.to(adt), rel.to(adt)
    k = l2_clip(k.view(S, Hk, dh), cfg.qk_clip, None).to(adt)
    v = v.view(S, Hk, dh)
    vf = v.float()
    v = (vf * torch.rsqrt(vf.square().mean(-1, keepdim = True) + cfg.eps)).to(adt)

    q_off = 0
    k_all, v_all = k, v
    if cache is not None:
        hk, hv = cache.history(adt)
        if hk is not None:
            q_off = hk.shape[0]
            k_all, v_all = torch.cat([hk, k], 0), torch.cat([hv, v], 0)
        cache.append(k, v)
    o = attention_core(
        q, k_all, v_all, rel, sinks, W.rel_proj, None if W.is_global else cfg.window, cfg.rel_extent, dh ** -0.5, q_off = q_off
    ).to(adt)
    o = o.reshape(S, H * dh)
    o = (o.float() * torch.sigmoid(gate.float())).to(adt)
    if taps is not None:
        taps["o"] = o
    o = _proj(W, "o", o, qlin)
    if cache is None:
        return short_conv(o, W.attn_sconv, cfg.conv_k, adt)
    out, cs["attn"] = short_conv(o, W.attn_sconv, cfg.conv_k, adt, cs["attn"])
    return out


def layer_mlp_input(W: LayerWeights, hidden: torch.Tensor, blocks: torch.Tensor | None, attn_out: torch.Tensor,
                    adt = torch.bfloat16):
    """
    Everything between the attention output and the experts. Returns (resid fp32 [S, D], x_mlp adt [S, D]).
    At block boundaries the incoming hidden state becomes a block state (blocks_local) and is not added to the residual.
    """
    cfg, i = W.cfg, W.i
    a = attn_out.float()
    if cfg.is_boundary(i):
        resid = a
        bl = hidden.unsqueeze(0) if blocks is None or blocks.shape[0] == 0 else torch.cat([blocks, hidden.unsqueeze(0)], 0)
    else:
        resid = hidden.float() + a
        bl = blocks
    x = attn_res(bl, resid, W.mlp_query, cfg.eps)
    x = rms_norm(x, W.mlp_norm, cfg.eps).to(adt)
    return resid, x


def route(W: LayerWeights, x: torch.Tensor):
    """Sigmoid router. x: [T, D]. Returns (indices [T, k], weights fp32 [T, k] incl. route_scale)"""
    cfg = W.cfg
    logits = F.linear(x.float(), W.router_w.float())
    scores = torch.sigmoid(logits)
    sel = torch.topk(scores + W.router_b.float(), cfg.topk, dim = -1, sorted = False).indices
    w = scores.gather(1, sel)
    w = w / (w.sum(-1, keepdim = True) + 1e-20)
    return sel, w * cfg.route_scale


def expert_act(gu: torch.Tensor, cfg: Inkling2Cfg, adt) -> torch.Tensor:
    """gu: [n, 2*inter] with gate in the first half, up in the second"""
    g, u = gu.float().chunk(2, dim = -1)
    return (softcap(g, cfg.gate_cap) * torch.sigmoid(g) * softcap(u, cfg.up_cap)).to(adt)


class DenseExperts:
    """Experts as dense tensors: w13 [E, 2*inter, D] (gate rows first), w2 [E, D, inter]"""

    def __init__(self, w13: torch.Tensor, w2: torch.Tensor):
        self.w13, self.w2 = w13, w2
        self.device = w13.device

    def gate_up(self, e, x): return F.linear(x, self.w13[e])
    def down(self, e, a): return F.linear(a, self.w2[e])


class HessianSink:
    """Accumulates per-expert input Hessians (packed fp32 upper triangle) and token statistics"""

    def __init__(self, cfg: Inkling2Cfg, device, weight_pow: float = 0.0, weight_eps: float = 0.0, experts = None):
        from .quant.exl3_lib.hessian_packed import packed_numel
        self.cfg, self.device = cfg, device
        E = cfg.experts
        self.weight_pow, self.weight_eps = weight_pow, weight_eps
        self.H13 = torch.zeros((E, packed_numel(cfg.hidden)), dtype = torch.float32, device = device)
        self.H2 = torch.zeros((E, packed_numel(cfg.inter)), dtype = torch.float32, device = device)
        self.Hs = torch.zeros(packed_numel(cfg.hidden), dtype = torch.float32, device = device)   # layer-wide input covariance
        self.n_tokens = torch.zeros(E, dtype = torch.float64)           # routed tokens per expert
        self.n_weight = torch.zeros(E, dtype = torch.float64)           # sum of Hessian row weights
        self.n_total = 0

    def add_shared(self, x):
        from .quant.exl3_lib.hessian_packed import accumulate_
        accumulate_(self.Hs, x)
        self.n_total += x.shape[0]

    def add(self, e, xe, ae, we):
        from .quant.exl3_lib.hessian_packed import accumulate_
        n = xe.shape[0]
        if self.weight_pow != 0.0:
            rw = (we.float().clamp_min(1e-3) ** self.weight_pow) + self.weight_eps
            rw = rw / rw.new_tensor(1.0 + self.weight_eps)                # roughly unit mean for w ~ 1
            sq = rw.sqrt().unsqueeze(1)
            xe_h = (xe.float() * sq).to(xe.dtype)
            ae_h = (ae.float() * sq).to(ae.dtype)
            self.n_weight[e] += rw.sum().item()
        else:
            xe_h, ae_h = xe, ae
            self.n_weight[e] += n
        accumulate_(self.H13[e], xe_h)
        accumulate_(self.H2[e], ae_h)
        self.n_tokens[e] += n


def moe_forward(x: torch.Tensor, sel: torch.Tensor, rw: torch.Tensor, ex, cfg: Inkling2Cfg, adt = torch.bfloat16,
                sink: HessianSink | None = None) -> torch.Tensor:
    """Routed experts. x: [T, D] adt, sel/rw: [T, k]. Returns [T, D] adt (fp32 accumulation like the reference)"""
    T, D = x.shape
    k = sel.shape[1]
    out = torch.zeros((T, D), dtype = torch.float32, device = x.device)
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable = True)
    counts = torch.bincount(flat, minlength = cfg.experts).tolist()
    tok_sorted = order // k
    w_sorted = rw.reshape(-1)[order]
    if sink is not None:
        sink.add_shared(x)
    pos = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        tok = tok_sorted[pos : pos + n]
        we = w_sorted[pos : pos + n]
        pos += n
        xe = x[tok]
        a = expert_act(ex.gate_up(e, xe), cfg, adt)
        if sink is not None:
            sink.add(e, xe, a, we)
        y = ex.down(e, a)
        out.index_add_(0, tok, y.float() * we.unsqueeze(1))
    return out.to(adt)


def layer_finish(W: LayerWeights, resid: torch.Tensor, moe_out: torch.Tensor, adt = torch.bfloat16, cache = None) -> torch.Tensor:
    if cache is None:
        y = short_conv(moe_out, W.mlp_sconv, W.cfg.conv_k, adt)
    else:
        y, cache.conv["mlp"] = short_conv(moe_out, W.mlp_sconv, W.cfg.conv_k, adt, cache.conv["mlp"])
    return (resid + y.float()).to(adt)


def layer_forward(W: LayerWeights, ex, hidden, blocks, adt = torch.bfloat16, sink = None, return_parts = False,
                  cache = None, pos0: int = 0):
    """One full decoder layer for one sequence. Returns (new_hidden, new_blocks)"""
    cfg, i = W.cfg, W.i
    a = layer_attention(W, hidden, blocks, adt, cache, pos0)
    resid, x = layer_mlp_input(W, hidden, blocks, a, adt)
    sel, rw = route(W, x)
    y = moe_forward(x, sel, rw, ex, cfg, adt, sink)
    new_hidden = layer_finish(W, resid, y, adt, cache)
    if cfg.is_boundary(i):
        blocks = hidden.unsqueeze(0) if blocks is None or blocks.shape[0] == 0 else torch.cat([blocks, hidden.unsqueeze(0)], 0)
    return new_hidden, blocks


class HeadWeights:
    def __init__(self, src: Inkling2Source, cfg: Inkling2Cfg, device, with_lm_head = True):
        self.cfg = cfg
        self.out_query = src.get("model.llm.attn_res.output_query", device)
        self.norm = src.get("model.llm.norm.weight", device)
        self.lm_head = src.get("model.llm.unembed.weight", device) if with_lm_head else None


def final_hidden(Hd: HeadWeights, hidden, blocks, adt = torch.bfloat16):
    x = attn_res(blocks, hidden, Hd.out_query, Hd.cfg.eps)
    return rms_norm(x, Hd.norm, Hd.cfg.eps).to(adt)


def logits_from_hidden(Hd: HeadWeights, h: torch.Tensor) -> torch.Tensor:
    z = F.linear(h, Hd.lm_head).float()
    z = softcap(z, Hd.cfg.final_cap)
    return z[..., : Hd.cfg.unpadded_vocab]


def embed_tokens(src_embed: torch.Tensor, embed_norm: torch.Tensor, ids: torch.Tensor, eps: float, adt = torch.bfloat16):
    return rms_norm(src_embed[ids], embed_norm, eps).to(adt)


# ----------------------------------------------------------------------------------------------------------------------
# Routed expert providers

class SourceExperts:
    """
    Packed (NVFP4 / MXFP8) routed experts of one layer of the source checkpoint, resident on `device` in their packed
    form (about 7 / 12.6 GB). Dequantizes single experts to fp32 on demand; `dense()` materializes the whole layer as bf16.
    """

    def __init__(self, src: Inkling2Source, cfg: Inkling2Cfg, layer: int, device):
        self.cfg, self.layer, self.device = cfg, layer, device
        p = f"model.llm.layers.{layer}.mlp.experts."
        self.nvfp4 = src.has(p + "w13_weight_amax")
        g = lambda k: src.get(p + k, device)
        self.w13, self.s13, self.w2, self.s2 = g("w13_weight"), g("w13_weight_scale"), g("w2_weight"), g("w2_weight_scale")
        if self.nvfp4:
            self.g13 = g("w13_weight_amax").float() / g("w13_weight_outer_scale_denominator").float()
            self.g2 = g("w2_weight_amax").float() / g("w2_weight_outer_scale_denominator").float()

    @staticmethod
    def deinterleave(w: torch.Tensor) -> torch.Tensor:
        """Checkpoint w13 rows alternate gate/up; returns [gate rows; up rows]"""
        return torch.cat([w[..., 0::2, :], w[..., 1::2, :]], dim = -2)

    def w13_f32(self, e: int) -> torch.Tensor:
        """[2*inter, D]: gate rows first"""
        if self.nvfp4:
            w = dequant_nvfp4(self.w13[e], self.s13[e], self.g13[e])
        else:
            w = dequant_mxfp8(self.w13[e], self.s13[e])
        return self.deinterleave(w)

    def w2_f32(self, e: int) -> torch.Tensor:
        """[D, inter]"""
        if self.nvfp4:
            return dequant_nvfp4(self.w2[e], self.s2[e], self.g2[e])
        return dequant_mxfp8(self.w2[e], self.s2[e])

    def dense(self, dtype = torch.bfloat16, device = None) -> DenseExperts:
        dev = device or self.device
        E, D, I = self.cfg.experts, self.cfg.hidden, self.cfg.inter
        w13 = torch.empty((E, 2 * I, D), dtype = dtype, device = dev)
        w2 = torch.empty((E, D, I), dtype = dtype, device = dev)
        for e in range(E):
            w13[e] = self.w13_f32(e).to(dtype)
            w2[e] = self.w2_f32(e).to(dtype)
        return DenseExperts(w13, w2)

    # Provider interface (slow, per-call dequant; used for tests only)
    def gate_up(self, e, x): return F.linear(x, self.w13_f32(e).to(x.dtype))
    def down(self, e, a): return F.linear(a, self.w2_f32(e).to(a.dtype))


class Exl3Experts:
    """
    Converted experts: one LinearEXL3 per expert and projection (gate_proj / up_proj / down_proj). kernel = True runs the
    EXL3 kernels; otherwise the layer is reconstructed once to dense tensors (`dense()`).
    """

    def __init__(self, src: Inkling2Source, cfg: Inkling2Cfg, layer: int, device, config = None):
        from .quant.exl3 import LinearEXL3
        self.cfg, self.layer, self.device = cfg, layer, device
        D, I = cfg.hidden, cfg.inter
        self.lin = {}
        self.bits = {}
        for e in range(cfg.experts):
            for name, (fi, fo) in {"gate_proj": (D, I), "up_proj": (D, I), "down_proj": (I, D)}.items():
                k = f"model.llm.layers.{layer}.mlp.experts.{e}.{name}"
                t = lambda s: src.get(f"{k}.{s}", device) if src.has(f"{k}.{s}") else None
                self.lin[(e, name)] = LinearEXL3(
                    config, fi, fo, suh = t("suh"), svh = t("svh"), trellis = t("trellis"),
                    mcg = t("mcg"), mul1 = t("mul1"), out_dtype = torch.half, key = k
                )
        self.params = {}

    def gate_up(self, e, x):
        xh = x.to(torch.half).contiguous()
        g = self.lin[(e, "gate_proj")].forward(xh, self.params, torch.half)
        u = self.lin[(e, "up_proj")].forward(xh, self.params, torch.half)
        return torch.cat([g, u], dim = -1).to(x.dtype)

    def down(self, e, a):
        return self.lin[(e, "down_proj")].forward(a.to(torch.half).contiguous(), self.params, torch.half).to(a.dtype)

    def dense(self, dtype = torch.bfloat16) -> DenseExperts:
        cfg = self.cfg
        E, D, I = cfg.experts, cfg.hidden, cfg.inter
        w13 = torch.empty((E, 2 * I, D), dtype = dtype, device = self.device)
        w2 = torch.empty((E, D, I), dtype = dtype, device = self.device)
        for e in range(E):
            w13[e, :I] = self.lin[(e, "gate_proj")].get_weight_tensor().t().to(dtype)
            w13[e, I:] = self.lin[(e, "up_proj")].get_weight_tensor().t().to(dtype)
            w2[e] = self.lin[(e, "down_proj")].get_weight_tensor().t().to(dtype)
        return DenseExperts(w13, w2)


# ----------------------------------------------------------------------------------------------------------------------
# Sliding-window aware KV cache

def kv_quant(x: torch.Tensor, bits: int):
    """x: [n, Hk, dh] -> (codes, fp16 scale [n, Hk]). bits 16 = raw, 8 = int8, 4 = packed int4; symmetric per token and head"""
    if bits == 16:
        return x, None
    qmax = (1 << (bits - 1)) - 1
    sc = (x.float().abs().amax(-1) / qmax).clamp_min(1e-8)
    q = torch.round(x.float() / sc.unsqueeze(-1)).clamp(-qmax - 1, qmax).to(torch.int8)
    if bits == 4:
        u = (q + 8).to(torch.uint8)
        q = u[..., 0::2] | (u[..., 1::2] << 4)
    return q, sc.half()


def kv_dequant(q: torch.Tensor, sc, bits: int, adt):
    if bits == 16:
        return q
    if bits == 4:
        lo, hi = (q & 15).to(torch.int8) - 8, (q >> 4).to(torch.int8) - 8
        q = torch.stack([lo, hi], -1).reshape(*q.shape[:-1], q.shape[-1] * 2)
    return (q.float() * sc.float().unsqueeze(-1)).to(adt)


class LayerCache:
    """
    Per-sequence state of one layer. Global layers keep every K/V token (growing), sliding-window layers a ring buffer of
    `window` tokens. K/V codes are stored with `bits` (16/8/4) per element and a per-(token, head) fp16 scale. Short-conv
    states hold the last k-1 raw conv inputs (K, V and the post-attention / post-MLP convolutions).
    """

    def __init__(self, cfg: Inkling2Cfg, is_global: bool, device, bits: int = 8, max_len: int = 4096, adt = torch.bfloat16):
        self.cfg, self.is_global, self.bits, self.device = cfg, is_global, bits, device
        Hk, dh = cfg.kv_heads, cfg.head_dim
        self.cap = max_len if is_global else cfg.window
        self.n = 0                          # tokens seen
        cdim = dh // 2 if bits == 4 else dh
        cdt = adt if bits == 16 else (torch.uint8 if bits == 4 else torch.int8)
        self.kq = torch.zeros((self.cap, Hk, cdim), dtype = cdt, device = device)
        self.vq = torch.zeros_like(self.kq)
        self.ks = None if bits == 16 else torch.zeros((self.cap, Hk), dtype = torch.half, device = device)
        self.vs = None if bits == 16 else torch.zeros_like(self.ks)
        k = cfg.conv_k - 1
        z = lambda c: torch.zeros((k, c), dtype = adt, device = device)
        self.conv = {"k": z(Hk * dh), "v": z(Hk * dh), "attn": z(cfg.hidden), "mlp": z(cfg.hidden)}

    def append(self, k, v):
        n = k.shape[0]
        qk, sk = kv_quant(k, self.bits); qv, sv = kv_quant(v, self.bits)
        if self.is_global:
            assert self.n + n <= self.cap, "global cache full"
            idx = torch.arange(self.n, self.n + n, device = self.device)
        else:
            if n > self.cap:             # only the last `cap` tokens can ever be attended
                qk, qv = qk[-self.cap:], qv[-self.cap:]
                sk = None if sk is None else sk[-self.cap:]; sv = None if sv is None else sv[-self.cap:]
                idx = torch.arange(self.n + n - self.cap, self.n + n, device = self.device) % self.cap
            else:
                idx = torch.arange(self.n, self.n + n, device = self.device) % self.cap
        self.kq[idx], self.vq[idx] = qk, qv
        if sk is not None: self.ks[idx], self.vs[idx] = sk, sv
        self.n += n

    def history(self, adt):
        """Dequantized K/V of the tokens still attendable, oldest first (None if empty)"""
        if self.n == 0:
            return None, None
        m = min(self.n, self.cap)
        idx = torch.arange(self.n - m, self.n, device = self.device) % self.cap
        ks = None if self.ks is None else self.ks[idx]
        vs = None if self.vs is None else self.vs[idx]
        return kv_dequant(self.kq[idx], ks, self.bits, adt), kv_dequant(self.vq[idx], vs, self.bits, adt)

    def nbytes(self, tokens: int | None = None):
        """(bytes per token for the K/V store, flat bytes per sequence) at the allocated geometry"""
        per_tok = (self.kq[0].numel() * self.kq.element_size() + (self.ks[0].numel() * 2 if self.ks is not None else 0)) * 2
        flat = sum(t.numel() * t.element_size() for t in self.conv.values())
        return per_tok, (per_tok * self.cap if not self.is_global else 0) + flat


class Inkling2SeqCache:
    def __init__(self, cfg: Inkling2Cfg, devices_per_layer, bits_global: int = 8, bits_swa: int = 8, max_len: int = 4096,
                 adt = torch.bfloat16):
        self.layers = [LayerCache(cfg, cfg.is_global(i), devices_per_layer[i], bits_global if cfg.is_global(i) else bits_swa,
                                  max_len, adt) for i in range(cfg.layers)]
        self.cfg = cfg

    def report(self):
        """Measured bytes: per-token growth (global layers) and flat bytes per sequence (ring buffers + conv states)"""
        per_tok = sum(c.nbytes()[0] for c in self.layers if c.is_global)
        flat = sum(c.nbytes()[1] for c in self.layers)
        return {"per_token_bytes": per_tok, "per_token_KiB": per_tok / 1024, "flat_per_seq_MiB": flat / 2**20,
                "global_layers": sum(c.is_global for c in self.layers), "swa_layers": sum(not c.is_global for c in self.layers)}
