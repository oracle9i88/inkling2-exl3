"""
Inkling-2 routed experts whose bitrates differ inside a layer (the two-Spark v2 build: every expert has its own gate/up
and down rate on 1.5 ... 4 bits, by its routing share).

exllamav3's fused MoE paths (the coop decode kernels, exl3_moe, the batched reconstruct tier) take one rate per projection
for all experts of a module (MultiLinear asserts it). A mixed layer is therefore split into rate classes, experts with
equal (K gate, K up, K down), each its own BlockSparseMLP over a contiguous range of router ids:

    route once with the layer's router (its columns permuted so that every class is a contiguous id range), then every
    class adds its experts' part of the routed sum. Picks outside a class's range are masked inside the kernels and
    contribute exact zeros, exactly as for an expert-parallel shard (rows with no pick in the range are written as zeros).

The permutation is an internal relabeling: the checkpoint keeps its expert ids, router id r is checkpoint expert perm[r].
Rank r of tp holds router ids [r * E / tp, (r + 1) * E / tp). Every class is dealt to the ranks half and half, so the
ranks hold the same bytes; within a class the experts have similar routing shares (rates were assigned by share), which
evens out the work. An optional per-expert weight (routing counts of a profile run) balances the work exactly.

Decode-sized calls (rows <= MAX_BSZN) do not go class by class: every class costs its own kernel launches (about 4.6 us
each on a 5090 even nearly empty, ~10 classes per layer), so they run all of this rank's experts through one pair of
the fused decode kernels (exl3_moe_coop_mk), whose blocks decode their expert at its own rate (a rate code per expert;
rates 1.5 ... 4 bpw, mul1 codebook). Prefill-sized calls take the classes (the per-class cost is small against them).
I2_MIXK_DECODE=0 sends decode calls through the classes too.

Uniform layers (one rate for all experts, e.g. the v1 build) do not come here: they keep the single BlockSparseMLP and
the identity order.
"""
from __future__ import annotations
import json
import os
import torch

from .block_sparse_mlp import BlockSparseMLP

PROJ = ("gate_proj", "up_proj", "down_proj")
MK_CODES = {1.5: 0, 2.0: 1, 2.5: 2, 3.0: 3, 3.5: 4, 4.0: 5}       # exl3_moe_coop_mk rate codes (gemv_tile_mk)
MOE_COOP_ACT_SILU_OAI = 3
_mk_ctr = {}                                                       # per device: the decode kernels' counter scratch


def _k_of_width(w: int):
    K = w / 16                      # LinearEXL3: 16 * K uint16 per tile (K = 1.5, 2.5, 3.5 have 24 / 40 / 56)
    return int(K) if float(K).is_integer() else K


def expert_rates(stc, key: str, num_experts: int):
    """Per expert (K gate, K up, K down) from the safetensors headers (no tensor data read), or None when the experts are
    not all EXL3 (then the caller keeps the single-module path)"""
    rates = []
    for e in range(num_experts):
        ks = []
        for p in PROJ:
            m = stc.get_tensor_meta(f"{key}.experts.{e}.{p}.trellis")
            if m is None:
                return None
            ks.append(_k_of_width(next(iter(m.values()))["shape"][-1]))
        rates.append(tuple(ks))
    return rates


def _num(c):
    return tuple(float(x) for x in c)


def plan_layout(rates, tp: int, weights = None):
    """rates: per checkpoint expert a hashable rate class. Returns (perm, parts): perm[r] = the checkpoint expert behind
    router id r; parts[rank] = [(first, last, class)], contiguous router-id ranges covering [rank * E / tp, ...).
    Deterministic (all ranks compute the same layout from the same headers)."""
    E = len(rates)
    assert E % tp == 0, f"{E} experts do not split over {tp} ranks"
    classes = sorted(set(rates), key = _num)
    # every class's quota per rank: an equal share (equal bytes); the remainder goes to the ranks with the smallest quota
    # total so far, so every rank ends with exactly E / tp
    quota, tot = {}, [0] * tp
    for c in classes:
        n = sum(1 for x in rates if x == c)
        q = [n // tp] * tp
        for r in sorted(range(tp), key = lambda r: (tot[r], r))[:n % tp]:
            q[r] += 1
        quota[c] = q
        tot = [a + b for a, b in zip(tot, q)]
    held = [[] for _ in range(tp)]
    given = {c: [0] * tp for c in classes}
    if weights is not None:
        # heaviest expert first (any class), each to the rank with the least routed work so far that has quota left in
        # its class: the hot experts are placed while the light ones can still even the work out
        work = [0.0] * tp
        for e in sorted(range(E), key = lambda e: (-float(weights[e]), e)):
            c = rates[e]
            r = min((r for r in range(tp) if given[c][r] < quota[c][r]), key = lambda r: (work[r], given[c][r], r))
            held[r].append(e)
            given[c][r] += 1
            work[r] += float(weights[e])
    else:
        for e in range(E):
            c = rates[e]
            r = min((r for r in range(tp) if given[c][r] < quota[c][r]), key = lambda r: (given[c][r], r))
            held[r].append(e)
            given[c][r] += 1
    perm, parts = [], []
    for r in range(tp):
        es = sorted(held[r], key = lambda e: (_num(rates[e]), e))
        base = len(perm)
        perm += es
        ranges, i = [], 0
        while i < len(es):
            j = i
            while j < len(es) and rates[es[j]] == rates[es[i]]:
                j += 1
            ranges.append((base + i, base + j, rates[es[i]]))
            i = j
        parts.append(ranges)
    assert sorted(perm) == list(range(E))
    return perm, parts


def load_weights(model_dir: str, layer: int):
    """Optional per-expert work weights (routing counts of a profile run): <model>/expert_load.json
    {"layers": {"<layer>": [count per checkpoint expert]}}. None when absent."""
    path = os.path.join(model_dir, "expert_load.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        w = json.load(fh).get("layers", {}).get(str(layer))
    return w


def _shared_selection(bsz, cfg, y, params):
    # BlockSparseMLP.forward routes through self.routing_fn when it holds a gate: every class takes the layer's selection
    return params["i2_routing"]


class RateClassMLP(BlockSparseMLP):
    """One rate class of a mixed layer. Only the layer's router class builds a routing config; the others route through
    the selection MixedRateMoE.forward hands over (they hold a reference to the router's gate, unregistered, so that
    BlockSparseMLP.forward takes its routing branch)."""

    i2_router = False

    def load_routing(self, **kwargs):
        if self.i2_router:
            return super().load_routing(**kwargs)
        self.routing_cfg = None


class MixedRateMoE:
    """
    Drop-in for a layer's BlockSparseMLP when its experts mix rates: forward(x, params) returns this rank's fp32 partial
    routed sum [T, D] like BlockSparseMLP.forward. parts[0] holds the layer's router (its columns permuted by `perm` at
    load); every part is a RateClassMLP over one contiguous router-id range.
    """

    mixed_rates = True

    def __init__(self, parts: list[RateClassMLP], perm: list[int], classes: list):
        assert parts, "a mixed layer needs at least one rate class on this rank"
        self.parts, self.perm, self.classes = parts, perm, classes
        self.router = parts[0]
        self.router.i2_router = True
        self.route = self.router.routing_fn            # the layer's real router (dots); kept before the override below
        for p in parts:
            p.routing_fn = _shared_selection
            if p is not self.router:
                p.routing_gate = self.router.routing_gate
        self.hidden = self.router.hidden_size
        self.first, self.last = classes[0][0], classes[-1][1]       # this rank's router-id range
        self.device = None
        self.mk = None                                              # decode path state (_setup_decode)
        self.mk_off = None                                          # why the decode path is off, if it is
        self.counts = None                                          # routing counts per router id (I2_EXPERT_COUNT=1)

    def load(self, device, **kwargs):
        stc = self.router.config.stc
        with stc.eager_load():                          # the router's values are read (permuted) right below
            self.router.load(device, **kwargs)
        self._permute_router()
        for p in self.parts[1:]:
            p.load(device, **kwargs)
        self.device = device
        self._setup_decode()
        if os.environ.get("I2_EXPERT_COUNT", "0") == "1":
            # every row this layer routes, padding and speculative drafts included: that is the work each expert
            # does, which is what expert_load.json balances (plan_layout). Plain tensor ops, so captured decode
            # graphs keep counting on replay
            self.counts = torch.zeros(len(self.perm), dtype = torch.long, device = device)

    def _setup_decode(self):
        """Pointer and rate-code tables over all of this rank's experts (router-id order) for exl3_moe_coop_mk, and the
        scratch: BlockSparseMLP's fused-decode buffers (shared by every MoE module on the device, used in stream
        order) and an own zeroed counter block (the kernels keep it consistent from call to call)"""
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache
        from .block_sparse_mlp import MAX_BSZN
        self.mk, self.mk_off = None, None
        if os.environ.get("I2_MIXK_DECODE", "1") == "0":
            self.mk_off = "I2_MIXK_DECODE=0"
            return
        if not hasattr(ext, "exl3_moe_coop_mk"):
            self.mk_off = "extension without exl3_moe_coop_mk (rebuild with the current patch)"
            return
        p0 = self.parts[0]
        g, u, d = [], [], []
        for p in self.parts:
            g += [l.inner for l in p.gates]
            u += [l.inner for l in p.ups]
            d += [l.inner for l in p.downs]
        if p0.experts_cfg is None or not p0.gated or p0.activation_fn != "swiglu_oai":
            self.mk_off = "not a gated swiglu_oai expert layout"
            return
        for x in g + u + d:
            if getattr(x, "trellis", None) is None or not x.mul1 or x.mcg or x.bias is not None or float(x.K) not in MK_CODES:
                self.mk_off = f"expert tensor {getattr(x, 'key', '?')}: K {getattr(x, 'K', '?')} / codebook / bias not supported"
                return
        if any(float(a.K) != float(b.K) for a, b in zip(g, u)):
            self.mk_off = "gate and up rates differ"
            return
        dev = g[0].trellis.device
        ptrs = lambda xs, f: torch.tensor([getattr(x, f).data_ptr() for x in xs], dtype = torch.long, device = dev)
        cfg = p0.experts_cfg
        Hi, I, Ho = p0.ups[0].in_features, p0.intermediate_size_padded, p0.downs[0].out_features
        rows = MAX_BSZN * p0.num_experts_per_tok
        n_ctr = rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 * rows + 3       # exl3_moe_coop_ctr_len
        ctr = _mk_ctr.get(dev)
        if ctr is None or ctr.numel() < n_ctr:
            ctr = _mk_ctr[dev] = torch.zeros((n_ctr,), dtype = torch.int, device = dev)
        self.mk = dict(
            tables = [ptrs(xs, f) for xs in (g, u, d) for f in ("trellis", "suh", "svh")],
            code_gu = torch.tensor([MK_CODES[float(x.K)] for x in g], dtype = torch.uint8, device = dev),
            code_d = torch.tensor([MK_CODES[float(x.K)] for x in d], dtype = torch.uint8, device = dev),
            Hi = Hi, act_limit = float(p0.act_limit),
            had_g = cfg.yh, had_u = g_tensor_cache.get(dev, (rows, Hi), torch.half, "moe1_temp_hidden_u"),
            gu_g = cfg.interm_g, gu_u = cfg.interm_u, act_out = cfg.interm_a, d_out = cfg.out_d, ctr = ctr,
            out = cfg.out_bszn, max_rows = MAX_BSZN, ext = ext,
        )
        assert self.mk["code_gu"].numel() == self.last - self.first

    def _forward_decode(self, y, sel, w):
        m = self.mk
        t = m["tables"]
        m["ext"].exl3_moe_coop_mk(
            y, sel, w, self.first, self.last, m["Hi"], *t, MOE_COOP_ACT_SILU_OAI, m["act_limit"],
            m["had_g"], m["had_u"], m["gu_g"], m["gu_u"], m["act_out"], m["d_out"], m["ctr"], m["out"],
            m["code_gu"], m["code_d"],
        )
        return m["out"][:y.shape[0]]

    def _permute_router(self):
        """Router id r := checkpoint expert perm[r]: permute the gate's columns (it is (hidden, E), logits = y @ W) and
        the selection bias in place (the routing config holds these very tensors), drop the lazily built copies"""
        r = self.router
        ig = r.routing_gate.inner
        pt = torch.tensor(self.perm, dtype = torch.long, device = ig.weight.device)
        assert ig.weight.shape[-1] == len(self.perm), f"{r.key}: router width {tuple(ig.weight.shape)}"
        ig.weight.copy_(ig.weight[:, pt])
        if getattr(ig, "bias", None) is not None:
            ig.bias.copy_(ig.bias[pt.to(ig.bias.device)])
        if r.e_score_correction_bias is not None:
            b = r.e_score_correction_bias
            b.copy_(b[pt.to(b.device)])
        assert r.per_expert_scale is None and getattr(r, "tid2eid", None) is None, \
            f"{r.key}: per-expert routing tensors besides the selection bias are not permuted"
        cfg = r.routing_cfg
        for f in ("gate_tensor_t", "gate_i8", "gate_sb", "e_score_bias_h", "gate_tensor_f32", "e_score_bias_f32"):
            if hasattr(cfg, f):
                setattr(cfg, f, None)

    def unload(self):
        self.mk = None
        for p in self.parts:
            p.unload()
        self.device = None

    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        y = x.view(-1, self.hidden)
        p = dict(params) if params else {}
        p["i2_routing"] = self.route(y.shape[0], self.router.routing_cfg, y, p)
        if self.counts is not None:
            sel = p["i2_routing"][0].reshape(-1)
            self.counts.index_add_(0, sel, torch.ones_like(sel))
        if self.mk is not None and y.shape[0] <= self.mk["max_rows"]:
            # all of this rank's experts in one launch pair (a view of the shared decode output buffer, as
            # BlockSparseMLP's own decode path returns)
            return self._forward_decode(y, *p["i2_routing"]).view(x.shape[:-1] + (self.hidden,))
        out = None
        for part in self.parts:
            o = part.forward(x, p)
            if out is None:
                out = o.clone()      # the fused decode path returns a view of a buffer every MoE module shares
            else:
                out.add_(o)
        return out

    def expert_counts(self, reset: bool = False):
        """routing counts per checkpoint expert (the expert_load.json layout), or None when counting is off"""
        if self.counts is None:
            return None
        c = self.counts.tolist()
        if reset:
            self.counts.zero_()
        out = [0] * len(self.perm)
        for r, e in enumerate(self.perm):
            out[e] = c[r]
        return out

    def describe(self) -> str:
        return " ".join(f"{'/'.join(str(k) for k in c)}x{l - f}" for f, l, c in self.classes)
