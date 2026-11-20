"""
CPU test of the mixed-rate expert layout (overlay/exllamav3/modules/inkling2_mixk.py), no GPU, no exllamav3 build:

  1. expert_rates: K per projection from trellis widths (whole and half rates), None when an expert is not EXL3
  2. plan_layout on the real two-spark v2 rate tables (48 layers, 2 and 3 ranks): perm is a permutation, every rank holds
     E / tp experts as contiguous single-class ranges in class order, every class split as evenly as possible (so the
     bytes per rank differ by at most a few experts), deterministic
  3. with per-expert work weights (heavy-tailed): the work per rank is balanced within a few percent
  4. MixedRateMoE on a small torch MoE: router columns permuted in place by perm, routed once, every class adds its range
     (picks outside masked) -> the same output as the original-order MoE, for every split of the classes over ranks

  python3 spark/test_mixk_cpu.py        (needs torch)
"""
import importlib.util, json, os, random, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "overlay", "exllamav3", "modules", "inkling2_mixk.py")


class _StubBSM:                      # stands in for exllamav3's BlockSparseMLP (the planner and wrapper do not need it)
    pass


def load_mixk():
    pkg = types.ModuleType("i2stub"); pkg.__path__ = []
    bsm = types.ModuleType("i2stub.block_sparse_mlp"); bsm.BlockSparseMLP = _StubBSM
    sys.modules["i2stub"], sys.modules["i2stub.block_sparse_mlp"] = pkg, bsm
    spec = importlib.util.spec_from_file_location("i2stub.inkling2_mixk", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MIXK = load_mixk()


def v2_rates():
    d = json.load(open(os.path.join(HERE, "data", "rates-two-spark-v2.json")))
    LV = d["levels"]
    num = lambda x: int(x) if float(x).is_integer() else x
    return {int(L): [(num(LV[int(s[2 * e])]), num(LV[int(s[2 * e])]), num(LV[int(s[2 * e + 1])])) for e in range(len(s) // 2)]
            for L, s in d["layers"].items()}


def expert_bytes(rate, D = 4096, I = 2304):    # bytes of one expert's trellis at its rates (the scales are the same for all)
    g, u, d = rate
    return (g + u) * D * I / 8 + d * I * D / 8


def test_expert_rates():
    shapes = {}
    for e, (g, d) in enumerate([(3, 4), (1.5, 2.5), (4, 4), (2, 3.5)]):
        for p, K in (("gate_proj", g), ("up_proj", g), ("down_proj", d)):
            shapes[f"m.experts.{e}.{p}.trellis"] = [256, 144, int(16 * K)]

    class Stc:
        def __init__(self, s): self.s = s
        def get_tensor_meta(self, k, optional = True):
            return {k: {"shape": self.s[k], "n_bytes": 0, "dtype": "torch.int16"}} if k in self.s else None

    r = MIXK.expert_rates(Stc(shapes), "m", 4)
    assert r == [(3, 3, 4), (1.5, 1.5, 2.5), (4, 4, 4), (2, 2, 3.5)], r
    assert all(isinstance(x, int) for x in r[0]) and isinstance(r[1][0], float)
    del shapes["m.experts.2.down_proj.trellis"]
    assert MIXK.expert_rates(Stc(shapes), "m", 4) is None
    print("1. expert_rates: whole and half rates from trellis widths, None for a non-EXL3 expert: PASS")


def check_layout(rates, tp, weights = None):
    E = len(rates)
    per = E // tp
    perm, parts = MIXK.plan_layout(rates, tp, weights)
    assert sorted(perm) == list(range(E))
    assert (perm, parts) == MIXK.plan_layout(rates, tp, weights), "not deterministic"
    classes = sorted(set(rates), key = MIXK._num)
    count = {c: [0] * tp for c in classes}
    byts = [0.0] * tp
    for r in range(tp):
        ranges = parts[r]
        assert ranges[0][0] == r * per and ranges[-1][1] == (r + 1) * per
        for (f, l, c), nxt in zip(ranges, ranges[1:] + [None]):
            assert l > f and all(rates[perm[i]] == c for i in range(f, l))
            if nxt is not None:
                assert nxt[0] == l and MIXK._num(nxt[2]) > MIXK._num(c), "ranges must be contiguous, one per class, in order"
            count[c][r] += l - f
            byts[r] += sum(expert_bytes(rates[perm[i]]) for i in range(f, l))
    for c in classes:
        assert max(count[c]) - min(count[c]) <= 1, (c, count[c])
    return perm, parts, byts


def test_v2_layouts():
    R = v2_rates()
    assert len(R) == 48 and all(len(v) == 384 for v in R.values())
    worst_b, n_parts = 0.0, []
    for tp in (2, 3, 1):
        for L, rates in sorted(R.items()):
            perm, parts, byts = check_layout(rates, tp)
            worst_b = max(worst_b, (max(byts) - min(byts)) / (sum(byts) / tp))
            if tp == 2:
                n_parts += [len(p) for p in parts]
    print(f"2. v2 layouts (48 layers x tp 1/2/3): contiguous one-class ranges, classes split evenly, bytes per rank within "
          f"{100 * worst_b:.2f}%; classes per rank and layer at tp 2: {min(n_parts)}-{max(n_parts)} "
          f"(mean {sum(n_parts) / len(n_parts):.1f}): PASS")


def test_weights():
    R = v2_rates()
    rnd = random.Random(7)
    worst_w, worst_nw = 0.0, 0.0
    for L, rates in sorted(R.items()):
        # heavy-tailed work, larger for higher-rate classes (rates were assigned by routing share)
        w = [rnd.lognormvariate(0, 1.0) * 2 ** (2 * (MIXK._num(c)[0] - 1.5)) for c in rates]
        tot = sum(w)
        for weights in (w, None):
            perm, parts, _ = check_layout(rates, 2, weights)
            work = [sum(w[perm[i]] for f, l, _ in parts[r] for i in range(f, l)) for r in range(2)]
            imb = (max(work) - min(work)) / (tot / 2)
            if weights is None:
                worst_nw = max(worst_nw, imb)
            else:
                worst_w = max(worst_w, imb)
    assert worst_w < 0.05, worst_w
    print(f"3. work weights: worst work imbalance between 2 ranks {100 * worst_w:.2f}% with weights "
          f"({100 * worst_nw:.1f}% without): PASS")


# ---- 4. the wrapper on a torch MoE ----------------------------------------------------------------------------------

def dots_route(bsz, cfg, y, params):
    """the dots router in torch (sigmoid scores, top-k on score + bias, weights normalized * scale)"""
    logits = y.float() @ cfg.gate.float()
    s = torch.sigmoid(logits)
    sel = torch.topk(s + cfg.bias.float(), cfg.k, dim = -1).indices
    w = s.gather(1, sel)
    return sel, (w / w.sum(-1, keepdim = True) * cfg.scale).half()


class FakePart:
    """one rate class: its experts' part of the routed sum, picks outside [f, l) masked (as the kernels do)"""

    def __init__(self, f, l, experts, perm, D, k, gate = None, bias = None, scale = 8.0):
        self.f, self.l, self.experts, self.perm = f, l, experts, perm
        self.hidden_size = D
        self.routing_fn = dots_route
        self.key = "fake"
        if gate is not None:
            self.routing_gate = types.SimpleNamespace(inner = types.SimpleNamespace(weight = gate, bias = None))
            self.e_score_correction_bias = bias
            self.routing_cfg = types.SimpleNamespace(gate = gate, bias = bias, k = k, scale = scale, gate_tensor_t = 1, gate_i8 = 1)
        else:
            self.routing_gate, self.e_score_correction_bias, self.routing_cfg = None, None, None
        self.per_expert_scale, self.tid2eid = None, None

    def forward(self, x, params):
        sel, w = params["i2_routing"]
        y = x.view(-1, self.hidden_size).float()
        out = torch.zeros_like(y)
        for t in range(y.shape[0]):
            for j in range(sel.shape[1]):
                r = int(sel[t, j])
                if self.f <= r < self.l:
                    out[t] += float(w[t, j]) * self.experts[self.perm[r]](y[t])
        return out


def test_wrapper():
    torch.manual_seed(0)
    D, E, k = 16, 24, 4
    rates = [(random.Random(e).choice([1.5, 2, 3]),) * 2 + (random.Random(e + 99).choice([2, 4]),) for e in range(E)]
    gate = torch.randn(D, E)
    bias = torch.randn(E) * 0.1
    mats = [torch.randn(D, D) / D ** 0.5 for _ in range(E)]
    experts = [lambda v, m = m: torch.tanh(v @ m) for m in mats]
    x = torch.randn(9, D)

    # reference: original expert order, one module
    cfg0 = types.SimpleNamespace(gate = gate, bias = bias, k = k, scale = 8.0)
    sel0, w0 = dots_route(9, cfg0, x, {})
    ref = torch.zeros(9, D)
    for t in range(9):
        for j in range(k):
            ref[t] += float(w0[t, j]) * experts[int(sel0[t, j])](x[t])

    for tp in (1, 2, 3):
        perm, parts = MIXK.plan_layout(rates, tp)
        total = torch.zeros(9, D)
        for r in range(tp):
            g, b = gate.clone(), bias.clone()           # every rank permutes its own copy
            ps = [FakePart(f, l, experts, perm, D, k, *((g, b) if i == 0 else ())) for i, (f, l, _) in enumerate(parts[r])]
            m = MIXK.MixedRateMoE(ps, perm, parts[r])
            m._permute_router()
            assert torch.equal(g, gate[:, perm]) and torch.equal(b, bias[perm])
            assert ps[0].routing_cfg.gate_tensor_t is None and ps[0].routing_cfg.gate_i8 is None
            total += m.forward(x, {})
        err = (total - ref).abs().max().item()
        assert err < 1e-5, (tp, err)
    print("4. MixedRateMoE on a torch MoE (router permuted in place, routed once, classes masked and summed) equals the "
          "original-order MoE at tp 1/2/3: PASS")


if __name__ == "__main__":
    test_expert_rates()
    test_v2_layouts()
    test_weights()
    test_wrapper()
    print("all PASS")
