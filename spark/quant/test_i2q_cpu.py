"""
CPU test of the expert-quantization pipeline (i2q.py + quantize_experts.py) on a tiny random Inkling-2 checkpoint
with the same tensor names, layouts and formats as the real one (NVFP4 experts in most layers, MXFP8 in the last two,
global and sliding layers, block attention residuals every 3 layers, a window shorter than the rows).

  1. unit checks: per-expert token lists, Hessian shrinkage, row partitioning
  2. identity "quantizer", two workers: the held-out rows of the quantized track must equal the reference track bit for
     bit (KLD exactly 0), and the reference track must equal a direct layer-by-layer forward pass
  3. resume: stopping after layer 3 and continuing gives the same final state, bit for bit
  4. a rounding "quantizer": KLD and the layer-local errors become positive
  5. the Hessians and weights handed to the quantizer, against a brute-force recomputation
  6. per-expert bits (v2): allocation optimal against brute force, layer means kept, bookkeeping unchanged (identity:
     same final state bit for bit), every expert quantized at its own rate

  ~/.venvs/mlxgen/bin/python spark/quant/test_i2q_cpu.py
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import i2q
from i2q import I2

D, H, HK, DH, DR, EXT, WIN, CK = 64, 4, 2, 16, 4, 16, 16, 4
V, VU, NL, BL, E, TOPK, INTER = 257, 250, 8, 3, 8, 2, 32
GLOBAL = (2, 5)
MXFP8_LAYERS = (6, 7)
R, S, HELD = 6, 24, (4, 5)


def make_checkpoint(d):
    g = torch.Generator().manual_seed(1234)
    rn = lambda *s, std = 1.0: torch.randn(*s, generator = g) * std
    t = {}
    bf = lambda x: x.to(torch.bfloat16)
    t["model.llm.embed.weight"] = bf(rn(V, D))
    t["model.llm.embed_norm.weight"] = bf(1 + 0.1 * rn(D))
    t["model.llm.norm.weight"] = bf(1 + 0.1 * rn(D))
    t["model.llm.unembed.weight"] = bf(rn(V, D, std = D ** -0.5))
    t["model.llm.attn_res.output_query"] = rn(D)
    for i in range(NL):
        if i > 0:
            t[f"model.llm.attn_res.attn_queries.{i - 1}"] = rn(3, D) if i in GLOBAL else rn(D)   # q/k/v queries on global layers
        t[f"model.llm.attn_res.mlp_queries.{i}"] = rn(D)
        p, a = f"model.llm.layers.{i}.", f"model.llm.layers.{i}.attn."
        s = D ** -0.5
        t[a + "wq_du.weight"] = bf(rn(H * DH, D, std = s))
        t[a + "wk_dv.weight"] = bf(rn(HK * DH, D, std = s))
        t[a + "wv_dv.weight"] = bf(rn(HK * DH, D, std = s))
        t[a + "wr_du.weight"] = bf(rn(H * DR, D, std = s))
        t[a + "wo_ud.weight"] = bf(rn(D, H * DH, std = (H * DH) ** -0.5))
        t[a + "gate_attn.weight"] = bf(rn(H * DH, D, std = s))
        t[a + "attn_sink_proj.weight"] = bf(rn(H, D, std = s))
        t[a + "q_l2_clip.scale"] = bf(1 + 0.1 * rn(H, DH))
        t[a + "rel_l2_clip.scale"] = bf(1 + 0.1 * rn(H, DR))
        t[a + "rel_logits_proj.proj"] = bf(rn(DR, EXT, std = 0.3))
        if i in GLOBAL:
            for kv in ("k", "v"):
                t[a + f"{kv}_sconv.static_w"] = bf(rn(CK, HK * DH, std = 0.3))
                t[a + f"{kv}_sconv.weight_proj"] = bf(rn(HK * CK, D, std = 0.05))
            for n in ("q", "k", "v"):
                t[p + f"{n}_input_norm.weight"] = bf(1 + 0.1 * rn(D))
        else:
            t[a + "k_sconv.weight"] = bf(rn(HK * DH, 1, CK, std = 0.3))
            t[a + "v_sconv.weight"] = bf(rn(HK * DH, 1, CK, std = 0.3))
            t[p + "attn_norm.weight"] = bf(1 + 0.1 * rn(D))
        t[p + "attn_sconv.weight"] = bf(rn(D, 1, CK, std = 0.3))
        t[p + "mlp_sconv.weight"] = bf(rn(D, 1, CK, std = 0.3))
        t[p + "mlp_norm.weight"] = bf(1 + 0.1 * rn(D))
        t[p + "mlp.gate.weight"] = bf(rn(E, D, std = s))
        t[p + "mlp.gate.bias"] = 0.1 * rn(E)
        x = p + "mlp.experts."
        if i in MXFP8_LAYERS:
            t[x + "w13_weight"] = (rn(E, 2 * INTER, D) * 2).to(torch.float8_e4m3fn)
            t[x + "w13_weight_scale"] = (122 + torch.randint(0, 3, (E, 2 * INTER, D // 32), generator = g)).to(torch.uint8)
            t[x + "w2_weight"] = (rn(E, D, INTER) * 2).to(torch.float8_e4m3fn)
            t[x + "w2_weight_scale"] = (122 + torch.randint(0, 3, (E, D, INTER // 32), generator = g)).to(torch.uint8)
        else:
            t[x + "w13_weight"] = torch.randint(0, 256, (E, 2 * INTER, D // 2), generator = g).to(torch.uint8)
            t[x + "w13_weight_scale"] = (0.02 + 0.03 * torch.rand(E, 2 * INTER, D // 16, generator = g)).to(torch.float8_e4m3fn)
            t[x + "w13_weight_amax"] = 1 + 0.2 * torch.rand(E, generator = g)
            t[x + "w13_weight_outer_scale_denominator"] = torch.tensor([1.25])
            t[x + "w2_weight"] = torch.randint(0, 256, (E, D, INTER // 2), generator = g).to(torch.uint8)
            t[x + "w2_weight_scale"] = (0.02 + 0.03 * torch.rand(E, D, INTER // 16, generator = g)).to(torch.float8_e4m3fn)
            t[x + "w2_weight_amax"] = 1 + 0.2 * torch.rand(E, generator = g)
            t[x + "w2_weight_outer_scale_denominator"] = torch.tensor([1.25])
    os.makedirs(d, exist_ok = True)
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in t}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"text_config": {
        "hidden_size": D, "num_hidden_layers": NL, "num_attention_heads": H, "num_key_value_heads": HK, "head_dim": DH,
        "d_rel": DR, "rel_extent": EXT, "sliding_window_size": WIN, "vocab_size": V, "unpadded_vocab_size": VU,
        "rms_norm_eps": 1e-6, "sconv_kernel_size": CK, "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0, "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1,
        "n_layers_per_attn_res_block": BL, "n_routed_experts": E, "num_experts_per_tok": TOPK, "intermediate_size": INTER,
        "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "final_logit_softcapping": 30.0,
        "local_layer_ids": [i for i in range(NL) if i not in GLOBAL]}}, open(os.path.join(d, "config.json"), "w"))


def make_calib(p):
    g = np.random.default_rng(7)
    ids = g.integers(0, VU, size = (R, S)).astype(np.int32)
    held = np.zeros(R, bool)
    held[list(HELD)] = True
    np.savez(p, ids = ids, heldout = held)


def run(args, check = True):
    cmd = [sys.executable, os.path.join(HERE, "quantize_experts.py")] + args
    r = subprocess.run(cmd, capture_output = True, text = True)
    if check and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("pipeline failed")
    return r


def load_state(work, track, name, n_tok):
    m = i2q.Mat(os.path.join(work, "state", f"{track}_{name}.bin"), n_tok, D, torch.bfloat16)
    return m.t.clone().view(torch.bfloat16)


def reference_forward(src_dir, calib_p):
    """direct pass: embed, layer_forward with the source experts in every layer -> (final hidden, blocks) per held row"""
    src, cfg = I2.Inkling2Source(src_dir), I2.Inkling2Cfg.from_dir(src_dir)
    z = np.load(calib_p)
    Em, En = src.get("model.llm.embed.weight"), src.get("model.llm.embed_norm.weight")
    out = {}
    for r in HELD:
        h = I2.embed_tokens(Em, En, torch.from_numpy(z["ids"][r]).long(), cfg.eps, torch.bfloat16)
        blocks = None
        for L in range(NL):
            W = I2.LayerWeights(src, cfg, L, "cpu")
            h, blocks = I2.layer_forward(W, I2.SourceExperts(src, cfg, L, "cpu"), h, blocks)
        out[r] = (h, blocks)
    return out


def unit_checks():
    sel = torch.tensor([[0, 2], [1, 2], [2, 0], [3, 3]], dtype = torch.int16)
    tok = torch.tensor([10, 11, 12, 13])
    lists = i2q.routed_tokens(sel, tok, 5)
    want = [[10, 12], [11], [10, 11, 12], [13, 13], []]
    assert [l.tolist() for l in lists] == want, [l.tolist() for l in lists]
    Hx, Hs = torch.eye(3) * 4, torch.eye(3) * 2
    assert torch.allclose(i2q.shrink_gate_up(Hx, 2, Hs, 2.0), torch.eye(3) * 2)          # (4 + 2*2) / 4
    Ha = torch.tensor([[4.0, 2.0], [2.0, 6.0]])
    P = torch.diag(torch.tensor([1.0, 3.0]))
    out = i2q.shrink_down(Ha, 2, P, 2.0, m = 2.0)        # dt = (diag(Ha) + 2 diag(P)) / 4 = [1.5, 3]
    assert torch.allclose(out, (Ha + 2 * torch.diag(torch.tensor([1.5, 3.0]))) / 4)
    assert torch.allclose(i2q.shrink_down(torch.zeros(2, 2), 0, P, 5.0), P)
    c = i2q.Calib.__new__(i2q.Calib)
    c.R, c.S, c.held = 7, 4, np.array([0, 0, 1, 0, 1, 0, 0], bool)
    c.r_rows = np.nonzero(c.held)[0]
    for tr in ("q", "r"):
        allr = sorted(x for p in range(3) for x in c.rows(tr, p, 3))
        assert allr == sorted(c.rows(tr, 0, 1)), tr
    # per-expert bits: the greedy allocation against brute force, and routing shares
    import itertools
    r = lambda K: 4.0 ** (-(K - 2.0))
    lv = i2q.PER_EXPERT_LEVELS
    for share in ([0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25], [0.9, 0.05, 0.03, 0.02], [0.0, 0.2, 0.3, 0.5]):
        for avg in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            K = i2q.expert_bits(share, avg)
            assert abs(sum(K) / 4 - avg) < 1e-9, (share, avg, K)
            best = min(sum(sh * r(k) for sh, k in zip(share, ks))
                       for ks in itertools.product(lv, repeat = 4) if abs(sum(ks) - 4 * avg) < 1e-9)
            got = sum(sh * r(k) for sh, k in zip(share, K))
            assert got <= best + 1e-12, (share, avg, K, got, best)
    assert i2q.expert_bits([0.25] * 4, 2.5) == [2.5] * 4
    sel = torch.tensor([[0, 1], [0, 2], [3, 0]], dtype = torch.int16)
    rw = torch.tensor([[0.5, 0.5], [1.0, 2.0], [1.0, 1.0]])
    sh = i2q.routing_share(sel, rw, 5)              # sums of squares: e0 .25+1+1, e1 .25, e2 4, e3 1, e4 0
    assert np.allclose(sh, np.array([2.25, 0.25, 4.0, 1.0, 0.0]) / 7.5), sh
    print("unit checks: ok")


class CaptureQuantizer:
    def __init__(self):
        self.calls = []

    def quantize(self, weights, H_datas, K, seeds, device):
        self.calls.append(([w.clone() for w in weights], H_datas, K))
        return [(0.0, {"weight": w.t().contiguous().to(torch.bfloat16)}) for w in weights]


def hessian_check(src_dir, calib_p, work_dir, layer, n0 = 1024.0):
    src, cfg = I2.Inkling2Source(src_dir), I2.Inkling2Cfg.from_dir(src_dir)
    calib, work = i2q.Calib(calib_p), i2q.Work(work_dir)
    cq = CaptureQuantizer()
    i2q.phase_experts(layer, 0, E, cfg, src, calib, work, "cpu", cq, {"gate_up": 3, "down": 4}, n0 = n0, group = 3,
                      log = lambda m: None)
    Nq = calib.n_tokens("q")
    x = i2q.Mat(work.tmp(layer, "q_x.bin"), Nq, D, torch.bfloat16).t.view(torch.bfloat16)
    sel = i2q.Mat(work.tmp(layer, "q_sel.bin"), Nq, TOPK, torch.int16).t
    tidx = calib.train_token_index()
    xt = x[tidx].float()
    Hs = xt.t() @ xt / xt.shape[0]
    se = I2.SourceExperts(src, cfg, layer, "cpu")
    gu = [c for c in cq.calls if c[2] == 3]
    dn = [c for c in cq.calls if c[2] == 4]
    W_gu = [w for c in gu for w in c[0]]; H_gu = [h for c in gu for h in c[1]]
    W_dn = [w for c in dn for w in c[0]]; H_dn = [h for c in dn for h in c[1]]
    assert len(W_gu) == 2 * E and len(W_dn) == E
    for e in range(E):
        rows = tidx[(sel[tidx].long() == e).any(1)]
        n = rows.numel()
        xe = x[rows]
        Hx = xe.float().t() @ xe.float()
        want13 = (Hx + n0 * Hs) / (n + n0)
        assert H_gu[2 * e] is H_gu[2 * e + 1], "gate and up must share one Hessian dict"
        assert torch.allclose(H_gu[2 * e]["H"], want13, rtol = 1e-5, atol = 1e-6), f"H13 expert {e}"
        W13 = se.w13_f32(e)
        assert torch.equal(W_gu[2 * e], W13[:INTER].t()) and torch.equal(W_gu[2 * e + 1], W13[INTER:].t()), f"gate/up weights {e}"
        assert torch.equal(W_dn[e], se.w2_f32(e).t()), f"down weight {e}"
        a = I2.expert_act(torch.nn.functional.linear(xe, W13.to(torch.bfloat16)), cfg, torch.bfloat16).float()
        Ha = a.t() @ a
        H2 = H_dn[e]["H"]
        off = H2 - torch.diag_embed(torch.diagonal(H2))
        assert torch.allclose(off, (Ha - torch.diag_embed(torch.diagonal(Ha))) / (n + n0), rtol = 1e-5, atol = 1e-6), f"H2 off-diagonal {e}"
        assert H_dn[e]["count"] == 1 and H_dn[e]["num_total"] == n
    print(f"hessians: {E} experts of layer {layer} match the brute-force recomputation")


def main():
    unit_checks()
    tmp = tempfile.mkdtemp(prefix = "i2q-test-")
    try:
        src, calib = os.path.join(tmp, "src"), os.path.join(tmp, "calib.npz")
        make_checkpoint(src)
        make_calib(calib)
        bits = os.path.join(tmp, "bits.json")
        json.dump({"layers": {str(L): {"gate_up": 3, "down": 4} for L in range(NL)}}, open(bits, "w"))
        base = ["--src", src, "--calib", calib, "--bits", bits, "--gpus", "cpu,cpu", "--group", "3"]

        # 2. identity
        w1 = os.path.join(tmp, "w-identity")
        run(["run", "--work", w1, "--fake", "identity"] + base)
        f = json.load(open(os.path.join(w1, "out", "final.json")))
        print("identity final:", f)
        assert f["kld"] == 0.0 and f["top1_agree"] == 1.0 and f["ppl_quant"] == f["ppl_ref"], f
        for L in range(NL):
            m = json.load(open(os.path.join(w1, "out", f"L{L:02d}.done.json")))
            assert m["moe_rel_mse"] == 0.0, (L, m)
        Nq, Nr = R * S, len(HELD) * S
        hq, hr = load_state(w1, "q", f"h{NL:02d}", Nq), load_state(w1, "r", f"h{NL:02d}", Nr)
        ref = reference_forward(src, calib)
        nb = (NL + BL - 1) // BL
        for j, r in enumerate(HELD):
            h_ref, b_ref = ref[r]
            assert torch.equal(hr[j * S:(j + 1) * S], h_ref), f"reference track row {r} differs from direct forward"
            assert torch.equal(hq[r * S:(r + 1) * S], h_ref), f"quantized track row {r} (identity) differs"
            for kb in range(nb):
                bq = load_state(w1, "r", f"b{kb}", Nr)[j * S:(j + 1) * S]
                assert torch.equal(bq, b_ref[kb]), f"block {kb} row {r}"
        print("identity: held-out rows equal the direct forward pass bit for bit; KLD 0")

        # 3. resume
        w2 = os.path.join(tmp, "w-resume")
        run(["run", "--work", w2, "--fake", "identity", "--layers", "0-3"] + base)
        assert not os.path.exists(os.path.join(w2, "out", "final.json"))
        run(["run", "--work", w2, "--fake", "identity"] + base)
        assert torch.equal(load_state(w2, "q", f"h{NL:02d}", Nq), hq), "resume changed the result"
        print("resume: same final state bit for bit")

        # 4. rounding
        w3 = os.path.join(tmp, "w-round")
        r = run(["run", "--work", w3, "--fake", "0.004"] + base)
        f3 = json.load(open(os.path.join(w3, "out", "final.json")))
        errs = [json.load(open(os.path.join(w3, "out", f"L{L:02d}.done.json")))["moe_rel_mse"] for L in range(NL)]
        print("rounding final:", f3)
        print("rounding moe rel mse per layer:", [f"{e:.2e}" for e in errs])
        assert f3["kld"] > 0 and all(e > 0 for e in errs)
        files = sorted(os.listdir(os.path.join(w3, "out")))
        assert sum(1 for x in files if x.startswith("experts-L")) == 2 * NL, files
        print(r.stdout.strip().splitlines()[-1])

        # 5. the Hessians and weights handed to the quantizer, against a brute-force recomputation
        w4 = os.path.join(tmp, "w-hess")
        run(["run", "--work", w4, "--fake", "identity", "--layers", "0-1", "--keep-tmp"] + base)
        hessian_check(src, calib, w4, layer = 1)

        # 6. per-expert bits (v2)
        lv = "1.5,2,2.5,3,3.5,4"
        w5 = os.path.join(tmp, "w-pe-identity")
        run(["run", "--work", w5, "--fake", "identity", "--per-expert", lv] + base)
        f5 = json.load(open(os.path.join(w5, "out", "final.json")))
        assert f5["kld"] == 0.0, f5
        assert torch.equal(load_state(w5, "q", f"h{NL:02d}", Nq), hq), "per-expert order/batching changed the result"
        for L in range(NL):
            pe = json.load(open(os.path.join(w5, "out", f"L{L:02d}.done.json")))["per_expert_bits"]
            assert abs(pe["gate_up"]["mean"] - 3) < 1e-9 and abs(pe["down"]["mean"] - 4) < 1e-9, (L, pe)
        bits2 = os.path.join(tmp, "bits-2.5.json")
        json.dump({"layers": {str(L): {"gate_up": 2.5, "down": 2.5} for L in range(NL)}}, open(bits2, "w"))
        base2 = [bits2 if x == bits else x for x in base]
        wu, wp = os.path.join(tmp, "w-k-uniform"), os.path.join(tmp, "w-k-pe")
        run(["run", "--work", wu, "--fake", "k0.01"] + base2)
        run(["run", "--work", wp, "--fake", "k0.01", "--per-expert", lv, "--layers", "0-1", "--keep-tmp"] + base2)
        run(["run", "--work", wp, "--fake", "k0.01", "--per-expert", lv] + base2)
        eu = [json.load(open(os.path.join(wu, "out", f"L{L:02d}.done.json")))["moe_rel_mse"] for L in range(NL)]
        ep = [json.load(open(os.path.join(wp, "out", f"L{L:02d}.done.json")))["moe_rel_mse"] for L in range(NL)]
        fu = json.load(open(os.path.join(wu, "out", "final.json")))
        fp = json.load(open(os.path.join(wp, "out", "final.json")))
        print(f"k-aware rounding at 2.5 bits: moe rel mse sum uniform {sum(eu):.3e}, per-expert {sum(ep):.3e}; "
              f"KLD uniform {fu['kld']:.3e}, per-expert {fp['kld']:.3e}")
        for L in (0, 1):                               # each expert's error matches its own rate
            st = {}
            for fn in sorted(os.listdir(os.path.join(wp, "tmp", f"L{L:02d}"))):
                if fn.startswith("experts-") and fn.endswith(".json"):
                    st.update(json.load(open(os.path.join(wp, "tmp", f"L{L:02d}", fn)))["experts"])
            assert len(st) == E, len(st)
            byk = {}
            for v in st.values():
                byk.setdefault(v["K_gate_up"], []).append(v["gate_proj"])
            ks = sorted(byk)
            assert len(ks) > 1, byk
            means = [sum(byk[k]) / len(byk[k]) for k in ks]
            assert all(a > b for a, b in zip(means, means[1:])), (L, ks, means)
            sh = np.array([st[str(e)]["share"] for e in range(E)])
            K = np.array([st[str(e)]["K_gate_up"] for e in range(E)])
            assert (sh * 4.0 ** -(K - 2)).sum() <= (sh * 4.0 ** -(2.5 - 2)).sum() + 1e-12
        assert sum(ep) <= 1.05 * sum(eu), (sum(ep), sum(eu))
        print("per-expert bits: allocation optimal, means kept, identity bit for bit, rates per expert")

        # 7. ablation of the rounding run: a<k> = layers < k quantized, s<k> = layers >= k quantized
        hq3, hr3 = load_state(w3, "q", f"h{NL:02d}", Nq), load_state(w3, "r", f"h{NL:02d}", Nr)
        starts = f"0,{BL},{2 * BL},{NL}"
        run(["ablate", "--work", w3, "--starts", starts] + base)
        ab = json.load(open(os.path.join(w3, "out", "ablate.json")))
        st = ab["tracks"]
        print("ablate:", {k: f"{v['kld']:.3e}" for k, v in st.items()})
        for g in ab["groups"]:
            print(f"  layers {g['layers']}: prefix {g['kld_prefix']:+.3e}, suffix {g['kld_suffix']:+.3e}, mean {g['kld']:+.3e}")
        close = lambda x, y: abs(x - y) <= 1e-12 * max(abs(x), abs(y), 1e-30)
        assert st["a00"]["kld"] == 0.0, st["a00"]                  # all layers back to the source == the reference
        assert close(st[f"a{NL:02d}"]["kld"], f3["kld"]), (st[f"a{NL:02d}"], f3)    # nothing restored == the run
        assert close(st["s00"]["kld"], f3["kld"]), (st["s00"], f3)                    # all quantized == the run
        for x in ("kld_prefix", "kld_suffix", "kld"):
            assert close(sum(g[x] for g in ab["groups"]), f3["kld"]), x
        assert torch.equal(load_state(w3, "q", f"h{NL:02d}", Nq), hq3), "ablate changed the run's quantized track"
        assert torch.equal(load_state(w3, "r", f"h{NL:02d}", Nr), hr3), "ablate changed the run's reference track"
        # a<BL> and s<BL> built independently: runs with layers 0..BL-1 rounded and the rest identity, and the reverse
        def split_run(name, first, rest):
            w = os.path.join(tmp, name)
            run(["run", "--work", w, "--fake", first, "--layers", f"0-{BL - 1}"] + base)
            mp = os.path.join(w, "run.json")
            meta = json.load(open(mp)); meta["fake"] = rest; json.dump(meta, open(mp, "w"))
            run(["run", "--work", w, "--fake", rest] + base)
            return json.load(open(os.path.join(w, "out", "final.json")))
        fa, fs = split_run("w-split-a", "0.004", "identity"), split_run("w-split-s", "identity", "0.004")
        ta, ts = st[f"a{BL:02d}"], st[f"s{BL:02d}"]
        assert close(fa["kld"], ta["kld"]) and fa["top1_agree"] == ta["top1_agree"], (fa, ta)
        assert close(fs["kld"], ts["kld"]) and fs["top1_agree"] == ts["top1_agree"], (fs, ts)
        # resume: a second call recomputes nothing and gives the same numbers
        r7 = run(["ablate", "--work", w3, "--starts", starts] + base)
        assert "tracks in" not in r7.stdout, r7.stdout
        assert json.load(open(os.path.join(w3, "out", "ablate.json")))["tracks"] == st
        print(f"ablate: a00 = reference (KLD 0), a{NL:02d} = s00 = the run, a{BL:02d} / s{BL:02d} = independent split runs "
              f"({fa['kld']:.6e} / {fs['kld']:.6e}), resume recomputes nothing")
        print("I2Q-CPU-TEST PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


if __name__ == "__main__":
    main()
