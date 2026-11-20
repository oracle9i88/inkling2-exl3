"""
One GPU: the EXL3 attention projections of v3 (spark/quant/quantize_attention.py), as the serving engine slices them
for tensor parallelism, against the whole tensors and against the bf16 source.

  PYTHONPATH=/work/build/exllamav3 python spark/test_attnq_engine.py --model /models/Inkling-2-Small-2spark-v2 \
      --attn /work/tmp/attn-L00.safetensors --layer 0 [--world 2] [--rows 37]

Per rank of `world`:
  1. input projections: the rank's tensor (its heads' 128-row blocks) gives the whole tensor's outputs in those rows
  2. wo: the ranks' partial outputs (their heads' input columns) add up to the whole tensor's output
  3. the engine's assembly (I2Engine.attn_proj_q) into the fused layouts equals the bf16 layout filled from the
     reconstructed weights, and stays close to the bf16 source weights (quantization error only)
"""
import argparse, glob, json, os, sys, types
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from exllamav3.modules import inkling2_attnq as AQ
from exllamav3.modules.inkling2_parallel import head_shard
from exllamav3.modules.inkling2_engine import I2Engine


class Stc:
    """has_tensor / get_tensor over a dict (enough for AQ.exl3_select)"""

    def __init__(self, tensors):
        self.t = tensors

    def has_tensor(self, k):
        return k in self.t

    def get_tensor(self, k, device, no_defer = True):
        return self.t[k].to(device)


def bf16_weights(model_dir, keys, device):
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "model-nonexpert-*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in keys:
                if k in s.keys():
                    out[k] = s.get_tensor(k).to(device)
    missing = [k for k in keys if k not in out]
    assert not missing, missing
    return out


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--attn", required = True)
    ap.add_argument("--layer", type = int, required = True)
    ap.add_argument("--world", type = int, default = 2)
    ap.add_argument("--rows", type = int, default = 37, help = "test rows (decode-sized by default)")
    ap.add_argument("--head", help = "head.safetensors of the v3 run: also check the EXL3 head per rank")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    t = cfg.get("text_config", cfg)
    H, Hk, D, dh = t["num_attention_heads"], t["num_key_value_heads"], t["hidden_size"], 128
    is_global = a.layer not in t["local_layer_ids"]
    p = f"model.llm.layers.{a.layer}.attn."
    stc = Stc(load_file(a.attn))
    src = bf16_weights(a.model, [p + n for n in ("wq_du.weight", "gate_attn.weight", "wk_dv.weight", "wv_dv.weight",
                                                 "wo_ud.weight", "wr_du.weight", "attn_sink_proj.weight")]
                       + ([p + "k_sconv.weight_proj", p + "v_sconv.weight_proj"] if is_global else []), dev)
    hq, hk = H * dh, Hk * dh
    T = a.rows
    xs = [torch.randn(T, D, device = dev, dtype = torch.bfloat16) for _ in range(3)]
    full = {}
    if is_global:
        full["qg"] = AQ.exl3_select(stc, p + "wqg", dev)
        full["k"], full["v"] = AQ.exl3_select(stc, p + "wk", dev), AQ.exl3_select(stc, p + "wv", dev)
    else:
        full["qgkv"] = AQ.exl3_select(stc, p + "wqgkv", dev)
    full["o"] = AQ.exl3_select(stc, p + "wo", dev)
    dq = {n: l.get_weight_tensor().t().contiguous() for n, l in full.items()}         # (out, in) fp16
    if is_global:
        Wq, Wg, Wk, Wv = dq["qg"][:hq], dq["qg"][hq:], dq["k"], dq["v"]
    else:
        Wq, Wg, Wk, Wv = dq["qgkv"][:hq], dq["qgkv"][hq:2 * hq], dq["qgkv"][2 * hq:2 * hq + hk], dq["qgkv"][2 * hq + hk:]
    for name, Wd, s in (("q", Wq, "wq_du"), ("gate", Wg, "gate_attn"), ("k", Wk, "wk_dv"), ("v", Wv, "wv_dv"), ("o", dq["o"], "wo_ud")):
        print(f"  {name:5s} reconstructed vs bf16 source: weight rel err {rel(Wd, src[p + s + '.weight']):.4f}")
    xo = torch.randn(T, hq, device = dev, dtype = torch.bfloat16)
    y_o_full = AQ.run(full["o"], xo).float()
    y_o_sum = torch.zeros_like(y_o_full)
    for r in range(a.world):
        sh = head_shard(H, Hk, r, a.world)
        rr = AQ.rank_ranges(sh, H, Hk)
        qs = lambda w, dim = 0: sh.q_rows(w, dim)
        ks = lambda w, dim = 0: sh.kv_rows(w, dim)
        # 1. input projections: rows of the whole tensor's output
        if is_global:
            lin_qg = AQ.exl3_select(stc, p + "wqg", dev, rows = [rr["q"], rr["gate"]])
            lin_k = AQ.exl3_select(stc, p + "wk", dev, rows = [rr["kv"]])
            lin_v = AQ.exl3_select(stc, p + "wv", dev, rows = [rr["kv"]])
            yf = AQ.run(full["qg"], xs[0])
            want = torch.cat([yf[:, rr["q"][0]:rr["q"][1]], yf[:, rr["gate"][0]:rr["gate"][1]]], 1)
            got = AQ.run(lin_qg, xs[0])
            e1 = max(rel(got, want), rel(AQ.run(lin_k, xs[1]), AQ.run(full["k"], xs[1])[:, rr["kv"][0]:rr["kv"][1]]),
                     rel(AQ.run(lin_v, xs[2]), AQ.run(full["v"], xs[2])[:, rr["kv"][0]:rr["kv"][1]]))
        else:
            lin_qgkv = AQ.exl3_select(stc, p + "wqgkv", dev, rows = [rr["q"], rr["gate"], rr["k"], rr["v"]])
            yf = AQ.run(full["qgkv"], xs[0])
            want = torch.cat([yf[:, r0:r1] for r0, r1 in (rr["q"], rr["gate"], rr["k"], rr["v"])], 1)
            e1 = rel(AQ.run(lin_qgkv, xs[0]), want)
        # 2. wo: partial sums
        lin_o = AQ.exl3_select(stc, p + "wo", dev, cols = [rr["q"]])
        y_o_sum += AQ.run(lin_o, xo[:, rr["q"][0]:rr["q"][1]]).float()
        # 3. the engine's assembly against the bf16 layout of the reconstructed (and of the source) weights
        wr, wsink = qs(src[p + "wr_du.weight"]), qs(src[p + "attn_sink_proj.weight"])
        Hl, Hkl = sh.hl, sh.hkl
        eng = types.SimpleNamespace(Hl = Hl, Hkl = Hkl, mm = lambda x, w: (x.float() @ w.float().t()).to(torch.bfloat16))
        L = types.SimpleNamespace(wrs = torch.cat([wr, wsink], 0).contiguous())
        lay = lambda q, g, k, v: torch.cat([q, wr, g, wsink] + ([k, v] if k is not None else []), 0)
        if is_global:
            L.lin_qg, L.lin_k, L.lin_v = lin_qg, lin_k, lin_v
            kp, vp = ks(src[p + "k_sconv.weight_proj"]), ks(src[p + "v_sconv.weight_proj"])
            L.kproj, L.vproj = kp, vp
            qb, kb, vb = I2Engine.attn_proj_q(eng, L, xs[0], xs[1], xs[2])
            ref = [(qb, xs[0], lay(qs(Wq), qs(Wg), None, None), lay(qs(src[p + "wq_du.weight"]), qs(src[p + "gate_attn.weight"]), None, None)),
                   (kb, xs[1], torch.cat([ks(Wk), kp], 0), torch.cat([ks(src[p + "wk_dv.weight"]), kp], 0)),
                   (vb, xs[2], torch.cat([ks(Wv), vp], 0), torch.cat([ks(src[p + "wv_dv.weight"]), vp], 0))]
        else:
            L.lin_qgkv = lin_qgkv
            qb = I2Engine.attn_proj_q(eng, L, xs[0])
            ref = [(qb, xs[0], lay(qs(Wq), qs(Wg), ks(Wk), ks(Wv)),
                    lay(qs(src[p + "wq_du.weight"]), qs(src[p + "gate_attn.weight"]), ks(src[p + "wk_dv.weight"]), ks(src[p + "wv_dv.weight"])))]
        e3 = max(rel(b, x.float() @ Wr.float().t()) for b, x, Wr, _ in ref)
        e4 = max(rel(b, x.float() @ Ws.float().t()) for b, x, _, Ws in ref)
        print(f"rank {r}/{a.world} ({'global' if is_global else 'sliding'} layer {a.layer}, heads {sh.q0}-{sh.q1}): "
              f"slice vs whole {e1:.2e}; assembly vs reconstructed {e3:.2e}; vs bf16 source {e4:.2e}")
        assert e1 < 2e-3, e1
        assert e3 < 5e-3, e3
        assert e4 < 0.05, e4
    e2 = rel(y_o_sum, y_o_full)
    print(f"wo: ranks' partial sums vs whole {e2:.2e}")
    assert e2 < 2e-3, e2
    # prefill-sized calls (> 144 rows take exllamav3's reconstruct path, which reads config.infer_params)
    cfg_ns = types.SimpleNamespace(infer_params = types.SimpleNamespace(no_reconstruct = False))
    sh = head_shard(H, Hk, 0, a.world)
    rr = AQ.rank_ranges(sh, H, Hk)
    key = p + ("wqg" if is_global else "wqgkv")
    lin_big = AQ.exl3_select(stc, key, dev, rows = [rr["q"]], config = cfg_ns)
    xb = torch.randn(600, D, device = dev, dtype = torch.bfloat16)
    yb = AQ.run(lin_big, xb).float()
    ys = AQ.run(AQ.exl3_select(stc, key, dev, rows = [rr["q"]]), xb[:100]).float()
    want = xb.float() @ (Wq[rr["q"][0]:rr["q"][1]] if not is_global else dq["qg"][rr["q"][0]:rr["q"][1]]).float().t()
    e5, e6 = rel(yb, want), rel(yb[:100], ys)
    print(f"600-row call (reconstruct path) vs reconstructed weights {e5:.2e}; vs the 100-row kernel path {e6:.2e}")
    assert e5 < 5e-3 and e6 < 5e-3, (e5, e6)
    if a.head:
        hs = Stc(load_file(a.head))
        Vp = hs.t[AQ.HEAD_KEY + ".svh"].shape[0]
        full_h = AQ.exl3_select(hs, AQ.HEAD_KEY, dev)
        Wh = full_h.get_weight_tensor().t().float()                                    # [Vp, D]
        Ws = bf16_weights(a.model, ["model.llm.unembed.weight"], dev)["model.llm.unembed.weight"].float()
        V = Ws.shape[0]
        xh = torch.randn(T, D, device = dev, dtype = torch.bfloat16)
        print(f"head: vocab {V} padded to {Vp}; reconstructed vs bf16 weight rel err {rel(Wh[:V], Ws):.4f}; "
              f"padding rows max |w| {Wh[V:].abs().max().item():.2e}")
        for r in range(a.world):
            r0, r1 = AQ.head_rows(hs, r, a.world)
            lin = AQ.exl3_select(hs, AQ.HEAD_KEY, dev, rows = [(r0, r1)], config = cfg_ns)
            z = AQ.run_f32(lin, xh)
            zb = AQ.run_f32(lin, torch.randn(300, D, device = dev, dtype = torch.bfloat16))   # chunked path
            e7 = rel(z, xh.float() @ Wh[r0:r1].t())
            print(f"  rank {r}: rows {r0}-{r1}, fp32 logits vs reconstructed {e7:.2e}, dtype {z.dtype}, chunked {tuple(zb.shape)}")
            assert z.dtype == torch.float32 and e7 < 2e-3 and zb.shape == (300, r1 - r0), (z.dtype, e7, zb.shape)
    print("ATTNQ-ENGINE-TEST PASS")


if __name__ == "__main__":
    main()
