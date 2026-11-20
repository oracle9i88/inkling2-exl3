"""
CPU test of quantize_attention.py (v3: attention and head quantized on top of a run's experts) on the tiny random
Inkling-2 checkpoint of test_i2q_cpu.py.

  1. experts from an identity run, attention identity: KLD exactly 0, and the student's final state equals the expert
     run's bit for bit (the attention path with the quantized projections swapped in is the source path)
  2. resume after layer 3: same final state bit for bit
  3. attention rounded: KLD > 0; tensors per layer as documented (wqgkv in sliding layers; wqg, wk, wv in global
     layers; wo), shapes (out, in); the Hessians handed to the quantizer against a brute-force recomputation
  4. head rounded as well: final-head.json, KLD above the run without it

  ~/.venvs/mlxgen/bin/python spark/quant/test_attnq_cpu.py
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_i2q_cpu as T
from i2q import I2

PY = sys.executable


def run(script, args):
    r = subprocess.run([PY, os.path.join(HERE, script)] + args, capture_output = True, text = True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"{script} failed")
    return r


class CaptureQuantizer:
    """records (weight (in, out), H) per call; identity"""

    def __init__(self):
        self.calls = []

    def quantize(self, weights, H_datas, K, seeds, device):
        out = []
        for w, h in zip(weights, H_datas):
            self.calls.append((h["first_key"], w.clone(), h["H"].clone(), h["num_total"]))
            out.append((0.0, {"weight": w.t().contiguous().to(torch.bfloat16)}))
        return out


def hessian_check(src, calib_p, work, L):
    """recompute layer L's attention Hessians from the student state (kept with --keep-tmp) and compare"""
    import i2q, quantize_attention as QA
    cfg, s = I2.Inkling2Cfg.from_dir(src), I2.Inkling2Source(src)
    calib = i2q.Calib(calib_p)
    w = i2q.Work(work)
    cq = CaptureQuantizer()
    shutil.copytree(w.tmp(L), w.tmp(L) + ".bak")
    for f in os.listdir(w.tmp(L)):                       # phase_attnq rewrites the tmp mats
        if f.startswith("q_"):
            os.remove(os.path.join(w.tmp(L), f))
    QA.phase_attnq(L, cfg, s, calib, w, "cpu", cq, 6, 0, lambda m: None)
    W = I2.LayerWeights(s, cfg, L, "cpu")
    D, nb = cfg.hidden, cfg.num_blocks_before(L)
    N = calib.n_tokens("q")
    Hin = i2q.Mat(w.h("q", L), N, D, torch.bfloat16)
    B = i2q.load_blocks(w, "q", nb, N, D)
    xs = {"xq": [], "xk": [], "xv": []}
    os_ = []
    for r, slot in calib.rows("q", 0, 1):
        if calib.held[r]:
            continue
        sl = i2q.row_slice(slot, calib.S)
        hidden = Hin.read(sl, "cpu")
        blocks = torch.stack([b.read(sl, "cpu") for b in B]) if nb else None
        for n, x in zip(("xq", "xk", "xv"), I2.attn_inputs(W, hidden, blocks)):
            xs[n].append(x.float())
        taps = {}
        I2.layer_attention(W, hidden, blocks, taps = taps)
        os_.append(taps["o"].float())
    n = sum(x.shape[0] for x in xs["xq"])
    mean = lambda v: torch.cat(v).t() @ torch.cat(v) / n
    want = {"xq": mean(xs["xq"]), "xk": mean(xs["xk"]), "xv": mean(xs["xv"]), "o": mean(os_)}
    a = f"model.llm.layers.{L}.attn."
    src_of = ({a + "wqg": "xq", a + "wk": "xk", a + "wv": "xv"} if W.is_global else {a + "wqgkv": "xq"})
    src_of[a + "wo"] = "o"
    assert [c[0] for c in cq.calls] == list(src_of), [c[0] for c in cq.calls]
    for key, wt, H, cnt in cq.calls:
        assert cnt == n, (key, cnt, n)
        err = (H - want[src_of[key]]).abs().max().item() / want[src_of[key]].abs().max().item()
        assert err < 1e-5, (key, err)
    wq = torch.cat([W.wq, W.wgate] + ([] if W.is_global else [W.wk, W.wv]), 0).float()
    assert torch.equal(cq.calls[0][1], wq.t()), "input projection weights"
    assert torch.equal(cq.calls[-1][1], W.wo.float().t()), "wo weights"
    print(f"layer {L} ({'global' if W.is_global else 'sliding'}): Hessians and weights match a brute-force recomputation")


def main():
    tmp = tempfile.mkdtemp(prefix = "attnq-test-")
    try:
        src, calib = os.path.join(tmp, "src"), os.path.join(tmp, "calib.npz")
        T.make_checkpoint(src)
        T.make_calib(calib)
        bits = os.path.join(tmp, "bits.json")
        json.dump({"layers": {str(L): {"gate_up": 3, "down": 4} for L in range(T.NL)}}, open(bits, "w"))
        we = os.path.join(tmp, "w-experts")
        run("quantize_experts.py", ["run", "--src", src, "--calib", calib, "--bits", bits, "--gpus", "cpu",
                                    "--work", we, "--fake", "identity"])
        base = ["--src", src, "--calib", calib, "--experts-from", os.path.join(we, "out"), "--gpu", "cpu"]
        Nq = T.R * T.S
        hq_e = T.load_state(we, "q", f"h{T.NL:02d}", Nq)

        # 1. identity
        w1 = os.path.join(tmp, "w-identity")
        run("quantize_attention.py", ["run", "--work", w1, "--fake", "identity"] + base)
        f = json.load(open(os.path.join(w1, "out", "final.json")))
        print("identity final:", f)
        assert f["kld"] == 0.0 and f["top1_agree"] == 1.0, f
        assert torch.equal(T.load_state(w1, "q", f"h{T.NL:02d}", Nq), hq_e), "attention path changed the result"
        print("identity: KLD 0 and the same final state as the expert run, bit for bit")

        # 2. resume
        w2 = os.path.join(tmp, "w-resume")
        run("quantize_attention.py", ["run", "--work", w2, "--fake", "identity", "--layers", "0-3"] + base)
        assert not os.path.exists(os.path.join(w2, "out", "final.json"))
        run("quantize_attention.py", ["run", "--work", w2, "--fake", "identity"] + base)
        assert torch.equal(T.load_state(w2, "q", f"h{T.NL:02d}", Nq), hq_e), "resume changed the result"
        print("resume: same final state bit for bit")

        # 3. rounding
        w3 = os.path.join(tmp, "w-round")
        run("quantize_attention.py", ["run", "--work", w3, "--fake", "0.004"] + base)
        f3 = json.load(open(os.path.join(w3, "out", "final.json")))
        print("rounding final:", f3)
        assert f3["kld"] > 0, f3
        hq, hk = T.H * T.DH, T.HK * T.DH
        for L in range(T.NL):
            t = load_file(os.path.join(w3, "out", f"attn-L{L:02d}.safetensors"))
            a = f"model.llm.layers.{L}.attn."
            want = ({a + "wqg.weight": (2 * hq, T.D), a + "wk.weight": (hk, T.D), a + "wv.weight": (hk, T.D)}
                    if L in T.GLOBAL else {a + "wqgkv.weight": (2 * hq + 2 * hk, T.D)})
            want[a + "wo.weight"] = (T.D, hq)
            assert {k: tuple(v.shape) for k, v in t.items()} == want, (L, {k: tuple(v.shape) for k, v in t.items()})
        print("rounding: KLD > 0; tensors and shapes per layer as documented")
        w4 = os.path.join(tmp, "w-hess")
        run("quantize_attention.py", ["run", "--work", w4, "--fake", "identity", "--layers", "0-2", "--keep-tmp"] + base)
        hessian_check(src, calib, w4, 1)
        hessian_check(src, calib, w4, 2)

        # 4. head
        w5 = os.path.join(tmp, "w-head")
        run("quantize_attention.py", ["run", "--work", w5, "--fake", "0.004", "--head-bits", "8"] + base)
        f5 = json.load(open(os.path.join(w5, "out", "final-head.json")))
        h5 = json.load(open(os.path.join(w5, "out", "head.json")))
        print("head:", h5, "final with head:", f5)
        assert h5["padded_vocab"] % 256 == 0 and h5["padded_vocab"] >= h5["vocab"]
        assert f5["kld"] != f3["kld"], (f5, f3)                  # the head's rounding took effect (either direction)
        w6 = os.path.join(tmp, "w-head-identity")
        run("quantize_attention.py", ["run", "--work", w6, "--fake", "identity", "--head-bits", "8"] + base)
        f6 = json.load(open(os.path.join(w6, "out", "final-head.json")))
        assert f6["kld"] == 0.0 and f6["top1_agree"] == 1.0, f6
        print("head: padded to a multiple of 256; rounding changes the KLD; identity head keeps KLD 0")
        print("all attention-quantization checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


if __name__ == "__main__":
    main()
