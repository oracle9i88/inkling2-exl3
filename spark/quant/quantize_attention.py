#!/usr/bin/env python3
"""
v3: quantize the attention projections (and optionally the output head) of Inkling-2-Small to EXL3 on top of an
earlier run's routed experts, layer by layer, and measure the held-out KL divergence against the original model.

Why: on two DGX Sparks a decode step reads ~15.4 GB of weights per token and 9.43 GB of it is the bf16 attention.
At 6 bits that part shrinks to ~3.5 GB.

Per layer, with the student track "q" (all rows; quantized so far) and the reference track "r" (held-out rows, the
original model throughout):
  attnq   student: Hessians of the projection inputs over the training rows, then quantize
            sliding layers: wqgkv = [wq; gate_attn; wk; wv]    (one input, one tensor)
            global layers:  wqg = [wq; gate_attn], wk, wv      (three inputs)
          then the Hessian of o (the gated input of wo) with those projections quantized, quantize wo, and run the
          attention half of every row with all of them quantized -> tmp/ inputs of the MoE (as i2q.phase_attn)
  attnr   reference: i2q.phase_attn on "r"
  moe     both tracks (i2q.phase_moe); the student's experts are the earlier run's, linked in, not requantized
final: KL(reference || student) over the held-out rows (out/final.json); with --head-bits the head is quantized
too, from the student's final hidden states, and out/final-head.json measures the whole thing.

Kept as stored (bf16): wr (16 rows per head), attn_sink_proj (1 per head), the global layers' k/v conv weight
projections, convolutions, norms, router. Their per-rank slices are not whole 128-row blocks. The quantized tensors
have whole 128-row blocks per head (EXL3's Hadamard transforms work in 128-blocks on both sides), so a tensor-parallel
rank takes its heads' row blocks of the input projections and its heads' column blocks of wo.

Tensors written to out/attn-LNN.safetensors as model.llm.layers.N.attn.{wqgkv|wqg|wk|wv|wo}.{trellis,suh,svh,...}
in exllamav3's layout; out/head.safetensors holds model.llm.unembed_q.* with the vocabulary padded to a multiple of 256
rows (zero rows; the logits beyond the real vocabulary are cut as before).

  python quantize_attention.py run --src SRC --calib calib.npz --experts-from work-v2/out --work work-v3 --bits 6 \
      [--head-bits 8] [--gpu 0] [--layers 0-1]
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEED_BASE = 900_000          # attention seeds stay apart from the experts' (i2q._seed)


def seed_of(seed: int, L: int, j: int) -> int:
    return (seed * 1_000_003 + L * 10_007 + SEED_BASE + j) % (2 ** 31 - 1)


def make_quantizer(a, root):
    from quantize_experts import Exl3Quantizer, FakeQuantizer
    if a.fake is None:
        return Exl3Quantizer(os.path.join(root, "logs", "debug"))
    if a.fake.startswith("k"):
        return FakeQuantizer(float(a.fake[1:]), k_aware = True)
    return FakeQuantizer(0.0 if a.fake == "identity" else float(a.fake))


def dense_of(out: dict, fi: int, fo: int, device):
    """(out, in) weight of a quantizer result: EXL3 tensors reconstructed (fp16), or the fake quantizer's bf16 weight"""
    import torch
    if "weight" in out:
        return out["weight"].to(device)
    from exllamav3.modules.quant.exl3 import LinearEXL3
    t = lambda s: out[s].to(device) if s in out else None
    lin = LinearEXL3(None, fi, fo, suh = t("suh"), svh = t("svh"), trellis = t("trellis"), mcg = t("mcg"),
                     mul1 = t("mul1"), out_dtype = torch.half, key = "attn")
    return lin.get_weight_tensor().t().contiguous()


def projector(w, adt):
    """x -> x @ w^T. EXL3 weights (fp16) run in fp16 like the kernels; the fake quantizer's bf16 weights as the source"""
    import torch
    import torch.nn.functional as F
    if w.dtype == torch.half:
        return lambda x: F.linear(x.half(), w).to(adt)
    return lambda x: F.linear(x, w.to(adt))


def quantize_one(quantizer, Wt, H, n, key, bits, seed, device):
    """Wt (out, in); H the summed Hessian of n inputs -> (proxy error, tensors dict, dense (out, in))"""
    from i2q import h_data
    (perr, out), = quantizer.quantize([Wt.float().t().contiguous()], [h_data(H / n, key, n, device)], bits, [seed], device)
    return float(perr), out, dense_of(out, Wt.shape[1], Wt.shape[0], device)


# ----------------------------------------------------------------------------------------------------------------------
# Phase attnq: the student's attention, quantized

def phase_attnq(L, cfg, src, calib, work, device, quantizer, bits, seed, log):
    import torch
    from safetensors.torch import save_file
    from i2q import I2, Mat, load_blocks, row_slice, tmp_mats, no_tf32
    adt = torch.bfloat16
    W = I2.LayerWeights(src, cfg, L, device, adt)
    D, k = cfg.hidden, cfg.topk
    hq, hk = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    nb = cfg.num_blocks_before(L)
    N = calib.n_tokens("q")
    Hin = Mat(work.h("q", L), N, D, torch.bfloat16)
    B = load_blocks(work, "q", nb, N, D)
    rows = calib.rows("q", 0, 1)
    train = [(r, s) for r, s in rows if not calib.held[r]]
    t0 = time.time()

    def row_in(slot):
        sl = row_slice(slot, calib.S)
        return sl, Hin.read(sl, device), (torch.stack([b.read(sl, device) for b in B]) if nb else None)

    with torch.inference_mode():
        # 1. Hessians of the projection inputs (xq = xk = xv in sliding layers)
        names = ("xq", "xk", "xv") if W.is_global else ("xq",)
        Hs = {n: torch.zeros((D, D), dtype = torch.float32, device = device) for n in names}
        n_tok = 0
        with no_tf32():
            for r, slot in train:
                _, hidden, blocks = row_in(slot)
                xs = dict(zip(("xq", "xk", "xv"), I2.attn_inputs(W, hidden, blocks, adt)))
                for n in names:
                    xf = xs[n].float()
                    Hs[n].addmm_(xf.t(), xf)
                n_tok += hidden.shape[0]

        # 2. the input projections
        a = f"model.llm.layers.{L}.attn."
        groups = ([("wqg", [W.wq, W.wgate], "xq"), ("wk", [W.wk], "xk"), ("wv", [W.wv], "xv")] if W.is_global else
                  [("wqgkv", [W.wq, W.wgate, W.wk, W.wv], "xq")])
        tensors, stats, dense = {}, {}, {}
        for j, (name, ws, hx) in enumerate(groups):
            Wc = torch.cat(ws, 0)
            perr, out, dense[name] = quantize_one(quantizer, Wc, Hs[hx], n_tok, a + name, bits, seed_of(seed, L, j), device)
            tensors.update({f"{a}{name}.{s}": t.detach().to("cpu").contiguous() for s, t in out.items()})
            stats[name] = {"proxy_rel_err": perr, "rows": int(Wc.shape[0])}
        del Hs
        if W.is_global:
            wq_, wg_, wk_, wv_ = dense["wqg"][:hq], dense["wqg"][hq:], dense["wk"], dense["wv"]
        else:
            d = dense["wqgkv"]
            wq_, wg_, wk_, wv_ = d[:hq], d[hq:2 * hq], d[2 * hq:2 * hq + hk], d[2 * hq + hk:]
        assert wq_.shape[0] == hq and wg_.shape[0] == hq and wk_.shape[0] == hk and wv_.shape[0] == hk
        qlin = {"q": projector(wq_, adt), "gate": projector(wg_, adt), "k": projector(wk_, adt), "v": projector(wv_, adt)}

        # 3. wo, from the gated attention outputs with the input projections quantized
        Ho = torch.zeros((hq, hq), dtype = torch.float32, device = device)
        with no_tf32():
            for r, slot in train:
                _, hidden, blocks = row_in(slot)
                taps = {}
                I2.layer_attention(W, hidden, blocks, adt, taps = taps, qlin = qlin)
                of = taps["o"].float()
                Ho.addmm_(of.t(), of)
        perr, out, wo_ = quantize_one(quantizer, W.wo, Ho, n_tok, a + "wo", bits, seed_of(seed, L, 9), device)
        tensors.update({f"{a}wo.{s}": t.detach().to("cpu").contiguous() for s, t in out.items()})
        stats["wo"] = {"proxy_rel_err": perr, "rows": int(W.wo.shape[0])}
        qlin["o"] = projector(wo_, adt)
        del Ho

        # 4. the attention half of every student row, everything quantized (as i2q.phase_attn writes it)
        if not os.path.exists(work.tmp(L, "q_x.bin.ready")):
            tmp_mats(work, L, "q", N, D, k, create = True)
            open(work.tmp(L, "q_x.bin.ready"), "w").close()
        T = tmp_mats(work, L, "q", N, D, k, create = False)
        for r, slot in rows:
            sl, hidden, blocks = row_in(slot)
            att = I2.layer_attention(W, hidden, blocks, adt, qlin = qlin)
            resid, x = I2.layer_mlp_input(W, hidden, blocks, att, adt)
            sel, rw = I2.route(W, x)
            T["x"].write(sl, x)
            T["resid"].write(sl, resid)
            T["sel"].write(sl, sel.to(torch.int16))
            T["rw"].write(sl, rw.float())
        for m in T.values():
            m.flush()

    save_file(tensors, work.tmp(L, "attn-q.safetensors"))
    json.dump({"layer": L, "global": bool(W.is_global), "bits": bits, "train_tokens": n_tok, "tensors": stats,
               "seconds": time.time() - t0}, open(work.tmp(L, "attn-q.json"), "w"), indent = 1)
    log(f"L{L:02d} attnq: {time.time() - t0:.1f}s, {n_tok} training tokens, proxy " +
        ", ".join(f"{n}={s['proxy_rel_err']:.5f}" for n, s in stats.items()))


# ----------------------------------------------------------------------------------------------------------------------
# The head, and KL divergence with a given student head

def student_logits(Hd, w, unpadded):
    """the student's logits with head weight w (out, in): fp16 EXL3 reconstruction or bf16"""
    import torch
    import torch.nn.functional as F
    from i2q import I2
    if w is None:
        return lambda h: I2.logits_from_hidden(Hd, h)
    if w.dtype == torch.half:
        return lambda h: I2.softcap(F.linear(h.half(), w).float(), Hd.cfg.final_cap)[..., :unpadded]
    return lambda h: I2.softcap(F.linear(h, w).float(), Hd.cfg.final_cap)[..., :unpadded]


def final_kl(cfg, src, calib, work, device, head_w, out_name, log, chunk = 512):
    """KL(reference || student) over the held-out rows, the student's head given (None: the source head)"""
    import torch
    from i2q import I2, Mat, load_blocks, row_slice
    adt = torch.bfloat16
    Hd = I2.HeadWeights(src, cfg, device)
    D = cfg.hidden
    nb = cfg.num_blocks_before(cfg.layers)
    acc = {"kld": 0.0, "nll_q": 0.0, "nll_r": 0.0, "top1": 0, "tokens": 0}
    Nq, Nr = calib.n_tokens("q"), calib.n_tokens("r")
    Hq, Hr = Mat(work.h("q", cfg.layers), Nq, D, torch.bfloat16), Mat(work.h("r", cfg.layers), Nr, D, torch.bfloat16)
    Bq, Br = load_blocks(work, "q", nb, Nq, D), load_blocks(work, "r", nb, Nr, D)
    lq_of = student_logits(Hd, head_w, cfg.unpadded_vocab)
    with torch.inference_mode():
        for row, j in calib.rows("r", 0, 1):
            sq, sr = row_slice(row, calib.S), row_slice(j, calib.S)
            hq = I2.final_hidden(Hd, Hq.read(sq, device), torch.stack([b.read(sq, device) for b in Bq]), adt)
            hr = I2.final_hidden(Hd, Hr.read(sr, device), torch.stack([b.read(sr, device) for b in Br]), adt)
            ids = torch.from_numpy(calib.ids[row]).to(device)
            for c0 in range(0, calib.S - 1, chunk):
                c1 = min(calib.S - 1, c0 + chunk)
                lq = torch.log_softmax(lq_of(hq[c0:c1]), -1)
                lr = torch.log_softmax(I2.logits_from_hidden(Hd, hr[c0:c1]), -1)
                tgt = ids[c0 + 1:c1 + 1]
                acc["kld"] += (lr.exp() * (lr - lq)).sum().item()
                acc["nll_q"] -= lq.gather(1, tgt[:, None]).sum().item()
                acc["nll_r"] -= lr.gather(1, tgt[:, None]).sum().item()
                acc["top1"] += (lq.argmax(-1) == lr.argmax(-1)).sum().item()
                acc["tokens"] += c1 - c0
    import math
    n = max(acc["tokens"], 1)
    res = {"tokens": acc["tokens"], "kld": acc["kld"] / n, "ppl_quant": math.exp(acc["nll_q"] / n),
           "ppl_ref": math.exp(acc["nll_r"] / n), "top1_agree": acc["top1"] / n}
    json.dump(res, open(work.out(out_name), "w"), indent = 1)
    log(f"{out_name}: KLD {res['kld']:.5f}, top-1 {res['top1_agree']:.4f}, PPL {res['ppl_quant']:.3f} vs "
        f"{res['ppl_ref']:.3f} over {res['tokens']} tokens")
    return res


def phase_head(cfg, src, calib, work, device, quantizer, bits, seed, log, pad_to = 256):
    import torch
    from safetensors.torch import save_file
    from i2q import I2, Mat, load_blocks, row_slice, no_tf32
    adt = torch.bfloat16
    t0 = time.time()
    Hd = I2.HeadWeights(src, cfg, device)
    D = cfg.hidden
    nb = cfg.num_blocks_before(cfg.layers)
    N = calib.n_tokens("q")
    Hq = Mat(work.h("q", cfg.layers), N, D, torch.bfloat16)
    Bq = load_blocks(work, "q", nb, N, D)
    Hh = torch.zeros((D, D), dtype = torch.float32, device = device)
    n_tok = 0
    with torch.inference_mode(), no_tf32():
        for r, slot in calib.rows("q", 0, 1):
            if calib.held[r]:
                continue
            sl = row_slice(slot, calib.S)
            h = I2.final_hidden(Hd, Hq.read(sl, device), torch.stack([b.read(sl, device) for b in Bq]), adt).float()
            Hh.addmm_(h.t(), h)
            n_tok += h.shape[0]
        Wh = Hd.lm_head
        V = Wh.shape[0]
        Vp = -(-V // pad_to) * pad_to
        if Vp != V:
            Wh = torch.cat([Wh, Wh.new_zeros((Vp - V, D))], 0)
        perr, out, dense = quantize_one(quantizer, Wh, Hh, n_tok, "model.llm.unembed_q", bits, seed_of(seed, 99, 0), device)
    tensors = {f"model.llm.unembed_q.{s}": t.detach().to("cpu").contiguous() for s, t in out.items()}
    save_file(tensors, work.out("head.safetensors"))
    json.dump({"bits": bits, "vocab": V, "padded_vocab": Vp, "train_tokens": n_tok, "proxy_rel_err": perr,
               "seconds": time.time() - t0}, open(work.out("head.json"), "w"), indent = 1)
    log(f"head: {bits} bits, vocab {V} padded to {Vp}, proxy {perr:.5f}, {time.time() - t0:.1f}s")
    final_kl(cfg, src, calib, work, device, dense[:V], "final-head.json", log)


# ----------------------------------------------------------------------------------------------------------------------
# One phase in this process

def run_phase(a):
    import torch
    from i2q import I2, Calib, Work, phase_embed, phase_attn, phase_moe
    device = torch.device("cpu" if a.device == "cpu" else "cuda:0")      # one visible GPU per process
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    work = Work(a.work)
    src = I2.Inkling2Source(a.src)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush = True)
    q = make_quantizer(a, a.work)
    if a.phase == "embed":
        phase_embed(cfg, src, calib, work, device, 0, 1, log = log)
    elif a.phase == "attnq":
        phase_attnq(a.layer, cfg, src, calib, work, device, q, a.bits, a.seed, log)
    elif a.phase == "attnr":
        phase_attn(a.layer, cfg, src, calib, work, device, 0, 1, log = log, tracks = ("r",))
    elif a.phase == "moe":
        phase_moe(a.layer, cfg, src, calib, work, device, 0, 1, q, log = log)
    elif a.phase == "final":
        final_kl(cfg, src, calib, work, device, None, "final.json", log)
    elif a.phase == "head":
        phase_head(cfg, src, calib, work, device, q, a.head_bits, a.seed, log)
    else:
        raise SystemExit(f"unknown phase {a.phase}")


# ----------------------------------------------------------------------------------------------------------------------
# Orchestration

def common_args(a):
    out = ["--src", a.src, "--calib", a.calib, "--work", a.work, "--experts-from", a.experts_from, "--bits", str(a.bits),
           "--seed", str(a.seed)]
    if a.head_bits:
        out += ["--head-bits", str(a.head_bits)]
    if a.fake is not None:
        out += ["--fake", a.fake]
    return out


def spawn(a, phase, layer = None):
    cmd = [sys.executable, os.path.abspath(__file__), "phase", "--phase", phase,
           "--device", "cpu" if a.gpu == "cpu" else "cuda"] + common_args(a)
    if layer is not None:
        cmd += ["--layer", str(layer)]
    env = dict(os.environ)
    if a.gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    tag = f"L{layer:02d}-" if layer is not None else ""
    logf = os.path.join(a.work, "logs", f"{tag}{phase}.log")
    p = subprocess.run(cmd, env = env, stdout = open(logf, "w"), stderr = subprocess.STDOUT)
    if p.returncode != 0:
        print(f"--- {logf} (tail)", flush = True)
        print("".join(open(logf).readlines()[-30:]), flush = True)
        raise SystemExit(f"phase {phase} failed (layer {layer})")


def run(a):
    from i2q import I2, Calib, Work
    from quantize_experts import link_or_copy, file_hash
    work = Work(a.work)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    meta = {"src": os.path.abspath(a.src), "calib": file_hash(a.calib), "experts_from": os.path.abspath(a.experts_from),
            "bits": a.bits, "head_bits": a.head_bits, "seed": a.seed, "fake": a.fake, "rows": calib.R,
            "row_len": calib.S, "heldout_rows": int(calib.held.sum())}
    mp = os.path.join(a.work, "run.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        diff = {k: (old.get(k), v) for k, v in meta.items() if k != "src" and old.get(k) != v}
        if diff:
            raise SystemExit(f"work dir belongs to a different run: {diff}")
    json.dump(meta, open(mp, "w"), indent = 1)
    say = lambda m: print(time.strftime("%m-%d %H:%M:%S"), m, flush = True)
    say(f"{calib.R} rows x {calib.S} tokens ({int(calib.held.sum())} held out); attention {a.bits} bits, "
        f"head {a.head_bits or 'as stored'}; experts from {a.experts_from}")
    if not os.path.exists(work.out("embed.done")):
        spawn(a, "embed")
        open(work.out("embed.done"), "w").close()
    layers = range(cfg.layers) if a.layers is None else range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1)
    t_start, done_here = time.time(), 0
    for L in layers:
        if os.path.exists(work.out(f"L{L:02d}.done.json")):
            continue
        t0 = time.time()
        ex = sorted(glob.glob(os.path.join(a.experts_from, f"experts-L{L:02d}-*.safetensors")))
        if not ex:
            raise SystemExit(f"no experts for layer {L} in {a.experts_from}")
        if L % cfg.block_layers == 0:                       # this layer's input becomes finished block L // 6
            for tr in ("q", "r"):
                if calib.n_tokens(tr):
                    link_or_copy(work.h(tr, L), work.b(tr, L // cfg.block_layers))
        for phase in ("attnq", "attnr"):
            if not os.path.exists(work.tmp(L, f"{phase}.done")):
                spawn(a, phase, L)
                open(work.tmp(L, f"{phase}.done"), "w").close()
        for p in ex:                                         # i2q.phase_moe reads tmp/LNN/experts-*.safetensors
            dst = work.tmp(L, os.path.basename(p).replace(f"experts-L{L:02d}-", "experts-"))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(p), dst)
        if not os.path.exists(work.tmp(L, "moe.done")):
            spawn(a, "moe", L)
            open(work.tmp(L, "moe.done"), "w").close()
        m = json.load(open(work.tmp(L, "attn-q.json")))
        mo = json.load(open(work.tmp(L, "moe-0.json")))
        m["moe_rel_mse"] = mo["num"] / max(mo["den"], 1e-30)
        os.replace(work.tmp(L, "attn-q.safetensors"), work.out(f"attn-L{L:02d}.safetensors"))
        m["seconds"] = time.time() - t0
        json.dump(m, open(work.out(f"L{L:02d}.done.json"), "w"), indent = 1)
        if not a.keep_tmp:
            shutil.rmtree(work.tmp(L))
            for tr in ("q", "r"):                           # block files are hard links and survive this
                for p in (work.h(tr, L), work.h(tr, L) + ".ready"):
                    if os.path.exists(p):
                        os.remove(p)
        done_here += 1
        left = sum(1 for l in layers if not os.path.exists(work.out(f"L{l:02d}.done.json")))
        eta = (time.time() - t_start) / done_here * left
        say(f"L{L:02d} done in {m['seconds']:.0f}s: proxy " +
            ", ".join(f"{n}={s['proxy_rel_err']:.5f}" for n, s in m["tensors"].items()) +
            f", moe rel mse {m['moe_rel_mse']:.5f}; {left} layers left, ETA {eta / 3600:.2f} h")
    if all(os.path.exists(work.out(f"L{l:02d}.done.json")) for l in range(cfg.layers)) and calib.n_tokens("r"):
        if not os.path.exists(work.out("final.json")):
            spawn(a, "final")
        say(f"final: {json.load(open(work.out('final.json')))}")
        if a.head_bits and not os.path.exists(work.out("final-head.json")):
            spawn(a, "head")
        if a.head_bits:
            say(f"final with the head at {a.head_bits} bits: {json.load(open(work.out('final-head.json')))}")


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices = ["run", "phase"])
    ap.add_argument("--src", required = True)
    ap.add_argument("--calib", required = True)
    ap.add_argument("--work", required = True)
    ap.add_argument("--experts-from", required = True, help = "out/ of the run whose routed experts are kept")
    ap.add_argument("--bits", type = float, default = 6.0, help = "attention bits")
    ap.add_argument("--head-bits", type = float, default = None, help = "also quantize the head (e.g. 8)")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--fake", help = "tests: 'identity', a rounding step, or k<step> instead of EXL3")
    ap.add_argument("--gpu", default = "0", help = "GPU index, or cpu")
    ap.add_argument("--layers", help = "e.g. 0-47 (default all)")
    ap.add_argument("--keep-tmp", action = "store_true")
    ap.add_argument("--phase")
    ap.add_argument("--layer", type = int, default = 0)
    ap.add_argument("--device", default = "cuda")
    a = ap.parse_args()
    a.bits = int(a.bits) if float(a.bits).is_integer() else a.bits
    if a.head_bits is not None and float(a.head_bits).is_integer():
        a.head_bits = int(a.head_bits)
    if a.cmd == "phase":
        run_phase(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
