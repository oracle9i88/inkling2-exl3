#!/usr/bin/env python3
"""
Requantize the routed experts of thinkingmachines/Inkling-2-Small to EXL3 (mul1 codebook, half-bit steps) from the
original NVFP4 / MXFP8 checkpoint, layer by layer, for a model that fits two DGX Sparks. Everything outside the routed
experts stays as it is (bf16), and the output tensors have exactly the names and layout of 0xSero's EXL3 quants, so the
engine reads them unchanged.

  python quantize_experts.py run --src /models/Inkling-2-Small --calib calib.npz --bits bits-2.55.json \
      --work work/q255 --gpus 0,1

Each phase of each layer runs in its own process (one per GPU; see i2q.py for what the phases do). The run resumes:
finished layers are skipped, and a layer restarts from its first unfinished phase.

  bench   time the quantization of a few experts of one layer (small calibration), before a full run
  run     the full conversion (embed, then per layer attn -> experts -> moe, then final KL divergence)
  phase   (internal) one phase of one layer on one device
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------------------------------------------------
# Quantizers

def _k(K):
    K = float(K)
    return int(K) if K.is_integer() else K


class Exl3Quantizer:
    """exllamav3's EXL3 quantizer, batched over same-shape tensors with per-tensor Hessians"""

    def __init__(self, debug_dir: str):
        from exllamav3.modules.quant.exl3_lib import quantize as Q
        self.Q, self.debug_dir = Q, debug_dir

    def quantize(self, weights, H_datas, K, seeds, device):
        import torch
        dev = torch.device(device)
        qas = [{"K": _k(K), "seed": s, "devices": [dev.index if dev.index is not None else 0], "device_ratios": None,
                "apply_out_scales": None, "mul1": True, "debug_dir": self.debug_dir} for s in seeds]
        return self.Q.quantize_exl3_batch(weights, H_datas, qas)

    def provider(self, tensors, cfg, L, device):
        from i2q import I2, DictSource
        return I2.Exl3Experts(DictSource(tensors), cfg, L, device)


class FakeQuantizer:
    """CPU tests: identity (step 0) or rounding to a uniform grid of `step`; with k_aware the step halves per bit above 3
    (so the error falls like EXL3's with the rate); stores the dense weight"""

    def __init__(self, step: float = 0.0, k_aware: bool = False):
        self.step, self.k_aware = step, k_aware

    def quantize(self, weights, H_datas, K, seeds, device):
        import torch
        step = self.step * 2.0 ** (-(float(K) - 3.0)) if self.k_aware else self.step
        out = []
        for w in weights:                                   # (in, out) fp32
            wq = w if step == 0 else torch.round(w / step) * step
            err = 0.0 if step == 0 else float((wq - w).square().sum() / w.square().sum().clamp_min(1e-30))
            out.append((err, {"weight": wq.t().contiguous().to(torch.bfloat16)}))
        return out

    def provider(self, tensors, cfg, L, device):
        import torch
        from i2q import I2, expert_key
        E, I, D = cfg.experts, cfg.inter, cfg.hidden
        w13 = torch.empty((E, 2 * I, D), dtype = torch.bfloat16)
        w2 = torch.empty((E, D, I), dtype = torch.bfloat16)
        for e in range(E):
            w13[e, :I] = tensors[expert_key(L, e, "gate_proj") + ".weight"]
            w13[e, I:] = tensors[expert_key(L, e, "up_proj") + ".weight"]
            w2[e] = tensors[expert_key(L, e, "down_proj") + ".weight"]
        return I2.DenseExperts(w13.to(device), w2.to(device))


def make_quantizer(a, work_root):
    if a.fake is None:
        return Exl3Quantizer(os.path.join(work_root, "logs", "debug"))
    if a.fake.startswith("k"):
        return FakeQuantizer(float(a.fake[1:]), k_aware = True)
    return FakeQuantizer(0.0 if a.fake == "identity" else float(a.fake))


# ----------------------------------------------------------------------------------------------------------------------
# One phase in this process

def run_phase(a):
    import torch
    from i2q import I2, Calib, Work, phase_embed, phase_attn, phase_experts, phase_moe, phase_final
    device = torch.device("cpu" if a.device == "cpu" else "cuda:0")      # one visible GPU per process
    if device.type == "cuda":
        torch.cuda.set_device(0)
    work = Work(a.work)
    src = I2.Inkling2Source(a.src)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush = True)
    if a.phase == "embed":
        phase_embed(cfg, src, calib, work, device, a.part, a.parts, log = log)
    elif a.phase == "attn":
        phase_attn(a.layer, cfg, src, calib, work, device, a.part, a.parts, log = log)
    elif a.phase == "experts":
        bits = json.load(open(a.bits))["layers"][str(a.layer)]
        e0, e1 = (int(x) for x in a.experts.split(":"))
        phase_experts(a.layer, e0, e1, cfg, src, calib, work, device, make_quantizer(a, a.work), bits,
                      n0 = a.n0, group = a.group, seed = a.seed, log = log, per_expert = a.per_expert)
    elif a.phase == "moe":
        phase_moe(a.layer, cfg, src, calib, work, device, a.part, a.parts, make_quantizer(a, a.work), log = log)
    elif a.phase == "final":
        phase_final(cfg, src, calib, work, device, a.part, a.parts, log = log)
    else:
        raise SystemExit(f"unknown phase {a.phase}")


# ----------------------------------------------------------------------------------------------------------------------
# Orchestration

def common_args(a):
    out = ["--src", a.src, "--calib", a.calib, "--work", a.work, "--n0", str(a.n0), "--group", str(a.group),
           "--seed", str(a.seed)]
    if a.bits:
        out += ["--bits", a.bits]
    if a.fake is not None:
        out += ["--fake", a.fake]
    if a.per_expert:
        out += ["--per-expert", a.per_expert]
    return out


def spawn(a, devs, phase, layer = None, tasks = None):
    """run `phase` on every device in parallel; tasks: per device extra args (default: --part i --parts n)"""
    procs = []
    n = len(devs)
    for i, dev in enumerate(devs):
        extra = tasks[i] if tasks is not None else ["--part", str(i), "--parts", str(n)]
        if extra is None:
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "phase", "--phase", phase, "--device", "cpu" if dev == "cpu" else "cuda"]
        cmd += common_args(a) + extra
        if layer is not None:
            cmd += ["--layer", str(layer)]
        env = dict(os.environ)
        if dev != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = dev
        tag = f"L{layer:02d}-" if layer is not None else ""
        logf = os.path.join(a.work, "logs", f"{tag}{phase}-{i}.log")
        procs.append((subprocess.Popen(cmd, env = env, stdout = open(logf, "w"), stderr = subprocess.STDOUT), logf))
    bad = []
    for p, logf in procs:
        if p.wait() != 0:
            bad.append(logf)
    if bad:
        for logf in bad:
            print(f"--- {logf} (tail)", flush = True)
            print("".join(open(logf).readlines()[-25:]), flush = True)
        raise SystemExit(f"phase {phase} failed (layer {layer})")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def layer_metrics(work, L, ranges, parts):
    from i2q import PROJ
    ex = {}
    secs = 0.0
    for e0, e1 in ranges:
        j = json.load(open(work.tmp(L, f"experts-{e0:03d}-{e1:03d}.json")))
        ex.update(j["experts"])
        secs = max(secs, j["seconds"])
        bits = j["bits"]
    mo = [json.load(open(work.tmp(L, f"moe-{p}.json"))) for p in range(parts)]
    num, den = sum(m["num"] for m in mo), sum(m["den"] for m in mo)
    toks = [v["tokens"] for v in ex.values()]
    perr = {p: sum(v.get(p, 0.0) for v in ex.values()) / max(len(ex), 1) for p in PROJ}
    pe = None
    if all("K_gate_up" in v for v in ex.values()):
        pe = {}
        for m in ("gate_up", "down"):
            ks = [v[f"K_{m}"] for v in ex.values()]
            pe[m] = {"mean": sum(ks) / len(ks), "hist": {f"{k:g}": ks.count(k) for k in sorted(set(ks))}}
    return {"layer": L, "bits": bits, "per_expert_bits": pe, "moe_rel_mse": num / den if den else None, "proxy_rel_err": perr,
            "tokens_per_expert": {"min": min(toks), "median": sorted(toks)[len(toks) // 2], "max": max(toks),
                                  "zero": sum(1 for t in toks if t == 0)},
            "experts_seconds": secs, "moe_seconds": max(m["seconds"] for m in mo)}


def wait_for_source(src, prefixes, say, poll = 60):
    """A layer-by-layer run may start before the source download ends: wait until every shard holding a tensor under
    one of `prefixes` exists in src (e.g. a directory of links to the shards of a download that have finished and been verified)"""
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    need = sorted({f for k, f in wm.items() if k.startswith(tuple(prefixes))})
    t0, told = time.time(), False
    while True:
        missing = [f for f in need if not os.path.exists(os.path.join(src, f))]
        if not missing:
            if told:
                say(f"  ... there after {(time.time() - t0) / 60:.0f} min")
            return
        if not told:
            say(f"waiting for {', '.join(missing)} ({', '.join(prefixes)})")
            told = True
        time.sleep(poll)


def run(a):
    from i2q import I2, Calib, Work, summarize_final
    devs = a.gpus.split(",")
    parts = len(devs)
    work = Work(a.work)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    meta = {"src": os.path.abspath(a.src), "calib": file_hash(a.calib), "bits": file_hash(a.bits), "per_expert": a.per_expert, "n0": a.n0,
            "seed": a.seed, "fake": a.fake, "rows": calib.R, "row_len": calib.S, "heldout_rows": int(calib.held.sum())}
    mp = os.path.join(a.work, "run.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        diff = {k: (old.get(k), v) for k, v in meta.items() if k != "src" and old.get(k) != v}
        if diff:
            raise SystemExit(f"work dir belongs to a different run: {diff}")
    json.dump(meta, open(mp, "w"), indent = 1)
    say = lambda m: print(time.strftime("%m-%d %H:%M:%S"), m, flush = True)
    say(f"{calib.R} rows x {calib.S} tokens ({int(calib.held.sum())} held out), devices {devs}")

    if not os.path.exists(work.out("embed.done")):
        wait_for_source(a.src, ("model.llm.embed", "model.llm.attn_res"), say)
        spawn(a, devs, "embed")
        open(work.out("embed.done"), "w").close()
    E = cfg.experts
    ranges = [(E * i // parts, E * (i + 1) // parts) for i in range(parts)]
    layers = range(cfg.layers) if a.layers is None else range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1)
    t_start = time.time()
    done_here = 0
    for L in layers:
        if os.path.exists(work.out(f"L{L:02d}.done.json")):
            continue
        wait_for_source(a.src, (f"model.llm.layers.{L}.",), say)
        t0 = time.time()
        if L % cfg.block_layers == 0:                       # this layer's input becomes finished block L // 6
            for tr in ("q", "r"):
                if calib.n_tokens(tr):
                    link_or_copy(work.h(tr, L), work.b(tr, L // cfg.block_layers))
        if not os.path.exists(work.tmp(L, "attn.done")):
            for f in os.listdir(work.tmp(L)):
                if f.startswith("Hs-"):
                    os.remove(os.path.join(work.tmp(L), f))
            spawn(a, devs, "attn", L)
            open(work.tmp(L, "attn.done"), "w").close()
        todo = [None if os.path.exists(work.tmp(L, f"experts-{e0:03d}-{e1:03d}.json")) else ["--experts", f"{e0}:{e1}"]
                for e0, e1 in ranges]
        if any(todo):
            spawn(a, devs, "experts", L, todo)
        if not os.path.exists(work.tmp(L, "moe.done")):
            spawn(a, devs, "moe", L)
            open(work.tmp(L, "moe.done"), "w").close()
        m = layer_metrics(work, L, ranges, parts)
        for e0, e1 in ranges:
            os.replace(work.tmp(L, f"experts-{e0:03d}-{e1:03d}.safetensors"), work.out(f"experts-L{L:02d}-{e0:03d}-{e1:03d}.safetensors"))
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
        pe = m.get("per_expert_bits")
        bits_txt = str(m["bits"]) if not pe else " ".join(f"{k} " + "/".join(f"{b}:{n}" for b, n in v["hist"].items())
                                                         for k, v in pe.items())
        say(f"L{L:02d} done in {m['seconds']:.0f}s: bits {bits_txt}, moe rel mse {m['moe_rel_mse']:.5f}, "
            f"proxy {', '.join(f'{k[0]}={v:.4f}' for k, v in m['proxy_rel_err'].items())}, "
            f"tokens/expert median {m['tokens_per_expert']['median']} (0: {m['tokens_per_expert']['zero']}); "
            f"{left} layers left, ETA {eta / 3600:.1f} h")
    if all(os.path.exists(work.out(f"L{l:02d}.done.json")) for l in range(cfg.layers)) and calib.n_tokens("r"):
        wait_for_source(a.src, ("model.llm.norm", "model.llm.unembed"), say)
        spawn(a, devs, "final")
        f = summarize_final(work, parts)
        say(f"final: KLD {f['kld']:.5f}, top-1 agreement {f['top1_agree']:.4f}, PPL {f['ppl_quant']:.3f} vs reference "
            f"{f['ppl_ref']:.3f} over {f['tokens']} held-out tokens")


def ablate(a):
    """After a run: where its KL divergence comes from (held-out rows only, from the run's saved block states).
      a<k>: the quantized experts of layers < k, the source experts from layer k on. KL(a<k>) = what layers < k cost.
      s<k>: the source experts of layers < k, the quantized experts from layer k on. KL(s<k>) = what layers >= k cost.
    Share of layers k1..k2-1: KL(a<k2>) - KL(a<k1>) (prefix order) and KL(s<k1>) - KL(s<k2>) (suffix order); their
    mean is the Shapley value of the group when errors interact only in pairs. Writes out/ablate.json. One process on
    one device; resumable per layer."""
    import torch
    from i2q import I2, Calib, Work, phase_attn, phase_moe, phase_final, summarize_final, ablate_init, ablate_track
    work = Work(a.work)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    src = I2.Inkling2Source(a.src)
    say = lambda m: print(time.strftime("%m-%d %H:%M:%S"), m, flush = True)
    if not os.path.exists(work.out("final.json")):
        raise SystemExit("ablate needs a finished run (out/final.json)")
    full = json.load(open(work.out("final.json")))
    if a.fake is None:                                      # rebuild the run's experts the way it stored them
        a.fake = json.load(open(os.path.join(a.work, "run.json"))).get("fake")
    starts = sorted({int(x) for x in a.starts.split(",")})
    kinds = [x for x in a.kinds.split(",") if x]
    bl, NL = cfg.block_layers, cfg.layers
    bad = [k for k in starts if not (k == NL or (0 <= k < NL and k % bl == 0))]
    if bad or not set(kinds) <= {"a", "s"}:
        raise SystemExit(f"starts must begin a block (multiples of {bl}) or be {NL}: {bad}; kinds a and/or s: {kinds}")
    tracks_all = [(kind, k) for kind in kinds for k in starts if not (kind == "s" and k == NL)]
    dev = a.gpus.split(",")[0]
    device = torch.device("cpu") if dev == "cpu" else torch.device("cuda:0")      # CUDA_VISIBLE_DEVICES picks the GPU
    if device.type == "cuda":
        torch.cuda.set_device(0)
    quantizer = make_quantizer(a, a.work) if "s" in kinds else None
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush = True)
    t_start = time.time()
    for kind, k in tracks_all:
        mark = work.out(f"ablate-{ablate_track(k, kind)}.init")
        if not os.path.exists(mark):
            ablate_init(cfg, calib, work, k, kind)
            open(mark, "w").close()
    for L in range(min(starts), NL):
        tracks = [ablate_track(k, kind) for kind, k in tracks_all if k <= L < NL]
        mark = work.out(f"ablate-{''.join(kinds)}-L{L:02d}.done")
        if not tracks or os.path.exists(mark):
            continue
        t0 = time.time()
        if L % bl == 0:                                     # this layer's input becomes finished block L // bl
            for t in tracks:
                link_or_copy(work.h(t, L), work.b(t, L // bl))
        if os.path.isdir(work.tmp(L)):                      # leftovers of an interrupted attempt at this layer
            shutil.rmtree(work.tmp(L))
        phase_attn(L, cfg, src, calib, work, device, 0, 1, log = log, tracks = tracks)
        phase_moe(L, cfg, src, calib, work, device, 0, 1, quantizer, log = log, tracks = tracks)
        shutil.rmtree(work.tmp(L))
        for t in tracks:                                    # block files are hard links and survive this
            for p in (work.h(t, L), work.h(t, L) + ".ready"):
                if os.path.exists(p):
                    os.remove(p)
        open(mark, "w").close()
        say(f"ablate L{L:02d}: {len(tracks)} tracks in {time.time() - t0:.0f}s")
    res = {}
    for kind, k in tracks_all:
        t = ablate_track(k, kind)
        if not os.path.exists(work.out(f"final-{t}.json")):
            phase_final(cfg, src, calib, work, device, 0, 1, log = log, student = t)
        res[t] = summarize_final(work, 1, student = t)
    F = full["kld"]
    pre = {0: 0.0, **{k: res[ablate_track(k, "a")]["kld"] for k in starts if "a" in kinds}, NL: F}
    suf = {0: F, **{k: res[ablate_track(k, "s")]["kld"] for k in starts if "s" in kinds and k < NL}, NL: 0.0}
    ks = sorted(set(pre) | set(suf))
    groups = []
    for k1, k2 in zip(ks, ks[1:]):
        g = {"layers": [k1, k2 - 1]}
        if "a" in kinds:
            g["kld_prefix"] = pre[k2] - pre[k1]
        if "s" in kinds:
            g["kld_suffix"] = suf[k1] - suf[k2]
        vals = [g[x] for x in ("kld_prefix", "kld_suffix") if x in g]
        g["kld"] = sum(vals) / len(vals)
        g["share"] = g["kld"] / F
        groups.append(g)
    out = {"full": full, "tracks": res, "groups": groups, "kinds": kinds, "seconds": time.time() - t_start,
           "note": "a<k>: layers < k quantized; s<k>: layers >= k quantized; group kld = mean of the prefix and suffix differences"}
    json.dump(out, open(work.out("ablate.json"), "w"), indent = 1)
    for g in groups:
        parts = ", ".join(f"{x[4:]} {g[x]:+.5f}" for x in ("kld_prefix", "kld_suffix") if x in g)
        say(f"layers {g['layers'][0]:2d}-{g['layers'][1]:2d}: KLD {g['kld']:+.5f} ({100 * g['share']:5.1f}% of {F:.5f}; {parts})")
    for t, want in (("a00", 0.0), (ablate_track(NL, "a"), F), ("s00", F)):
        if t in res:
            say(f"check {t}: KLD {res[t]['kld']:.6g} (expected {want:.6g})")


def bench(a):
    """embed + attn of one layer on a small calibration, then quantize a few experts at the layer's bits"""
    import torch
    from i2q import I2, Calib, Work, phase_embed, phase_attn, phase_experts
    device = torch.device("cpu" if a.device == "cpu" else "cuda:0")
    work = Work(a.work)
    src = I2.Inkling2Source(a.src)
    cfg = I2.Inkling2Cfg.from_dir(a.src)
    calib = Calib(a.calib)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush = True)
    assert a.layer == 0, "bench runs on layer 0 (its input is the embedding)"
    t = time.time(); phase_embed(cfg, src, calib, work, device, 0, 1, log = log); log(f"embed {time.time() - t:.1f}s")
    t = time.time(); phase_attn(0, cfg, src, calib, work, device, 0, 1, log = log); log(f"attn {time.time() - t:.1f}s")
    bits = json.load(open(a.bits))["layers"]["0"] if a.bits else {"gate_up": 3, "down": 4}
    if a.bench_bits:
        g, d = a.bench_bits.split(",")
        bits = {"gate_up": float(g), "down": float(d)}
    e0, e1 = (int(x) for x in a.experts.split(":"))
    t = time.time()
    phase_experts(0, e0, e1, cfg, src, calib, work, device, make_quantizer(a, a.work), bits, n0 = a.n0,
                  group = a.group, seed = a.seed, log = log, per_expert = a.per_expert)
    dt = time.time() - t
    st = json.load(open(work.tmp(0, f"experts-{e0:03d}-{e1:03d}.json")))
    n = e1 - e0
    log(f"{n} experts at {bits}: {dt:.1f}s ({dt / n:.2f}s per expert, {dt / (3 * n):.2f}s per matrix); "
        f"x 384 x 48 = {dt / n * 384 * 48 / 3600:.1f} h on one GPU")
    for e, v in list(st["experts"].items())[:8]:
        log(f"  expert {e}: {v}")


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices = ["run", "phase", "bench", "ablate"])
    ap.add_argument("--src", required = True)
    ap.add_argument("--calib", required = True)
    ap.add_argument("--work", required = True)
    ap.add_argument("--bits")
    ap.add_argument("--gpus", default = "0,1", help = "comma-separated GPU indices, or cpu,cpu")
    ap.add_argument("--layers", help = "e.g. 0-47 (default all)")
    ap.add_argument("--n0", type = float, default = 1024.0, help = "Hessian shrinkage pseudo-count (tokens)")
    ap.add_argument("--group", type = int, default = 8, help = "experts quantized per batch")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--fake", help = "tests: 'identity', a rounding step, or k<step> (step halving per bit) instead of EXL3")
    ap.add_argument("--per-expert", help = "v2: bits per expert by routing share on these levels, e.g. "
                                           "1.5,2,2.5,3,3.5,4 (each layer's means stay those of --bits)")
    ap.add_argument("--keep-tmp", action = "store_true")
    ap.add_argument("--starts", default = "0,6,12,18,24,30,36,42,48",
                    help = "ablate: track starts k (block starts, or the layer count)")
    ap.add_argument("--kinds", default = "a,s", help = "ablate: a = layers < k quantized, s = layers >= k quantized")
    # phase / bench
    ap.add_argument("--phase")
    ap.add_argument("--layer", type = int, default = 0)
    ap.add_argument("--part", type = int, default = 0)
    ap.add_argument("--parts", type = int, default = 1)
    ap.add_argument("--experts", default = "0:8")
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--bench-bits", help = "bench: gate_up,down bits (default: the layer's entry in --bits)")
    a = ap.parse_args()
    if a.cmd == "run":
        if not a.bits:
            raise SystemExit("run needs --bits")
        run(a)
    elif a.cmd == "phase":
        run_phase(a)
    elif a.cmd == "ablate":
        ablate(a)
    else:
        bench(a)


if __name__ == "__main__":
    main()
