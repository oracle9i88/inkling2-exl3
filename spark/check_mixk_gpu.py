"""
GPU check of mixed-rate experts (overlay/exllamav3/modules/inkling2_mixk.py) on one layer of a real checkpoint, all
ranks of a tp split simulated in one process (each rank's Inkling2Layer loads its own experts and router):

  correctness   the summed rank outputs against the reference (modules/inkling2.py: route + moe_forward over one
                LinearEXL3 per expert and projection, every expert at its own rate), for row counts on the fused decode
                path (<= 64) and on the prefill tiers (> 64):
                  sel    tokens whose 8 experts differ from the fp32 reference router (fp16 router kernels: rare near-ties)
                  same   relative error against the reference fed the engine's own selection (expert compute alone)
                  own    relative error against the reference with its own routing
  speed         rank 0's MoE call captured in a CUDA graph as the server does, replayed with a fresh input each time
                (different experts per replay, so the weights come from DRAM, not L2)

  --split N     force N rate classes on a uniform checkpoint (the v1 build) to measure what the per-class launches cost
                with the bytes unchanged
  --mk-off      decode-sized calls through the rate classes too (I2_MIXK_DECODE=0) instead of the multi-rate kernels
  --dense       also against fp32 math on the dequantized weights (the referee: the reference's own small-batch GEMV is
                ~1% off, the engine's paths ~0.1%)

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<exllamav3 with the overlay> python check_mixk_gpu.py --model DIR --layer 12
"""
import argparse, json, os, time, types
import torch

from exllamav3.model.config import Config
from exllamav3.architecture.inkling2 import Inkling2Layer
from exllamav3.modules import inkling2 as I2
from exllamav3.modules import inkling2_mixk as MIXK


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


class StSource:
    """the Exl3Experts source interface (has / get) over the directory's safetensors"""

    def __init__(self, cfg):
        self.stc = cfg.stc

    def has(self, k):
        return self.stc.has_tensor(k)

    def get(self, k, device):
        return self.stc.get_tensor(k, device, allow_bf16 = True, no_defer = True)


def build(d, L, tp, dev, split = 0):
    orig = MIXK.expert_rates
    if split:
        MIXK.expert_rates = lambda stc, key, E: (lambda r: None if r is None else [tuple(x) + (e % split,) for e, x in enumerate(r)])(orig(stc, key, E))
    try:
        cfg = Config.from_directory(d)
        layers = [Inkling2Layer(cfg, L, r, tp) for r in range(tp)]
    finally:
        MIXK.expert_rates = orig
    t = time.time()
    for Lr in layers:
        Lr.mlp.load(dev)
    torch.cuda.synchronize()
    return cfg, layers, time.time() - t


def engine_routing(m, x):
    """the engine router's selection in checkpoint ids and its weights (copies: the router writes reused buffers)"""
    y = x.view(-1, x.shape[-1])
    if getattr(m, "mixed_rates", False):
        s, w = m.route(y.shape[0], m.router.routing_cfg, y, {})
        return torch.tensor(m.perm, device = s.device)[s].clone(), w.float().clone()
    s, w = m.routing_fn(y.shape[0], m.routing_cfg, y, {})
    return s.clone(), w.float().clone()


@torch.no_grad()
def correctness(cfg, layers, L, dev, rows, seed = 0):
    c = cfg.i2
    ex = I2.Exl3Experts(StSource(cfg), c, L, dev, config = cfg)
    key = f"model.llm.layers.{L}.mlp.gate"
    W = types.SimpleNamespace(cfg = c, router_w = cfg.stc.get_tensor(key + ".weight", dev, allow_bf16 = True, no_defer = True),
                              router_b = cfg.stc.get_tensor(key + ".bias", dev, allow_bf16 = True, no_defer = True))
    g = torch.Generator(device = dev).manual_seed(seed)
    res = []
    for T in rows:
        x = torch.randn((T, c.hidden), generator = g, device = dev).half()
        new = None
        for Lr in layers:                       # each rank's partial sum (clone: the decode path returns a shared buffer)
            o = Lr.mlp.forward(x, {}).clone()
            new = o if new is None else new + o
        sel_e, w_e = engine_routing(layers[0].mlp, x)
        sel_r, w_r = I2.route(W, x.float())
        diff = int((sel_e.sort(-1).values != sel_r.sort(-1).values).any(-1).sum())
        ref_same = I2.moe_forward(x.float(), sel_e, w_e, ex, c, adt = torch.float32)
        ref_own = I2.moe_forward(x.float(), sel_r, w_r, ex, c, adt = torch.float32)
        res.append({"rows": T, "sel_diff_tokens": diff, "rel_same": rel(new, ref_same), "rel_own": rel(new, ref_own),
                    "max_abs_same": (new - ref_same).abs().max().item(), "ref_rms": ref_same.float().pow(2).mean().sqrt().item()})
        print(f"  T={T:5d}  sel differs on {diff:3d} tokens   rel err (same routing) {res[-1]['rel_same']:.2e}   "
              f"(own routing) {res[-1]['rel_own']:.2e}", flush = True)
    del ex
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def dense(cfg, layers, L, dev, rows, seed = 1):
    """engine and reference against fp32 math on the dequantized expert weights (one expert at a time)"""
    c = cfg.i2
    ex = I2.Exl3Experts(StSource(cfg), c, L, dev, config = cfg)
    g = torch.Generator(device = dev).manual_seed(seed)
    res = []
    for T in rows:
        x = torch.randn((T, c.hidden), generator = g, device = dev).half()
        new = None
        for Lr in layers:
            o = Lr.mlp.forward(x, {}).clone()
            new = o if new is None else new + o
        sel, w = engine_routing(layers[0].mlp, x)
        ref = I2.moe_forward(x.float(), sel, w, ex, c, adt = torch.float32)
        d = torch.zeros((T, c.hidden), dtype = torch.float32, device = dev)
        xf = x.float()
        for e in sorted(set(sel.view(-1).tolist())):
            wg, wu, wd = (ex.lin[(e, p)].get_weight_tensor().float() for p in ("gate_proj", "up_proj", "down_proj"))
            ti, ji = (sel == e).nonzero(as_tuple = True)
            a = I2.expert_act(torch.cat([xf[ti] @ wg, xf[ti] @ wu], -1), c, torch.float32)
            d.index_add_(0, ti, (a @ wd) * w[ti, ji].unsqueeze(1))
            del wg, wu, wd
        res.append({"rows": T, "engine_vs_fp32": rel(new, d), "reference_vs_fp32": rel(ref, d)})
        print(f"  T={T:5d}  engine vs fp32 {res[-1]['engine_vs_fp32']:.2e}   reference vs fp32 {res[-1]['reference_vs_fp32']:.2e}",
              flush = True)
    del ex
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def bench(m, D, T, dev, n = 200, pool_n = 64):
    xs = torch.randn((pool_n, T, D), device = dev).half()
    x = xs[0].clone()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            m.forward(x, {})
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        m.forward(x, {})
    for i in range(5):
        x.copy_(xs[i % pool_n]); graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(n):
        x.copy_(xs[i % pool_n])
        graph.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for i in range(n):
        x.copy_(xs[i % pool_n])
    torch.cuda.synchronize()
    return (dt - (time.perf_counter() - t) / n) * 1e6     # microseconds per MoE call, input copy subtracted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, help = "dir with config.json, the layer's router and its experts file")
    ap.add_argument("--layer", type = int, required = True)
    ap.add_argument("--tp", type = int, default = 2)
    ap.add_argument("--split", type = int, default = 0, help = "force this many rate classes (uniform checkpoints)")
    ap.add_argument("--rows", default = "1,2,3,5,8,9,16,33,64,65,128,192,300,1024")
    ap.add_argument("--bench-rows", default = "1,2,5,8,9,16,64")
    ap.add_argument("--no-check", action = "store_true")
    ap.add_argument("--mk-off", action = "store_true", help = "decode calls through the classes (I2_MIXK_DECODE=0)")
    ap.add_argument("--dense", default = "", help = "row counts to check against fp32 math, e.g. 1,8,64,300")
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    if a.mk_off:
        os.environ["I2_MIXK_DECODE"] = "0"
    cfg, layers, tload = build(a.model, a.layer, a.tp, dev, a.split)
    m0 = layers[0].mlp
    mixed = getattr(m0, "mixed_rates", False)
    desc = [Lr.mlp.describe() if mixed else "uniform" for Lr in layers]
    print(f"{a.model} layer {a.layer}: tp {a.tp}, {'mixed' if mixed else 'uniform'}, loaded in {tload:.1f}s, "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    for r, s in enumerate(desc):
        print(f"  rank {r}: {s}")
    decode = None
    if mixed:
        decode = "multi-rate kernels" if m0.mk is not None else f"rate classes ({m0.mk_off})"
        print(f"  decode-sized calls: {decode}")
    out = {"model": a.model, "layer": a.layer, "tp": a.tp, "split": a.split, "mixed": mixed, "ranks": desc, "decode": decode}
    if not a.no_check:
        out["check"] = correctness(cfg, layers, a.layer, dev, [int(x) for x in a.rows.split(",")])
    if a.dense:
        out["dense"] = dense(cfg, layers, a.layer, dev, [int(x) for x in a.dense.split(",")])
    out["bench_us_rank0"] = {}
    for T in [int(x) for x in a.bench_rows.split(",")]:
        us = bench(m0, cfg.i2.hidden, T, dev)
        out["bench_us_rank0"][T] = us
        print(f"  rank 0 MoE call, T={T:3d}: {us:8.1f} us", flush = True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent = 1)


if __name__ == "__main__":
    main()
