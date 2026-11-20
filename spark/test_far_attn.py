"""
Global-layer attention for long prefill chunks: the far/near split (ops._attn_far_kernel for the keys more than
rel_extent behind the chunk's first row, the per-row kernel for the rest) against the per-row kernel alone, on one GB10
with a synthetic KV cache in the served format (4-bit, 16-level grid, 4 fp16 scale groups per vector, rotation,
shuffled pages, fused relative bias). Prints the time of one call and the output difference per configuration.

  PYTHONPATH=/work/exllamav3 python spark/test_far_attn.py --starts 0,1000,8192,65536,131072,300000 --rows 2048,512
"""
import argparse, math
from types import SimpleNamespace
import torch
from exllamav3.modules import inkling2_ops as ops

ap = argparse.ArgumentParser()
ap.add_argument("--starts", default = "0,1000,1100,8192,65536,131072,300000", help = "position of the chunk's first row")
ap.add_argument("--rows", default = "2048,512")
ap.add_argument("--configs", default = "8,0,0;4,4,64;2,2,64;1,3,64;1,7,64;4,4,128",
                help = "near splits,far splits,M (far 0 = the per-row kernel alone, the first one is the reference)")
ap.add_argument("--iters", type = int, default = 5)
ap.add_argument("--seqs", type = int, default = 1, help = "sequences in one call, --rows rows each (the second starts 777 "
                "positions before the first, and so on), each with its own page table")
ap.add_argument("--verify", action = "store_true", help = "far kernel with one program per sequence (far_seq_rows = rows)")
a = ap.parse_args()
dev = torch.device("cuda:0")
torch.cuda.set_device(0)
torch.manual_seed(0)

G, HKL, DH, PS = 6, 4, 128, 64
HL = G * HKL
cfg = SimpleNamespace(rel_extent = 1024, window = 1024)
starts = [int(x) for x in a.starts.split(",")]
rows = [int(x) for x in a.rows.split(",")]
configs = [tuple(int(v) for v in c.split(",")) for c in a.configs.split(";")]
maxpos = max(starts) + max(rows)
pages = math.ceil(maxpos / PS) + 1
kq = torch.randint(0, 256, (pages, PS, HKL, DH // 2), dtype = torch.uint8, device = dev)
vq = torch.randint(0, 256, (pages, PS, HKL, DH // 2), dtype = torch.uint8, device = dev)
ks = (torch.rand((pages, PS, HKL, 4), device = dev) * 0.2 + 0.2).half()
vs = (torch.rand((pages, PS, HKL, 4), device = dev) * 0.2 + 0.2).half()
page_table = torch.zeros((max(2, a.seqs), pages), dtype = torch.int32, device = dev)
for i in range(a.seqs):
    page_table[i] = torch.randperm(pages, device = dev).to(torch.int32)
rot = ops.make_kv_rotation(dev)
rel_proj = (torch.randn((16, cfg.rel_extent), device = dev) * 0.3).float().contiguous()
print(f"KV cache: {pages} pages ({maxpos} positions), G {G}, kv heads {HKL}", flush = True)


def run(T, c0, near, far, M, q, sink, gate, relq, o):
    R = T // a.seqs
    slot = torch.arange(a.seqs, dtype = torch.int32, device = dev).repeat_interleave(R)
    pos = torch.cat([torch.arange(max(0, c0 - i * 777), max(0, c0 - i * 777) + R) for i in range(a.seqs)]).to(torch.int32).to(dev)
    ns = near + far
    pm = torch.empty((T, HL, ns), dtype = torch.float32, device = dev)
    pl = torch.empty_like(pm)
    po = torch.empty((T, HL, ns, DH), dtype = torch.float32, device = dev)
    kw = dict(ns_far = far, far_m = M) if far else {}
    if far and a.verify:
        kw["far_seq_rows"] = R
    ops.attention(q, None, sink, gate, kq, vq, ks, vs, page_table, slot, pos, (pm, pl, po), o, cfg, True, 4, 0, PS,
                  near, HKL, G, rot = rot, ng = 4, q16 = True, rel = (relq, rel_proj), **kw)


ok = True
for T in rows:
    T = T * a.seqs
    q = (torch.randn((T, HL, DH), device = dev) * (22.0 / DH ** 0.5)).bfloat16()
    sink = (torch.randn((T, HL), device = dev) * 2).bfloat16()
    gate = torch.randn((T, HL * DH), device = dev).bfloat16()
    relq = (torch.randn((T, HL, 16), device = dev) * 0.5).bfloat16()
    for c0 in starts:
        res = []
        ref = None
        for (near, far, M) in configs:
            o = torch.empty((T, HL * DH), dtype = torch.bfloat16, device = dev)
            run(T, c0, near, far, M, q, sink, gate, relq, o)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(a.iters):
                run(T, c0, near, far, M, q, sink, gate, relq, o)
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / a.iters
            if ref is None:
                ref, ref_ms = o.float(), ms
                res.append(f"{near},{far},{M}: {ms:.2f} ms (reference)")
                continue
            d = (o.float() - ref)
            rel = (d.norm() / ref.norm()).item()
            mx = d.abs().max().item()
            bad = not (rel < 2e-3 and mx < 5e-2) or not torch.isfinite(o.float()).all().item()
            ok &= not bad
            res.append(f"{near},{far},{M}: {ms:.2f} ms ({ref_ms / ms:.2f}x) rel {rel:.1e} max {mx:.1e}{'  <-- MISMATCH' if bad else ''}")
        print(f"T {T} ({a.seqs} x {T // a.seqs}) from {c0}: " + " | ".join(res), flush = True)
print("ALL MATCH" if ok else "SOME MISMATCH")
