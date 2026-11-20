"""
One GB10: where a prefill chunk's time goes on one rank of the two-Spark split. Rank 0's shard of the first
--layers layers (tp_size 2), no collectives (world 1, so the all-reduces are skipped and the numbers are one rank's
compute only). A cold prefill of --ctx tokens in --chunk-row calls through those layers with the engine's own code;
prints ms per chunk and per layer, tokens/s extrapolated to 48 layers, and a torch.profiler kernel table of one chunk.

  PYTHONPATH=/work/build/exllamav3 python spark/bench_prefill.py --model /models/X --layers 6 --ctx 16384 --chunk 512
"""
import argparse, json, os, time
import torch
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required = True)
ap.add_argument("--layers", type = int, default = 6)
ap.add_argument("--ctx", type = int, default = 16384)
ap.add_argument("--chunk", type = int, default = 512)
ap.add_argument("--profile-chunk", type = int, default = 3, help = "which chunk to profile (-1: none)")
ap.add_argument("--rows", type = int, default = 40)
ap.add_argument("--verify-check", type = int, default = 0, metavar = "R",
                help = "after the prefill: one speculative-verify-shaped call of R rows at the end of the context, far "
                       "kernel off vs on (Batch.seq_rows); prints the output difference and both timings")
ap.add_argument("--compare-far", action = "store_true",
                help = "run the prefill with the far-key kernel off, then on; print both timings and the final hidden-state difference")
a = ap.parse_args()
dev = torch.device("cuda:0")
torch.cuda.set_device(0)

t0 = time.time()
cfg = Config.from_directory(a.model)
model = Model.from_config(cfg, tp_rank = 0, tp_size = 2)
model.embed.load(dev)
layers = model.layers[:a.layers]
for L in layers:
    L.load(dev)
print(f"loaded embed + {len(layers)} layers in {time.time() - t0:.0f}s; quantized attention: "
      f"{[bool(getattr(L, 'attn_q', False)) for L in layers]}; global: {[bool(L.is_global) for L in layers]}", flush = True)
PS = 64
pages = a.ctx // PS + 2
eng = I2Engine(model, dev, 0, 1, max_slots = 2, num_pages = pages + 2, page_size = PS, bits_g = 4, bits_l = 8,
               max_pages_per_seq = pages, kv_rot = True, kv_ng = 4, kv_q16 = True, max_rows = a.chunk)
eng.page_table[1, :2] = 0
eng.page_table[0, :pages] = torch.arange(2, 2 + pages, dtype = torch.int32, device = dev)
ids = torch.randint(1000, 150000, (a.ctx,), device = dev)

ev = lambda: torch.cuda.Event(enable_timing = True)
n_chunks = a.ctx // a.chunk


def prefill(profile_chunk, nsplit_g = 8):
    per_chunk, per_layer = [], {L.idx: 0.0 for L in layers}
    h = None
    with torch.no_grad():
        for k in range(n_chunks):
            c0, n = k * a.chunk, a.chunk
            B = Batch(torch.zeros(n, dtype = torch.int32, device = dev), torch.arange(c0, c0 + n, dtype = torch.int32, device = dev),
                      torch.arange(n, dtype = torch.int32, device = dev), True, nsplit_g = nsplit_g, nsplit_l = 4)
            prof = None
            if k == profile_chunk:
                prof = torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            e_start = ev(); e_start.record()
            h = eng.embed_tokens(ids[c0:c0 + n])
            blocks = eng.new_blocks(n)
            marks = []
            for L in layers:
                h = eng.layer(L, h, blocks, B)
                m = ev(); m.record(); marks.append((L.idx, m))
            torch.cuda.synchronize()
            if prof is not None:
                prof.__exit__(None, None, None)
            t_prev = e_start
            tot = e_start.elapsed_time(marks[-1][1])
            for idx, m in marks:
                per_layer[idx] += t_prev.elapsed_time(m)
                t_prev = m
            per_chunk.append(tot)
            if prof is not None:
                print(f"--- profile of chunk {k} (positions {c0}-{c0 + n}):")
                print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = a.rows, max_name_column_width = 70))
    return per_chunk, per_layer, h


def report(tag, per_chunk, per_layer):
    warm = per_chunk[1:] if len(per_chunk) > 1 else per_chunk
    ms = sum(warm) / len(warm)
    nL = len(layers)
    print(f"{tag}chunk {a.chunk} rows: {ms:.1f} ms for {nL} layers (first chunk {per_chunk[0]:.1f} ms; last {per_chunk[-1]:.1f} ms)")
    print(f"{tag}per layer, mean over chunks (ms): " + ", ".join(f"L{i} {v / n_chunks:.1f}" for i, v in per_layer.items()))
    print(f"{tag}extrapolated to 48 layers on one rank: {ms / nL * 48:.0f} ms per chunk -> {a.chunk / (ms / nL * 48 / 1000):.0f} tokens/s "
          f"(the two-Spark server adds 2 all-reduces per layer)", flush = True)


if a.compare_far:
    far = eng.far
    eng.far = None
    pc0, pl0, h0 = prefill(-1)
    report("[far off] ", pc0, pl0)
    eng.far = far
    pc1, pl1, h1 = prefill(-1)
    report(f"[far {far}] ", pc1, pl1)
    d = (h1.float() - h0.float())
    print(f"final hidden state of the last chunk: rel diff {(d.norm() / h0.float().norm()).item():.2e}, "
          f"max {d.abs().max().item():.2e} (|h| max {h0.float().abs().max().item():.1f}); "
          f"last chunk {pc0[-1]:.1f} -> {pc1[-1]:.1f} ms", flush = True)
    eng.far = None
    _, _, h2 = prefill(-1, nsplit_g = 4)
    d = (h2.float() - h0.float())
    print(f"noise floor (far off, 4 vs 8 key splits): rel diff {(d.norm() / h0.float().norm()).item():.2e}, "
          f"max {d.abs().max().item():.2e}", flush = True)
    raise SystemExit
per_chunk, per_layer, _ = prefill(a.profile_chunk)
report("", per_chunk, per_layer)
if a.verify_check:
    R = a.verify_check
    c0 = n_chunks * a.chunk
    Bv = Batch(torch.zeros(R, dtype = torch.int32, device = dev), torch.arange(c0, c0 + R, dtype = torch.int32, device = dev),
               torch.arange(R, dtype = torch.int32, device = dev), True, nsplit_g = 8, nsplit_l = 4,
               nafter = (R - 1 - torch.arange(R, device = dev)).to(torch.int32), seq_rows = R)
    out = {}
    far_v = eng.far_v
    with torch.no_grad():
        for tag, fv in (("off", None), ("on", far_v), ("off", None), ("on", far_v)):
            eng.far_v = fv
            torch.cuda.synchronize()
            e0, e1 = ev(), ev()
            e0.record()
            h = eng.embed_tokens(ids[:R])
            blocks = eng.new_blocks(R)
            for L in layers:
                h = eng.layer(L, h, blocks, Bv)
            e1.record()
            torch.cuda.synchronize()
            out[tag] = (h.float(), e0.elapsed_time(e1))
    d = out["on"][0] - out["off"][0]
    print(f"verify-shaped call, {R} rows at position {c0}: far {far_v} {out['on'][1]:.2f} ms vs off {out['off'][1]:.2f} ms "
          f"for {len(layers)} layers; output rel diff {(d.norm() / out['off'][0].norm()).item():.2e}", flush = True)
