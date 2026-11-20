"""
Held-out check of an assembled build on the Sparks. Runs the calibration rows the quantization held out (7 x 4096
tokens) through the engine (the serving kernels, tensor-parallel over the nodes) and compares, on rank 0, with the
states the quantization run saved on the 5090 (acceptance/<state>, copied from the portable disk):

  engine vs q   the same quantized model as computed by the quantization pipeline (reference ops, one GPU). Should
                agree closely: a wrong expert file or a wrong router permutation shows up here at once
  engine vs r   the source model: KL(r || engine), top-1 agreement, NLL. Should be near the run's own KL (final.json)
  q vs r        recomputed from the files, which checks this script against the run's final.json

Every row goes through the engine in four segments, so both MoE paths are covered: positions up to S-512 in 512-row
prefill chunks (rate classes), then 64-row and 8-row chunks (the multi-rate decode kernels), and the last 32
positions one token per sequence with all rows at once (plain decode, NR sequences per call).

  SCRIPT=spark/eval/heldout_check.py spark/launch_2spark.sh start [--kv 16,8] [--rows N] [--locate 1]

--kv global,local KV bits: 16,8 keeps the global-layer KV in bf16 (isolates the experts); 4,8 is how the server runs.
--locate 1 also compares the inputs of layers 0, 6, ..., 42 and the last layer's output with q_b0..q_b7 and q_h48.
Rank 0's log is the report; the numbers also go to /work/logs/heldout-kv<g>-<l>-<time>.json.
"""
import os, sys, time, json, math, argparse
import numpy as np
import torch, torch.distributed as dist
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch
from exllamav3.modules.inkling2_parallel import local_device_index
from exllamav3.modules import inkling2 as I2

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "/model")
ap.add_argument("--acc", default = "/models/inkling2-disk/acceptance")
ap.add_argument("--state", default = "state-v2")
ap.add_argument("--kv", default = "16,8", help = "global,local KV bits")
ap.add_argument("--rows", type = int, default = 0, help = "only the first N held-out rows (0 = all)")
ap.add_argument("--locate", type = int, default = 0)
ap.add_argument("--engine-head", default = "auto", help = "auto: the engine's own head for the engine's logits (EXL3 when the folder has model.llm.unembed_q); bf16: the source head for every track")
args, _ = ap.parse_known_args()          # the launch script also passes --port and --reserve-gb
BG, BL = (int(x) for x in args.kv.split(","))
CHUNK = 512

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
_gpu = local_device_index()
torch.cuda.set_device(_gpu)
dev = torch.device(f"cuda:{_gpu}")
dist.init_process_group("nccl", device_id = dev)


def log(*x):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *x, flush = True)


# ------------------------------------------------------------------------------------------------- held-out rows
if rank == 0:
    z = np.load(os.path.join(args.acc, "calib", "calib.npz"))
    ids_all, held = z["ids"].astype(np.int64), z["heldout"].astype(bool)
    R, S = ids_all.shape
    r_rows = np.nonzero(held)[0]
    if args.rows:
        r_rows = r_rows[:args.rows]
    hdr = torch.tensor([R, S, len(r_rows)], dtype = torch.long, device = dev)
else:
    hdr = torch.zeros(3, dtype = torch.long, device = dev)
dist.broadcast(hdr, 0)
R, S, NR = (int(v) for v in hdr.tolist())
ROWS = torch.from_numpy(ids_all[r_rows]).to(dev) if rank == 0 else torch.zeros((NR, S), dtype = torch.long, device = dev)
dist.broadcast(ROWS, 0)
SEGS = [(0, S - 512, CHUNK, "512-row prefill chunks (rate classes)"),
        (S - 512, S - 256, 64, "64-row chunks (multi-rate kernels)"),
        (S - 256, S - 32, 8, "8-row chunks (multi-rate kernels)"),
        (S - 32, S, 1, f"plain decode, {NR} sequences per call")]
log(f"{NR} held-out rows x {S} tokens (of {R} calibration rows); KV bits global {BG} local {BL}; world {world}")

# ------------------------------------------------------------------------------------------------- model + engine
t0 = time.time()
cfg = Config.from_directory(args.model)
model = Model.from_config(cfg, tp_rank = rank, tp_size = world)
model.embed.load(dev)
for L in model.layers:
    L.load(dev)
model.head.load(dev)
_mixed = [L for L in model.layers if getattr(L.mlp, "mixed_rates", False)]
log(f"weights loaded in {time.time() - t0:.0f}s; mixed-rate layers {len(_mixed)}"
    + (f", decode-sized calls: {'multi-rate kernels' if all(L.mlp.mk is not None for L in _mixed) else 'rate classes'}"
       if _mixed else ""))
for f in os.listdir(args.model):                # hand the weight files' page cache back (unified memory, like the server)
    if f.endswith(".safetensors"):
        try:
            fd = os.open(os.path.join(args.model, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
PS = 64
PPR = S // PS                                   # pages per row
D = cfg.i2.hidden
eng = I2Engine(model, dev, rank, world, max_slots = NR + 1, num_pages = NR * PPR + 2, page_size = PS, bits_g = BG,
               bits_l = BL, max_pages_per_seq = PPR + 2, kv_rot = True, kv_ng = 4, kv_q16 = True, max_rows = CHUNK)
eng.page_table[NR, :2] = 0                      # spare slot owns page 0, like the server's padding slot
for j in range(NR):
    eng.page_table[j, :PPR] = torch.arange(1 + j * PPR, 1 + (j + 1) * PPR, dtype = torch.int32, device = dev)


@torch.no_grad()
def run_all(stop_layer = None):
    """every row through the engine, segment by segment; returns [NR, S, D] (final hidden, or h after stop_layer)"""
    X = torch.empty((NR, S, D), dtype = torch.bfloat16, device = dev)
    fwd = lambda ids, b: eng.forward(ids, b, stop_layer = stop_layer, return_hidden = stop_layer is None)
    for j in range(NR):
        for a, e, step, _ in SEGS[:3]:
            for c0 in range(a, e, step):
                n = min(step, e - c0)
                b = Batch(torch.full((n,), j, dtype = torch.int32, device = dev),
                          torch.arange(c0, c0 + n, dtype = torch.int32, device = dev),
                          torch.arange(n, dtype = torch.int32, device = dev), True, nsplit_g = 8, nsplit_l = 4)
                X[j, c0:c0 + n] = fwd(ROWS[j, c0:c0 + n], b)
    slots = torch.arange(NR, dtype = torch.int32, device = dev)
    zeros = torch.zeros(NR, dtype = torch.int32, device = dev)
    for p in range(SEGS[3][0], S):              # like the server's plain decode step: one row per sequence
        b = Batch(slots, torch.full((NR,), p, dtype = torch.int32, device = dev), zeros, False, nsplit_g = 8, nsplit_l = 4)
        X[:, p] = fwd(ROWS[:, p], b)
    torch.cuda.synchronize()
    return X


# ------------------------------------------------------------------------------------------------- reference states
if rank == 0:
    hcfg = I2.Inkling2Cfg.from_dir(args.model)
    Hd = I2.HeadWeights(I2.Inkling2Source(args.model), hcfg, dev)
    NB = hcfg.num_blocks_before(hcfg.layers)
    sd = os.path.join(args.acc, args.state)
    n_r = int(held.sum()) * S                   # the r track holds every held-out row, in order
    mm = lambda name, N: np.memmap(os.path.join(sd, name), dtype = np.int16, mode = "r", shape = (N, D))
    Q48, R48 = mm("q_h48.bin", R * S), mm("r_h48.bin", n_r)
    QB = [mm(f"q_b{k}.bin", R * S) for k in range(NB)]
    RB = [mm(f"r_b{k}.bin", n_r) for k in range(NB)]
    rd = lambda m, sl: torch.from_numpy(np.ascontiguousarray(m[sl])).view(torch.bfloat16).to(dev)
    fj = os.path.join(args.acc, "out-" + args.state.split("-", 1)[1] if "-" in args.state else "out-v1", "final.json")   # state-vN -> out-vN
    final_ref = json.load(open(fj)) if os.path.exists(fj) else None
    head_e = None                               # the engine's EXL3 head (v3), whole padded vocabulary on rank 0
    from exllamav3.modules import inkling2_attnq as AQ
    if args.engine_head == "auto" and AQ.has_quantized_head(cfg.stc):
        head_e = AQ.exl3_select(cfg.stc, AQ.HEAD_KEY, dev, config = cfg)
        log(f"engine logits from the EXL3 head ({head_e.out_features} rows, {head_e.K} bits); q and r from the bf16 head")
else:
    NB = 8

t1 = time.time()
XE = run_all()
log(f"engine pass: {time.time() - t1:.0f}s")

if rank == 0:
    keys = ("kl_re", "kl_rq", "kl_qe", "nll_e", "nll_q", "nll_r", "t1_er", "t1_qr", "t1_eq", "sq_eq", "sq_q", "sq_qr", "sq_r")
    acc = {k: torch.zeros((NR, S), dtype = torch.float64) for k in keys}
    for j in range(NR):
        row = int(r_rows[j])
        sq, sr = slice(row * S, (row + 1) * S), slice(j * S, (j + 1) * S)
        xe = XE[j]
        xq = I2.final_hidden(Hd, rd(Q48, sq), torch.stack([rd(b, sq) for b in QB]))
        xr = I2.final_hidden(Hd, rd(R48, sr), torch.stack([rd(b, sr) for b in RB]))
        f = lambda t: t.double().cpu()
        acc["sq_eq"][j] = f((xe.float() - xq.float()).square().sum(-1))
        acc["sq_qr"][j] = f((xq.float() - xr.float()).square().sum(-1))
        acc["sq_q"][j], acc["sq_r"][j] = f(xq.float().square().sum(-1)), f(xr.float().square().sum(-1))
        ids = ROWS[j]
        for c0 in range(0, S, 256):              # [256, 200k] fp32 logits per track keeps rank 0's peak small
            c1 = min(S, c0 + 256)
            if head_e is None:
                le = torch.log_softmax(I2.logits_from_hidden(Hd, xe[c0:c1]), -1)
            else:
                ze = AQ.run_f32(head_e, xe[c0:c1])
                le = torch.log_softmax((torch.tanh(ze / hcfg.final_cap) * hcfg.final_cap)[..., :hcfg.unpadded_vocab], -1)
            lq = torch.log_softmax(I2.logits_from_hidden(Hd, xq[c0:c1]), -1)
            lr = torch.log_softmax(I2.logits_from_hidden(Hd, xr[c0:c1]), -1)
            pr, pq = lr.exp(), lq.exp()
            acc["kl_re"][j, c0:c1] = f((pr * (lr - le)).sum(-1))
            acc["kl_rq"][j, c0:c1] = f((pr * (lr - lq)).sum(-1))
            acc["kl_qe"][j, c0:c1] = f((pq * (lq - le)).sum(-1))
            m = min(c1, S - 1) - c0                  # positions with a next token
            if m > 0:
                tgt = ids[c0 + 1:c0 + 1 + m, None]
                acc["nll_e"][j, c0:c0 + m] = f(-le[:m].gather(1, tgt)[:, 0])
                acc["nll_q"][j, c0:c0 + m] = f(-lq[:m].gather(1, tgt)[:, 0])
                acc["nll_r"][j, c0:c0 + m] = f(-lr[:m].gather(1, tgt)[:, 0])
            ae, aq, ar = le.argmax(-1), lq.argmax(-1), lr.argmax(-1)
            acc["t1_er"][j, c0:c1] = f(ae == ar)
            acc["t1_qr"][j, c0:c1] = f(aq == ar)
            acc["t1_eq"][j, c0:c1] = f(ae == aq)
        log(f"row {row:3d}: engine vs q rel {math.sqrt(acc['sq_eq'][j].sum() / acc['sq_q'][j].sum()):.2e}, "
            f"KL(q||e) {acc['kl_qe'][j, :S - 1].mean():.5f} | KL(r||e) {acc['kl_re'][j, :S - 1].mean():.5f}  "
            f"KL(r||q) {acc['kl_rq'][j, :S - 1].mean():.5f}  top-1 e/r {acc['t1_er'][j, :S - 1].mean():.4f}")

    def summary(a, e):
        sl = (slice(None), slice(a, min(e, S - 1)))          # the run's convention: positions 0..S-2
        g = lambda k: acc[k][sl]
        return {"tokens": int(g("kl_re").numel()),
                "engine_vs_q": {"rel_hidden": math.sqrt(acc["sq_eq"][:, a:e].sum() / acc["sq_q"][:, a:e].sum()),
                                "kld_q_e": g("kl_qe").mean().item(), "top1": g("t1_eq").mean().item()},
                "engine_vs_ref": {"kld": g("kl_re").mean().item(), "top1": g("t1_er").mean().item(),
                                  "ppl_engine": math.exp(g("nll_e").mean().item()), "ppl_ref": math.exp(g("nll_r").mean().item())},
                "q_vs_ref": {"kld": g("kl_rq").mean().item(), "top1": g("t1_qr").mean().item(),
                             "ppl_q": math.exp(g("nll_q").mean().item()),
                             "rel_hidden": math.sqrt(acc["sq_qr"][:, a:e].sum() / acc["sq_r"][:, a:e].sum())}}

    res = {"kv_bits": [BG, BL], "rows": [int(r) for r in r_rows], "all": summary(0, S),
           "segments": [{"positions": [a, e], "how": how, **summary(a, e)} for a, e, _, how in SEGS],
           "run_final_json": final_ref, "seconds": time.time() - t1}
    for s_ in res["segments"]:
        log(f"  {s_['how']:<44s} {s_['tokens']:6d} tok: engine vs q rel {s_['engine_vs_q']['rel_hidden']:.2e} "
            f"KL(q||e) {s_['engine_vs_q']['kld_q_e']:.5f} | KL(r||e) {s_['engine_vs_ref']['kld']:.5f} "
            f"KL(r||q) {s_['q_vs_ref']['kld']:.5f}")
    a_ = res["all"]
    log("== engine vs the quantization run (same model):  hidden rel {rel_hidden:.2e}  KL(q||e) {kld_q_e:.5f}  "
        "top-1 {top1:.4f}".format(**a_["engine_vs_q"]))
    log("== engine vs source:  KL(r||e) {kld:.5f}  top-1 {top1:.4f}  PPL {ppl_engine:.3f} (source {ppl_ref:.3f})"
        .format(**a_["engine_vs_ref"]))
    log("== run's own, recomputed:  KL(r||q) {kld:.5f}  top-1 {top1:.4f}  PPL {ppl_q:.3f}".format(**a_["q_vs_ref"])
        + (f"   final.json: {final_ref['kld']:.5f} / {final_ref['top1_agree']:.4f} / {final_ref['ppl_quant']:.3f}"
           if final_ref else ""))

if args.locate:   # where along the stack the engine leaves the quantization run's track
    errs = []
    for k in range(NB + 1):                     # inputs of layers 0, 6, ..., 42, then the output of the last layer
        if k == 0:
            H_ = torch.stack([torch.cat([eng.embed_tokens(ROWS[j, c0:c0 + CHUNK]) for c0 in range(0, S, CHUNK)])
                              for j in range(NR)])
        else:
            H_ = run_all(stop_layer = 6 * k - 1 if k < NB else cfg.i2.layers - 1)
        if rank == 0:
            e = [((H_[j].float() - rd(QB[k] if k < NB else Q48, slice(int(r) * S, (int(r) + 1) * S)).float()).norm()
                  / rd(QB[k] if k < NB else Q48, slice(int(r) * S, (int(r) + 1) * S)).float().norm()).item()
                 for j, r in enumerate(r_rows)]
            errs.append(e)
            log(f"locate: {'input of layer ' + str(6 * k) if k < NB else 'output of the last layer'}: rel error vs q "
                + " ".join(f"{x:.1e}" for x in e))
    if rank == 0:
        res["locate"] = errs

if rank == 0:
    os.makedirs("/work/logs", exist_ok = True)
    out = f"/work/logs/heldout-kv{BG}-{BL}-{time.strftime('%Y%m%d-%H%M')}.json"
    json.dump(res, open(out, "w"), indent = 1)
    log(f"wrote {out}; DONE")
dist.barrier()
dist.destroy_process_group()
