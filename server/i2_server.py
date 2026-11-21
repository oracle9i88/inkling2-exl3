"""
OpenAI-compatible server for Inkling-2-Small EXL3 (v3: uniform K4 mcg experts): TP4 attention + EP4 experts over NCCL,
paged global KV + SWA rings, CUDA-graphed DSpark speculative decoding at every batch size (always on, no plain fallback).

    torchrun --nproc_per_node=4 server/i2_server.py --port 8000 [--kv-gb 18]

Rank 0 runs the HTTP server and the scheduler; ranks 1..3 follow rank 0's command stream.
Sampling: temperature (Gumbel-max, speculative sampling with deterministic drafts stays exact); top_p / top_k are ignored.
"""
import os, sys, time, json, math, threading, queue, argparse, asyncio, uuid, collections, bisect
import torch, torch.distributed as dist
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch
from exllamav3.modules.inkling2_draft import I2Draft
from exllamav3.modules.inkling2_prefix import SlotSnapshots, copy_page

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-4bpw"))
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--slots", type=int, default=16)
ap.add_argument("--kv-gb", type=float, default=0.0, help="KV cache budget per GPU for global-layer pages (0 = auto from free memory)")
ap.add_argument("--reserve-gb", type=float, default=2.0, help="free memory kept for workspace when sizing KV automatically (measured peak extra use: 0.6 GiB)")
ap.add_argument("--max-model-len", type=int, default=380000)
ap.add_argument("--bits-g", type=int, default=4); ap.add_argument("--bits-l", type=int, default=8)
ap.add_argument("--burst", type=int, default=1, help="max back-to-back speculative steps per scheduler round (1 = one host sync per step)")
ap.add_argument("--spec-k", default="1:4,2:3,4:2,8:2,16:2", help="draft length per decode bucket: one K (1..7) or a map like 1:2,2:2,4:2,8:3,16:4")
ap.add_argument("--kv-rot", type=int, default=1, help="global KV: randomized Hadamard rotation along head_dim before quantizing")
ap.add_argument("--kv-ng", type=int, default=4, help="global KV: fp16 scale groups per 128-vector (1, 2, 4)")
ap.add_argument("--kv-q16", type=int, default=1, help="global 4-bit KV: 16-level symmetric grid")
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--served-name", default="inkling-2-small-exl3-v3")
ap.add_argument("--default-max-tokens", type=int, default=4096, help="max_tokens of requests that do not set it")
ap.add_argument("--default-thinking-budget", type=int, default=-1, help="thinking_budget of requests that do not set it (-1 = none)")
ap.add_argument("--answer-reserve", type=int, default=8192, help="thinking stops at least this many tokens (at most a quarter "
                "of max_tokens) before max_tokens, so a long thought still leaves room for the answer; 0 = off")
ap.add_argument("--think-loop-guard", type=float, default=0.5, help="end the thinking block when this fraction of the last "
                "--think-loop-window thinking tokens repeats 32-token runs seen earlier in that window (a loop); 0 = off")
ap.add_argument("--think-loop-window", type=int, default=2048)
ap.add_argument("--prefix-cache", type=int, default=1, help="reuse the state of earlier prompts that a new prompt starts with")
ap.add_argument("--prefix-snapshots", type=int, default=32, help="snapshot pool size (per-slot ring state, ~70 MB each per GPU)")
ap.add_argument("--prefix-min", type=int, default=1024, help="shortest prefix worth a snapshot")
ap.add_argument("--prefix-every", type=int, default=8192, help="also snapshot long prompts every N tokens (0 = only at the end)")
ap.add_argument("--spec-mode", default="auto", choices=("auto", "spec", "plain"),
                help="decode steps: speculative, plain (one token per step), or auto: per step, whichever yields more tokens/s "
                     "given the drafts the active requests have been accepting")
ap.add_argument("--probe-every", type=int, default=32, help="auto mode: one speculative step after this many plain steps, to re-measure acceptance")
ap.add_argument("--vision", type=int, default=1, help="accept images (reference vision tower from transformers, on rank 0)")
ap.add_argument("--max-image-tokens", type=int, default=0, help="downscale images above this many tokens (40x40 px each; 0 = never, like the reference)")
ap.add_argument("--chat-template", default="", help="chat template file (default: <model>/chat_template.jinja)")
ap.add_argument("--b12x", type=int, default=int(os.environ.get("I2_B12X", "0")), help="routed experts of the speculative verify through the b12x fused Trellis MoE (needs the b12x stack: cutlass-dsl + FlashInfer tree at I2_B12X_PATH)")
args = ap.parse_args()
if args.b12x:
    os.environ["EXL3_LOAD_ARENA"] = "0"     # b12x repack frees the per-expert tensors; arena slabs would pin a second copy
SPEC = True               # speculative decoding is always on (every batch size); there is no plain-decode path
NSPLIT_DEC = int(os.environ.get('I2_DEC_NSPLIT', 32))   # key splits per row for global layers in a plain decode step (was 8; 1.5x at 65K-190K positions)
_SK = {int(k): int(v) for k, v in (x.split(":") for x in args.spec_k.split(","))} if ":" in args.spec_k else None
def spec_k(B):
    """draft length of decode bucket B (the smallest mapped bucket >= B, else the largest)"""
    if _SK is None: return int(args.spec_k)
    ks = sorted(_SK)
    return _SK[next((k for k in ks if k >= B), ks[-1])]
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
from exllamav3.modules.inkling2_parallel import head_shard, local_device_index
_gpu = local_device_index()       # one GPU per node on DGX Spark: LOCAL_RANK, not the global rank
torch.cuda.set_device(_gpu); dev = torch.device(f"cuda:{_gpu}")
dist.init_process_group("nccl", device_id=dev)
def log(*x):
    if rank == 0: print(time.strftime("%H:%M:%S"), *x, flush=True)

# ---------------------------------------------------------------------------------------------------------- load
cfg = Config.from_directory(args.model)
model = Model.from_config(cfg, tp_rank=rank, tp_size=world)
if args.b12x:
    from exllamav3.modules.inkling2_b12x import prepare_model, attach_moe
    prepare_model(model)
t0 = time.time()
model.embed.load(dev)
for L in model.layers: L.load(dev)
model.head.load(dev)
_mixed = [L for L in model.layers if getattr(L.mlp, "mixed_rates", False)]
if _mixed:   # experts of different rates inside a layer (modules/inkling2_mixk.py)
    _off = sorted({L.mlp.mk_off for L in _mixed if L.mlp.mk is None})
    log(f"mixed-rate experts in {len(_mixed)} layers ({sum(len(L.mlp.parts) for L in _mixed)} rate classes on rank 0); "
        f"decode-sized calls: {'one multi-rate kernel pair per layer' if not _off else 'rate classes (' + '; '.join(_off) + ')'}")
VISION = None
if args.vision and rank == 0:
    from i2_vision import VisionEncoder
    VISION = VisionEncoder(args.model, dev, max_tokens=args.max_image_tokens)
    log(f"vision tower loaded ({VISION.n_params/1e6:.0f}M parameters, reference implementation)")
S = args.slots
PS = 64
def drop_weight_cache(model_dir):
    """Hand the page cache the weight files left behind back to the system (no root needed). On a unified-memory GPU
    (DGX Spark / GB10) that cache sits in the same memory the KV cache is allocated from."""
    n = 0
    for f in os.listdir(model_dir):
        if f.endswith(".safetensors"):
            try:
                fd = os.open(os.path.join(model_dir, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError:
                pass
    return n

def free_bytes():
    """Free memory for allocations. cudaMemGetInfo on an integrated GPU (GB10) does not count reclaimable page cache;
    there the kernel's MemAvailable is the better estimate of what an allocation can get."""
    free, _ = torch.cuda.mem_get_info()
    if torch.cuda.get_device_properties(dev).is_integrated:
        try:
            with open("/proc/meminfo") as fh:
                avail = next(int(l.split()[1]) * 1024 for l in fh if l.startswith("MemAvailable:"))
            free = max(free, avail)
        except (OSError, StopIteration, ValueError):
            pass
    return free

ndrop = drop_weight_cache(args.model)
free0, total = free_bytes(), torch.cuda.mem_get_info()[1]
log(f"weights loaded in {time.time()-t0:.0f}s; GPU free {free0/2**30:.1f} GiB")
# KV sizing: bytes per global page on this rank (target 4-bit + draft bf16)
HKL = head_shard(cfg.i2.heads, cfg.i2.kv_heads, rank, world).hkl   # this rank's KV heads (3 / 3 / 2 on 3 ranks)
def page_bytes():
    tgt = 8 * 2 * PS * HKL * ((64 if args.bits_g == 4 else 128 if args.bits_g == 8 else 256) + (2 * args.kv_ng if args.bits_g != 16 else 0))
    drf = PS * I2Draft.global_bytes_per_token(HKL) if SPEC else 0
    return tgt + drf
fixed = (S + 1) * (40 * 2 * (1024 + 64) * HKL * ((64 if args.bits_l == 4 else 128) + 2) + 48 * 2 * 4096 * 16 * 2 // 1) * 0  # accounted by free-memory probe below
pt_len = args.max_model_len // PS + 2
# measure the fixed per-slot state by constructing the engine with a tiny page pool first
eng = I2Engine(model, dev, rank, world, max_slots=S + 1, num_pages=8, page_size=PS, bits_g=args.bits_g, bits_l=args.bits_l,
               max_pages_per_seq=pt_len, kv_rot=bool(args.kv_rot), kv_ng=args.kv_ng, kv_q16=bool(args.kv_q16),
               max_rows=args.chunk)
draft = I2Draft(args.model + "/mtp.safetensors", eng, dev, rank, world, S + 1, 8, PS) if SPEC else None
if args.b12x:
    from exllamav3.modules.inkling2_b12x import B12xMoE
    tb = time.time()
    attach_moe(eng, model, args.model, cfg, dev, rank, world, sizes=sorted(set((spec_k(b) + 1) * b for b in (1, 2, 4, 8, 16) if b <= S)))
    log(f"b12x fused Trellis MoE attached for verify T={sorted(eng.b12x.sizes)} in {time.time()-tb:.0f}s (tile {eng.b12x.tile})")
drop_weight_cache(args.model)
free1 = free_bytes()
budget = args.kv_gb * 2**30 if args.kv_gb > 0 else max(0.0, free1 - args.reserve_gb * 2**30)
SNAP_BYTES = SlotSnapshots.bytes_per_snapshot(eng, draft) if args.prefix_cache else 0
if args.prefix_cache:
    budget = max(0.0, budget - args.prefix_snapshots * SNAP_BYTES)
NP = max(16, int(budget // page_bytes()))
NP = int(torch.tensor(NP, device=dev).min().item()) if world == 1 else int(NP)
t = torch.tensor([NP], device=dev, dtype=torch.long); dist.all_reduce(t, op=dist.ReduceOp.MIN); NP = int(t.item())
# rebuild the page-based tensors at the final size
for i, (kq, vq, ks, vs) in list(eng.kv.items()):
    if cfg.i2.is_global(i):
        shape = (NP, PS, HKL, kq.shape[-1])
        sshape = shape[:-1] + ((args.kv_ng,) if args.kv_ng > 1 else ())
        eng.kv[i] = (torch.zeros(shape, dtype=kq.dtype, device=dev), torch.zeros(shape, dtype=vq.dtype, device=dev),
                     None if ks is None else torch.zeros(sshape, dtype=ks.dtype, device=dev),
                     None if vs is None else torch.zeros(sshape, dtype=vs.dtype, device=dev))
if SPEC:
    draft.alloc(NP)
snaps = SlotSnapshots(eng, draft, args.prefix_snapshots) if args.prefix_cache else None
torch.cuda.empty_cache()
if args.prefix_cache:
    log(f"prefix cache: {args.prefix_snapshots} snapshots x {SNAP_BYTES/2**20:.1f} MiB per GPU")
log(f"KV pages: {NP} x {PS} = {NP*PS:,} tokens (per GPU {NP*page_bytes()/2**30:.1f} GiB, {page_bytes()/PS:.0f} B/token)")

# ---------------------------------------------------------------------------------------------------------- state
BUCKETS = [b for b in (1, 2, 4, 8, 16) if b <= S]
DUMMY = S                               # scratch slot for padded rows; owns page 0
PROF_AT = int(os.environ.get("I2_PROFILE_AT", "0")); PROF_STEP = 0
BURST_EMIT = torch.zeros((max(1, args.burst), S + 1, 8), dtype=torch.long, device=dev)
BURST_N = torch.zeros((max(1, args.burst), S + 1), dtype=torch.long, device=dev)
anchor_all = torch.zeros(S + 1, dtype=torch.long, device=dev)
d_all = torch.zeros((S + 1, 7), dtype=torch.long, device=dev)
P_all = torch.zeros(S + 1, dtype=torch.int32, device=dev)
temp_all = torch.zeros(S + 1, dtype=torch.float32, device=dev)
eng.page_table[DUMMY, :2] = 0
EOS = 200006
V = cfg.i2.unpadded_vocab
ar8 = torch.arange(8, dtype=torch.int32, device=dev)

def sample_rows(z, temps):
    """z [T, Vshard] fp32 logits shard, temps [T] -> token ids [T] (greedy where temp == 0, Gumbel-max otherwise)"""
    if True:
        u = torch.rand_like(z).clamp_(1e-20, 1.0)
        g = -torch.log(-torch.log(u))
        t = temps[:, None]
        z = torch.where(t > 0, z / t.clamp_min(1e-4) + g, z)
    return eng.global_argmax(z)

class Bucket:
    def __init__(self, B, spec):
        self.B = B
        self.spec = spec and SPEC
        self.K = spec_k(B)
        R = self.K + 1
        self.slots = torch.full((B,), DUMMY, dtype=torch.int32, device=dev)
        self.emit = torch.zeros((B, R), dtype=torch.long, device=dev)
        self.n = torch.zeros(B, dtype=torch.long, device=dev)
        self.taps = torch.empty((B * R if self.spec else B, 8, 4608), dtype=torch.bfloat16, device=dev) if SPEC else None
        self.graph = None

    @torch.no_grad()
    def step(self):
        B, K = self.B, self.K
        R = K + 1
        arR = ar8[:R]
        sl = self.slots.long()
        slot8 = self.slots.repeat_interleave(R)
        nprev = arR.repeat(B); nafter = (K - nprev).to(torch.int32)
        P = P_all[sl]
        anchor = anchor_all[sl]
        temps = temp_all[sl].repeat_interleave(R)
        pos = (P[:, None] + arR[None]).reshape(-1).to(torch.int32)
        nsplit_g = 8
        if self.spec:
            d = d_all[sl][:, :K]
            ids = torch.cat([anchor[:, None], d], 1).reshape(-1)
            bt = Batch(slot8, pos, nprev, True, nsplit_g=nsplit_g, nsplit_l=4, nafter=nafter, seq_rows=R)
            z = eng.forward(ids, bt, taps=self.taps)
            g = sample_rows(z, temps).view(B, R)
            n = (d == g[:, :K]).int().cumprod(1).sum(1)
            draft.store_context(self.taps, slot8, pos)
            newanchor = g.gather(1, n[:, None])[:, 0]
            emit = torch.cat([d, g[:, :1]], 1)
            emit.scatter_(1, n[:, None], newanchor[:, None])
            self.emit.copy_(emit); self.n.copy_(n)
            Pn = P + (n + 1).to(torch.int32)
            f_last = self.taps.view(B, R, 8, 4608)[torch.arange(B, device=dev), n]
            dn = draft.draft(newanchor, self.slots, Pn, f_last, nsplit_g=nsplit_g, K=K)
            anchor_all[sl] = newanchor; P_all[sl] = Pn; d_all[sl, :K] = dn
        else:
            bt = Batch(self.slots, P, torch.zeros(B, dtype=torch.int32, device=dev), False, nsplit_g=NSPLIT_DEC, nsplit_l=4)
            z = eng.forward(anchor, bt, taps=self.taps)
            if SPEC:       # keep the draft's context K/V complete so speculative steps can resume at any time
                draft.store_context(self.taps, self.slots, P)
            tok = sample_rows(z, temp_all[sl])
            self.emit[:, 0] = tok; self.n.zero_()
            anchor_all[sl] = tok; P_all[sl] = P + 1

# speculative buckets (draft K per bucket) and plain buckets (one token per step); --spec-mode picks which exist
sbuckets = {B: Bucket(B, True) for B in BUCKETS} if args.spec_mode != "plain" else {}
buckets = {B: Bucket(B, False) for B in BUCKETS} if args.spec_mode != "spec" else {}
pool = None
def capture(b):
    global pool
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): b.step()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        b.step()
    pool = g.pool()
    b.graph = g
    anchor_all.zero_(); P_all.zero_(); d_all.zero_()

t0 = time.time()
with torch.no_grad():
    for B in sbuckets:
        capture(sbuckets[B])
    for B in buckets:
        capture(buckets[B])

@torch.no_grad()
def calibrate(b, n=6):
    """seconds per replay of a bucket's graph on the dummy slot (every rank replays: the graphs hold collectives)"""
    b.slots.fill_(DUMMY)
    for _ in range(2):
        P_all[DUMMY] = 0; b.graph.replay()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(n):
        P_all[DUMMY] = 0; b.graph.replay()
    torch.cuda.synchronize()
    P_all[DUMMY] = 0
    return (time.perf_counter() - t) / n

# step-time model for --spec-mode auto, refined online with the measured decode steps
T_STEP = {("spec", B): calibrate(b) for B, b in sbuckets.items()}
T_STEP.update({("plain", B): calibrate(b) for B, b in buckets.items()})
log(f"decode graphs captured for buckets {BUCKETS} (spec: {bool(sbuckets)}, draft K {[spec_k(b) for b in BUCKETS]}; plain: {bool(buckets)}) "
    f"in {time.time()-t0:.0f}s; step ms " + ", ".join(f"{m}{B} {1000*v:.1f}" for (m, B), v in sorted(T_STEP.items())) +
    f"; GPU free {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB")

# ---------------------------------------------------------------------------------------------------------- commands
OP_NOOP, OP_PREFILL, OP_DECODE, OP_FORCE, OP_SNAPSHOT, OP_RESTORE, OP_STOP = 0, 1, 2, 3, 4, 5, 9
HDR = torch.zeros(16, dtype=torch.long, device=dev)
IMGIDX = torch.zeros(args.chunk, dtype=torch.long, device=dev)              # rows of a prefill chunk that are image tokens
IMG = torch.zeros((args.chunk, 4608), dtype=torch.bfloat16, device=dev)    # and their features (computed on rank 0)
PAY = torch.zeros(args.max_model_len // PS + 4096 + 64, dtype=torch.long, device=dev)
FPAY = torch.zeros(64, dtype=torch.float32, device=dev)

def bcast(hdr_vals=None, pay=None, fpay=None, img=None):
    """rank 0 fills (hdr_vals, pay, fpay, img); everyone receives; returns (hdr list, payload view, float payload view).
    h[8] = image rows in a prefill chunk: their chunk offsets go to IMGIDX and their features to IMG"""
    if rank == 0:
        HDR.zero_(); HDR[:len(hdr_vals)] = torch.tensor(hdr_vals, dtype=torch.long, device=dev)
    dist.broadcast(HDR, 0)
    h = HDR.tolist()
    npay, nf = h[6], h[7]
    if npay:
        if rank == 0: PAY[:npay] = torch.tensor(pay, dtype=torch.long, device=dev)
        dist.broadcast(PAY[:npay], 0)
    if nf:
        if rank == 0: FPAY[:nf] = torch.tensor(fpay, dtype=torch.float32, device=dev)
        dist.broadcast(FPAY[:nf], 0)
    nimg = h[8]
    if nimg:
        if rank == 0:
            IMGIDX[:nimg].copy_(img[0]); IMG[:nimg].copy_(img[1])
        dist.broadcast(IMGIDX[:nimg], 0)
        dist.broadcast(IMG[:nimg], 0)
    return h, PAY[:npay], FPAY[:nf]

@torch.no_grad()
def do_prefill(h, pay, fpay):
    slot, start, n, last, npg, total_len = h[1], h[2], h[3], h[4], h[5], int(h[7] and 0)
    # pay = [token ids (n)] + [page ids (npg)] (page ids only on the first chunk)
    ids = pay[:n]
    if npg:
        eng.page_table[slot, :npg] = pay[n:n + npg].to(torch.int32)
    pos = torch.arange(start, start + n, dtype=torch.int32, device=dev)
    b = Batch(torch.full((n,), slot, dtype=torch.int32, device=dev), pos, torch.arange(n, dtype=torch.int32, device=dev), True,
              nsplit_g=8, nsplit_l=4)
    taps = torch.empty((n, 8, 4608), dtype=torch.bfloat16, device=dev) if SPEC else None
    nimg = h[8]
    img = (IMGIDX[:nimg], IMG[:nimg]) if nimg else None
    z = eng.forward(ids, b, taps=taps, img=img)
    if SPEC:
        draft.store_context(taps, b.slot, b.pos)
    if last:
        temp = float(fpay[0]) if fpay.numel() else 0.0
        temp_all[slot] = temp
        tok = sample_rows(z[-1:], temp_all[slot:slot + 1])
        anchor_all[slot] = tok[0]; P_all[slot] = start + n
        if SPEC:
            sl = torch.tensor([slot], dtype=torch.int32, device=dev)
            d_all[slot] = draft.draft(tok, sl, P_all[slot:slot + 1], taps[-1:], nsplit_g=8)[0]
        return int(tok[0]) if rank == 0 else None
    return None

@torch.no_grad()
def do_decode(h, pay, fpay):
    """h[4] = number of back-to-back speculative steps (burst; one host sync for all of them)"""
    B, nact, nsteps = h[1], h[2], max(1, h[4])
    b = sbuckets[B] if h[3] else buckets[B]
    sl = pay[:nact].to(torch.int32)
    b.slots.fill_(DUMMY); b.slots[:nact] = sl
    P_all[DUMMY] = 0
    R = b.emit.shape[1]
    global PROF_STEP
    PROF_STEP += nsteps
    if PROF_AT and PROF_AT <= PROF_STEP < PROF_AT + nsteps:
        # diagnostics: 20 replays on every rank with all rows on the dummy slot (live sequences untouched)
        from torch.profiler import profile, ProfilerActivity
        saved = b.slots.clone(); b.slots.fill_(DUMMY)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(20):
                P_all[DUMMY] = 0; b.graph.replay()
            torch.cuda.synchronize()
        P_all[DUMMY] = 0; b.slots.copy_(saved)
        if rank == 0:
            print(f"PROFILE bucket {B} (20 dummy-slot replays):\n" + pr.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=80), flush=True)
    for i in range(nsteps):
        b.graph.replay()
        if rank == 0:
            BURST_EMIT[i, :B, :R].copy_(b.emit); BURST_N[i, :B].copy_(b.n)
    if rank == 0:
        return BURST_EMIT[:nsteps, :B, :R].cpu(), BURST_N[:nsteps, :B].cpu()
    return None

@torch.no_grad()
def do_force(h):
    """h = [OP_FORCE, slot, token, pos]: the slot's next step feeds `token` at position `pos`; the stale drafts are
    replaced by the token itself (they are only proposals: the verify step rejects whatever is wrong)"""
    slot, t, pos = h[1], h[2], h[3]
    anchor_all[slot] = t; P_all[slot] = pos; d_all[slot] = t

@torch.no_grad()
def do_snapshot(h):
    """h = [OP_SNAPSHOT, slot, sid]: copy the slot's ring state (local KV, conv rings, draft local KV) into snapshot sid"""
    snaps.save(h[1], h[2])

@torch.no_grad()
def do_restore(h):
    """h = [OP_RESTORE, slot, sid, cow_src, cow_dst]: copy the page that holds the prefix boundary (if any), then put
    snapshot sid into the slot; the slot's page table (shared pages + the copy + fresh pages) arrives with the first
    prefill chunk"""
    if h[3] >= 0:
        copy_page(eng, draft, h[3], h[4])
    snaps.load(h[2], h[1])

def worker_loop():
    while True:
        h, pay, fpay = bcast()
        op = h[0]
        if op == OP_STOP: break
        if op == OP_PREFILL: do_prefill(h, pay, fpay)
        elif op == OP_DECODE: do_decode(h, pay, fpay)
        elif op == OP_FORCE: do_force(h)
        elif op == OP_SNAPSHOT: do_snapshot(h)
        elif op == OP_RESTORE: do_restore(h)

if rank != 0:
    worker_loop()
    sys.exit(0)

# ====================================================================================================== rank 0 only
from tokenizers import Tokenizer
import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

tok = Tokenizer.from_file(args.model + "/tokenizer.json")
def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def _raise(msg): raise jinja2.TemplateError(msg)
jenv = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=False, lstrip_blocks=False)
jenv.filters["tojson"] = _tojson
jenv.globals["raise_exception"] = _raise
template = jenv.from_string(open(args.chat_template or args.model + "/chat_template.jinja").read())

T_MSG_MODEL, T_TEXT, T_THINK, T_END, T_EOS, T_TOOL = 200001, 200004, 200008, 200010, 200006, 200049
ROLE_TOKENS = (200000, 200001, 200002, 200003)          # <|message_user|> <|message_model|> <|message_system|> <|message_tool|>

def last_message_start(ids):
    """position where the prompt's last message begins (0 if none): a snapshot there serves new sessions with the same
    system prompt and tools, and exact repeats of a prompt"""
    for i in range(len(ids) - 1, 0, -1):
        if ids[i] in ROLE_TOKENS:
            return i
    return 0

pending = queue.Queue()
free_slots = list(range(S))
active = {}                           # slot -> Req
prefilling = []                       # Reqs with remaining prompt
loop_ref = {}
STATS = {'dec_steps': 0, 'dec_time': 0.0, 'dec_rows': 0, 'dec_tok': 0, 'pre_time': 0.0, 'pre_tok': 0, 'idle_time': 0.0,
         'spec_steps': 0, 'plain_steps': 0}
PLAIN_STREAK = [0]
PROBE_LEFT = [0]          # speculative steps left in a probe: the 1st refreshes stale drafts, the 2nd measures
PROBE_GAP = [4]           # plain steps before the next probe: 4, doubling while probes keep confirming plain, up to --probe-every

def choose_spec(reqs, nb):
    """auto mode: speculative step if its expected tokens/s beats a plain step's for this batch"""
    if not sbuckets: return False
    if not buckets: return True
    if any(r.n_ema is None for r in reqs): return True            # still measuring a new request's acceptance
    if PLAIN_STREAK[0] >= PROBE_GAP[0]:                            # re-measure: soon after switching, then less often
        PROBE_LEFT[0] = 2; PROBE_GAP[0] = min(2 * PROBE_GAP[0], args.probe_every)
    if PROBE_LEFT[0] > 0: return True
    Bs = next(b for b in sbuckets if b >= nb); Bp = next(b for b in buckets if b >= nb)
    gain = sum(1.0 + r.n_ema for r in reqs) / len(reqs)          # tokens per row per speculative step
    spec = gain * T_STEP[("plain", Bp)] > T_STEP[("spec", Bs)]
    if spec: PROBE_GAP[0] = 4
    return spec

# ---------------------------------------------------------------------------------------------------------- prefix cache
# reference-counted pages + LRU index of prefix snapshots (server/i2_prefix_cache.py); the callbacks run on every rank
from i2_prefix_cache import PagePool, PrefixCache

def _snapshot_all(slot, sid):
    bcast([OP_SNAPSHOT, slot, sid]); do_snapshot(HDR.tolist())

def _restore_all(slot, sid, cow_src, cow_dst):
    bcast([OP_RESTORE, slot, sid, cow_src, cow_dst]); do_restore(HDR.tolist())

page_pool = PagePool(range(1, NP))      # page 0 belongs to the dummy slot
pcache = PrefixCache(page_pool, PS, args.prefix_snapshots if args.prefix_cache else 0, args.prefix_min,
                     _snapshot_all, _restore_all, STATS)

class Req:
    def __init__(self, ids, max_tokens, temperature, out_q, think_budget = None):
        self.ids, self.max_tokens, self.temperature, self.out_q = ids, max_tokens, temperature, out_q
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.slot = None; self.pages = []; self.done_prompt = 0; self.gen = []; self.t_arrive = time.time()
        self.first_token_time = None; self.finish = None
        self.t_http = self.t_admit = self.t_pre0 = self.t_first_emit = None    # timing (I2_TIMING=1 logs them)
        # thinking budget: at most `think_budget` tokens inside a <|content_thinking|> block, then <|end_message|> is
        # forced (see force_end_thinking); last_pos = KV position of the last token handed to the client
        self.think_budget = think_budget; self.think_mode = False; self.think_count = 0
        self.think_ids = []; self.loop_max = 0.0; self.loop_cut = False   # loop guard: recent thinking tokens, max ratio seen
        self.gone = False                                # the client hung up: drop it from the queue / end it
        self.last_pos = None; self.force_pending = False; self.n_forced = 0
        self.pages_sent = False; self.cached = 0       # prefix cache: page table sent to the ranks; tokens restored
        self.use_cache = True
        self.key_ids = ids                               # prefix-cache key (image runs replaced by image-hash ids)
        self.images = []                                 # [(pixel_values, (rows, cols))] in prompt order
        self.img_pos = []                                # positions of image tokens in ids (sorted)
        self.img_feats = None                            # [len(img_pos), 4608] bf16, computed on rank 0 when first needed
        self.n_ema = None                                # auto spec mode: drafts accepted per speculative step (EMA)
        self.fresh_drafts = True                         # False after a plain step: the slot's drafts are stale
        self.boundary = last_message_start(ids)          # extra snapshot point (prefix cache)

def emit(req, item):
    loop_ref["loop"].call_soon_threadsafe(req.out_q.put_nowait, item)

def drop_gone():
    """11-21: a client that hung up (stopped reply, timed-out retry) used to keep its slot until max_tokens. Queued
    ones are dropped; decoding ones end now. One still reading its prompt is ended at its first decode step, so the
    prefix cache never sees a half-written prompt."""
    while not pending.empty() and pending.queue[0].gone:
        pending.get(); STATS["gone"] = STATS.get("gone", 0) + 1
    for req in [r for r in active.values() if r.gone]:
        STATS["gone"] = STATS.get("gone", 0) + 1
        finish(req, "client_gone")


def try_admit():
    drop_gone()
    while not pending.empty() and free_slots:
        req = pending.queue[0]
        if req.gone:
            pending.get(); continue
        need = (len(req.ids) + req.max_tokens + 24 + PS - 1) // PS
        if need > pt_len - 2 or need > NP - 1:
            pending.get(); emit(req, ("error", "request longer than the KV cache / max_model_len")); continue
        got = pcache.admit(free_slots[0], req.key_ids, need, req.use_cache)    # may restore a cached prefix into the slot
        if got is None:
            break                                            # wait for running requests to free pages
        pending.get()
        req.t_admit = time.time()
        req.slot = free_slots.pop(0)
        req.pages, start = got
        req.done_prompt = req.cached = start
        prefilling.append(req)

TIMING = os.environ.get("I2_TIMING", "0") == "1"
# 11-21: a streamed reply is silent while it waits in the queue, while its prompt is read and while a tool call is
# generated (tool calls are sent whole). Agent clients give up after a few minutes without a chunk (pi: 300 s), so an
# empty chunk goes out after KEEPALIVE seconds of silence. It also lets the server see a client that left while queued.
KEEPALIVE = float(os.environ.get("I2_KEEPALIVE", "15"))

# 11-20: rank 0's resident memory grew ~100 MB per long request (request bodies, prompt text and token lists of 100K+
# token conversations) until the node ran out of memory twice. glibc keeps such freed blocks in the heap; give them back
# with malloc_trim after requests finish: when idle, or under steady load at most every TRIM_EVERY seconds. I2_TRIM=0: off.
import ctypes
_LIBC = ctypes.CDLL("libc.so.6") if os.environ.get("I2_TRIM", "1") == "1" else None
TRIM_EVERY = float(os.environ.get("I2_TRIM_EVERY", "60"))
TRIM = {"pending": 0, "last": 0.0}
STATS["trims"] = 0; STATS["trim_ms_last"] = 0.0; STATS["trim_ms_max"] = 0.0

def maybe_trim(idle):
    if _LIBC is None or not TRIM["pending"]: return
    now = time.time()
    if now - TRIM["last"] < (5.0 if idle else TRIM_EVERY): return
    _LIBC.malloc_trim(0)
    dt = (time.time() - now) * 1000
    TRIM["last"] = time.time(); TRIM["pending"] = 0
    STATS["trims"] += 1; STATS["trim_ms_last"] = round(dt, 1); STATS["trim_ms_max"] = round(max(STATS["trim_ms_max"], dt), 1)

def finish(req, reason):
    req.finish = reason
    if TIMING:
        t = lambda a, b: f"{(b - a):.3f}" if a and b else "-"
        print(f"TIMING {req.id[-8:]} prompt {len(req.ids)} cached {req.cached} gen {len(req.gen)} | http->arrive {t(req.t_http, req.t_arrive)}"
              f" queue {t(req.t_arrive, req.t_admit)} admit->prefill {t(req.t_admit, req.t_pre0)} prefill {t(req.t_pre0, req.first_token_time)}"
              f" ->first emit {t(req.first_token_time, req.t_first_emit)} total {t(req.t_arrive, time.time())}"
              f" | {reason} think {req.think_count} loop_max {req.loop_max:.2f}{' CUT' if req.loop_cut else ''}", flush=True)
    active.pop(req.slot, None)
    free_slots.append(req.slot); pcache.release(req.pages)
    emit(req, ("done", reason))
    TRIM["pending"] += 1

def scheduler():
    last_hb = time.time()
    while True:
        try_admit()
        did = False
        if prefilling:
            req = prefilling[0]
            n = min(args.chunk, len(req.ids) - req.done_prompt)
            if args.prefix_cache and req.done_prompt < req.boundary < req.done_prompt + n:
                n = req.boundary - req.done_prompt        # stop at the last message's start to snapshot it
            first = not req.pages_sent                    # a restored request starts mid-prompt
            if first: req.t_pre0 = time.time()
            req.pages_sent = True
            last = req.done_prompt + n == len(req.ids)
            pay = req.ids[req.done_prompt:req.done_prompt + n] + (req.pages if first else [])
            img = None
            if req.img_pos:
                if req.img_feats is None:                # vision tower once per request, on rank 0
                    pv = torch.cat([x[0] for x in req.images])
                    req.img_feats = VISION.encode(pv, [x[1] for x in req.images])
                lo = bisect.bisect_left(req.img_pos, req.done_prompt); hi = bisect.bisect_left(req.img_pos, req.done_prompt + n)
                if hi > lo:
                    img = (torch.tensor(req.img_pos[lo:hi], dtype=torch.long, device=dev) - req.done_prompt, req.img_feats[lo:hi])
            h = [OP_PREFILL, req.slot, req.done_prompt, n, int(last), len(req.pages) if first else 0, len(pay), 1 if last else 0,
                 0 if img is None else img[0].numel()]
            tp0 = time.time()
            t, _, _ = bcast(h, pay, [req.temperature] if last else None, img)
            res = do_prefill(HDR.tolist(), PAY[:len(pay)], FPAY[:1 if last else 0])
            prev = req.done_prompt
            req.done_prompt += n
            STATS['pre_time'] += time.time() - tp0; STATS['pre_tok'] += n
            if args.prefix_cache and (last or req.done_prompt == req.boundary or
                                      (args.prefix_every and prev // args.prefix_every != req.done_prompt // args.prefix_every)):
                pcache.add(req.slot, req.key_ids, req.done_prompt, req.pages)
            if last:
                prefilling.pop(0)
                req.first_token_time = time.time()
                active[req.slot] = req
                req.last_pos = len(req.ids) - 1          # the first sampled token sits at position len(ids)
                handle_tokens(req, [res])
            did = True
        if active:
            slots = sorted(active)
            nb = len(slots)
            use_spec = choose_spec([active[s] for s in slots], nb)
            B = next(b for b in (sbuckets if use_spec else buckets) if b >= nb)
            # burst of back-to-back steps when nothing waits for admission/prefill; bounded by every active request's
            # page headroom (24 positions past max_tokens) so overshooting steps never write outside its pages
            R = spec_k(B) + 1 if use_spec else 1
            nsteps = 1
            if not prefilling and (pending.empty() or not free_slots) and args.burst > 1 \
                    and not any(r.think_budget is not None and r.think_mode for r in active.values()):
                head = min((r.max_tokens + 24 - R - len(r.gen)) // R for r in active.values())
                nsteps = max(1, min(args.burst, head))
            h = [OP_DECODE, B, nb, int(use_spec), nsteps, 0, nb, 0]
            td0 = time.time()
            bcast(h, slots)
            emit_t, n_t = do_decode(HDR.tolist(), PAY[:nb], None)
            dt = time.time() - td0
            key = ("spec" if use_spec else "plain", B)
            T_STEP[key] = 0.9 * T_STEP[key] + 0.1 * dt / nsteps
            if use_spec:
                STATS['spec_steps'] += nsteps; PLAIN_STREAK[0] = 0; PROBE_LEFT[0] = max(0, PROBE_LEFT[0] - nsteps)
                for i, s in enumerate(slots):
                    r = active.get(s)
                    if r is not None:
                        if r.fresh_drafts:       # drafts made by the previous speculative step: a fair sample
                            n = float(n_t[:, i].float().mean())
                            r.n_ema = n if r.n_ema is None else 0.7 * r.n_ema + 0.3 * n
                        r.fresh_drafts = True
            else:
                STATS['plain_steps'] += nsteps; PLAIN_STREAK[0] += nsteps
                for s in slots:
                    if s in active: active[s].fresh_drafts = False
            STATS['dec_steps'] += nsteps; STATS['dec_time'] += dt; STATS['dec_rows'] += nb * nsteps
            STATS['dec_tok'] += int(n_t[:, :nb].sum()) + nb * nsteps
            for st in range(nsteps):
                for i, s in enumerate(slots):
                    req = active.get(s)
                    if req is None: continue
                    k = int(n_t[st, i]) + 1
                    handle_tokens(req, emit_t[st, i, :k].tolist())
            did = True
        maybe_trim(idle = not did and not active and not prefilling)
        if not did:
            time.sleep(0.002)
            if time.time() - last_hb > 1.0:
                bcast([OP_NOOP]); last_hb = time.time()

def force_end_thinking(req, pos):
    """Thinking budget reached: put <|end_message|> at KV position `pos` as the slot's next anchor on every rank.
    Positions >= pos that speculative steps already wrote are overwritten by the following steps (every cache - paged
    global KV, SWA rings, conv rings, draft context - is indexed by position), so rewinding the slot is exact."""
    bcast([OP_FORCE, req.slot, T_END, pos])
    do_force(HDR.tolist())
    req.force_pending = False; req.think_mode = False; req.n_forced += 1
    req.gen.append(T_END); req.last_pos = pos
    emit(req, ("tok", T_END))
    if len(req.gen) >= req.max_tokens:
        finish(req, "length")

LOOP_N = 32
def loop_ratio(ids, n=LOOP_N):
    """fraction of the n-token runs in ids that already appeared earlier in ids"""
    seen, hit = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in seen: hit += 1
        else: seen.add(g)
    return hit / max(1, len(ids) - n + 1)

def check_loop(req, t):
    """11-21: at high reasoning effort a thought can fall into a loop and repeat itself until max_tokens (60K+ tokens,
    most of an hour, no answer). Every 256 thinking tokens past one window, measure the repetition of the last window;
    over the guard the thinking block is ended exactly like a spent thinking budget, and the model answers."""
    W = args.think_loop_window
    req.think_ids.append(t)
    if len(req.think_ids) > 2 * W: del req.think_ids[:-W]
    if req.think_count < W or req.think_count % 256 or req.loop_cut: return
    r = loop_ratio(req.think_ids[-W:])
    req.loop_max = max(req.loop_max, r)
    if r >= args.think_loop_guard:
        req.loop_cut = True; STATS["loop_cut"] = STATS.get("loop_cut", 0) + 1
        req.think_budget = req.think_count          # the budget check right after this ends the thinking block
        print(f"thinking loop: {req.id[-8:]} at {req.think_count} thinking tokens, {r:.0%} of the last {W} repeated; "
              f"thinking ended", flush=True)

def handle_tokens(req, toks):
    """toks: the tokens one step produced for this slot, in order; token i sits at KV position last_pos + 1 + i.
    All but the last are already in the KV cache (accepted drafts); the last is the slot's next anchor (written by the
    next step)."""
    for i, t in enumerate(toks):
        if req.slot not in active or req.finish: return
        if req.force_pending:
            # the budget was hit on the previous step's anchor, which this step has now written: drop this step's
            # tokens and force the end of the thinking block right after it
            force_end_thinking(req, req.last_pos + 1); return
        if t in (T_EOS, 199999):
            finish(req, "stop"); return
        prev = req.gen[-1] if req.gen else None
        req.gen.append(t); req.last_pos += 1
        if t == T_THINK and prev == T_MSG_MODEL:
            req.think_mode = True          # the count is cumulative: a second thinking block gets no fresh budget
        elif t == T_END:
            req.think_mode = False
        elif req.think_mode:
            req.think_count += 1
            if args.think_loop_guard > 0:
                check_loop(req, t)
        emit(req, ("tok", t))
        if req.t_first_emit is None: req.t_first_emit = time.time()
        if len(req.gen) >= req.max_tokens:
            finish(req, "length"); return
        if req.think_mode and req.think_budget is not None and req.think_count >= req.think_budget:
            if i + 1 < len(toks):
                force_end_thinking(req, req.last_pos + 1)      # next token's position already holds a draft: overwrite
            else:
                req.force_pending = True                     # t is the next anchor, not in the cache yet
            return

# ---------------------------------------------------------------------------------------------------------- output parsing
class Parser:
    """Token-level parser of Inkling's typed blocks -> OpenAI deltas (reasoning_content / content / tool_calls)"""
    def __init__(self):
        self.mode = None; self.buf = []; self.sent = 0; self.tool_idx = 0
    def _text(self):
        return tok.decode(self.buf, skip_special_tokens=False)
    def feed(self, t):
        out = []
        if t == T_MSG_MODEL:
            self.mode = "hdr"; self.buf = []; self.sent = 0
        elif t in (T_TEXT, T_THINK, T_TOOL) and self.mode in ("hdr", None):
            self.mode = {T_TEXT: "text", T_THINK: "think", T_TOOL: "tool"}[t]; self.buf = []; self.sent = 0
        elif t == T_END:
            if self.mode == "tool":
                body = self._text()
                try:
                    obj = json.loads(body)
                    out.append(("tool", {"index": self.tool_idx, "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                                         "function": {"name": obj.get("name", ""), "arguments": json.dumps(obj.get("args", {}))}}))
                except Exception:
                    out.append(("content", body))
                self.tool_idx += 1
            else:
                out += self._flush(final=True)
            self.mode = None; self.buf = []; self.sent = 0
        elif self.mode in ("text", "think"):
            self.buf.append(t)
            out += self._flush()
        elif self.mode == "tool":
            self.buf.append(t)
        return out
    def _flush(self, final=False):
        s = self._text()
        if not final and s.endswith("�"):
            return []
        d = s[self.sent:]; self.sent = len(s)
        if not d: return []
        return [("reasoning" if self.mode == "think" else "content", d)]
    def finish(self):
        out = []
        if self.mode in ("text", "think"): out += self._flush(final=True)
        return out

def render_prompt(messages, tools, reasoning_effort):
    kw = dict(messages=messages, tools=tools or None, add_generation_prompt=True)
    if reasoning_effort is not None: kw["reasoning_effort"] = reasoning_effort
    return template.render(**kw)

def encode(text):
    return tok.encode(text, add_special_tokens=False).ids

app = FastAPI()

@app.get("/health")
async def health(): return {"status": "ok", "stats": STATS, "spec": SPEC, "kv_tokens": NP * PS, "active": len(active), "queued": pending.qsize(),
                            "prefix": pcache.info()}

@app.get("/admin/expert_load")
async def expert_load(save: int = 0, reset: int = 0):
    """routing counts per layer and checkpoint expert since start (or the last reset), when the server runs with
    I2_EXPERT_COUNT=1: the expert_load.json that balances a mixed-rate build's experts over the ranks. save=1 also
    writes /work/logs/expert_load-<time>.json on rank 0; reset=1 zeroes the counts after reading them"""
    layers = {str(L.idx): L.mlp.expert_counts(reset=bool(reset)) for L in model.layers if hasattr(L.mlp, "expert_counts")}
    layers = {k: v for k, v in layers.items() if v is not None}
    if not layers:
        return {"error": "expert counting is off (start the server with -e I2_EXPERT_COUNT=1)"}
    res = {"layers": layers, "time": time.strftime("%F %T"), "stats": dict(STATS)}
    totals = sorted({sum(v) for v in layers.values()})
    summary = {"layers_counted": len(layers), "rows_x_topk_per_layer": totals[:3] + (["..."] if len(totals) > 3 else [])}
    if save:
        path = f"/work/logs/expert_load-{time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(path, "w") as fh:
            json.dump(res, fh)
        return {"saved": path, **summary}
    return {**summary, **res}

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local",
                                                         "max_model_len": args.max_model_len}]}

@app.post("/tokenize")
@app.post("/v1/tokenize")
async def tokenize(request: Request):
    """vLLM-style token count (after 0xSero's 8ba4513): {"prompt": text} or {"messages": [...], "tools": ...,
    "reasoning_effort": ...}, rendered with the chat template as a chat request would be. Images count with their
    expanded placeholder tokens, so `count` is what the prompt will take in the KV cache."""
    body = await request.json()
    urls = [] if VISION is not None else None
    if body.get("messages"):
        try:
            text = render_prompt(norm_messages(body["messages"], urls), body.get("tools"), body.get("reasoning_effort"))
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code = 400)
    else:
        text = body.get("prompt") or body.get("content") or body.get("text") or ""
        if isinstance(text, list): text = text[0] if text else ""
    ids = encode(text)
    if urls:
        from i2_vision import expand_image_tokens
        counts = []
        for u in urls:
            im, _ = VISION.load_image(u)
            _, g = VISION.prepare(im)
            counts.append(g[0] * g[1])
        ids, _ = expand_image_tokens(ids, counts)
    return {"tokens": list(ids), "count": len(ids), "max_model_len": args.max_model_len}

def norm_messages(msgs, images=None):
    """OpenAI messages -> template input. Image parts become {"type": "image"} (the template emits one placeholder each)
    and their URLs are appended to `images` in prompt order."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "developer":                 # OpenAI's newer name for the system role; the template knows "system"
            m["role"] = "system"
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"): parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") in ("image_url", "input_image", "image"):
                    if images is None:
                        raise ValueError("this server was started without --vision")
                    u = p.get("image_url")
                    u = u.get("url") if isinstance(u, dict) else (u or p.get("url") or p.get("image"))
                    images.append(u); parts.append({"type": "image"})
                elif isinstance(p, str): parts.append(p)
            m["content"] = parts
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try: tc["function"]["arguments"] = json.loads(a)
                except Exception: tc["function"]["arguments"] = {}
        out.append(m)
    return out

async def run_request(body, chat):
    loop = asyncio.get_running_loop()
    loop_ref["loop"] = loop
    urls = [] if VISION is not None else None
    if chat:
        prompt = render_prompt(norm_messages(body["messages"], urls), body.get("tools"), body.get("reasoning_effort"))
    else:
        prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = encode(prompt)
    images, digests = [], []
    if urls:
        from i2_vision import expand_image_tokens, cache_key_ids
        for u in urls:
            im, dg = VISION.load_image(u)
            images.append(VISION.prepare(im)); digests.append(dg)
        counts = [g[0] * g[1] for _, g in images]
        ids, img_pos = expand_image_tokens(ids, counts)
        key_ids = cache_key_ids(ids, img_pos, counts, digests)
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or args.default_max_tokens)
    max_tokens = max(1, min(max_tokens, args.max_model_len - len(ids) - 32))   # never more than the window has left
    temp = float(body.get("temperature", 0.0) or 0.0)
    # hard cap on thinking tokens (the template's reasoning_effort only asks the model to think less)
    tb = body.get("thinking_budget", body.get("max_thinking_tokens"))
    if tb is None and args.default_thinking_budget >= 0:
        tb = args.default_thinking_budget
    think_budget = None if tb is None or int(tb) < 0 else int(tb)
    if args.answer_reserve > 0:                    # leave room for the answer however long the thought runs
        cap = max(0, max_tokens - min(args.answer_reserve, max_tokens // 4))
        think_budget = cap if think_budget is None else min(think_budget, cap)
    out_q = asyncio.Queue()
    req = Req(ids, max_tokens, temp, out_q, think_budget)
    req.use_cache = body.get("prefix_cache", True) is not False      # "prefix_cache": false skips the lookup
    if images:
        req.images, req.img_pos, req.key_ids = images, img_pos, key_ids
    pending.put(req)
    return req, out_q, len(ids)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t_http = time.time()
    body = await request.json()
    try:
        req, out_q, nprompt = await run_request(body, True)
    except (ValueError, OSError, jinja2.TemplateError) as e:   # bad image / unsupported content / message the template rejects
        return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})
    req.t_http = t_http
    stream = bool(body.get("stream"))
    created = int(time.time())
    parser = Parser()
    def usage():
        return {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen),
                "prompt_tokens_details": {"cached_tokens": req.cached}}
    def chunk(delta, finish=None):
        return {"id": req.id, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    async def events():
        sent_role = False
        has_tool = False
        try:
            async for item in _events():
                yield item
        finally:
            if not getattr(req, "finish", None):
                req.gone = True                       # generator closed before "done": the client went away
                state = "active" if req.slot in active and active.get(req.slot) is req else \
                        "prefilling" if req in prefilling else "queued"
                print(f"client gone: {req.id[-8:]} {state}, prompt {len(req.ids)}, generated {len(req.gen)}", flush=True)
    async def _events():
        sent_role = False
        has_tool = False
        last = time.time()
        while True:
            try:
                kind, val = await asyncio.wait_for(out_q.get(), max(0.5, KEEPALIVE - (time.time() - last)))
            except asyncio.TimeoutError:
                last = time.time(); yield {}, None     # keep-alive: queued, reading the prompt, or no token for a while
                continue
            if kind == "tok":
                for k, v in parser.feed(val):
                    d = {}
                    if not sent_role: d["role"] = "assistant"; sent_role = True
                    if k == "reasoning": d["reasoning_content"] = v
                    elif k == "content": d["content"] = v
                    elif k == "tool": d["tool_calls"] = [v]; has_tool = True
                    last = time.time(); yield d, None
                if time.time() - last >= KEEPALIVE:     # tokens arrive but nothing is sendable (a tool call in progress)
                    last = time.time(); yield {}, None
            elif kind == "done":
                for k, v in parser.finish():
                    yield ({"reasoning_content": v} if k == "reasoning" else {"content": v}), None
                yield {}, ("tool_calls" if has_tool else val)
                return
            elif kind == "error":
                yield {"content": ""}, "error"
                return
    if stream:
        async def gen():
            async for d, fin in events():
                yield f"data: {json.dumps(chunk(d, fin))}\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                u = chunk({}); u["choices"] = []; u["usage"] = usage()
                yield f"data: {json.dumps(u)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    content, reasoning, tools, fin = "", "", [], "stop"
    async for d, f in events():
        content += d.get("content", ""); reasoning += d.get("reasoning_content", "")
        tools += d.get("tool_calls", [])
        if f: fin = f
    msg = {"role": "assistant", "content": content or None}
    if reasoning: msg["reasoning_content"] = reasoning
    if tools: msg["tool_calls"] = tools
    return {"id": req.id, "object": "chat.completion", "created": created, "model": args.served_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": usage()}

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    try:
        req, out_q, nprompt = await run_request(body, False)
    except (ValueError, OSError, jinja2.TemplateError) as e:
        return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})
    text, fin = "", "stop"
    while True:
        kind, val = await out_q.get()
        if kind == "tok": pass
        elif kind in ("done", "error"):
            fin = val; break
    text = tok.decode(req.gen, skip_special_tokens=True)
    return {"id": req.id, "object": "text_completion", "created": int(time.time()), "model": args.served_name,
            "choices": [{"index": 0, "text": text, "finish_reason": fin}],
            "usage": {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen)}}

threading.Thread(target=scheduler, daemon=True).start()
log(f"serving on http://{args.host}:{args.port}  (spec decoding {'ON' if SPEC else 'OFF'}, slots {S}, buckets {BUCKETS})")
uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
