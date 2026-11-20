"""
i2q: building blocks of the routed-expert quantizer for Inkling-2-Small (driver: quantize_experts.py).

Plain PyTorch. The EXL3 quantizer is passed in as an object, so every phase also runs on the CPU with a stand-in
quantizer (spark/quant/test_i2q_cpu.py checks the bookkeeping that way, bit for bit).

Pipeline per decoder layer L (layer by layer, like GPTQ: the inputs of layer L come from layers 0..L-1 with their
experts already quantized, so the error of earlier layers is part of what layer L is calibrated on):

  attn     for every calibration row: attention half of the layer, the MoE input x, the residual and the routing
           (top-8 experts and weights). Also sum x^T x over the training rows (the layer-wide input covariance).
  experts  for one range of experts (one process per GPU): per expert, the Hessians of its inputs over the training
           tokens routed to it (gate/up: x; down: the expert's activations under the source weights), shrunk toward
           a prior when few tokens reach it, then EXL3 quantization of gate, up and down.
  moe      for every row: the routed experts with the quantized weights, the layer's output -> input of layer L + 1.
           Held-out rows are also run with the source experts on the same input (layer-local error), and a reference
           track of the held-out rows runs every layer with the source experts (for the final KL divergence).

Two tracks of state, both token-major on disk (bf16 stored as int16):
  q   every calibration row, quantized experts in all layers done so far
  r   held-out rows only, source experts everywhere (reference)
Block attention residuals: the input of every boundary layer (L % 6 == 0) becomes a finished block, kept as a file.
"""
from __future__ import annotations
import importlib.util, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

PROJ = ("gate_proj", "up_proj", "down_proj")
PER_EXPERT_LEVELS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)        # v2: the kernels' rates, half-bit steps


# ----------------------------------------------------------------------------------------------------------------------
# Inkling-2 reference ops (0xSero's overlay module, numerics checked against transformers)

def load_i2():
    """exllamav3.modules.inkling2 when the overlay is installed, else the overlay file itself (CPU tests)"""
    try:
        from exllamav3.modules import inkling2 as m
        return m
    except Exception:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "overlay", "exllamav3", "modules", "inkling2.py")
        spec = importlib.util.spec_from_file_location("inkling2_overlay", p)
        m = importlib.util.module_from_spec(spec)
        import sys
        sys.modules["inkling2_overlay"] = m                 # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(m)
        return m


I2 = load_i2()


def expert_key(L: int, e: int, proj: str) -> str:
    return f"model.llm.layers.{L}.mlp.experts.{e}.{proj}"


# ----------------------------------------------------------------------------------------------------------------------
# Calibration rows and on-disk state

class Calib:
    """calib.npz: ids int32 [R, S] (token ids, fixed row length), heldout bool [R]"""

    def __init__(self, path: str):
        z = np.load(path)
        self.ids = z["ids"].astype(np.int64)
        self.held = z["heldout"].astype(bool)
        self.R, self.S = self.ids.shape
        self.r_rows = np.nonzero(self.held)[0]                  # reference track: held-out rows, in order
        self.r_pos = {int(r): j for j, r in enumerate(self.r_rows)}
        self.train_rows = np.nonzero(~self.held)[0]

    def rows(self, track: str, part: int, parts: int):
        """rows of a track handled by worker `part` of `parts` (round robin); entries are (row, slot)"""
        if track == "q":
            rows = [(int(r), int(r)) for r in range(self.R)]
        else:
            rows = [(int(r), j) for j, r in enumerate(self.r_rows)]
        return rows[part::parts]

    def n_tokens(self, track: str) -> int:
        return (self.R if track == "q" else len(self.r_rows)) * self.S

    def train_token_index(self) -> torch.Tensor:
        """token indices (q track) of the training rows"""
        base = torch.from_numpy(self.train_rows).long() * self.S
        return (base[:, None] + torch.arange(self.S)[None, :]).reshape(-1)


_NP = {torch.bfloat16: np.int16, torch.float32: np.float32, torch.int16: np.int16}


class Mat:
    """token-major matrix on disk [N, C]; bf16 is stored as int16"""

    def __init__(self, path: str, N: int, C: int, dtype, create: bool = False):
        if create:
            os.makedirs(os.path.dirname(path), exist_ok = True)
        mode = "w+" if create else "r+"
        self.mm = np.memmap(path, dtype = _NP[dtype], mode = mode, shape = (N, C))
        self.t = torch.from_numpy(self.mm)
        self.dtype, self.path = dtype, path

    def read(self, idx, device):
        x = self.t[idx]
        if self.dtype == torch.bfloat16:
            x = x.view(torch.bfloat16)
        return x.to(device)

    def write(self, idx, x: torch.Tensor):
        x = x.detach().to("cpu", self.dtype)
        if self.dtype == torch.bfloat16:
            x = x.view(torch.int16)
        self.t[idx] = x

    def flush(self):
        self.mm.flush()


class Work:
    """Layout of the work directory"""

    def __init__(self, root: str):
        self.root = root
        for d in ("state", "tmp", "out", "logs"):
            os.makedirs(os.path.join(root, d), exist_ok = True)

    def state(self, name):
        return os.path.join(self.root, "state", name)

    def tmp(self, L, name = ""):
        d = os.path.join(self.root, "tmp", f"L{L:02d}")
        os.makedirs(d, exist_ok = True)
        return os.path.join(d, name)

    def out(self, name):
        return os.path.join(self.root, "out", name)

    # state files
    def h(self, track, L):
        return self.state(f"{track}_h{L:02d}.bin")

    def b(self, track, k):
        return self.state(f"{track}_b{k}.bin")


def row_slice(slot: int, S: int):
    return slice(slot * S, (slot + 1) * S)


# ----------------------------------------------------------------------------------------------------------------------
# Phase: embeddings (input of layer 0)

@torch.inference_mode()
def phase_embed(cfg, src, calib: Calib, work: Work, device, part: int, parts: int, adt = torch.bfloat16, log = print):
    E = src.get("model.llm.embed.weight", device)
    En = src.get("model.llm.embed_norm.weight", device)
    D = cfg.hidden
    for track in ("q", "r"):
        N = calib.n_tokens(track)
        if N == 0:
            continue
        if part == 0 and not os.path.exists(work.h(track, 0) + ".ready"):
            Mat(work.h(track, 0), N, D, torch.bfloat16, create = True).flush()
            open(work.h(track, 0) + ".ready", "w").close()
    _barrier_files([work.h(t, 0) + ".ready" for t in ("q", "r") if calib.n_tokens(t)])
    for track in ("q", "r"):
        N = calib.n_tokens(track)
        if N == 0:
            continue
        H = Mat(work.h(track, 0), N, D, torch.bfloat16)
        for row, slot in calib.rows(track, part, parts):
            ids = torch.from_numpy(calib.ids[row]).to(device)
            H.write(row_slice(slot, calib.S), I2.embed_tokens(E, En, ids, cfg.eps, adt))
        H.flush()
    log(f"embed part {part}/{parts} done")


def _barrier_files(paths, timeout = 600):
    t0 = time.time()
    while not all(os.path.exists(p) for p in paths):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {paths}")
        time.sleep(0.2)


# ----------------------------------------------------------------------------------------------------------------------
# Phase: attention half + routing

def tmp_mats(work: Work, L: int, track: str, N: int, D: int, k: int, create: bool):
    return {
        "x": Mat(work.tmp(L, f"{track}_x.bin"), N, D, torch.bfloat16, create),
        "resid": Mat(work.tmp(L, f"{track}_resid.bin"), N, D, torch.float32, create),
        "sel": Mat(work.tmp(L, f"{track}_sel.bin"), N, k, torch.int16, create),
        "rw": Mat(work.tmp(L, f"{track}_rw.bin"), N, k, torch.float32, create),
    }


def load_blocks(work: Work, track: str, nb: int, N: int, D: int):
    return [Mat(work.b(track, k), N, D, torch.bfloat16) for k in range(nb)]


@torch.inference_mode()
def phase_attn(L: int, cfg, src, calib: Calib, work: Work, device, part: int, parts: int, adt = torch.bfloat16,
               log = print, tracks = ("q", "r")):
    """tracks: "q" (all rows, quantized experts so far), "r" (held-out rows, source experts) and the ablation tracks
    a<k> (held-out rows, see ablate_init), which run like "r" """
    W = I2.LayerWeights(src, cfg, L, device, adt)
    D, k = cfg.hidden, cfg.topk
    nb = cfg.num_blocks_before(L)
    Hs = torch.zeros((D, D), dtype = torch.float32, device = device)
    n_hs = 0
    t0 = time.time()
    for track in tracks:
        N = calib.n_tokens(track)
        if N == 0:
            continue
        if part == 0 and not os.path.exists(work.tmp(L, f"{track}_x.bin.ready")):
            tmp_mats(work, L, track, N, D, k, create = True)
            open(work.tmp(L, f"{track}_x.bin.ready"), "w").close()
        _barrier_files([work.tmp(L, f"{track}_x.bin.ready")])
        T = tmp_mats(work, L, track, N, D, k, create = False)
        Hin = Mat(work.h(track, L), N, D, torch.bfloat16)
        B = load_blocks(work, track, nb, N, D)
        for row, slot in calib.rows(track, part, parts):
            sl = row_slice(slot, calib.S)
            hidden = Hin.read(sl, device)
            blocks = torch.stack([b.read(sl, device) for b in B]) if nb else None
            a = I2.layer_attention(W, hidden, blocks, adt)
            resid, x = I2.layer_mlp_input(W, hidden, blocks, a, adt)
            sel, rw = I2.route(W, x)
            T["x"].write(sl, x)
            T["resid"].write(sl, resid)
            T["sel"].write(sl, sel.to(torch.int16))
            T["rw"].write(sl, rw.float())
            if track == "q" and not calib.held[row]:
                xf = x.float()
                Hs.addmm_(xf.t(), xf)
                n_hs += x.shape[0]
        for m in T.values():
            m.flush()
    torch.save({"H": Hs.cpu(), "n": n_hs}, work.tmp(L, f"Hs-{part}.pt"))
    log(f"L{L:02d} attn part {part}/{parts}: {time.time() - t0:.1f}s, {n_hs} training tokens")


# ----------------------------------------------------------------------------------------------------------------------
# Source experts for a range [e0, e1)

class SourceExpertRange:
    """Like inkling2.SourceExperts, for experts e0..e1-1 only (packed NVFP4 / MXFP8 on `device`)"""

    def __init__(self, src, cfg, L: int, e0: int, e1: int, device):
        self.cfg, self.L, self.e0, self.e1, self.device = cfg, L, e0, e1, device
        p = f"model.llm.layers.{L}.mlp.experts."
        self.nvfp4 = src.has(p + "w13_weight_amax")
        s = lambda k: src.slice(p + k, e0, e1, device)
        self.w13, self.s13, self.w2, self.s2 = s("w13_weight"), s("w13_weight_scale"), s("w2_weight"), s("w2_weight_scale")
        if self.nvfp4:
            self.g13 = s("w13_weight_amax").float() / src.get(p + "w13_weight_outer_scale_denominator", device).float()
            self.g2 = s("w2_weight_amax").float() / src.get(p + "w2_weight_outer_scale_denominator", device).float()

    def w13_f32(self, e):
        i = e - self.e0
        w = I2.dequant_nvfp4(self.w13[i], self.s13[i], self.g13[i]) if self.nvfp4 else I2.dequant_mxfp8(self.w13[i], self.s13[i])
        return I2.SourceExperts.deinterleave(w)

    def w2_f32(self, e):
        i = e - self.e0
        return I2.dequant_nvfp4(self.w2[i], self.s2[i], self.g2[i]) if self.nvfp4 else I2.dequant_mxfp8(self.w2[i], self.s2[i])


# ----------------------------------------------------------------------------------------------------------------------
# Hessians

def routed_tokens(sel: torch.Tensor, token_index: torch.Tensor, E: int):
    """sel [n, k] (experts of the tokens token_index) -> per expert: sorted token indices (int64 tensors)"""
    k = sel.shape[1]
    flat = sel.reshape(-1).long()
    order = torch.argsort(flat, stable = True)
    counts = torch.bincount(flat, minlength = E).tolist()
    tok = token_index[order // k]
    out, pos = [], 0
    for n in counts:
        out.append(torch.sort(tok[pos:pos + n]).values)
        pos += n
    return out


def routing_share(sel: torch.Tensor, rw: torch.Tensor, E: int) -> np.ndarray:
    """each expert's share of the summed squared routing weights over the tokens given (sel, rw: [n, k]): the weight of
    its quantization error in the layer's output error"""
    w = torch.bincount(sel.reshape(-1).long(), weights = rw.reshape(-1).double().square(), minlength = E)
    w = w.numpy().astype(np.float64)
    return w / max(float(w.sum()), 1e-300)


def expert_bits(share, avg: float, levels = PER_EXPERT_LEVELS):
    """bits per expert on `levels` (half-bit steps) with mean exactly `avg`, minimizing sum_e share_e * 4^-(K_e - 2):
    the relative error of an EXL3 tensor halves per half bit (measured on this model, 2.0 -> 2.5 -> 4.0). Greedy over
    half-bit steps, which is optimal for this convex model; deterministic (ties go to the lower expert id)."""
    import heapq
    lv = sorted(float(x) for x in levels)
    assert all(abs(b - a - 0.5) < 1e-9 for a, b in zip(lv, lv[1:])), "levels must be consecutive half-bit steps"
    E = len(share)
    assert lv[0] - 1e-9 <= avg <= lv[-1] + 1e-9, (avg, lv)
    steps = round((avg - lv[0]) * 2 * E)
    assert abs(steps - (avg - lv[0]) * 2 * E) < 1e-6, "the mean must be reachable in half-bit steps"
    r = lambda K: 4.0 ** (-(K - 2.0))
    idx = [0] * E
    heap = [(-float(share[e]) * (r(lv[0]) - r(lv[1])), e) for e in range(E)]
    heapq.heapify(heap)
    for _ in range(steps):
        _, e = heapq.heappop(heap)
        idx[e] += 1
        if idx[e] + 1 < len(lv):
            heapq.heappush(heap, (-float(share[e]) * (r(lv[idx[e]]) - r(lv[idx[e] + 1])), e))
    return [lv[i] for i in idx]


def shrink_gate_up(Hx: torch.Tensor, n: int, Hs_mean: torch.Tensor, n0: float) -> torch.Tensor:
    """mean input covariance of one expert, shrunk toward the layer-wide one: (sum x x^T + n0 * Hs) / (n + n0)"""
    return (Hx + n0 * Hs_mean) / (n + n0)


def shrink_down(Ha: torch.Tensor, n: int, P: torch.Tensor, n0: float, m: float = 256.0) -> torch.Tensor:
    """
    Down projection: the expert's own activations, no layer-wide counterpart. Shrink toward a diagonal target whose
    per-channel energy comes from the routed tokens, blended with a proxy sample P (the expert run on random training
    tokens, pseudo-count m) so that it is defined when few or no tokens are routed.
    """
    dp = torch.diagonal(P)
    if n > 0:
        dt = (torch.diagonal(Ha) + m * dp) / (n + m)
    else:
        dt = dp
    return (Ha + n0 * torch.diag_embed(dt)) / (n + n0)


def h_data(H: torch.Tensor, key: str, n: int, device) -> dict:
    """Hessian dict in the form exllamav3's quantize_exl3 expects (H already the mean; count 1)"""
    return {"H": H, "count": 1, "finalized": False, "num_total": int(n), "first_key": key,
            "inf_nan": torch.zeros(2, dtype = torch.long, device = device), "device": device}


def no_tf32():
    class _C:
        def __enter__(self):
            self.p = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        def __exit__(self, *a):
            torch.backends.cuda.matmul.allow_tf32 = self.p
    return _C()


# ----------------------------------------------------------------------------------------------------------------------
# Phase: per-expert Hessians and quantization of one expert range

@torch.inference_mode()
def phase_experts(L: int, e0: int, e1: int, cfg, src, calib: Calib, work: Work, device, quantizer, bits: dict,
                  n0: float = 1024.0, group: int = 8, proxy_tokens: int = 512, adt = torch.bfloat16, seed: int = 0,
                  log = print, per_expert: str | None = None):
    """per_expert (v2): comma-separated levels; each expert gets its own bits (gate/up and down separately) by its
    share of the layer's routing (expert_bits), the layer means staying those of `bits`. Experts are processed in order
    of their bits so that a batch shares one rate where possible (the batch quantizer needs one K per call)."""
    D, I, k = cfg.hidden, cfg.inter, cfg.topk
    Nq = calib.n_tokens("q")
    T = tmp_mats(work, L, "q", Nq, D, k, create = False)
    hs = [torch.load(os.path.join(work.tmp(L), f), map_location = "cpu") for f in sorted(os.listdir(work.tmp(L)))
          if f.startswith("Hs-") and f.endswith(".pt")]
    n_train = sum(h["n"] for h in hs)
    Hs_mean = (sum(h["H"] for h in hs) / max(n_train, 1)).to(device)
    tidx = calib.train_token_index()
    lists = routed_tokens(T["sel"].t[tidx], tidx, cfg.experts)
    g = torch.Generator().manual_seed(seed * 1000 + L)
    proxy_idx = tidx[torch.randperm(tidx.numel(), generator = g)[:proxy_tokens]].sort().values
    x_proxy = T["x"].read(proxy_idx, device)
    ex = SourceExpertRange(src, cfg, L, e0, e1, device)
    E = cfg.experts
    if per_expert:
        levels = tuple(float(x) for x in per_expert.split(","))
        share = routing_share(T["sel"].t[tidx], T["rw"].t[tidx], E)
        K_gu_e = expert_bits(share, float(bits["gate_up"]), levels)
        K_d_e = expert_bits(share, float(bits["down"]), levels)
    else:
        share, K_gu_e, K_d_e = None, [bits["gate_up"]] * E, [bits["down"]] * E
    order = sorted(range(e0, e1), key = lambda e: (K_gu_e[e], K_d_e[e], e))
    tensors, stats = {}, {}
    t0, done = time.time(), 0
    for g0 in range(0, len(order), group):
        es = order[g0:g0 + group]
        gu, dn = {}, {}                                   # expert -> (weights, H data, seeds, keys)
        for e in es:
            idx = lists[e]
            n = idx.numel()
            W13, W2 = ex.w13_f32(e), ex.w2_f32(e)                      # [2I, D], [D, I] fp32
            W13a = W13.to(adt)
            with no_tf32():
                Hx = torch.zeros((D, D), dtype = torch.float32, device = device)
                Ha = torch.zeros((I, I), dtype = torch.float32, device = device)
                for c0 in range(0, n, 16384):
                    xe = T["x"].read(idx[c0:c0 + 16384], device)
                    a = I2.expert_act(F.linear(xe, W13a), cfg, adt)        # as moe_forward with the source weights
                    xf, af = xe.float(), a.float()
                    Hx.addmm_(xf.t(), xf)
                    Ha.addmm_(af.t(), af)
                ap = I2.expert_act(F.linear(x_proxy, W13a), cfg, adt).float()
                P = ap.t() @ ap / ap.shape[0]
            H13 = shrink_gate_up(Hx, n, Hs_mean, n0)
            H2 = shrink_down(Ha, n, P, n0)
            del Hx, Ha, P
            kg, ku, kd = (expert_key(L, e, p) for p in PROJ)
            hd = h_data(H13, kg, n, device)
            gu[e] = ([W13[:I].t().contiguous(), W13[I:].t().contiguous()],     # exllamav3 layout: (in, out)
                     [hd, hd], [_seed(seed, L, e, 0), _seed(seed, L, e, 1)], [kg, ku])   # gate and up share inputs and H
            dn[e] = ([W2.t().contiguous()], [h_data(H2, kd, n, device)], [_seed(seed, L, e, 2)], [kd])
            stats[e] = {"tokens": n}
            if per_expert:
                stats[e].update({"K_gate_up": K_gu_e[e], "K_down": K_d_e[e], "share": float(share[e])})
            del W13, W2, W13a
        keys, res = [], []
        for recs, Ks in ((gu, K_gu_e), (dn, K_d_e)):
            for K in sorted({Ks[e] for e in es}):
                sub = [e for e in es if Ks[e] == K]
                keys += [x for e in sub for x in recs[e][3]]
                res += quantizer.quantize([x for e in sub for x in recs[e][0]], [x for e in sub for x in recs[e][1]], K,
                                          [x for e in sub for x in recs[e][2]], device)
        for key, (perr, out) in zip(keys, res):
            for name, t in out.items():
                tensors[f"{key}.{name}"] = t.detach().to("cpu").contiguous()
            parts_ = key.split(".")                  # model.llm.layers.L.mlp.experts.e.proj
            e, proj = int(parts_[6]), parts_[7]
            stats[e][proj] = float(perr)
        del gu, dn, res
        done += len(es)
        log(f"L{L:02d} experts {e0}-{e1 - 1}: {done}/{e1 - e0} in {time.time() - t0:.0f}s")
    from safetensors.torch import save_file
    save_file(tensors, work.tmp(L, f"experts-{e0:03d}-{e1:03d}.safetensors"))
    json.dump({"layer": L, "e0": e0, "e1": e1, "bits": bits, "per_expert": per_expert, "n0": n0, "train_tokens": n_train,
               "seconds": time.time() - t0, "experts": {str(e): s for e, s in stats.items()}},
              open(work.tmp(L, f"experts-{e0:03d}-{e1:03d}.json"), "w"))


def _seed(seed: int, L: int, e: int, j: int) -> int:
    return (seed * 1_000_003 + L * 10_007 + e * 7 + j) % (2 ** 31 - 1)


# ----------------------------------------------------------------------------------------------------------------------
# Phase: routed experts with the quantized weights -> next layer's input

class DictSource:
    """has/get over a dict of tensors (inkling2.Exl3Experts reads its tensors through this interface)"""

    def __init__(self, tensors: dict):
        self.t = tensors

    def has(self, key):
        return key in self.t

    def get(self, key, device = "cpu"):
        return self.t[key].to(device)


def load_layer_tensors(work: Work, L: int, where: str = "tmp"):
    from safetensors.torch import load_file
    d = work.tmp(L) if where == "tmp" else os.path.dirname(work.out("x"))
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors") and f.startswith("experts-") and (where == "tmp" or f.startswith(f"experts-L{L:02d}-")):
            out.update(load_file(os.path.join(d, f)))
    return out


@torch.inference_mode()
def phase_moe(L: int, cfg, src, calib: Calib, work: Work, device, part: int, parts: int, quantizer,
              adt = torch.bfloat16, log = print, tracks = ("q", "r")):
    W = I2.LayerWeights(src, cfg, L, device, adt)
    D, k = cfg.hidden, cfg.topk
    quantized = lambda t: t == "q" or t.startswith("s")           # suffix ablation tracks s<k> run the quantized experts
    qex = None
    if any(quantized(t) for t in tracks):                          # during the run from tmp/, afterwards from out/
        qex = quantizer.provider(load_layer_tensors(work, L, "tmp" if "q" in tracks else "out"), cfg, L, device)
    sex = I2.SourceExperts(src, cfg, L, device)
    num = den = 0.0
    t0 = time.time()
    for track in tracks:
        N = calib.n_tokens(track)
        if N == 0:
            continue
        if part == 0 and not os.path.exists(work.h(track, L + 1) + ".ready"):
            Mat(work.h(track, L + 1), N, D, torch.bfloat16, create = True).flush()
            open(work.h(track, L + 1) + ".ready", "w").close()
        _barrier_files([work.h(track, L + 1) + ".ready"])
        T = tmp_mats(work, L, track, N, D, k, create = False)
        Hout = Mat(work.h(track, L + 1), N, D, torch.bfloat16)
        for row, slot in calib.rows(track, part, parts):
            sl = row_slice(slot, calib.S)
            x = T["x"].read(sl, device)
            resid = T["resid"].read(sl, device)
            sel = T["sel"].read(sl, device).long()
            rw = T["rw"].read(sl, device)
            if track == "q":
                y = I2.moe_forward(x, sel, rw, qex, cfg, adt)
                if calib.held[row]:
                    ys = I2.moe_forward(x, sel, rw, sex, cfg, adt).float()
                    num += (y.float() - ys).square().sum().item()
                    den += ys.square().sum().item()
            elif quantized(track):
                y = I2.moe_forward(x, sel, rw, qex, cfg, adt)
            else:
                y = I2.moe_forward(x, sel, rw, sex, cfg, adt)
            Hout.write(sl, I2.layer_finish(W, resid, y, adt))
        Hout.flush()
    json.dump({"num": num, "den": den, "seconds": time.time() - t0}, open(work.tmp(L, f"moe-{part}.json"), "w"))
    log(f"L{L:02d} moe part {part}/{parts}: {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------------------------------------------------
# Final: logits of the held-out rows, quantized track vs reference track

@torch.inference_mode()
def phase_final(cfg, src, calib: Calib, work: Work, device, part: int, parts: int, adt = torch.bfloat16,
                chunk: int = 512, log = print, student: str = "q"):
    """KL(reference || student) over the held-out rows; student "q" (the run) or an ablation track a<k>"""
    Hd = I2.HeadWeights(src, cfg, device)
    D = cfg.hidden
    nb = cfg.num_blocks_before(cfg.layers)
    acc = {"kld": 0.0, "nll_q": 0.0, "nll_r": 0.0, "top1": 0, "tokens": 0}
    Nq, Nr = calib.n_tokens(student), calib.n_tokens("r")
    Hq, Hr = Mat(work.h(student, cfg.layers), Nq, D, torch.bfloat16), Mat(work.h("r", cfg.layers), Nr, D, torch.bfloat16)
    Bq, Br = load_blocks(work, student, nb, Nq, D), load_blocks(work, "r", nb, Nr, D)
    for row, j in calib.rows("r", part, parts):
        sq, sr = row_slice(row if student == "q" else j, calib.S), row_slice(j, calib.S)
        hq = I2.final_hidden(Hd, Hq.read(sq, device), torch.stack([b.read(sq, device) for b in Bq]), adt)
        hr = I2.final_hidden(Hd, Hr.read(sr, device), torch.stack([b.read(sr, device) for b in Br]), adt)
        ids = torch.from_numpy(calib.ids[row]).to(device)
        for c0 in range(0, calib.S - 1, chunk):
            c1 = min(calib.S - 1, c0 + chunk)
            lq = torch.log_softmax(I2.logits_from_hidden(Hd, hq[c0:c1]), -1)
            lr = torch.log_softmax(I2.logits_from_hidden(Hd, hr[c0:c1]), -1)
            tgt = ids[c0 + 1:c1 + 1]
            acc["kld"] += (lr.exp() * (lr - lq)).sum().item()
            acc["nll_q"] -= lq.gather(1, tgt[:, None]).sum().item()
            acc["nll_r"] -= lr.gather(1, tgt[:, None]).sum().item()
            acc["top1"] += (lq.argmax(-1) == lr.argmax(-1)).sum().item()
            acc["tokens"] += c1 - c0
    json.dump(acc, open(work.out(_final_name(student, part)), "w"))
    log(f"final {student} part {part}/{parts}: {acc['tokens']} tokens")


def _final_name(student: str, part = None) -> str:
    base = "final" if student == "q" else f"final-{student}"
    return f"{base}.json" if part is None else f"{base}-{part}.json"


def summarize_final(work: Work, parts: int, student: str = "q") -> dict:
    a = [json.load(open(work.out(_final_name(student, p)))) for p in range(parts)]
    n = sum(x["tokens"] for x in a)
    out = {"tokens": n, "kld": sum(x["kld"] for x in a) / max(n, 1),
           "ppl_quant": math.exp(sum(x["nll_q"] for x in a) / max(n, 1)),
           "ppl_ref": math.exp(sum(x["nll_r"] for x in a) / max(n, 1)),
           "top1_agree": sum(x["top1"] for x in a) / max(n, 1)}
    json.dump(out, open(work.out(_final_name(student)), "w"), indent = 1)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Ablation: where the run's KL divergence comes from

def ablate_track(k: int, kind: str = "a") -> str:
    return f"{kind}{k:02d}"


def ablate_init(cfg, calib: Calib, work: Work, k: int, kind: str = "a"):
    """Start state of an ablation track at the input of layer k, held-out rows only (slot j = j-th held-out row, like
    the reference track):
      a<k>  from the run's quantized track: layers < k quantized, then layers >= k run with the source experts, so
            KL(reference || a<k>) is what the quantized layers < k cause on their own;
      s<k>  from the reference track: layers < k exact, then layers >= k run with the run's quantized experts, so
            KL(reference || s<k>) is what the quantized layers >= k cause on their own.
    Needs the run's block files (each track's input of every block-start layer, kept as hard links) and, for k = layers,
    its final hidden states. k must start a block (k % block_layers == 0) or be the layer count."""
    bl, D, S = cfg.block_layers, cfg.hidden, calib.S
    assert k == cfg.layers or (0 <= k < cfg.layers and k % bl == 0), k
    assert kind in ("a", "s"), kind
    t = ablate_track(k, kind)
    base = "q" if kind == "a" else "r"
    Nb, Na = calib.n_tokens(base), calib.n_tokens(t)
    nb = cfg.num_blocks_before(cfg.layers) if k == cfg.layers else k // bl
    pairs = [(work.h(base, k) if k == cfg.layers else work.b(base, k // bl), work.h(t, k))]
    pairs += [(work.b(base, i), work.b(t, i)) for i in range(nb)]
    for sp, dp in pairs:
        s = Mat(sp, Nb, D, torch.bfloat16)
        d = Mat(dp, Na, D, torch.bfloat16, create = True)
        for j, r in enumerate(calib.r_rows):
            src_slot = int(r) if base == "q" else j
            d.t[j * S:(j + 1) * S] = s.t[src_slot * S:(src_slot + 1) * S]
        d.flush()
