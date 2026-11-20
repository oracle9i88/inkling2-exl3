"""
Inkling-2 routed experts through the b12x fused Trellis W4A16 MoE kernels (Local Inference Lab, Apache-2.0; now
FlashInfer flashinfer/experimental/b12x, imported as `b12x`), for uniform-rate mcg EXL3 experts (v3: K4).

In-memory adapter, no requantization and no second copy of the weights: per layer the rank's local experts are
repacked into the b12x native trellis_t256 storage (w13 [2, E, H/16, I/16, 16K] projection-major, w2 [E, I/16, H/16,
16K], int16 words byte-identical to exllamav3's per-expert `trellis` tiles) and exllamav3's per-expert tensors become
views into it (the model must be loaded with EXL3_LOAD_ARENA=0 so the old per-expert allocations are really freed),
so the original exl3_moe_coop / wide paths keep working on the same storage (fallback for prefill-sized
batches). The fp16 rotation vectors are repacked the same way: gate/up suh -> [E, H], [gate svh | up svh | down suh]
-> [E, 3I] (b12x's intermediate_rotations order), down svh -> [E, H].

Kernel contract used (validated against fp32 truth on v3 layer 5, rel err 8.6e-4): activation "situ" (beta 4, linear
beta 25 = Inkling's softcap(g,4)*sigmoid(g)*softcap(u,25)), full_rotation + intermediate_rotation, fp16 I/O and
rotation input, expert_map for the EP slice (global id -> local id or -1), router weights fp32 applied after FC2.
"""
from __future__ import annotations
import os, sys
import torch
from .inkling2_gemm import route_sigmoid_topk

_FI = os.environ.get("I2_B12X_PATH", "/opt/flashinfer")


def _import_b12x():
    for p in (_FI, _FI + "/flashinfer/experimental/b12x/_compat"):
        if p not in sys.path:
            sys.path.append(p)
    from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_fused_moe, compile_w4a16_topk_sum, run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    return make_w4a16_packed_buffers, compile_w4a16_fused_moe, compile_w4a16_topk_sum, run_w4a16_moe, prepare_trellis256_moe_weights


# (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) per verify row count; override with I2_B12X_TILES="T:a,b,c,d;..."
DEFAULT_TILE = (128, 128, 32, 512)
DEFAULT_TILES = {5: (128, 128, 64, 256), 8: (128, 128, 64, 256)}   # all-48-layer realistic-routing sweep (serving/t_moe_sweep_all.py)
DEFAULT_STAGES = {12: 4}                                            # pipeline stages override per T (same sweep)


def _tiles_from_env():
    t = dict(DEFAULT_TILES)
    s = os.environ.get("I2_B12X_TILES", "")
    for part in filter(None, s.split(";")):
        k, v = part.split(":")
        t[int(k)] = tuple(int(x) for x in v.split(","))
    return t


def _rebuild_linear_bc(inner, ext, g_tensor_cache):
    inner.bc = ext.BC_LinearEXL3(inner.trellis, inner.suh, inner.svh, inner.K, inner.bias, inner.mcg, inner.mul1,
                                 g_tensor_cache.get(*inner.bsz1_xh_args))


class B12xLayer:
    """Repacked storage + prepared b12x weights of one layer's local experts"""

    def __init__(self, mlp, prep, block_m, tile):
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache
        dev = mlp.device
        E = mlp.num_local_experts
        gates, ups, downs = [l.inner for l in mlp.gates], [l.inner for l in mlp.ups], [l.inner for l in mlp.downs]
        assert all(x.mcg and not x.mul1 for x in gates + ups + downs), "b12x path needs mcg experts"
        K = gates[0].K
        assert all(x.K == K for x in gates + ups + downs), "b12x path needs a uniform bitrate"
        assert all(x.bias is None for x in gates + ups + downs)
        H, I = gates[0].in_features, gates[0].out_features
        self.H, self.I, self.E, self.K = H, I, E, K
        w13 = torch.empty((2, E) + tuple(gates[0].trellis.shape), dtype=torch.int16, device=dev)
        w2 = torch.empty((E,) + tuple(downs[0].trellis.shape), dtype=torch.int16, device=dev)
        suh_g = torch.empty((E, H), dtype=torch.half, device=dev)
        suh_u = torch.empty((E, H), dtype=torch.half, device=dev)
        iv = torch.empty((E, 3, I), dtype=torch.half, device=dev)
        svh_d = torch.empty((E, H), dtype=torch.half, device=dev)
        for e in range(E):
            g, u, d = gates[e], ups[e], downs[e]
            w13[0, e].copy_(g.trellis); g.trellis = w13[0, e]
            w13[1, e].copy_(u.trellis); u.trellis = w13[1, e]
            w2[e].copy_(d.trellis); d.trellis = w2[e]
            suh_g[e].copy_(g.suh); g.suh = suh_g[e]
            suh_u[e].copy_(u.suh); u.suh = suh_u[e]
            iv[e, 0].copy_(g.svh); g.svh = iv[e, 0]
            iv[e, 1].copy_(u.svh); u.svh = iv[e, 1]
            iv[e, 2].copy_(d.suh); d.suh = iv[e, 2]
            svh_d[e].copy_(d.svh); d.svh = svh_d[e]
            for x in (g, u, d):
                _rebuild_linear_bc(x, ext, g_tensor_cache)
        # Rebuild exllamav3's fused-path state on the new views: BC_BlockSparseMLP holds the per-expert
        # BC_LinearEXL3 objects and its own gate/up pointer tables, which would otherwise pin the old copies
        mlp.bc = None
        mlp.multi_gate = mlp.multi_up = mlp.multi_down = None
        if getattr(mlp, "batch_recon", None) not in (None, False):
            mlp.batch_recon = None          # rebuilt lazily from the (new) per-expert tensors
        mlp.load_local()
        self.w13, self.w2 = w13, w2
        self.prep, self.scales = prep, (suh_g, suh_u, iv.view(E, 3 * I), svh_d)
        self.tiers = {}
        self.tier = self.tier_for(tile)

    def tier_for(self, tile):
        """prepared b12x weights for a tile's (fc1 N, fc2 N) pair: zero-copy views of the same storage"""
        key = (tile[1], tile[3])
        if key not in self.tiers:
            suh_g, suh_u, iv, svh_d = self.scales
            self.tiers[key] = self.prep(self.w13, self.w2, hidden_size=self.H, intermediate_size=self.I, num_experts=self.E,
                activation="situ", fc1_tile_n=tile[1], fc2_tile_n=tile[3], params_dtype=torch.float16,
                w13_layout="trellis_t256_proj", codebook="mcg", gate_suh=suh_g, up_suh=suh_u, intermediate_rotations=iv,
                down_svh=svh_d, tile_config=tuple(tile), workspace=B12xMoE.workspace(self.w13.device))
        return self.tiers[key]


class B12xMoE:
    """Engine hook: routed-expert sum of the rank's EP slice for a token batch of one of the prepared sizes"""

    _ws = {}

    @classmethod
    def workspace(cls, dev):
        if dev not in cls._ws:
            from b12x.moe._shared.kernels.w4a16.prepare import _make_workspace
            cls._ws[dev] = _make_workspace(dev, max_blocks_per_sm=4)
        return cls._ws[dev]

    def __init__(self, layers, dev, sizes=(8, 16, 32, 64, 128), block_m=8, tile=None):
        (self.make_buffers, self.compile_moe, self.compile_sum, self.run, prep) = _import_b12x()
        self.dev = dev
        self.block_m = int(os.environ.get("I2_B12X_BLOCK_M", block_m))
        self.fused_router = os.environ.get("I2_B12X_ROUTER", "1") == "1"
        self.record = None
        # [E, K] contiguous fp16 router weights for the fused router (exllamav3 keeps its K-major copy for prefill): 3.5 MB/layer
        self.router_w = {L.idx: L.mlp.routing_gate.inner.weight.t().contiguous() for L in layers} if self.fused_router else {}
        tiles = _tiles_from_env()
        self.tile = tuple(tile or tiles.get(0, DEFAULT_TILE))
        self.tiles = {T: tuple(tiles.get(T, self.tile)) for T in sizes}
        # one prepared weight object per (layer, fc tile N pair); the tile N pair is a preparation contract
        self.layers = {}
        mlp0 = layers[0].mlp
        self.E_all = mlp0.num_experts
        self.lo, self.hi = mlp0.routing_first, mlp0.routing_first + mlp0.num_local_experts
        assert os.environ.get("EXL3_LOAD_ARENA") == "0", \
            "load the model with EXL3_LOAD_ARENA=0: arena-sliced expert tensors are never freed by the repack (2x weights)"
        for L in layers:
            self.layers[L.idx] = B12xLayer(L.mlp, prep, self.block_m, self.tile)
            torch.cuda.empty_cache()        # return the layer's old per-expert segments before the next layer's stack
        lay = next(iter(self.layers.values()))
        self.H, self.I, self.E, self.K = lay.H, lay.I, lay.E, lay.K
        self.topk = mlp0.num_experts_per_tok
        self.expert_map = torch.full((self.E_all,), -1, dtype=torch.int32, device=dev)
        self.expert_map[self.lo:self.hi] = torch.arange(self.E, dtype=torch.int32, device=dev)
        props = torch.cuda.get_device_properties(dev)
        self.sizes = {}
        for T in sizes:
            tile_T = self.tiles[T]
            for L in self.layers.values():
                L.tier_for(tile_T)
            buf = self.make_buffers(lay.tier_for(tile_T), m=T, topk=self.topk, dtype=torch.float16, device=dev,
                                    route_num_experts=self.E_all, full_rotation=True, block_size_m=self.block_m)
            launch = self.compile_moe(size_m=T, hidden_size=self.H, intermediate_size=self.I, num_experts=self.E,
                top_k=self.topk, activation="situ", apply_router_weight_on_input=False, zero_fc2_output=False,
                moe_block_size=self.block_m, max_m_blocks=int(buf.block_expert_ids.numel()), element_dtype="fp16",
                sms=int(props.multi_processor_count), max_shared_mem=int(props.shared_memory_per_block_optin),
                weight_layout="trellis_t256", scale_format="e4m3_k32", w13_layout="trellis_t256_proj",
                trellis_bits=int(self.K), trellis_codebook="mcg", force_tile_config=tile_T, intermediate_rotation=True,
                full_rotation=True, rotation_input_dtype="fp16", pipeline_stages=DEFAULT_STAGES.get(T))
            tsum = self.compile_sum(m=T, topk=self.topk, hidden_size=self.H, element_dtype="fp16", full_rotation=True,
                                    num_experts=self.E, route_num_experts=self.E_all, route_ids_dtype=torch.int32,
                                    use_expert_map=True)
            self.sizes[T] = (buf, launch, tsum, (tile_T[1], tile_T[3]))

    def supports(self, T):
        return T in self.sizes

    def forward(self, layer_idx, mlp, x, out_dtype = torch.float32):
        """x: [T, H] fp16 (the MoE input). Returns [T, H] routed sum over this rank's experts (kernel output fp16)"""
        T = x.shape[0]
        buf, launch, tsum, tkey = self.sizes[T]
        lay = self.layers[layer_idx]
        tier = lay.tiers[tkey]
        if self.fused_router:
            # fused router (inkling2_gemm): fp32 logits from the fp16 router weight, sigmoid + bias top-k, normalized
            # weights; matches inkling2.route() math, int32 ids / fp32 weights straight into the kernel
            route = route_sigmoid_topk if T <= 64 else _route_large
            ids, w = route(x, self.router_w[layer_idx], mlp.e_score_correction_bias, self.topk, mlp.routed_scaling_factor)
        else:
            sel, rw = mlp.routing_fn(T, mlp.routing_cfg, x, {})
            ids = sel.to(torch.int32)
            w = rw.float()
        if self.record is not None:     # eager routing capture (serving/ceiling.py)
            self.record.append((layer_idx, ids.long().clone()))
        out = self.run(x, tier, w, ids, activation="situ", intermediate_cache13=buf.intermediate_cache13,
            intermediate_cache2=buf.intermediate_cache2, output=buf.output, fc1_c_tmp=buf.fc1_c_tmp,
            fc2_c_tmp=buf.fc2_c_tmp, packed_route_indices=buf.packed_route_indices,
            block_expert_ids=buf.block_expert_ids, packed_route_count=buf.packed_route_count,
            expert_offsets=buf.expert_offsets, expert_counts=buf.expert_counts, expert_map=self.expert_map,
            output_expert_map=self.expert_map, fused_launch=launch, topk_sum_launch=tsum,
            route_block_size_m=self.block_m, intermediate_rotation_scales=tier.intermediate_rotations,
            full_rotation=True, suh_gate_table=tier.gate_suh, suh_up_table=tier.up_suh,
            svh_table=tier.down_svh, rotation_a_gate=buf.rotation_a_gate, rotation_a_up=buf.rotation_a_up)
        y = out if out is not None else buf.output
        y = y.view(T, -1)[:, :self.H].contiguous()
        return y if y.dtype == out_dtype else y.to(out_dtype)


# ---------------------------------------------------------------------------------------------------------------------
# EP2 x TP2 expert layout: rank r holds expert group g = r // tp (E / groups experts) and intermediate half h = r % tp of
# each of them (I / tp channels = whole 128-wide Hadamard blocks), read straight from the checkpoint (no exllamav3 expert
# modules). Every MoE batch size runs through b12x (padded to a prepared size); the 4-rank all-reduce of the routed output
# sums groups and halves (the activation is per channel and the down projection is linear in the intermediate).

def strip_experts(model):
    """drop the exllamav3 expert modules before loading (EP2xTP2 loads its own expert shards)"""
    for L in model.layers:
        L.mlp = None


class B12xMoEShard:
    PREFILL_SIZES = (64, 128, 256, 512)

    def __init__(self, model_dir, cfg, layer_ids, dev, rank, world, groups = 2, sizes = (5, 8, 12, 24, 48)):
        import json
        from safetensors import safe_open
        (self.make_buffers, self.compile_moe, self.compile_sum, self.run, prep) = _import_b12x()
        c = cfg.i2
        self.dev, self.topk, self.route_scale = dev, c.topk, float(c.route_scale)
        tp = world // groups
        g, h = rank // tp, rank % tp
        E_all, H, I = c.experts, c.hidden, c.inter
        Ih = I // tp
        assert Ih % 128 == 0, "intermediate shard must be whole 128-wide Hadamard blocks"
        # group boundaries: I2_EP2_SPLIT = experts in group 0 (default E/2); an uneven split offloads a slower GPU pair
        n0 = int(os.environ.get("I2_EP2_SPLIT", E_all // groups)) if groups == 2 else E_all // groups
        bounds = [0, n0, E_all] if groups == 2 else [k * E_all // groups for k in range(groups + 1)]
        self.lo, self.hi = bounds[g], bounds[g + 1]
        Eg = self.hi - self.lo
        self.E_all, self.E, self.H, self.I = E_all, Eg, H, Ih
        self.block_m = 8
        wm = json.load(open(model_dir + "/model.safetensors.index.json"))["weight_map"]
        handles = {}
        def get(key, sl = None):
            f = wm[key]
            if f not in handles:
                handles[f] = safe_open(model_dir + "/" + f, "pt", device = "cpu")
            t = handles[f].get_slice(key)
            return t[sl] if sl is not None else t[:]
        n16 = Ih // 16
        tiles = _tiles_from_env()
        self.tile = tuple(tiles.get(0, DEFAULT_TILE))
        self.tiles = {T: tuple(tiles.get(T, self.tile)) for T in tuple(sizes) + self.PREFILL_SIZES}
        self.layers, self.router_w, self.router_b = {}, {}, {}
        for LI in layer_ids:
            pre = f"model.llm.layers.{LI}.mlp."
            K = None
            for e_i, e in enumerate(range(self.lo, self.hi)):
                p = pre + f"experts.{e}."
                gt = get(p + "gate_proj.trellis", (slice(None), slice(h * n16, (h + 1) * n16)))
                ut = get(p + "up_proj.trellis", (slice(None), slice(h * n16, (h + 1) * n16)))
                dt_ = get(p + "down_proj.trellis", (slice(h * n16, (h + 1) * n16),))
                if K is None:
                    K = gt.shape[-1] // 16
                    w13 = torch.empty((2, Eg) + tuple(gt.shape), dtype = torch.int16, device = dev)
                    w2 = torch.empty((Eg,) + tuple(dt_.shape), dtype = torch.int16, device = dev)
                    suh_g = torch.empty((Eg, H), dtype = torch.half, device = dev)
                    suh_u = torch.empty((Eg, H), dtype = torch.half, device = dev)
                    iv = torch.empty((Eg, 3, Ih), dtype = torch.half, device = dev)
                    svh_d = torch.empty((Eg, H), dtype = torch.half, device = dev)
                w13[0, e_i].copy_(gt); w13[1, e_i].copy_(ut); w2[e_i].copy_(dt_)
                suh_g[e_i].copy_(get(p + "gate_proj.suh")); suh_u[e_i].copy_(get(p + "up_proj.suh"))
                iv[e_i, 0].copy_(get(p + "gate_proj.svh", (slice(h * Ih, (h + 1) * Ih),)))
                iv[e_i, 1].copy_(get(p + "up_proj.svh", (slice(h * Ih, (h + 1) * Ih),)))
                iv[e_i, 2].copy_(get(p + "down_proj.suh", (slice(h * Ih, (h + 1) * Ih),)))
                svh_d[e_i].copy_(get(p + "down_proj.svh"))
            lay = _ShardLayer(w13, w2, (suh_g, suh_u, iv.view(Eg, 3 * Ih), svh_d), prep, H, Ih, Eg, K, dev)
            for t in set(self.tiles.values()):
                lay.tier_for(t)
            self.layers[LI] = lay
            self.router_w[LI] = get(pre + "gate.weight").to(torch.float16).contiguous().to(dev)       # [E, K]
            self.router_b[LI] = get(pre + "gate.bias").float().to(dev)
        self.K = K
        self.expert_map = torch.full((E_all,), -1, dtype = torch.int32, device = dev)
        self.expert_map[self.lo:self.hi] = torch.arange(Eg, dtype = torch.int32, device = dev)
        self.pad_expert = self.hi if self.hi < E_all else 0   # an expert of another group: maps to -1 (skipped) on this rank
        props = torch.cuda.get_device_properties(dev)
        lay0 = next(iter(self.layers.values()))
        self.sizes = {}
        for T in sorted(set(sizes) | set(self.PREFILL_SIZES)):
            tile_T = self.tiles[T]
            tier = lay0.tier_for(tile_T)
            buf = self.make_buffers(tier, m = T, topk = self.topk, dtype = torch.float16, device = dev,
                                    route_num_experts = E_all, full_rotation = True, block_size_m = self.block_m)
            launch = self.compile_moe(size_m = T, hidden_size = H, intermediate_size = Ih, num_experts = Eg, top_k = self.topk,
                activation = "situ", apply_router_weight_on_input = False, zero_fc2_output = False, moe_block_size = self.block_m,
                max_m_blocks = int(buf.block_expert_ids.numel()), element_dtype = "fp16", sms = int(props.multi_processor_count),
                max_shared_mem = int(props.shared_memory_per_block_optin), weight_layout = "trellis_t256", scale_format = "e4m3_k32",
                w13_layout = "trellis_t256_proj", trellis_bits = int(K), trellis_codebook = "mcg", force_tile_config = tile_T,
                intermediate_rotation = True, full_rotation = True, rotation_input_dtype = "fp16",
                pipeline_stages = DEFAULT_STAGES.get(T))
            tsum = self.compile_sum(m = T, topk = self.topk, hidden_size = H, element_dtype = "fp16", full_rotation = True,
                                    num_experts = Eg, route_num_experts = E_all, route_ids_dtype = torch.int32, use_expert_map = True)
            self.sizes[T] = (buf, launch, tsum, (tile_T[1], tile_T[3]))
        self.record = None

    def supports(self, T):
        return T <= max(self.sizes)

    def forward(self, layer_idx, mlp, x, out_dtype = torch.float32):
        T = x.shape[0]
        S = next(s for s in sorted(self.sizes) if s >= T)
        buf, launch, tsum, tkey = self.sizes[S]
        tier = self.layers[layer_idx].tiers[tkey]
        ids, w = route_sigmoid_topk(x, self.router_w[layer_idx], self.router_b[layer_idx], self.topk, self.route_scale) \
            if T <= 64 else _route_large(x, self.router_w[layer_idx], self.router_b[layer_idx], self.topk, self.route_scale)
        if self.record is not None:
            self.record.append((layer_idx, ids.long().clone()))
        if S != T:
            xp = torch.zeros((S, x.shape[1]), dtype = x.dtype, device = x.device); xp[:T] = x
            ip = torch.full((S, self.topk), self.pad_expert, dtype = torch.int32, device = x.device); ip[:T] = ids
            wp = torch.zeros((S, self.topk), dtype = torch.float32, device = x.device); wp[:T] = w
            x, ids, w = xp, ip, wp
        out = self.run(x, tier, w, ids, activation = "situ", intermediate_cache13 = buf.intermediate_cache13,
            intermediate_cache2 = buf.intermediate_cache2, output = buf.output, fc1_c_tmp = buf.fc1_c_tmp,
            fc2_c_tmp = buf.fc2_c_tmp, packed_route_indices = buf.packed_route_indices,
            block_expert_ids = buf.block_expert_ids, packed_route_count = buf.packed_route_count,
            expert_offsets = buf.expert_offsets, expert_counts = buf.expert_counts, expert_map = self.expert_map,
            output_expert_map = self.expert_map, fused_launch = launch, topk_sum_launch = tsum,
            route_block_size_m = self.block_m, intermediate_rotation_scales = tier.intermediate_rotations,
            full_rotation = True, suh_gate_table = tier.gate_suh, suh_up_table = tier.up_suh,
            svh_table = tier.down_svh, rotation_a_gate = buf.rotation_a_gate, rotation_a_up = buf.rotation_a_up)
        y = (out if out is not None else buf.output).view(S, -1)[:T, :self.H]
        y = y.contiguous()
        return y if y.dtype == out_dtype else y.to(out_dtype)


class _ShardLayer:
    def __init__(self, w13, w2, scales, prep, H, I, E, K, dev):
        self.w13, self.w2, self.scales, self.prep = w13, w2, scales, prep
        self.H, self.I, self.E, self.K = H, I, E, K
        self.tiers = {}

    tier_for = B12xLayer.tier_for


def _route_large(x, w_ek, bias, topk, scale):
    """router for prefill-sized batches: fp32 logits via cuBLAS, same sigmoid + bias top-k math as route_sigmoid_topk"""
    z = torch.mm(x.float(), w_ek.float().t())
    s = torch.sigmoid(z)
    sel = torch.topk(s + bias[None], topk, dim = -1).indices
    wv = s.gather(1, sel)
    wv = wv / (wv.sum(-1, keepdim = True) + 1e-20) * scale
    return sel.to(torch.int32), wv.float()


def ep_layout():
    return os.environ.get("I2_EP_LAYOUT", "ep4")


def prepare_model(model):
    """call after Model.from_config, before loading layers: EP2xTP2 loads its own expert shards"""
    mixed = [L.idx for L in model.layers if getattr(L.mlp, "mixed_rates", False)]
    if mixed:
        raise ValueError(f"b12x runs one rate per layer (K4 mcg); the experts of layers {mixed} mix rates "
                         f"(modules/inkling2_mixk.py): serve this model with --b12x 0")
    if ep_layout() == "ep2tp2":
        strip_experts(model)


def attach_moe(eng, model, model_dir, cfg, dev, rank, world, sizes):
    """b12x routed experts for the engine: EP4 (exllamav3 experts repacked in place) or EP2xTP2 (checkpoint shards)"""
    if ep_layout() == "ep2tp2":
        eng.b12x = B12xMoEShard(model_dir, cfg, [L.idx for L in model.layers], dev, rank, world, groups=2, sizes=sizes)
    else:
        eng.b12x = B12xMoE(model.layers, dev, sizes=sizes)
    return eng.b12x
