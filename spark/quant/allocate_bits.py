"""
Bit allocation for the routed experts of a smaller Inkling-2-Small EXL3 quant (two DGX Sparks).

Anchor: 0xSero's 3.48 bpw quant. Its quantization_details.json lists, per layer, the bitrate of gate/up and of down in
half-bit steps; these are the real tile widths of his tensors (checked against the safetensors headers: trellis widths
80/64/56/48/40 = 5/4/3.5/3/2.5 bits). The table is read as the solution of

    minimize  sum_u c_u * 4^(-b_u)     subject to   sum_u P_u * b_u  <=  his budget

(u = one projection group of one layer: gate+up, or down; P_u its parameter count; every extra bit cuts a tensor's
quantization error by about 4x, i.e. each half bit by 2x). At an optimum the marginal gain per byte is equal for all
units, which fixes c_u = P_u * 4^(b0_u) up to a common factor. The same problem is then solved again for a smaller
budget on the half-bit grid, with a floor (2 bits by default), by greedy marginal gain from the floor upward. Without
the floor the answer is "every unit gives up the same number of bits"; where the floor binds, the others give up less.

Measured error curves (--curves, {layer: {"gate_up": {bits: err}, "down": {bits: err}}}) replace the model for the units
they cover.

Measured KL shares (--measured out/ablate.json of a finished run, with --v1 the bit table of that run): the anchor
model predicts each layer group's share of the run's error, sum over its units of c_u * 4^(-b1_u); `quantize_experts.py
ablate` measures it (the KL divergence the group's quantized experts add). Each group's c_u are scaled by measured /
predicted (relative to the median group, clamped to x8 either way; a group measured at or below zero counts as 5% of
the mean positive group), then the allocation is solved again. Equal factors give back the anchor allocation exactly.

Usage:
  python allocate_bits.py --anchor anchor-0xSero-3.48bpw.json --avg-bits 2.55 -o bits-2.55.json
  python allocate_bits.py --anchor ... --gb-per-node 96 -o bits.json      # budget in GB (1e9) of expert tensors per node
  python allocate_bits.py --anchor ... --avg-bits 2.55 --measured work/out/ablate.json --v1 bits-2.55.json -o bits-2.55-measured.json
"""
from __future__ import annotations
import argparse, json, math

HIDDEN, INTER, EXPERTS, LAYERS = 4608, 2304, 384, 48
P_PROJ = HIDDEN * INTER                         # parameters of one expert projection
UNITS = {"gate_up": 2, "down": 1}               # projections per unit
GRID = 0.5
# 11-16: rates the kernels actually have. Half-integer rates exist only at 1.5, 2.5 and 3.5: quantize_tiles_frac is
# instantiated for KA = 1..3, and exl3_moe_coop maps any other half rate to the 3.5-bit kernel (a 4.5-bit tensor would
# run silently wrong). Above 3.5 the grid is whole bits.
HALF_RATES = (1.5, 2.5, 3.5)


def next_bits(b: float) -> float:
    n = b + GRID
    return n if n in HALF_RATES or float(n).is_integer() else b + 1.0


def unit_params(kind: str) -> int:
    return UNITS[kind] * EXPERTS * P_PROJ


def unit_overhead_bytes(kind: str) -> int:
    """suh + svh (fp16, in + out) and the mul1 marker (int32) per tensor"""
    return UNITS[kind] * EXPERTS * ((HIDDEN + INTER) * 2 + 4)


def unit_bytes(kind: str, bits: float) -> float:
    return unit_params(kind) * bits / 8 + unit_overhead_bytes(kind)


def load_anchor(path: str) -> dict:
    d = json.load(open(path))
    L = d["layers"]
    return {int(i): {"gate_up": float(v["bits_gate_up"]), "down": float(v["bits_down"])} for i, v in L.items()}


def model_err(b0: float, P: int):
    """E(b) = P * 4^(b0 - b)  (common factor dropped)"""
    return lambda b: P * 4.0 ** (b0 - b)


def curve_err(points: dict):
    """log-linear interpolation of measured errors {bits: err}; extrapolates with the end slopes"""
    xs = sorted(float(k) for k in points)
    ys = [math.log(max(float(points[str(x)] if str(x) in points else points[x]), 1e-30)) for x in xs]
    def f(b):
        if len(xs) == 1:
            return math.exp(ys[0] - (b - xs[0]) * math.log(4))
        if b <= xs[0]:
            i = 0
        elif b >= xs[-1]:
            i = len(xs) - 2
        else:
            i = max(j for j in range(len(xs) - 1) if xs[j] <= b)
        t = (b - xs[i]) / (xs[i + 1] - xs[i])
        return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    return f


def measured_scale(anchor: dict, v1: dict, groups: list, clamp: float = 8.0):
    """per-layer factors on the anchor model's c_u so that each layer group's predicted error at the v1 bits is in
    proportion to its measured KL share; groups: [{"layers": [lo, hi], "kld": x}, ...] covering the layers"""
    pos = [g["kld"] for g in groups if g["kld"] > 0]
    floor_kld = 0.05 * sum(pos) / len(pos) if pos else 1.0
    rows = []
    for g in groups:
        lo, hi = g["layers"]
        pred = sum(unit_params(k) * 4.0 ** (anchor[L][k] - v1[L][k]) for L in range(lo, hi + 1) for k in UNITS)
        rows.append({"layers": [lo, hi], "kld": g["kld"], "pred": pred, "f": max(g["kld"], floor_kld) / pred})
    med = sorted(r["f"] for r in rows)[len(rows) // 2]
    scale = {}
    for r in rows:
        r["f_rel"] = min(max(r["f"] / med, 1.0 / clamp), clamp)
        for L in range(r["layers"][0], r["layers"][1] + 1):
            scale[L] = r["f_rel"]
    missing = sorted(set(anchor) - set(scale))
    if missing:
        raise SystemExit(f"--measured groups do not cover layers {missing}")
    return scale, rows


def allocate(anchor: dict, budget_bytes: float, floor: float = 2.0, cap: float = 8.0, curves: dict | None = None,
             scale: dict | None = None):
    units = [(L, k) for L in sorted(anchor) for k in ("gate_up", "down")]
    err = {}
    for L, k in units:
        c = (curves or {}).get(str(L), {}).get(k) or (curves or {}).get(L, {}).get(k)
        e = curve_err(c) if c else model_err(anchor[L][k], unit_params(k))
        if scale is not None:
            e = (lambda f, s: lambda b: s * f(b))(e, scale[L])
        err[(L, k)] = e
    bits = {u: floor for u in units}
    used = sum(unit_bytes(k, floor) for _, k in units)
    if used > budget_bytes:
        raise SystemExit(f"budget {budget_bytes / 1e9:.2f} GB is below the floor allocation ({used / 1e9:.2f} GB)")
    while True:
        best, best_key = None, None
        for u in units:
            b = bits[u]
            n = next_bits(b)
            if n > cap + 1e-9:
                continue
            cost = unit_bytes(u[1], n) - unit_bytes(u[1], b)
            if used + cost > budget_bytes:
                continue
            gain = (err[u](b) - err[u](n)) / cost
            # ties (common with the model): down before gate/up, earlier layers first
            key = (gain, u[1] == "down", -u[0])
            if best is None or key > best_key:
                best, best_key = u, key
        if best is None:
            break
        n = next_bits(bits[best])
        used += unit_bytes(best[1], n) - unit_bytes(best[1], bits[best])
        bits[best] = n
    return bits, used


def summary(bits: dict) -> dict:
    P = sum(unit_params(k) for _, k in bits)
    avg = sum(unit_params(k) * b for (_, k), b in bits.items()) / P
    total = sum(unit_bytes(k, b) for (_, k), b in bits.items())
    return {"avg_bits": round(avg, 4), "expert_bytes": int(total), "expert_GB": round(total / 1e9, 3),
            "per_node_GB_2way": round(total / 2e9, 3)}


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--anchor", required = True)
    ap.add_argument("--avg-bits", type = float, help = "target average bits over all expert parameters")
    ap.add_argument("--gb-per-node", type = float, help = "budget: GB (1e9) of expert tensors per node, two nodes")
    ap.add_argument("--floor", type = float, default = 2.0)
    ap.add_argument("--cap", type = float, default = 8.0)
    ap.add_argument("--curves", help = "measured error curves json (optional)")
    ap.add_argument("--measured", help = "out/ablate.json of a finished run: scale the model by the measured KL share of each layer group")
    ap.add_argument("--v1", help = "with --measured: the bit table that run used")
    ap.add_argument("--clamp", type = float, default = 8.0, help = "with --measured: largest factor either way, relative to the median group")
    ap.add_argument("-o", "--out", required = True)
    a = ap.parse_args()
    anchor = load_anchor(a.anchor)
    assert sorted(anchor) == list(range(LAYERS)), "anchor must cover layers 0..47"
    if (a.avg_bits is None) == (a.gb_per_node is None):
        raise SystemExit("give exactly one of --avg-bits / --gb-per-node")
    if a.avg_bits is not None:
        budget = sum(unit_bytes(k, a.avg_bits) for _ in range(LAYERS) for k in UNITS)
    else:
        budget = a.gb_per_node * 2e9
    curves = json.load(open(a.curves)) if a.curves else None
    scale = rows = v1 = None
    if a.measured:
        if not a.v1:
            raise SystemExit("--measured needs --v1 (the bit table of the measured run)")
        v1 = {int(L): {k: float(v[k]) for k in UNITS} for L, v in json.load(open(a.v1))["layers"].items()}
        scale, rows = measured_scale(anchor, v1, json.load(open(a.measured))["groups"], a.clamp)
    bits, used = allocate(anchor, budget, a.floor, a.cap, curves, scale)
    model = "measured curves" if curves else "anchor-optimal, error ~ 4^-bits"
    if scale is not None:
        model += ", scaled per layer group by the measured KL shares (ablate)"
    out = {
        "layers": {str(L): {"gate_up": bits[(L, "gate_up")], "down": bits[(L, "down")]} for L in range(LAYERS)},
        **summary(bits),
        "budget_GB": round(budget / 1e9, 3),
        "anchor": a.anchor, "anchor_summary": summary({(L, k): anchor[L][k] for L in anchor for k in UNITS}),
        "floor": a.floor, "model": model,
    }
    if rows is not None:
        out.update({"measured": a.measured, "v1": a.v1, "groups": rows, "v1_summary": summary(
            {(L, k): v1[L][k] for L in v1 for k in UNITS})})
    json.dump(out, open(a.out, "w"), indent = 1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("layers", "groups")}, indent = 1))
    for r in rows or []:
        print(f"layers {r['layers'][0]:2d}-{r['layers'][1]:2d}: measured KLD {r['kld']:+.5f}, model {r['pred']:.3e}, factor {r['f_rel']:.2f}")
    ref = v1 if v1 is not None else anchor
    for L in range(LAYERS):
        b0, b = ref[L], out["layers"][str(L)]
        print(f"L{L:02d}  gate/up {b0['gate_up']:.1f} -> {b['gate_up']:.1f}   down {b0['down']:.1f} -> {b['down']:.1f}")


if __name__ == "__main__":
    main()
