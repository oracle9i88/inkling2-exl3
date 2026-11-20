"""
Checks of allocate_bits.py --measured on the real anchor (0xSero 3.48 bpw) and the v1 table (bits-2.55.json):

1. Measured KL shares exactly as the anchor model predicts -> the v1 allocation comes back unchanged.
2. The last group (layers 42-47) measured 4x above the model -> those layers gain bits, others give some up, the
   budget holds.
3. The same with the first group instead -> the first layers gain.
4. A group measured below zero (noise) does not break the solve; it is treated as a small share.

  python test_allocate_bits.py
"""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allocate_bits as ab


def main():
    anchor = ab.load_anchor(os.path.join(HERE, "anchor-0xSero-3.48bpw.json"))
    v1 = {int(L): {k: float(v[k]) for k in ab.UNITS} for L, v in json.load(open(os.path.join(HERE, "bits-2.55.json")))["layers"].items()}
    budget = sum(ab.unit_bytes(k, 2.55) for _ in range(ab.LAYERS) for k in ab.UNITS)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, flush = True)
        ok &= bool(cond)

    base, used0 = ab.allocate(anchor, budget, 2.0)
    check("anchor allocation == bits-2.55.json", all(base[(L, k)] == v1[L][k] for L in v1 for k in ab.UNITS))

    def groups(mult):
        out = []
        for g in range(8):
            lo, hi = 6 * g, 6 * g + 5
            pred = sum(ab.unit_params(k) * 4.0 ** (anchor[L][k] - v1[L][k]) for L in range(lo, hi + 1) for k in ab.UNITS)
            out.append({"layers": [lo, hi], "kld": 1e-12 * pred * mult.get(g, 1.0)})
        return out

    def mean_bits(bits, layers):
        P = sum(ab.unit_params(k) for L in layers for k in ab.UNITS)
        return sum(ab.unit_params(k) * bits[(L, k)] for L in layers for k in ab.UNITS) / P

    s, rows = ab.measured_scale(anchor, v1, groups({}))
    b1, u1 = ab.allocate(anchor, budget, 2.0, scale = s)
    check("measured == model: factors all 1", all(abs(r["f_rel"] - 1) < 1e-9 for r in rows))
    check("measured == model: v1 allocation unchanged", all(b1[(L, k)] == v1[L][k] for L in v1 for k in ab.UNITS))

    late = range(42, 48)
    s, rows = ab.measured_scale(anchor, v1, groups({7: 4.0}))
    b2, u2 = ab.allocate(anchor, budget, 2.0, scale = s)
    m_v1, m_new = mean_bits(base, late), mean_bits(b2, late)
    check(f"late group 4x: layers 42-47 {m_v1:.3f} -> {m_new:.3f} bits", m_new > m_v1 + 0.25)
    check(f"late group 4x: budget holds ({u2 / 1e9:.3f} <= {budget / 1e9:.3f} GB)", u2 <= budget)
    others = [L for L in range(48) if L not in late]
    check(f"late group 4x: other layers give up ({mean_bits(base, others):.3f} -> {mean_bits(b2, others):.3f})",
          mean_bits(b2, others) < mean_bits(base, others))

    early = range(0, 6)
    s, rows = ab.measured_scale(anchor, v1, groups({0: 4.0}))
    b3, u3 = ab.allocate(anchor, budget, 2.0, scale = s)
    check(f"first group 4x: layers 0-5 {mean_bits(base, early):.3f} -> {mean_bits(b3, early):.3f} bits",
          mean_bits(b3, early) > mean_bits(base, early) + 0.25 and u3 <= budget)

    g = groups({})
    g[3]["kld"] = -1e-6
    s, rows = ab.measured_scale(anchor, v1, g)
    b4, u4 = ab.allocate(anchor, budget, 2.0, scale = s)
    mid = range(18, 24)
    check(f"negative group: solved, layers 18-23 {mean_bits(base, mid):.3f} -> {mean_bits(b4, mid):.3f} bits (factor {rows[3]['f_rel']:.3f})",
          u4 <= budget and mean_bits(b4, mid) <= mean_bits(base, mid) and rows[3]["f_rel"] >= 1 / 8 - 1e-12)

    print("ALLOCATE-TEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
