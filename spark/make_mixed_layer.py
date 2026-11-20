"""
A mixed-rate test layer from a checkpoint whose layers each have one rate (the 3.48 bpw build on the Sparks: gate/up and
down rates differ from layer to layer, not inside one): expert e of the test layer is expert e of one of several source
layers with different rates the multi-rate kernels cover (gate = up, all in 1.5 ... 4 bpw), renamed to the test layer,
plus the test layer's own router. Meaningless as a model layer, but a valid MoE whose experts have different rates: lets
spark/check_mixk_gpu.py check the mixed-rate paths on a GPU where no mixed checkpoint is at hand (GB10, before the
two-Spark build arrives).

  python make_mixed_layer.py --src /models/Inkling-2-Small-EXL3-3.48bpw --layer 20 --out /work/mixk-test
"""
import argparse, json, os, shutil, struct

from safetensors import safe_open
from safetensors.torch import save_file

LEVELS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required = True)
    ap.add_argument("--layer", type = int, default = 20)
    ap.add_argument("--out", required = True)
    ap.add_argument("--max-sources", type = int, default = 6)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    shapes, by_expert = {}, {}
    for f in sorted(set(idx.values())):
        for k, v in header(os.path.join(a.src, f)).items():
            if k == "__metadata__":
                continue
            shapes[k] = (f, v["shape"])
            p = k.split(".")
            if len(p) > 6 and p[4] == "mlp" and p[5] == "experts":          # model.llm.layers.L.mlp.experts.e.proj.name
                by_expert.setdefault((int(p[3]), int(p[6])), []).append(k)
    num = lambda w: int(w / 16) if (w / 16).is_integer() else w / 16
    rate = {}
    for L in sorted({L for L, _ in by_expert}):
        ks = [shapes.get(f"model.llm.layers.{L}.mlp.experts.0.{p}.trellis") for p in ("gate_proj", "up_proj", "down_proj")]
        if all(ks):
            rate[L] = tuple(num(s[1][-1]) for s in ks)
    usable = {L: r for L, r in rate.items() if r[0] == r[1] and all(float(x) in LEVELS for x in r)}
    first = {}
    for L, r in sorted(usable.items()):
        first.setdefault(r, L)
    sources = sorted(first.values())[:a.max_sources]
    print("rates per layer (gate, up, down):", " ".join(f"{L}:{'/'.join(map(str, r))}" for L, r in sorted(rate.items())))
    print("source layers:", " ".join(f"{L}:{'/'.join(map(str, rate[L]))}" for L in sources))
    assert len(sources) >= 2, "need at least two source layers with different rates"
    handles, out, assign = {}, {}, []
    def get(k):
        f = shapes[k][0]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.src, f), "pt")
        return handles[f].get_tensor(k)
    T = a.layer
    for e in range(384):
        S = sources[e % len(sources)]
        assign.append(S)
        pre = f"model.llm.layers.{S}.mlp.experts.{e}."
        for k in by_expert[(S, e)]:
            out[f"model.llm.layers.{T}.mlp.experts.{e}." + k[len(pre):]] = get(k)
    for k in (f"model.llm.layers.{T}.mlp.gate.weight", f"model.llm.layers.{T}.mlp.gate.bias"):
        out[k] = get(k)
    os.makedirs(a.out, exist_ok = True)
    save_file(out, os.path.join(a.out, "mixed.safetensors"))
    shutil.copy(os.path.join(a.src, "config.json"), os.path.join(a.out, "config.json"))
    json.dump({"layer": T, "src": a.src, "sources": {str(L): list(rate[L]) for L in sources}, "expert_source": assign},
              open(os.path.join(a.out, "mixed.json"), "w"))
    print(f"wrote {len(out)} tensors, {sum(t.numel() * t.element_size() for t in out.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
