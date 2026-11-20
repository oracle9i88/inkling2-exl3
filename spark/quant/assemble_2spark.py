#!/usr/bin/env python3
"""
Assemble the two-Spark build on a node: the new routed experts (copied from the portable disk to
~/models/inkling2-disk/<export>) plus everything else from the 3.48 bpw build already on the node (attention,
embeddings, head, vision tower, mtp drafter, tokenizer: the source checkpoint's own tensors, untouched). Files are hard
links, so the build costs no extra space. The engine reads every *.safetensors in the directory, so the old expert
files must not be in it; config.json, the index and quantization_details.json are written new.

  python3 -I assemble_2spark.py [--base DIR] [--experts DIR] [--out DIR] [--force]
"""
import argparse, collections, glob, json, os, shutil, struct, sys, time

H = os.path.expanduser("~/models")


def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def nbytes(v):
    return v["data_offsets"][1] - v["data_offsets"][0]


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--base", default = os.path.join(H, "Inkling-2-Small-EXL3-3.48bpw"))
    ap.add_argument("--experts", default = os.path.join(H, "inkling2-disk", "two-spark-v2"))
    ap.add_argument("--out", default = os.path.join(H, "Inkling-2-Small-2spark-v2"))
    ap.add_argument("--force", action = "store_true", help = "replace an earlier assembly in --out")
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    old_experts = {k for k, f in wm.items() if f.startswith("model-experts-")}
    keep = {k: f for k, f in wm.items() if not f.startswith("model-experts-")}
    if any(".mlp.experts." in k for k in keep):
        sys.exit("the base has expert tensors outside its model-experts-* files")

    # new experts: names, sizes, rates
    efiles = sorted(f for f in os.listdir(a.experts) if f.startswith("experts-L") and f.endswith(".safetensors"))
    new_map, exp_bytes, wsum, bsum = {}, 0, 0, 0
    rates = collections.defaultdict(dict)          # layer -> expert -> {proj: K}
    for f in efiles:
        for k, v in header(os.path.join(a.experts, f)).items():
            if k in new_map:
                sys.exit(f"{k} is in {new_map[k]} and {f}")
            new_map[k] = f
            exp_bytes += nbytes(v)
            if k.endswith(".trellis"):
                parts = k.split(".")                 # model.llm.layers.L.mlp.experts.E.proj.trellis
                K = v["shape"][-1] / 16              # trellis [in/16, out/16, 16 K]: K bits per weight
                w = v["shape"][0] * v["shape"][1] * 256
                wsum, bsum = wsum + w, bsum + K * w
                rates[int(parts[3])][int(parts[6])] = {**rates[int(parts[3])].get(int(parts[6]), {}), parts[7]: K}
    if set(new_map) != old_experts:
        miss, extra = old_experts - set(new_map), set(new_map) - old_experts
        sys.exit(f"expert tensors differ from the base: {len(miss)} missing (e.g. {sorted(miss)[:3]}), "
                 f"{len(extra)} extra (e.g. {sorted(extra)[:3]})")
    avg = bsum / wsum

    # output directory
    if os.path.exists(a.out):
        if not a.force:
            sys.exit(f"{a.out} exists (--force to rebuild it)")
        if not os.path.exists(os.path.join(a.out, ".assembled.json")):
            sys.exit(f"{a.out} was not made by this script; not touching it")
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    skip = {"model.safetensors.index.json", "config.json", "quantization_details.json", "README.md"}
    linked = []
    for f in sorted(os.listdir(a.base)):
        p = os.path.join(a.base, f)
        if f.startswith("model-experts-") or f in skip or not os.path.isfile(p):
            continue
        os.link(p, os.path.join(a.out, f))
        linked.append(f)
    for f in efiles:
        os.link(os.path.join(a.experts, f), os.path.join(a.out, f))

    # index
    keep_bytes = 0
    for f in sorted(set(keep.values())):
        h = header(os.path.join(a.base, f))
        keep_bytes += sum(nbytes(v) for k, v in h.items() if keep.get(k) == f)
    new_idx = {"metadata": {"total_size": keep_bytes + exp_bytes}, "weight_map": dict(sorted({**keep, **new_map}.items()))}
    json.dump(new_idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent = 2)

    # config: the base's, with the experts' average rate
    cfg = json.load(open(os.path.join(a.base, "config.json")))
    q = dict(cfg.get("quantization_config") or {})
    q.pop("bits", None)   # the Hub wants an int there; the average rate goes in bits_per_weight
    q.update({"quant_method": "exl3", "bits_per_weight": round(avg, 3),
              "note": "only routed MoE experts are EXL3, each expert with its own rates (gate/up and down, 1.5 to 4 bpw "
                      "in half-bit steps); all other tensors are the source checkpoint's"})
    cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent = 2)

    # quantization details: what the export carries
    det = {"source": "thinkingmachines/Inkling-2-Small, revision 2ad1157",
           "experts": {"avg_bpw": round(avg, 4), "bytes": exp_bytes, "files": len(efiles)},
           "non_expert_tensors": "unchanged from the source checkpoint (taken from the 3.48 bpw build's files)"}
    for name in ("final.json", "run.json", "calib.json"):
        p = os.path.join(a.experts, name)
        if os.path.exists(p):
            det[name[:-5]] = json.load(open(p))
    bits = sorted(glob.glob(os.path.join(a.experts, "bits-*.json")))
    if bits:
        det["bit_table_file"] = os.path.basename(bits[0])
        shutil.copy2(bits[0], os.path.join(a.out, os.path.basename(bits[0])))
    json.dump(det, open(os.path.join(a.out, "quantization_details.json"), "w"), indent = 1)
    open(os.path.join(a.out, "README.md"), "w").write(
        "# Inkling-2-Small, two-Spark build (local)\n\nRouted experts re-quantized from the source checkpoint "
        f"(EXL3, {avg:.3f} bpw on average, a rate per expert); all other tensors are the source's. Assembled "
        f"{time.strftime('%Y-%m-%d %H:%M')} from {a.experts} and {a.base}. Not the published model card.\n")

    # check the result the way the engine will read it: every *.safetensors in the directory
    seen = {}
    for p in sorted(glob.glob(os.path.join(a.out, "*.safetensors"))):
        for k in header(p):
            if k in seen:
                sys.exit(f"{k} appears in {seen[k]} and {os.path.basename(p)}")
            seen[k] = os.path.basename(p)
    missing = [k for k in new_idx["weight_map"] if seen.get(k) != new_idx["weight_map"][k]]
    if missing:
        sys.exit(f"{len(missing)} index entries not found where the index says, e.g. {missing[:3]}")

    classes = {L: len({(r["gate_proj"], r["up_proj"], r["down_proj"]) for r in ex.values()}) for L, ex in rates.items()}
    json.dump({"base": a.base, "experts": a.experts, "time": time.strftime("%F %T"), "avg_bpw": avg,
               "linked_from_base": linked, "expert_files": efiles, "rate_classes_per_layer": classes},
              open(os.path.join(a.out, ".assembled.json"), "w"), indent = 1)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"{a.out}: {len(linked)} files from the base, {len(efiles)} expert files, {len(seen)} tensors "
          f"({len(new_idx['weight_map'])} in the index, {len(seen) - len(new_idx['weight_map'])} others: mtp), {total / 1e9:.1f} GB (hard links), "
          f"experts {avg:.3f} bpw, rate classes per layer {min(classes.values())}-{max(classes.values())} "
          f"({sum(classes.values())} in all)")


if __name__ == "__main__":
    main()
