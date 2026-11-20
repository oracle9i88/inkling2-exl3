#!/usr/bin/env python3
"""
The release folder of an assembled two-Spark build: the same tensors the engine serves, without the bf16 copies that
the EXL3 attention and head replace, with a complete index, config and quantization details.

- A layer whose EXL3 attention is present (attn.wqgkv or attn.wqg, and attn.wo) loses its bf16 attention matrices
  (attn.wq_du, wk_dv, wv_dv, gate_attn, wo_ud). With the EXL3 head present (model.llm.unembed_q), the bf16
  model.llm.unembed.weight goes. The engine reads neither once the EXL3 versions are there; the release folder then
  needs the EXL3 paths (I2_ATTN_Q / I2_HEAD_Q on, the default).
- Files without such tensors are hard links (copies across file systems). Files with them are rewritten without them,
  and every kept tensor is compared byte for byte with the source.
- Only model files and the known side files are taken: local notes, hidden files and download lists stay behind.
  Absolute paths in quantization_details.json become base names.

  python3 -I make_release.py --src DIR --out DIR [--attn-final final.json] [--head-final final-head.json] [--sums]
"""
import argparse, concurrent.futures, hashlib, json, os, re, shutil, struct, sys, time

SIDE_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
              "processor_config.json", "preprocessor_config.json", "generation_config.json", "expert_load.json"}
REPLACED = ("wq_du.weight", "wk_dv.weight", "wv_dv.weight", "gate_attn.weight", "wo_ud.weight")
HEAD_Q, HEAD_BF16 = "model.llm.unembed_q", "model.llm.unembed.weight"


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def tensors(h):
    return {k: v for k, v in h.items() if k != "__metadata__"}


def bits_of(v):
    return v["shape"][-1] / 16          # EXL3 trellis [in/16, out/16, 16 K]: K bits per weight


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite(src, dst, drop):
    """src without the tensors in drop; returns (kept, dropped bytes). Kept tensors keep their order and bytes."""
    off, h = read_header(src)
    meta = h.get("__metadata__")
    items = sorted(tensors(h).items(), key = lambda kv: kv[1]["data_offsets"][0])
    new, pos, dropped = {}, 0, 0
    for k, v in items:
        a, b = v["data_offsets"]
        if k in drop:
            dropped += b - a
            continue
        new[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [pos, pos + b - a]}
        pos += b - a
    hd = ({"__metadata__": meta} if meta is not None else {}) | new
    hb = json.dumps(hd, separators = (",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for k, v in items:
            if k in drop:
                continue
            a, b = v["data_offsets"]
            fi.seek(off + a)
            left = b - a
            while left:
                buf = fi.read(min(left, 1 << 26))
                fo.write(buf)
                left -= len(buf)
    os.replace(dst + ".part", dst)
    # check: every kept tensor has the source's bytes
    off2, h2 = read_header(dst)
    with open(src, "rb") as fi, open(dst, "rb") as fo:
        for k, v in tensors(h2).items():
            a, b = h[k]["data_offsets"]
            a2, b2 = v["data_offsets"]
            assert b - a == b2 - a2 and h[k]["dtype"] == v["dtype"] and h[k]["shape"] == v["shape"], k
            fi.seek(off + a)
            fo.seek(off2 + a2)
            left = b - a
            while left:
                n = min(left, 1 << 26)
                if fi.read(n) != fo.read(n):
                    sys.exit(f"{dst}: {k} differs from the source")
                left -= n
    return len(new), dropped


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str) and re.match(r"^/[^\s]+/[^\s]*$", x):
        return os.path.basename(x.rstrip("/")) or x
    return x


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--src", required = True, help = "assembled model folder (experts, attn-LNN, head, the rest)")
    ap.add_argument("--out", required = True)
    ap.add_argument("--attn-final", help = "final.json of the attention run (held-out KL), added to the details")
    ap.add_argument("--head-final", help = "final-head.json of the head run, added to the details")
    ap.add_argument("--sums", action = "store_true", help = "write SHA256SUMS of every file")
    ap.add_argument("--force", action = "store_true", help = "replace an earlier release folder in --out")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    tc = cfg.get("text_config", cfg)
    local = set(tc["local_layer_ids"])
    nl = tc["num_hidden_layers"]
    st_files = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
    heads = {f: read_header(os.path.join(a.src, f))[1] for f in st_files}
    where = {}
    for f, h in heads.items():
        for k in tensors(h):
            if k in where:
                sys.exit(f"{k} is in {where[k]} and {f}")
            where[k] = f

    # what the EXL3 attention and head replace
    drop, qlayers = set(), []
    for L in range(nl):
        a_ = f"model.llm.layers.{L}.attn."
        first = a_ + ("wqgkv" if L in local else "wqg") + ".trellis"
        if first in where and a_ + "wo.trellis" in where and (L in local or (a_ + "wk.trellis" in where and a_ + "wv.trellis" in where)):
            qlayers.append(L)
            drop |= {a_ + r for r in REPLACED if a_ + r in where}
    head_q = HEAD_Q + ".trellis" in where
    if head_q and HEAD_BF16 in where:
        drop.add(HEAD_BF16)
    if len(qlayers) not in (0, nl):
        sys.exit(f"EXL3 attention in {len(qlayers)} of {nl} layers: the release expects all or none")

    if os.path.exists(a.out):
        if not a.force:
            sys.exit(f"{a.out} exists (--force to rebuild it)")
        if not os.path.exists(os.path.join(a.out, ".release.json")):
            sys.exit(f"{a.out} was not made by this script; not touching it")
        shutil.rmtree(a.out)
    os.makedirs(a.out)

    # model files
    rewritten, dropped_bytes = [], 0
    for f in st_files:
        src, dst = os.path.join(a.src, f), os.path.join(a.out, f)
        if drop & set(tensors(heads[f])):
            kept, db = rewrite(src, dst, drop)
            dropped_bytes += db
            rewritten.append(f)
            if kept == 0:
                os.remove(dst)
        else:
            link_or_copy(src, dst)
    side = sorted(f for f in os.listdir(a.src) if f in SIDE_FILES or re.match(r"^bits-.*\.json$", f))
    for f in side:
        link_or_copy(os.path.join(a.src, f), os.path.join(a.out, f))
    skipped = sorted(set(os.listdir(a.src)) - set(st_files) - set(side)
                     - {"config.json", "model.safetensors.index.json", "quantization_details.json"})

    # index: every tensor except the mtp drafter's (as in the source checkpoint's layout)
    out_st = sorted(f for f in os.listdir(a.out) if f.endswith(".safetensors"))
    wm, total = {}, 0
    for f in out_st:
        if f == "mtp.safetensors":
            continue
        for k, v in tensors(read_header(os.path.join(a.out, f))[1]).items():
            wm[k] = f
            total += v["data_offsets"][1] - v["data_offsets"][0]
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent = 2)

    # rates, from the trellis shapes
    def avg_bits(pred):
        w = b = 0
        for f, h in heads.items():
            for k, v in tensors(h).items():
                if k.endswith(".trellis") and pred(k):
                    n = v["shape"][0] * v["shape"][1] * 256
                    w, b = w + n, b + bits_of(v) * n
        return b / w if w else None
    exp_bits = avg_bits(lambda k: ".mlp.experts." in k)
    attn_bits = avg_bits(lambda k: ".attn." in k)
    head_bits = avg_bits(lambda k: k.startswith(HEAD_Q + "."))
    vocab_rows = heads[where[HEAD_Q + ".svh"]][HEAD_Q + ".svh"]["shape"][0] if head_q else None
    q = dict(cfg.get("quantization_config") or {})
    q.pop("bits", None)   # the Hub wants an int there; the average rate goes in bits_per_weight
    q.update({"quant_method": "exl3", "bits_per_weight": round(exp_bits, 3), "codebook": "mul1", "out_scales": "always"})
    parts = [f"routed MoE experts: EXL3, a rate per expert (gate/up and down, 1.5 to 4 bpw in half-bit steps), "
             f"{exp_bits:.3f} bpw on average"]
    if attn_bits:
        q["attention_bits"] = round(attn_bits, 3)
        parts.append(f"attention projections: EXL3 {attn_bits:g} bpw (attn-LNN.safetensors)")
    if head_bits:
        q["head_bits"] = round(head_bits, 3)
        parts.append(f"output head: EXL3 {head_bits:g} bpw, vocabulary padded to {vocab_rows} rows (head.safetensors)")
    parts.append("embeddings, vision tower, MTP draft, router, norms and small projections: the source checkpoint's bf16")
    q["note"] = "; ".join(parts)
    cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent = 2)

    # quantization details
    det = json.load(open(os.path.join(a.src, "quantization_details.json"))) if os.path.exists(
        os.path.join(a.src, "quantization_details.json")) else {}
    det = scrub(det)
    det["non_expert_tensors"] = ("the source checkpoint's, except the bf16 attention projections and output head, "
                                 "which the EXL3 files below replace")
    if attn_bits:
        det["attention"] = {"bpw": attn_bits, "layers": len(qlayers), "files": "attn-L00..attn-L%02d.safetensors" % (nl - 1)}
        if a.attn_final:
            det["attention"]["final"] = scrub(json.load(open(a.attn_final)))
    if head_bits:
        det["head"] = {"bpw": head_bits, "vocab_rows": vocab_rows, "file": "head.safetensors"}
        if a.head_final:
            det["head"]["final"] = scrub(json.load(open(a.head_final)))
    json.dump(det, open(os.path.join(a.out, "quantization_details.json"), "w"), indent = 1)

    # the engine reads every *.safetensors: no tensor twice, nothing replaced left, the index where it says
    seen = {}
    for f in out_st:
        for k in tensors(read_header(os.path.join(a.out, f))[1]):
            if k in seen:
                sys.exit(f"{k} appears in {seen[k]} and {f}")
            seen[k] = f
    assert not (drop & set(seen)), "a replaced tensor is still there"
    assert set(seen) == set(where) - drop, "tensors went missing"
    assert all(seen[k] == f for k, f in wm.items())

    files = sorted(os.listdir(a.out))
    if a.sums:
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            sums = dict(zip(files, ex.map(lambda f: sha256(os.path.join(a.out, f)), files)))
        with open(os.path.join(a.out, "SHA256SUMS"), "w") as fo:
            for f in files:
                fo.write(f"{sums[f]}  {f}\n")
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in files)
    info = {"src": os.path.basename(os.path.normpath(a.src)), "time": time.strftime("%F %T"),
            "dropped_tensors": len(drop), "dropped_bytes": dropped_bytes, "rewritten": rewritten,
            "side_files": side, "skipped": skipped, "files": len(files), "bytes": size}
    json.dump(info, open(os.path.join(a.out, ".release.json"), "w"), indent = 1)
    print(f"{a.out}: {len(files)} files, {size / 1e9:.1f} GB; {len(seen)} tensors ({len(drop)} replaced bf16 tensors, "
          f"{dropped_bytes / 1e9:.2f} GB, left out of {len(rewritten)} rewritten files); experts {exp_bits:.3f} bpw, "
          f"attention {attn_bits} bpw, head {head_bits} bpw; skipped: {skipped}")


if __name__ == "__main__":
    main()
