#!/usr/bin/env python3
"""
Pack calibration rows for quantize_experts.py: fixed-length rows of Inkling-2 token ids drawn from several sources in
set proportions, with a share of rows held out (never used for Hessians; used for the KL divergence at the end).

Sources (--source NAME:SHARE:PATH[,PATH...]), each file one of
  *.jsonl   one object per line: {"messages": [...], "tools": [...]?} (rendered with the chat template, as the server
            renders a conversation), or {"text": "..."} (raw text)
  *.txt     raw text; documents separated by blank lines

Within a source, documents are shuffled (seeded), tokenized, separated by the end-of-text token and cut into rows of
--row-len tokens; a source contributes round(SHARE * rows) rows. Documents that contain any line of --exclude (e.g. the
questions of an evaluation set) are dropped.

  python calib_pack.py --tokenizer tokenizer.json --template chat_template.jinja --tokens 500000 --row-len 4096 \
      --source legal:0.4:cail.jsonl --source reasoning:0.2:inkling_legal_reasoning.jsonl \
      --source general:0.2:wiki_zh.txt,wiki_en.txt --source agent:0.2:code.txt,toolcalls.jsonl \
      --exclude exam_questions.txt -o calib.npz
"""
from __future__ import annotations
import argparse, hashlib, json, os, random
import numpy as np

EOT = "<|endoftext|>"


def load_template(path):
    import jinja2

    def _tojson(value, ensure_ascii = False, indent = None, separators = None, sort_keys = False):
        return json.dumps(value, ensure_ascii = ensure_ascii, indent = indent, separators = separators, sort_keys = sort_keys)

    def _raise(msg):
        raise jinja2.TemplateError(msg)

    env = jinja2.Environment(loader = jinja2.BaseLoader(), trim_blocks = False, lstrip_blocks = False)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env.from_string(open(path).read())


def read_docs(path):
    """yields ("chat", obj) or ("text", str)"""
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if "messages" in o:
                    yield "chat", o
                elif "text" in o:
                    yield "text", o["text"]
    else:
        buf = []
        for line in open(path):
            if line.strip():
                buf.append(line)
            elif buf:
                yield "text", "".join(buf)
                buf = []
        if buf:
            yield "text", "".join(buf)


def doc_text(kind, d, template):
    if kind == "text":
        return d
    for m in d["messages"]:                      # the template wants parsed tool arguments, as the server passes them
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    return template.render(messages = d["messages"], tools = d.get("tools"), add_generation_prompt = False)


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required = True)
    ap.add_argument("--template", required = True)
    ap.add_argument("--source", action = "append", required = True, help = "NAME:SHARE:PATH[,PATH...]")
    ap.add_argument("--tokens", type = int, default = 500_000, help = "total calibration tokens (rounded to rows)")
    ap.add_argument("--row-len", type = int, default = 4096)
    ap.add_argument("--heldout", type = float, default = 0.05, help = "share of rows held out (at least one per source)")
    ap.add_argument("--exclude", help = "file: one string per line; documents containing any of them are dropped")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("-o", "--out", required = True)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id(EOT)
    assert eot is not None, f"{EOT} not in the tokenizer"
    template = load_template(a.template)
    excl = [l.strip() for l in open(a.exclude)] if a.exclude else []
    excl = [x for x in excl if len(x) >= 8]
    rng = random.Random(a.seed)

    srcs = []
    for s in a.source:
        name, share, paths = s.split(":", 2)
        srcs.append((name, float(share), paths.split(",")))
    tot_share = sum(x[1] for x in srcs)
    R = max(1, round(a.tokens / a.row_len))
    want = {name: max(1, round(R * share / tot_share)) for name, share, _ in srcs}

    rows, names, report = [], [], {}
    for name, share, paths in srcs:
        docs = [d for p in paths for d in read_docs(p)]
        rng.shuffle(docs)
        stream, used, dropped = [], 0, 0
        need = want[name] * a.row_len
        for kind, d in docs:
            text = doc_text(kind, d, template)
            if excl and any(x in text for x in excl):
                dropped += 1
                continue
            stream.extend(tok.encode(text, add_special_tokens = False).ids)
            stream.append(eot)
            used += 1
            if len(stream) >= need:
                break
        n = len(stream) // a.row_len
        if n < want[name]:
            print(f"!! source {name}: only {n} of {want[name]} rows ({len(stream)} tokens from {used} documents)")
        n = min(n, want[name])
        for i in range(n):
            rows.append(stream[i * a.row_len:(i + 1) * a.row_len])
            names.append(name)
        report[name] = {"rows": n, "documents": used, "dropped_by_exclude": dropped,
                        "files": {p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in paths}}
    ids = np.array(rows, dtype = np.int32)
    held = np.zeros(len(rows), bool)
    for name in report:
        idx = [i for i, n in enumerate(names) if n == name]
        k = max(1, round(len(idx) * a.heldout)) if len(idx) > 1 else 0
        for i in rng.sample(idx, k):
            held[i] = True
    order = list(range(len(rows)))
    rng.shuffle(order)
    ids, held, names = ids[order], held[order], [names[i] for i in order]
    np.savez(a.out, ids = ids, heldout = held)
    meta = {"rows": len(rows), "row_len": a.row_len, "tokens": int(ids.size), "heldout_rows": int(held.sum()),
            "sources": report, "row_source": names, "seed": a.seed, "tokenizer": a.tokenizer, "template": a.template}
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), ensure_ascii = False, indent = 1)
    print(json.dumps({k: v for k, v in meta.items() if k != "row_source"}, ensure_ascii = False, indent = 1))


if __name__ == "__main__":
    main()
