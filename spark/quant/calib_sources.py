#!/usr/bin/env python3
"""
Public calibration sources for the expert requantization (the mix of 记录/阶段4 §9), written as calib_pack.py inputs:
  general.jsonl    Wikipedia zh + en (wikimedia/wikipedia 20231101, CC BY-SA): random articles of 2,000+ characters,
                   cut at a paragraph boundary to at most 8,000 characters
  code.jsonl       source files of open-source projects: transformers (Apache-2.0) from a local install, and the
                   exllamav3 (MIT) CUDA/C++ extension from a local copy of its tree
  toolcalls.jsonl  NousResearch/hermes-function-calling-v1 (Apache-2.0), single- and multi-turn function calling,
                   converted to OpenAI messages + tools (calib_pack.py renders them with the Inkling template)
  agent.jsonl      nebius/SWE-agent-trajectories (CC BY 4.0), resolved runs only: every action becomes a call of one
                   "run" tool and the interface output its result
  legal.jsonl      CAIL2018 case facts (public), minus the facts of the legal exam (md5 list)
  exam_exclude.txt one line per held-out exam question (its longest line), for calib_pack.py --exclude
Rows come from the Hugging Face datasets-server API (no pyarrow needed). Offsets are seeded, so a rerun picks the same
documents. Provenance (datasets, licenses, row ids, file hashes) goes to <out>/sources.json.

  python calib_sources.py --out calib-data/sources --transformers <site-packages>/transformers \
      --exllamav3-src calib-data/sources/exllamav3-src --cail raw/cail2018_pred.jsonl --cail-exclude holdout_fact_md5.txt \
      --eval-dir calib-data/gen
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, random, re, time, urllib.parse, urllib.request

API = "https://datasets-server.huggingface.co/rows?"


def rows(ds, cfg, off, n, split = "train"):
    q = urllib.parse.urlencode({"dataset": ds, "config": cfg, "split": split, "offset": off, "length": n})
    err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(API + q, timeout = 90) as r:
                d = json.load(r)
            if "rows" not in d:
                raise RuntimeError(str(d)[:200])
            return d.get("num_rows_total"), [x["row"] for x in d["rows"]]
        except Exception as e:                            # rate limits and timeouts: back off and retry
            err = e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{ds}/{cfg} @{off}: {err}")


def write_jsonl(path, docs):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii = False) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------------

def wiki(lang, n_docs, rng, log):
    ds, cfg = "wikimedia/wikipedia", f"20231101.{lang}"
    total, _ = rows(ds, cfg, 0, 1)
    docs, ids = [], []
    while len(docs) < n_docs:
        _, rs = rows(ds, cfg, rng.randrange(0, total - 20), 20)
        for r in rs:
            t = r["text"].strip()
            if len(t) < 2000:
                continue
            if len(t) > 8000:
                cut = t.rfind("\n\n", 2000, 8000)
                t = t[:cut if cut > 0 else 8000]
            docs.append({"text": f"{r['title']}\n\n{t}"})
            ids.append(r["id"])
            if len(docs) >= n_docs:
                break
    log(f"wiki {lang}: {len(docs)} articles")
    return docs, {"dataset": ds, "config": cfg, "license": "CC BY-SA 3.0 / GFDL", "ids": ids}


def code(transformers_dir, ext_dir, n_py, n_cu, rng, log):
    def pick(root, exts, n, lo = 3000, hi = 60000):
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
                       if f.endswith(exts) and lo <= os.path.getsize(os.path.join(dp, f)) <= hi)
        return rng.sample(files, min(n, len(files)))
    docs, used = [], {}
    for root, exts, n, cm, name in ((transformers_dir, (".py",), n_py, "#", "transformers"),
                                    (ext_dir, (".cu", ".cuh", ".cpp", ".h"), n_cu, "//", "exllamav3")):
        if not root:
            continue
        sel = pick(root, exts, n)
        for p in sel:
            rel = os.path.relpath(p, os.path.dirname(root))
            docs.append({"text": f"{cm} {rel}\n{open(p, errors = 'replace').read()}"})
        used[name] = sorted(os.path.relpath(p, os.path.dirname(root)) for p in sel)
        log(f"code {name}: {len(sel)} files")
    rng.shuffle(docs)
    return docs, {"transformers": "Apache-2.0", "exllamav3": "MIT", "files": used}


TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def _obj(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def hermes_messages(row):
    """None when the conversation does not convert cleanly"""
    tools = row["tools"]
    tools = json.loads(tools) if isinstance(tools, str) else tools
    tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    msgs, last_ids, k = [], [], 0
    for m in row["conversations"]:
        who, v = m["from"], m["value"]
        if who == "system":
            continue                                      # Hermes' own tool prompt; the template renders the tools
        if who == "human":
            msgs.append({"role": "user", "content": v})
        elif who == "gpt":
            calls = []
            for c in TOOL_CALL.findall(v):
                o = _obj(c)
                k += 1
                calls.append({"id": f"call_{k}", "type": "function",
                              "function": {"name": o["name"], "arguments": o.get("arguments") or {}}})
            text = TOOL_CALL.sub("", v).strip()
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
            last_ids = [(c["id"], c["function"]["name"]) for c in calls]
        elif who == "tool":
            for i, r in enumerate(TOOL_RESP.findall(v)):
                o = _obj(r)
                cid, name = last_ids[i] if i < len(last_ids) else (None, o.get("name"))
                content = o.get("content", o)
                msgs.append({"role": "tool", "tool_call_id": cid, "name": name or o.get("name"),
                             "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii = False)})
        else:
            return None
    if not any(m.get("tool_calls") for m in msgs):
        return None
    return {"messages": msgs, "tools": tools}


def toolcalls(n_docs, rng, log):
    ds, docs, ids = "NousResearch/hermes-function-calling-v1", [], []
    for cfg, share in (("func_calling", 0.6), ("func_calling_singleturn", 0.4)):
        total, _ = rows(ds, cfg, 0, 1)
        want, got = round(n_docs * share), 0
        while got < want:
            off = rng.randrange(0, total - 20)
            _, rs = rows(ds, cfg, off, 20)
            for r in rs:
                try:
                    d = hermes_messages(r)
                except Exception:
                    d = None
                if d:
                    docs.append(d)
                    ids.append(f"{cfg}:{r['id']}")
                    got += 1
                    if got >= want:
                        break
        log(f"toolcalls {cfg}: {got}")
    rng.shuffle(docs)
    return docs, {"dataset": ds, "license": "Apache-2.0", "ids": ids}


RUN_TOOL = {"type": "function", "function": {
    "name": "run",
    "description": "Run one command of the agent interface (a bash command, or one of the editor commands described in "
                   "the system prompt) and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"}},
                   "required": ["command"]}}}
ACTION = re.compile(r"```[a-zA-Z]*\n(.*?)\n?```\s*$", re.S)


def swe_messages(traj):
    msgs, pending, k = [], None, 0
    for s in traj:
        role, text = s["role"], s.get("text") or ""
        if role == "system":
            msgs.append({"role": "system", "content": s.get("system_prompt") or text})
        elif role == "user":
            if pending:
                msgs.append({"role": "tool", "tool_call_id": pending, "name": "run", "content": text})
                pending = None
            else:
                msgs.append({"role": "user", "content": text})
        elif role == "ai":
            m = ACTION.search(text)
            if m:
                k += 1
                pending = f"call_{k}"
                msgs.append({"role": "assistant", "content": text[:m.start()].strip(), "tool_calls": [
                    {"id": pending, "type": "function", "function": {"name": "run", "arguments": {"command": m.group(1).strip()}}}]})
            else:
                msgs.append({"role": "assistant", "content": text})
                pending = None
    return msgs


def agent(n_docs, rng, log):
    ds, cfg = "nebius/SWE-agent-trajectories", "default"
    total, _ = rows(ds, cfg, 0, 1)
    docs, ids, seen = [], [], 0
    while len(docs) < n_docs:
        _, rs = rows(ds, cfg, rng.randrange(0, total - 10), 10)
        for r in rs:
            seen += 1
            if not r.get("target") or len(r["trajectory"]) < 8:
                continue
            msgs = swe_messages(r["trajectory"])
            if sum(1 for m in msgs if m.get("tool_calls")) < 3:
                continue
            docs.append({"messages": msgs, "tools": [RUN_TOOL]})
            ids.append(f"{r['instance_id']}|{r['model_name']}")
            if len(docs) >= n_docs:
                break
    log(f"agent: {len(docs)} resolved trajectories (of {seen} looked at)")
    return docs, {"dataset": ds, "license": "CC BY 4.0", "ids": ids}


def legal(cail, exclude_md5, n_docs, rng, log):
    held = {l.strip() for l in open(exclude_md5) if l.strip()} if exclude_md5 else set()
    facts, dropped = [], 0
    for line in open(cail):
        o = json.loads(line)
        f = (o.get("fact") or "").strip()
        if len(f) < 80:
            continue
        if hashlib.md5(f.encode()).hexdigest() in held or hashlib.md5(o.get("fact", "").encode()).hexdigest() in held:
            dropped += 1
            continue
        facts.append((o.get("case_id"), f))
    sel = rng.sample(facts, min(n_docs, len(facts)))
    log(f"legal: {len(sel)} CAIL2018 facts ({dropped} exam facts excluded, {len(facts)} available)")
    return [{"text": f} for _, f in sel], {"dataset": "CAIL2018 (public)", "file": os.path.basename(cail),
                                           "exam_facts_excluded": dropped, "case_ids": [c for c, _ in sel]}


def exam_lines(eval_dir):
    """the longest line of each question that no other question shares (the instruction line every prompt starts
    with would exclude all of the model's own reasoning on the calibration questions)"""
    prompts = [json.loads(l)["prompt"] for src in ("jecqa", "math") for l in open(os.path.join(eval_dir, f"eval-{src}.jsonl"))]
    count = {}
    for p in prompts:
        for l in {x.strip() for x in p.split("\n")}:
            count[l] = count.get(l, 0) + 1
    out = []
    for p in prompts:
        own = [l.strip() for l in p.split("\n") if count.get(l.strip()) == 1 and len(l.strip()) >= 20]
        if own:
            out.append(max(own, key = len))
    return out


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--out", required = True)
    ap.add_argument("--transformers", help = "a transformers package directory (Apache-2.0)")
    ap.add_argument("--exllamav3-src", help = "a copy of exllamav3/exllamav3_ext (MIT)")
    ap.add_argument("--cail", help = "CAIL2018 jsonl with a fact field")
    ap.add_argument("--cail-exclude", help = "md5 (utf-8) of facts to leave out, one per line")
    ap.add_argument("--eval-dir", help = "eval-jecqa.jsonl / eval-math.jsonl, for exam_exclude.txt")
    ap.add_argument("--only", help = "comma-separated subset: general,code,toolcalls,agent,legal,exclude")
    ap.add_argument("--seed", type = int, default = 20261116)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok = True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush = True)
    only = set(a.only.split(",")) if a.only else {"general", "code", "toolcalls", "agent", "legal", "exclude"}
    prov_path = os.path.join(a.out, "sources.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}

    def save(name, docs, meta):
        meta["documents"] = len(docs)
        meta["sha256_16"] = write_jsonl(os.path.join(a.out, f"{name}.jsonl"), docs)
        prov[name] = meta
        json.dump(prov, open(prov_path, "w"), ensure_ascii = False, indent = 1)

    if "general" in only:
        rng = random.Random(a.seed)
        zh, mz = wiki("zh", 60, rng, log)
        en, me = wiki("en", 50, rng, log)
        docs = zh + en
        rng.shuffle(docs)
        save("general", docs, {"zh": mz, "en": me})
    if "code" in only:
        docs, meta = code(a.transformers, a.exllamav3_src, 45, 30, random.Random(a.seed + 1), log)
        save("code", docs, meta)
    if "toolcalls" in only:
        docs, meta = toolcalls(260, random.Random(a.seed + 2), log)
        save("toolcalls", docs, meta)
    if "agent" in only:
        docs, meta = agent(14, random.Random(a.seed + 3), log)
        save("agent", docs, meta)
    if "legal" in only and a.cail:
        docs, meta = legal(a.cail, a.cail_exclude, 1500, random.Random(a.seed + 4), log)
        save("legal", docs, meta)
    if "exclude" in only and a.eval_dir:
        lines = exam_lines(a.eval_dir)
        with open(os.path.join(a.out, "exam_exclude.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"exam_exclude.txt: {len(lines)} lines")
    log("done: " + ", ".join(f"{k} {v['documents']}" for k, v in prov.items()))


if __name__ == "__main__":
    main()
