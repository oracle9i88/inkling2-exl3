#!/usr/bin/env python3
"""
Collect Inkling's own long reasoning on hard questions, as calibration text for the expert quantization (the
activations of the model's own thinking are what matters most in use). Public question sets, each split once into a
calibration part (used here) and an evaluation part (never shown to the model here; kept for comparing the two-Spark
quant with the three-Spark one):
  jecqa  case-analysis questions of the Chinese national judicial examination (JEC-QA via AGIEval), 300 held out
  math   MATH-500, levels 4 and 5, 100 held out

Polite to whoever else uses the server:
  - starts only after --not-before, and only once the server has had no request of anyone else for --quiet-minutes;
    while other requests show up it submits nothing new (requests already running finish)
  - at most --concurrency requests of ours at a time (default 3 of the server's 4 slots)
  - prompts stay under the server's prefix-snapshot minimum (1024 tokens), so these requests never take a snapshot and
    never evict the cached prefixes of long agent sessions; "prefix_cache": false also skips the lookup
Results are appended as they come (resumable): {"id", "source", "messages": [user, assistant with reasoning_content]}.

  python gen_reasoning.py --source jecqa:jecqa_ca.json --source math:MATH-500.json --out reasoning.jsonl \
      --target-tokens 100000 --not-before 07:30
"""
from __future__ import annotations
import argparse, datetime, json, os, random, re, threading, time, urllib.request

PROMPTS = {
    "jecqa": ("下面是一道法律职业资格考试的案例分析题（可能有多个正确选项）。请仔细分析案情，逐项判断每个选项涉及的法律规定是否成立，"
              "最后给出答案和简要理由。\n\n{q}"),
    "math": "Solve the following problem. Reason carefully and put the final answer in \\boxed{{}}.\n\n{q}",
}
EVAL_N = {"jecqa": 300, "math": 100}
SPLIT_SEED = 20261116


def load_source(name, path):
    """-> list of (id, prompt, gold)"""
    rows = json.load(open(path))
    out = []
    if name == "jecqa":
        for i, r in enumerate(rows):
            q = re.sub(r"\s*答案：从A到D, 我们应选择\s*$", "", r["query"]).strip()
            q = re.sub(r"^问题：", "", q)
            out.append((f"jecqa-{i:04d}", PROMPTS[name].format(q = q), r["gold"]))
    elif name == "math":
        for r in rows:
            if int(r["level"]) >= 4:
                out.append((f"math-{r['unique_id']}", PROMPTS[name].format(q = r["problem"]), r["answer"]))
    else:
        raise SystemExit(f"unknown source {name}")
    return out


def split(name, items, split_dir):
    """fixed calibration / evaluation split per source, stored once and reused"""
    p = os.path.join(split_dir, f"split-{name}.json")
    if os.path.exists(p):
        s = json.load(open(p))
    else:
        ids = sorted(i for i, _, _ in items)
        random.Random(SPLIT_SEED).shuffle(ids)
        s = {"eval": sorted(ids[:EVAL_N[name]]), "calib": sorted(ids[EVAL_N[name]:]), "seed": SPLIT_SEED}
        json.dump(s, open(p, "w"), indent = 1)
        with open(os.path.join(split_dir, f"eval-{name}.jsonl"), "w") as f:
            ev = set(s["eval"])
            for i, prompt, gold in items:
                if i in ev:
                    f.write(json.dumps({"id": i, "prompt": prompt, "gold": gold}, ensure_ascii = False) + "\n")
    cal = set(s["calib"])
    assert not cal & set(s["eval"])
    return [x for x in items if x[0] in cal]


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--server", default = os.environ.get("I2_SERVER", "http://127.0.0.1:18100"))
    ap.add_argument("--model", default = "inkling-2-small")
    ap.add_argument("--source", action = "append", required = True, help = "NAME:FILE (jecqa, math)")
    ap.add_argument("--out", required = True)
    ap.add_argument("--target-tokens", type = int, default = 100_000, help = "stop after this many generated tokens")
    ap.add_argument("--concurrency", type = int, default = 3)
    ap.add_argument("--thinking-budget", type = int, default = 12_000)
    ap.add_argument("--max-tokens", type = int, default = 16_000)
    ap.add_argument("--temperature", type = float, default = 0.8)
    ap.add_argument("--max-prompt-chars", type = int, default = 1200, help = "skip longer prompts (stay under 1024 tokens)")
    ap.add_argument("--not-before", help = "HH:MM local time")
    ap.add_argument("--quiet-minutes", type = float, default = 10.0)
    ap.add_argument("--seed", type = int, default = 0)
    a = ap.parse_args()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # never through a local proxy

    def call(path, body = None, timeout = 30):
        req = urllib.request.Request(a.server + path, data = None if body is None else json.dumps(body).encode(),
                                     headers = {"Content-Type": "application/json", "Authorization": "Bearer local"})
        with opener.open(req, timeout = timeout) as r:
            return json.load(r)

    split_dir = os.path.dirname(os.path.abspath(a.out))
    pools = []
    for s in a.source:
        name, path = s.split(":", 1)
        items = [x for x in split(name, load_source(name, path), split_dir) if len(x[1]) <= a.max_prompt_chars]
        random.Random(a.seed).shuffle(items)
        pools.append([(name,) + x for x in items])
    order = [x for grp in zip(*pools) for x in grp] if len(pools) > 1 else pools[0]     # round robin over sources
    done, total = set(), 0
    if os.path.exists(a.out):
        for line in open(a.out):
            o = json.loads(line)
            done.add(o["id"])
            total += o["usage"]["completion_tokens"]
    todo = [x for x in order if x[1] not in done]
    log = lambda m: print(time.strftime("%m-%d %H:%M:%S"), m, flush = True)
    log(f"{len(order)} calibration questions ({', '.join(f'{len(p)} {p[0][0]}' for p in pools if p)}), "
        f"{len(done)} done ({total} tokens), target {a.target_tokens}")

    if a.not_before:
        hh, mm = (int(x) for x in a.not_before.split(":"))
        now = datetime.datetime.now()
        start = now.replace(hour = hh, minute = mm, second = 0, microsecond = 0)
        if start < now - datetime.timedelta(hours = 12):
            start += datetime.timedelta(days = 1)
        if start > now:
            log(f"waiting until {start:%m-%d %H:%M}")
            time.sleep((start - now).total_seconds())

    lock = threading.Lock()
    state = {"total": total, "inflight": 0, "last_other": time.time()}

    def others():
        try:
            h = call("/health")
            with lock:
                return h.get("active", 0) + h.get("queued", 0) - state["inflight"]
        except Exception as e:
            log(f"health check failed: {e}")
            return 1

    def worker(src, qid, text):
        body = {"model": a.model, "messages": [{"role": "user", "content": text}], "temperature": a.temperature,
                "max_tokens": a.max_tokens, "thinking_budget": a.thinking_budget, "prefix_cache": False}
        t0 = time.time()
        try:
            r = call("/v1/chat/completions", body, timeout = 3600)
            m = r["choices"][0]["message"]
            out = {"id": qid, "source": src,
                   "messages": [{"role": "user", "content": text},
                                {"role": "assistant", "reasoning_content": m.get("reasoning_content") or "",
                                 "content": m.get("content") or ""}],
                   "usage": r["usage"], "finish_reason": r["choices"][0].get("finish_reason"),
                   "seconds": round(time.time() - t0)}
            with lock:
                with open(a.out, "a") as f:
                    f.write(json.dumps(out, ensure_ascii = False) + "\n")
                state["total"] += r["usage"]["completion_tokens"]
                tot = state["total"]
            log(f"{qid}: {r['usage']['completion_tokens']} tokens in {out['seconds']}s, total {tot}")
        except Exception as e:
            log(f"{qid} failed: {e}")
        finally:
            with lock:
                state["inflight"] -= 1

    threads, waiting_logged = [], False
    for src, qid, text, _ in todo:
        while True:
            with lock:
                reached, inflight = state["total"] >= a.target_tokens, state["inflight"]
            if reached:
                break
            if inflight >= a.concurrency:
                time.sleep(2)
                continue
            o = others()
            if o > 0:
                state["last_other"] = time.time()
            quiet = time.time() - state["last_other"]
            if quiet >= a.quiet_minutes * 60:
                waiting_logged = False
                break
            if not waiting_logged:
                log(f"waiting for {a.quiet_minutes:g} quiet minutes (other requests on the server: {max(o, 0)})")
                waiting_logged = True
            time.sleep(30)
        with lock:
            if state["total"] >= a.target_tokens:
                break
            state["inflight"] += 1
        t = threading.Thread(target = worker, args = (src, qid, text), daemon = True)
        t.start()
        threads.append(t)
        time.sleep(1)
    for t in threads:
        t.join()
    log(f"done: {state['total']} generated tokens in {a.out}")


if __name__ == "__main__":
    main()
