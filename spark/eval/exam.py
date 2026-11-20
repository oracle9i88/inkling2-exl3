#!/usr/bin/env python3
"""
Held-out hard-question exam for an Inkling-2 server (three-Spark build now, the two-Spark quant later; same questions,
same settings, so the difference is the quantization):
  jecqa  300 case-analysis questions of the Chinese national judicial examination (JEC-QA via AGIEval), multiple
         answers possible, all-or-nothing
  math   100 MATH-500 problems of levels 4-5, final answer in \\boxed{}
Both sets were split off with gen_reasoning.py (seed 20261116) and are never used for calibration.

Greedy decoding (temperature 0), thinking capped at --thinking-budget tokens. Questions alternate between the sets
(3 jecqa : 1 math), so a partial run is representative. Up to --concurrency requests; one slot is left free whenever
someone else's request is on the server. Resumable; answers and grades go to <out>/answers-<tag>.jsonl.

  python exam.py --eval-dir calib-data/gen --out exam --tag 3spark
  python exam.py --eval-dir calib-data/gen --out exam --tag 3spark --report
"""
from __future__ import annotations
import argparse, http.client, json, os, re, socket, threading, time, urllib.request
from urllib.parse import urlsplit

JECQA_SUFFIX = "\n\n请在回答的最后一行单独写出“答案：”加选项字母（例如“答案：ABD”）。"


class KeepAliveConnection(http.client.HTTPConnection):
    """TCP keepalive on, so a request whose connection died silently (on 11-16 08:01 the server finished four answers
    that never arrived, and the client sat waiting for its read timeout) fails within minutes and is retried; a
    server that is still thinking answers the probes, so long generations are unaffected"""

    def connect(self):
        super().connect()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, val in (("TCP_KEEPALIVE", 60), ("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
            if hasattr(socket, name):
                self.sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), val)


# ----------------------------------------------------------------------------------------------------------------------
# grading

def jecqa_letters(text: str):
    t = text.replace("：", ":").replace("（", "(").replace("）", ")")
    for pat in (r"答案\s*[:为是]?\s*\**\s*([A-D](?:[\s、,，和与及/]*[A-D])*)",
                r"(?:正确|应)选(?:项|择)?\s*(?:为|是)?\s*:?\s*([A-D](?:[\s、,，和与及/]*[A-D])*)",
                r"\\boxed\{\s*([A-D](?:[\s,，、]*[A-D])*)\s*\}"):
        m = re.findall(pat, t)
        if m:
            return "".join(sorted(set(re.findall(r"[A-D]", m[-1]))))
    tail = t[-80:]
    m = re.findall(r"(?<![A-Za-z])([A-D](?:[\s、,，]*[A-D])*)(?![A-Za-z])", tail)
    return "".join(sorted(set(re.findall(r"[A-D]", m[-1])))) if m else ""


def last_boxed(text: str):
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _fix_fracs(s):
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub[:1] == "{" or len(sub) < 2:
            out += sub
            continue
        a, b, rest = sub[0], sub[1], sub[2:]
        out += "{" + a + "}" + ("{" + b + "}" + rest if b != "{" else b + rest)
    return out


def norm_math(s: str):
    """the usual MATH answer normalization (after Hendrycks et al.), slightly extended"""
    if s is None:
        return None
    s = s.strip().replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"(?<=\S)\s*\\(?:text|mbox)\{\s+[^{}]*\}\s*$", "", s)      # trailing unit: 6\text{ trinkets} -> 6
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if re.fullmatch(r"[a-zA-Z]\s*=\s*.+", s) and len(s.split("=")[0].strip()) == 1:
        s = s.split("=", 1)[1]
    s = _fix_fracs(s.replace(" ", ""))
    s = re.sub(r"\\sqrt(?=[0-9a-zA-Z])(\w)", r"\\sqrt{\1}", s)          # \sqrt5 -> \sqrt{5}
    s = re.sub(r"(?<=\d)(?:\^\{?(?:st|nd|rd|th)\}?|st|nd|rd|th)$", "", s)   # 12^{th} -> 12
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        s = f"\\frac{{{m[1]}}}{{{m[2]}}}"
    s = s.rstrip(".")
    if s.endswith("}") and s.startswith("{") and s.count("{") == 1:
        s = s[1:-1]
    return s


def _num(s):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        v = float(m[2]) / float(m[3])
        return -v if m[1] else v
    return float(s)


def _eq1(pred, gold):
    a, b = norm_math(pred), norm_math(gold)
    if a is None:
        return False
    if a == b:
        return True
    ta, tb = _terms(a), _terms(b)
    if len(ta) > 1 and ta == tb:                          # same sum in another order: 11+11\sqrt{5} = 11\sqrt{5}+11
        return True
    try:                                                  # same value: 0.5 = \frac{1}{2}; approximations stay wrong
        x, y = _num(a), _num(b)
        return abs(x - y) < 1e-6 * max(1.0, abs(y))
    except (ValueError, ZeroDivisionError):
        return False


def _terms(s):
    """the terms of a sum at brace depth 0, sorted; a minus sign stays with its term"""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch in "([{") - (ch in ")]}")
        if depth == 0 and ch in "+-" and cur and cur[-1] not in "^_":
            out.append(cur)
            cur = "-" if ch == "-" else ""
        else:
            cur += ch
    out.append(cur)
    return sorted(out)


def _top_split(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch in "([{") - (ch in ")]}")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def _answer_set(s):
    """several answers: '1 \\pm \\sqrt{19}' and '1-\\sqrt{19}, 1+\\sqrt{19}' give the same set"""
    s = s.strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):  # 10,000 is one number
        return [s.replace(",", "")]
    out = []
    for x in _top_split(s):
        if x.count("\\pm") == 1:
            a, b = (y.strip() for y in x.split("\\pm"))
            out += [f"{a}+{b}", f"{a}-{b}"] if a else [b, f"-{b}"]
        else:
            out.append(x)
    return out


def math_equal(pred, gold):
    if pred is None:
        return False
    if _eq1(pred, gold):
        return True
    A, B = _answer_set(pred), _answer_set(gold)
    return len(A) == len(B) and all(any(_eq1(x, y) for y in B) for x in A) and all(any(_eq1(x, y) for x in A) for y in B)


def grade(q, content, reasoning):
    if q["source"] == "jecqa":
        gold = "".join("ABCD"[i] for i in sorted(q["gold"]))
        pred = jecqa_letters(content) or jecqa_letters(reasoning[-400:])
        return pred, gold, pred == gold
    pred = last_boxed(content) or last_boxed(reasoning[-2000:])
    return pred, q["gold"], math_equal(pred, q["gold"])


# ----------------------------------------------------------------------------------------------------------------------

def load_questions(eval_dir):
    qs = {}
    for src in ("jecqa", "math"):
        qs[src] = []
        for line in open(os.path.join(eval_dir, f"eval-{src}.jsonl")):
            o = json.loads(line)
            o["source"] = src
            if src == "jecqa":
                o["prompt"] += JECQA_SUFFIX
            qs[src].append(o)
    order, j, m = [], qs["jecqa"], qs["math"]
    while j or m:
        order += j[:3] + m[:1]
        j, m = j[3:], m[1:]
    return order


def report(path):
    """math is re-graded from the stored answers with the current grader (rows keep what was graded at the time)"""
    rows = [json.loads(l) for l in open(path)]
    for x in rows:
        if x.get("finish_reason") == "length" and x["pred"]:
            x["pred"], x["correct"], x["truncated"] = "", False, True
            continue
        if x["source"] == "math":
            ok = math_equal(x["pred"], x["gold"])
            if ok != x["correct"]:
                x["regraded"] = True
            x["correct"] = ok
    out = {}
    for src in ("jecqa", "math"):
        r = [x for x in rows if x["source"] == src]
        if not r:
            continue
        c = sum(x["correct"] for x in r)
        n = len(r)
        p = c / n
        se = (p * (1 - p) / n) ** 0.5
        out[src] = {"n": n, "correct": c, "acc": round(p, 4), "se": round(se, 4),
                    "no_answer": sum(1 for x in r if not x["pred"]),
                    "hit_thinking_cap": sum(1 for x in r if x.get("thinking_capped")),
                    "regraded": sum(1 for x in r if x.get("regraded")),
                    "truncated": sum(1 for x in r if x.get("truncated") or x.get("finish_reason") == "length"),
                    "mean_completion_tokens": round(sum(x["usage"]["completion_tokens"] for x in r) / n)}
    return out


def main():
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--server", default = os.environ.get("I2_SERVER", "http://127.0.0.1:18100"))
    ap.add_argument("--model", default = "inkling-2-small")
    ap.add_argument("--eval-dir", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--concurrency", type = int, default = 4)
    ap.add_argument("--thinking-budget", type = int, default = 24_000)
    ap.add_argument("--answer-tokens", type = int, default = 4_000)
    ap.add_argument("--limit", type = int, default = 0)
    ap.add_argument("--request-timeout", type = float, default = 2400, help = "backstop read timeout per request, s")
    ap.add_argument("--retries", type = int, default = 2)
    ap.add_argument("--report", action = "store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok = True)
    path = os.path.join(a.out, f"answers-{a.tag}.jsonl")
    if a.report:
        print(json.dumps(report(path), ensure_ascii = False, indent = 1))
        return

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(p, body = None, timeout = 30):
        req = urllib.request.Request(a.server + p, data = None if body is None else json.dumps(body).encode(),
                                     headers = {"Content-Type": "application/json", "Authorization": "Bearer local"})
        with opener.open(req, timeout = timeout) as r:
            return json.load(r)

    srv = urlsplit(a.server)

    def post(p, body, timeout):
        c = KeepAliveConnection(srv.hostname, srv.port or 80, timeout = timeout)
        try:
            c.request("POST", p, body = json.dumps(body).encode(),
                      headers = {"Content-Type": "application/json", "Authorization": "Bearer local"})
            r = c.getresponse()
            data = r.read()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
            return json.loads(data)
        finally:
            c.close()

    order = load_questions(a.eval_dir)
    if a.limit:
        order = order[:a.limit]
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path)}
    todo = [q for q in order if q["id"] not in done]
    log = lambda m: print(time.strftime("%m-%d %H:%M:%S"), m, flush = True)
    log(f"exam {a.tag}: {len(order)} questions, {len(done)} done, thinking budget {a.thinking_budget}")
    lock = threading.Lock()
    st = {"inflight": 0, "n": 0, "c": {"jecqa": [0, 0], "math": [0, 0]}, "tok": 0, "t0": time.time()}

    def worker(q):
        body = {"model": a.model, "messages": [{"role": "user", "content": q["prompt"]}], "temperature": 0.0,
                "max_tokens": a.thinking_budget + a.answer_tokens, "thinking_budget": a.thinking_budget,
                "prefix_cache": False}
        try:
            for attempt in range(a.retries + 1):
                t0 = time.time()
                try:
                    r = post("/v1/chat/completions", body, a.request_timeout)
                    break
                except Exception as e:
                    log(f"{q['id']} failed after {time.time() - t0:.0f}s (attempt {attempt + 1}): {e!r}")
                    if attempt == a.retries:
                        return
                    time.sleep(30)
            m = r["choices"][0]["message"]
            content, reasoning = m.get("content") or "", m.get("reasoning_content") or ""
            pred, gold, ok = grade(q, content, reasoning)
            if r["choices"][0].get("finish_reason") == "length":
                pred, ok = "", False                      # cut off before a final answer: letters found are drafts
            out = {"id": q["id"], "source": q["source"], "pred": pred, "gold": gold, "correct": bool(ok),
                   "content": content, "reasoning_chars": len(reasoning), "reasoning_tail": reasoning[-600:],
                   "thinking_capped": len(reasoning) > 0 and r["usage"]["completion_tokens"] >= a.thinking_budget,
                   "usage": r["usage"], "finish_reason": r["choices"][0].get("finish_reason"),
                   "seconds": round(time.time() - t0)}
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(out, ensure_ascii = False) + "\n")
                cc = st["c"][q["source"]]
                cc[0] += int(ok); cc[1] += 1
                st["n"] += 1; st["tok"] += r["usage"]["completion_tokens"]
                left = len(todo) - st["n"]
                rate = st["n"] / (time.time() - st["t0"])
                summary = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in st["c"].items() if v[1])
            log(f"{q['id']}: {'right' if ok else 'wrong'} (pred {pred!r}, gold {gold!r}), "
                f"{r['usage']['completion_tokens']} tokens, {out['seconds']}s | {summary} | "
                f"{left} left, ETA {left / rate / 3600:.1f} h")
        except Exception as e:
            log(f"{q['id']} failed: {e}")
        finally:
            with lock:
                st["inflight"] -= 1

    threads = []
    for q in todo:
        while True:
            with lock:
                inflight = st["inflight"]
            cap = a.concurrency
            if inflight >= cap - 1:
                try:
                    h = call("/health")
                    if h.get("active", 0) + h.get("queued", 0) - inflight > 0:
                        cap = a.concurrency - 1           # someone else is on the server: leave them a slot
                except Exception:
                    pass
            if inflight < cap:
                break
            time.sleep(3)
        with lock:
            st["inflight"] += 1
        t = threading.Thread(target = worker, args = (q,), daemon = True)
        t.start()
        threads.append(t)
        time.sleep(1)
    for t in threads:
        t.join()
    log("finished: " + json.dumps(report(path), ensure_ascii = False))


if __name__ == "__main__":
    main()
