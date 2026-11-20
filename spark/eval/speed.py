"""
Decode speed of a running server, the way the README numbers were taken: one request each of Chinese prose, Chinese
legal text, English prose and code (temperature 0, no thinking, max_tokens 500), then 2 and 4 concurrent Chinese
requests. tokens/s come from the server's own /health counters (decode tokens / decode time), so client latency and
prefill do not count. Run it on an idle server.

  python spark/eval/speed.py [http://rank0-host:18100]
"""
import json, os, sys, time, threading, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("I2_SERVER", "http://127.0.0.1:18100")
OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open     # direct: no client proxy in the numbers
def health(): return json.loads(OPEN(URL + "/health", timeout=20).read())
def chat(p, **kw):
    body = {"model": "i", "messages": [{"role": "user", "content": p}], "temperature": 0, "reasoning_effort": "none", **kw}
    r = json.loads(OPEN(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    return r["choices"][0]["message"].get("content") or ""
def rate(a, b):
    s = {k: b["stats"][k] - a["stats"][k] for k in a["stats"]}
    return (f"{s['dec_tok']/max(1e-9,s['dec_time']):.1f} tok/s, {1000*s['dec_time']/max(1,s['dec_steps']):.1f} ms/step, "
            f"{s['dec_tok']/max(1,s['dec_rows']):.2f} tok/row-step, spec {s['spec_steps']} plain {s['plain_steps']}")
P = {"zh-essay": "请写一篇大约600字的散文，题目是《秋天的河》。",
     "zh-legal": "请用约400字解释刑法中盗窃罪与诈骗罪的区别，并各举一例。",
     "en-prose": "Write about 400 words on why rivers matter to cities.",
     "code": "Write a Python function that parses an ISO-8601 date string without using datetime.fromisoformat, with docstring and tests."}
for name, p in P.items():
    a = health(); out = chat(p, max_tokens=500); b = health()
    print(f"[c1 {name}] {rate(a, b)} | {out[:40]!r}", flush=True)
for C in (2, 4):
    a = health(); ts = [threading.Thread(target=chat, args=(P["zh-essay"],), kwargs=dict(max_tokens=400)) for _ in range(C)]
    [t.start() for t in ts]; [t.join() for t in ts]; b = health()
    print(f"[c{C} zh-essay] {rate(a, b)}", flush=True)
