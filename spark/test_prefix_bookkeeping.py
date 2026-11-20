"""
Randomized test of the prefix-cache bookkeeping (server/i2_prefix_cache.py), no GPU: page contents are simulated as
the token written at each KV position, slot state as the token list a slot has absorbed.

Workload: agent-like conversations (each request = an earlier prompt + generated text + a new message), fresh prompts,
several requests in flight, small page size and snapshot pool so eviction and copy-on-write happen constantly.
Checked after every operation:
  * every cache entry's pages still hold exactly its tokens at positions [0, R) (nobody wrote into a shared prefix);
  * a restored request sees its own prompt at [0, start) through its page table, and the slot state equals it;
  * page refcounts equal the number of holders (requests + entries); the free list holds exactly the unreferenced pages.

  python3 spark/test_prefix_bookkeeping.py [iterations]
"""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from i2_prefix_cache import PagePool, PrefixCache

PS, NPAGES, NSNAP, MINLEN, SLOTS = 4, 160, 5, 6, 3


def run(seed, steps):
    rnd = random.Random(seed)
    content = {}                     # page -> {offset: token}
    slot_state = {s: [] for s in range(SLOTS)}
    snap_state = {}

    def snapshot(slot, sid):
        snap_state[sid] = list(slot_state[slot])

    def restore(slot, sid, cow_src, cow_dst):
        if cow_src >= 0:
            content[cow_dst] = dict(content.get(cow_src, {}))
        slot_state[slot] = list(snap_state[sid])

    stats = {}
    pool = PagePool(range(1, NPAGES))
    cache = PrefixCache(pool, PS, NSNAP, MINLEN, snapshot, restore, stats)
    free_slots = list(range(SLOTS))
    active = {}                      # slot -> dict(ids, pages, pos)
    history = [[rnd.randrange(50) for _ in range(rnd.randrange(8, 30))] for _ in range(3)]

    def write(slot, req, upto):
        """absorb tokens of req['seq'] from req['pos'] to `upto` (prefill or generation)"""
        for p in range(req["pos"], upto):
            pg = req["pages"][p // PS]
            content.setdefault(pg, {})[p % PS] = req["seq"][p]
            slot_state[slot].append(req["seq"][p])
        req["pos"] = upto

    def check():
        for e in cache.entries.values():
            got = [content.get(e.pages[p // PS], {}).get(p % PS) for p in range(e.R)]
            assert got == e.tokens, f"entry sid {e.sid} R {e.R} corrupted"
        holders = {}
        for r in active.values():
            for pg in r["pages"]:
                holders[pg] = holders.get(pg, 0) + 1
        for e in cache.entries.values():
            for pg in e.pages:
                holders[pg] = holders.get(pg, 0) + 1
        for pg, n in pool.ref.items():
            assert n == holders.get(pg, 0), f"page {pg}: ref {n} != holders {holders.get(pg, 0)}"
        assert sorted(pool.free) == sorted(pg for pg, n in pool.ref.items() if n == 0), "free list mismatch"
        assert len(set(pool.free)) == len(pool.free), "duplicate free page"

    restored = 0
    for step in range(steps):
        op = rnd.random()
        if op < 0.45 and free_slots:
            # new request: extend an earlier conversation (agent turn) or start fresh
            if rnd.random() < 0.8 and history:
                base = rnd.choice(history)
                cut = len(base) if rnd.random() < 0.7 else rnd.randrange(1, len(base) + 1)   # sometimes the history was edited
                ids = base[:cut] + [rnd.randrange(50) for _ in range(rnd.randrange(1, 12))]
            else:
                ids = [rnd.randrange(50) for _ in range(rnd.randrange(4, 40))]
            gen = rnd.randrange(1, 10)
            need = (len(ids) + gen + PS - 1) // PS
            slot = free_slots[0]
            got = cache.admit(slot, ids, need)
            if got is None:
                continue
            free_slots.pop(0)
            pages, start = got
            if start:
                restored += 1
                seen = [content.get(pages[p // PS], {}).get(p % PS) for p in range(start)]
                assert seen == ids[:start], "restored request does not see its prefix"
                assert slot_state[slot] == ids[:start], "restored slot state wrong"
            else:
                slot_state[slot] = []
            seq = ids + [rnd.randrange(50) for _ in range(gen)]
            active[slot] = dict(ids = ids, seq = seq, pages = pages, pos = start, gen = gen)
        elif op < 0.85 and active:
            slot = rnd.choice(list(active)); r = active[slot]
            if r["pos"] < len(r["ids"]):
                upto = min(len(r["ids"]), r["pos"] + rnd.randrange(1, 9))        # prefill a chunk
                mid = r["pos"]
                write(slot, r, upto)
                if upto == len(r["ids"]) or (upto // 8 != mid // 8):            # snapshot at the end / every 8 tokens
                    cache.add(slot, r["ids"], upto, r["pages"])
            elif r["pos"] < len(r["seq"]):
                write(slot, r, min(len(r["seq"]), r["pos"] + rnd.randrange(1, 4)))   # generate
            else:
                cache.release(r["pages"]); del active[slot]; free_slots.append(slot)
                history.append(r["seq"])
                if len(history) > 12:
                    history.pop(0)
        elif active and rnd.random() < 0.3:
            slot = rnd.choice(list(active)); r = active.pop(slot)                # client gave up mid-way
            cache.release(r["pages"]); free_slots.append(slot)
        check()
    return stats, restored


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tot = {}
    for seed in range(n):
        st, restored = run(seed, 400)
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        tot["restored_requests"] = tot.get("restored_requests", 0) + restored
    print(f"{n} seeds x 400 steps: all invariants held; totals {tot}")
