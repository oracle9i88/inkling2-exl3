"""
Rank-0 bookkeeping of the prefix cache (pure Python, no torch): reference-counted KV pages and an LRU index of prefix
snapshots. The server passes two callbacks that run on every rank: `snapshot(slot, sid)` copies a slot's ring state into
snapshot sid, `restore(slot, sid, cow_src, cow_dst)` copies page cow_src to cow_dst (unless -1) and loads snapshot sid
into the slot.

An entry covers the first R tokens of some prompt: its token list, the pages holding positions [0, R) (the last one
partial when R is not a multiple of the page size) and a snapshot of the slot state at R. A request whose prompt starts
with an entry's tokens (and is longer) maps the entry's full pages, gets a copy of the partial page (it writes the rest
of that page), restores the snapshot and prefills from R. Pages are freed when the last holder (request or entry) lets
go; entries are evicted least recently used first when pages or snapshot slots run out.
"""
from __future__ import annotations
import collections


class PagePool:
    def __init__(self, pages):
        self.free = list(pages)
        self.ref = {p: 0 for p in self.free}

    def take(self, n):
        out = [self.free.pop() for _ in range(n)]
        for p in out:
            self.ref[p] = 1
        return out

    def hold(self, pages):
        for p in pages:
            assert self.ref[p] > 0, f"hold on free page {p}"
            self.ref[p] += 1

    def release(self, pages):
        for p in pages:
            assert self.ref[p] > 0, f"release of free page {p}"
            self.ref[p] -= 1
            if self.ref[p] == 0:
                self.free.append(p)


class Entry:
    __slots__ = ("tokens", "R", "pages", "sid", "hits")

    def __init__(self, tokens, R, pages, sid):
        self.tokens, self.R, self.pages, self.sid, self.hits = tokens, R, pages, sid, 0


class PrefixCache:
    def __init__(self, pool: PagePool, page_size: int, nsnap: int, min_len: int, snapshot, restore, stats = None):
        self.pool, self.ps, self.min_len = pool, page_size, min_len
        self.snapshot_cb, self.restore_cb = snapshot, restore
        self.entries = collections.OrderedDict()        # sid -> Entry, least recently used first
        self.free_sids = list(range(nsnap))
        self.enabled = nsnap > 0
        self.stats = stats if stats is not None else {}
        for k in ("prefix_lookups", "prefix_hits", "prefix_hit_tok", "prefix_snapshots", "prefix_evictions"):
            self.stats.setdefault(k, 0)

    # ---- index
    def lookup(self, ids):
        """longest entry whose tokens start `ids` and leave at least one token to prefill"""
        best = None
        for e in self.entries.values():
            if e.R < len(ids) and (best is None or e.R > best.R) and ids[:e.R] == e.tokens:
                best = e
        return best

    def _same(self, ids, R):
        for e in self.entries.values():
            if e.R == R and ids[:R] == e.tokens:
                return e
        return None

    def evict_one(self, keep = None):
        for sid, e in self.entries.items():
            if e is not keep:
                del self.entries[sid]
                self.pool.release(e.pages)
                self.free_sids.append(sid)
                self.stats["prefix_evictions"] += 1
                return True
        return False

    # ---- pages
    def take_pages(self, n, keep = None):
        """n fresh pages, evicting entries (never `keep`) until enough are free; None if they cannot be found"""
        while len(self.pool.free) < n and self.evict_one(keep):
            pass
        if len(self.pool.free) < n:
            return None
        return self.pool.take(n)

    # ---- requests
    def admit(self, slot, ids, need_pages, use_cache = True):
        """Pages for a new request in `slot` needing `need_pages` pages in total (positions from 0). Returns
        (pages, start) where start is the number of prompt tokens restored from the cache (0 = none), or None when
        there are not enough free pages yet (the caller retries later). use_cache=False skips the lookup."""
        e = None
        if self.enabled and use_cache:
            self.stats["prefix_lookups"] += 1
            e = self.lookup(ids)
        nshared = e.R // self.ps if e is not None else 0
        own = self.take_pages(need_pages - nshared, keep = e)
        if own is None:
            return None
        if e is None:
            return own, 0
        shared = e.pages[:nshared]
        self.pool.hold(shared)
        cow_src = e.pages[nshared] if e.R % self.ps else -1
        cow_dst = own[0] if cow_src >= 0 else -1
        self.restore_cb(slot, e.sid, cow_src, cow_dst)
        e.hits += 1
        self.entries.move_to_end(e.sid)
        self.stats["prefix_hits"] += 1
        self.stats["prefix_hit_tok"] += e.R
        return shared + own, e.R

    def add(self, slot, ids, R, pages):
        """The request in `slot` has prefilled exactly its first R tokens into `pages`: keep them as an entry"""
        if not self.enabled or R < self.min_len:
            return
        e = self._same(ids, R)
        if e is not None:
            self.entries.move_to_end(e.sid)
            return
        if not self.free_sids and not self.evict_one():
            return
        sid = self.free_sids.pop()
        self.snapshot_cb(slot, sid)
        e = Entry(list(ids[:R]), R, list(pages[:(R + self.ps - 1) // self.ps]), sid)
        self.pool.hold(e.pages)
        self.entries[sid] = e
        self.stats["prefix_snapshots"] += 1

    def release(self, pages):
        self.pool.release(pages)

    def info(self):
        return {"entries": len(self.entries), "free_pages": len(self.pool.free),
                "lengths": [e.R for e in self.entries.values()], "hits": [e.hits for e in self.entries.values()]}
