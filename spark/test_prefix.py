"""
Prefix-cache restore test (one GPU; world 1 with `python`, or 3 ranks on GPU 0 with `torchrun --nproc_per_node 3`).

Run A (slot 0): prefill tokens [0, L1) in 512-token chunks, snapshot the slot, keep prefilling [L1, L2).
Run B (slot 1): map A's full pages below L1, copy the page that holds L1 (copy-on-write, done *after* A wrote past L1,
the worst case), restore the snapshot into slot 1, prefill [L1, L2) with the same chunking.
B's logits over [L1, L2) must equal A's bit for bit (the engine is deterministic for a fixed batch shape).

  python spark/test_prefix.py --model M --layers 12
  torchrun --standalone --nproc_per_node 3 spark/test_prefix.py --model M --layers 12
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "/models/Inkling-2-Small-EXL3-3.48bpw")
ap.add_argument("--layers", type = int, default = 12)
ap.add_argument("--l1", type = int, default = 1500)       # not a multiple of the page size: exercises copy-on-write
ap.add_argument("--l2", type = int, default = 3000)
ap.add_argument("--chunk", type = int, default = 512)
a = ap.parse_args()

import torch
import torch.distributed as dist


def main():
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group("gloo")
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    from exllamav3.modules.inkling2_engine import I2Engine, Batch
    from exllamav3.modules.inkling2_prefix import SlotSnapshots, copy_page
    from tokenizers import Tokenizer

    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg, tp_rank = rank, tp_size = world)
    model.layers = model.layers[:a.layers]
    model.embed.load(dev)
    for L in model.layers:
        L.load(dev)
    model.head.load(dev)
    PS = 64
    PP = (a.l2 + PS - 1) // PS + 2
    eng = I2Engine(model, dev, rank, world, max_slots = 2, num_pages = 2 * PP + 4, page_size = PS, bits_g = 4, bits_l = 8,
                   max_rows = a.chunk, kv_rot = True, kv_ng = 4, kv_q16 = True)
    snaps = SlotSnapshots(eng, None, 2)
    pagesA = list(range(PP))
    for p, pg in enumerate(pagesA):
        eng.page_table[0, p] = pg

    tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    text = open(os.path.join(a.model, "README.md"), encoding = "utf-8").read() * 8
    ids = torch.tensor(tk.encode(text).ids[:a.l2], dtype = torch.long, device = dev)
    assert ids.numel() == a.l2

    def prefill(slot, s0, s1):
        zs = []
        for c0 in range(s0, s1, a.chunk):
            c = ids[c0:min(s1, c0 + a.chunk)]; m = c.shape[0]
            b = Batch(torch.full((m,), slot, dtype = torch.int32, device = dev), torch.arange(c0, c0 + m, dtype = torch.int32, device = dev),
                      torch.arange(m, dtype = torch.int32, device = dev), True, nsplit_g = 8, nsplit_l = 4)
            zs.append(eng.forward(c, b))
        return torch.cat(zs) if zs else None

    with torch.no_grad():
        prefill(0, 0, a.l1)
        snaps.save(0, 0)
        zA = prefill(0, a.l1, a.l2)
        # run B: full pages below L1 shared, the boundary page copied, everything after it fresh
        nfull = a.l1 // PS
        cow = PP
        copy_page(eng, None, pagesA[nfull], cow)
        pagesB = pagesA[:nfull] + [cow] + list(range(PP + 1, PP + 1 + (PP - nfull - 1)))
        for p, pg in enumerate(pagesB):
            eng.page_table[1, p] = pg
        snaps.load(0, 1)
        zB = prefill(1, a.l1, a.l2)
        # control: restoring a *stale* snapshot (taken later) must break equality -> the test can fail
        snaps.save(0, 1)
        snaps.load(1, 1)
        zC = prefill(1, a.l1, a.l2)
    torch.cuda.synchronize()
    same = torch.equal(zA, zB)
    diff = (zA - zB).abs().max().item()
    stale = (zA - zC).abs().max().item()
    print(f"[rank {rank}/{world}] L1={a.l1} L2={a.l2} layers={a.layers}: restored == straight-through: {same} "
          f"(max |diff| {diff:.3g}); stale-snapshot control max |diff| {stale:.3g}; "
          f"snapshot {SlotSnapshots.bytes_per_snapshot(eng, None) / 2**20:.1f} MiB/rank (no draft)", flush = True)
    ok = same and stale > 0
    if world > 1:
        t = torch.tensor([int(ok)]); dist.all_reduce(t, op = dist.ReduceOp.MIN); ok = bool(t.item())
        dist.destroy_process_group()
    if rank == 0:
        print("PREFIX-TEST", "PASS" if ok else "FAIL", flush = True)


if __name__ == "__main__":
    main()
