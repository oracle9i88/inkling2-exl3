"""
Tensor-parallel equivalence test for the KV-group split (stage 1, one GPU).

Runs the first N decoder layers of Inkling-2 (embedding -> N layers -> head) on a fixed prompt, either unsplit
(world 1, plain `python`) or split over 3 ranks that all share GPU 0 (`torchrun --nproc_per_node 3`, gloo backend,
standing in for the 3 Spark nodes). Each run saves selected rows of its logits; `--compare` concatenates the 3 vocab
shards and reports max |diff|, KL divergence and top-1 agreement against the unsplit run.

  python spark/test_tp_equiv.py --model M --layers 12 --out /work/tmp/tp
  torchrun --standalone --nproc_per_node 3 spark/test_tp_equiv.py --model M --layers 12 --out /work/tmp/tp
  python spark/test_tp_equiv.py --compare --out /work/tmp/tp

All-reduces run in fp32 (I2_AR_DTYPE) and the KV cache in bf16 unless --bits-g/--bits-l say otherwise, so the two runs
differ only in the order of the partial sums.
"""
import argparse, os, sys, time, json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "/models/Inkling-2-Small-EXL3-3.48bpw")
ap.add_argument("--layers", type = int, default = 12)
ap.add_argument("--tokens", type = int, default = 1500)
ap.add_argument("--chunk", type = int, default = 512)
ap.add_argument("--bits-g", type = int, default = 16)
ap.add_argument("--bits-l", type = int, default = 16)
ap.add_argument("--ar-dtype", default = "fp32")
ap.add_argument("--out", default = "/work/tmp/tp")
ap.add_argument("--compare", action = "store_true")
ap.add_argument("--tag", default = "", help = "suffix of the saved run name (repeat runs of the same world size)")
ap.add_argument("--ref", default = "world1", help = "--compare: reference run name (e.g. world1)")
ap.add_argument("--cmp", default = "world3", help = "--compare: run compared against --ref (e.g. world3, world1b)")
ap.add_argument("--backend", default = "gloo", help = "gloo: ranks share GPU 0 on one node; nccl: one rank per node (multi-node)")
a, _unknown = ap.parse_known_args()       # the 3-node launcher appends server flags
os.environ.setdefault("I2_AR_DTYPE", a.ar_dtype)

import torch


def rows_to_keep(T):
    keep = sorted(set(range(0, T, 8)) | set(range(max(0, T - 32), T)))
    return torch.tensor(keep, dtype = torch.long)


def load_run(name):
    import glob
    files = sorted(glob.glob(os.path.join(a.out, f"{name}_rank*.pt")), key = lambda f: int(f.rsplit("rank", 1)[1][:-3]))
    assert files, f"no saved run {name} in {a.out}"
    return [torch.load(f) for f in files]


def compare():
    refs, shards = load_run(a.ref), load_run(a.cmp)
    ref = refs[0]
    assert all(torch.equal(s["rows"], ref["rows"]) for s in refs + shards)
    z1 = torch.cat([s["logits"] for s in refs], 1).double()
    z3 = torch.cat([s["logits"] for s in shards], 1).double()
    V = ref["unpadded_vocab"]
    z1, z3 = z1[:, :V], z3[:, :V]
    p1 = torch.log_softmax(z1, -1); p3 = torch.log_softmax(z3, -1)
    kld = (p1.exp() * (p1 - p3)).sum(-1)
    top1 = (z1.argmax(-1) == z3.argmax(-1)).float()
    h1, h3 = ref["hidden"].float(), shards[0]["hidden"].float()
    res = {
        "ref": a.ref, "cmp": a.cmp,
        "rows": int(z1.shape[0]), "layers": ref["layers"], "tokens": ref["tokens"],
        "logits_max_abs_diff": float((z1 - z3).abs().max()), "logits_mean_abs_diff": float((z1 - z3).abs().mean()),
        "logits_scale": float(z1.abs().mean()),
        "kld_mean": float(kld.mean()), "kld_max": float(kld.max()),
        "top1_agree": float(top1.mean()),
        "hidden_rel_max_diff": float((h1 - h3).abs().max() / h1.abs().max()),
        "hidden_ranks_identical": all(torch.equal(shards[0]["hidden"], s["hidden"]) for s in shards[1:]),
        "kv_split": [s["hkl"] for s in shards], "q_heads": [s["hl"] for s in shards],
    }
    print(json.dumps(res, indent = 1))
    json.dump(res, open(os.path.join(a.out, f"compare_{a.ref}_vs_{a.cmp}.json"), "w"), indent = 1)


def run():
    import torch.distributed as dist
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device("cuda:0")              # every rank on GPU 0: this test stands in for 3 nodes on one GPU
    torch.cuda.set_device(dev)
    if world > 1:
        if a.backend == "nccl":
            dist.init_process_group("nccl", device_id = dev)
        else:
            dist.init_process_group("gloo")
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    from exllamav3.modules.inkling2_engine import I2Engine, Batch
    from tokenizers import Tokenizer

    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg, tp_rank = rank, tp_size = world)
    model.layers = model.layers[:a.layers]
    t0 = time.time()
    model.embed.load(dev)
    for L in model.layers:
        L.load(dev)
    model.head.load(dev)
    if world > 1:
        dist.barrier()
    print(f"[rank {rank}/{world}] loaded {a.layers} layers in {time.time() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB, heads q {model.layers[0].Hl} kv {model.layers[0].Hkl}, "
          f"experts [{model.layers[0].first}, {model.layers[0].last})", flush = True)

    PS = 64
    PP = (a.tokens + PS - 1) // PS + 2
    eng = I2Engine(model, dev, rank, world, max_slots = 1, num_pages = PP + 8, page_size = PS,
                   bits_g = a.bits_g, bits_l = a.bits_l, max_rows = a.chunk)
    for p in range(PP):
        eng.page_table[0, p] = p

    tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    text = open(os.path.join(a.model, "README.md"), encoding = "utf-8").read() * 4
    ids = torch.tensor(tk.encode(text).ids[:a.tokens], dtype = torch.long)
    assert ids.numel() == a.tokens, f"prompt has only {ids.numel()} tokens"
    ids = ids.to(dev)

    keep = rows_to_keep(a.tokens)
    out_z, out_h = [], []
    with torch.no_grad():
        for s0 in range(0, a.tokens, a.chunk):
            c = ids[s0:s0 + a.chunk]; m = c.shape[0]
            pos = torch.arange(s0, s0 + m, dtype = torch.int32, device = dev)
            b = Batch(torch.zeros((m,), dtype = torch.int32, device = dev), pos,
                      torch.arange(m, dtype = torch.int32, device = dev), True, nsplit_g = 4, nsplit_l = 4)
            x = eng.forward(c, b, return_hidden = True)                         # [m, D] after the final norm
            z = torch.tanh(torch.mm(x, eng.head.unembed.t(), out_dtype = torch.float32) / eng.cfg.final_cap) * eng.cfg.final_cap
            sel = keep[(keep >= s0) & (keep < s0 + m)] - s0
            out_z.append(z[sel.to(dev)].cpu()); out_h.append(x[sel.to(dev)].cpu())
    torch.cuda.synchronize()
    os.makedirs(a.out, exist_ok = True)
    torch.save({"rows": keep, "logits": torch.cat(out_z), "hidden": torch.cat(out_h), "layers": a.layers,
                "tokens": a.tokens, "unpadded_vocab": eng.cfg.unpadded_vocab, "hkl": eng.Hkl, "hl": eng.Hl},
               os.path.join(a.out, f"world{world}{a.tag}_rank{rank}.pt"))
    print(f"[rank {rank}/{world}] saved {len(keep)} rows; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush = True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    compare() if a.compare else run()
