"""
One rank's shard of a model folder on one GPU (tp_size 2, no collectives): load it the way the server does
(Model.load_rank: embedding, every layer, head), run a fixed random prompt through the engine in prompt chunks and save
the logits of each chunk's last rows. Two folders holding the same served tensors give bit-identical logits; a folder
missing a tensor the engine reads fails at load. The logits are one rank's partial sums, not the model's output.

  PYTHONPATH=/work/build/exllamav3 python spark/release_load_check.py --model /models/X --out /work/x.pt [--rank 0]
  python spark/release_load_check.py --compare a.pt b.pt
"""
import argparse, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--out")
ap.add_argument("--rank", type = int, default = 0)
ap.add_argument("--tokens", type = int, default = 4096)
ap.add_argument("--chunk", type = int, default = 2048)
ap.add_argument("--compare", nargs = 2, metavar = ("A", "B"))
a = ap.parse_args()

if a.compare:
    x, y = torch.load(a.compare[0]), torch.load(a.compare[1])
    same = torch.equal(x["logits"], y["logits"])
    d = (x["logits"] - y["logits"]).abs().max().item()
    print(f"logits {tuple(x['logits'].shape)}: {'bit-identical' if same else f'DIFFERENT (max abs diff {d:.3g})'}")
    raise SystemExit(0 if same else 1)

from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch

dev = torch.device("cuda:0")
torch.cuda.set_device(0)
t0 = time.time()
cfg = Config.from_directory(a.model)
model = Model.from_config(cfg, tp_rank = a.rank, tp_size = 2)
model.load_rank(dev)
nq = sum(bool(getattr(L, "attn_q", False)) for L in model.layers)
print(f"loaded rank {a.rank} of 2 in {time.time() - t0:.0f} s: EXL3 attention in {nq} of {len(model.layers)} layers, "
      f"{'EXL3' if getattr(model.head, 'lin_head', None) is not None else 'bf16'} head, "
      f"GPU free {torch.cuda.mem_get_info()[0] / 2 ** 30:.1f} GiB", flush = True)

PS = 64
pages = a.tokens // PS + 4
eng = I2Engine(model, dev, a.rank, 1, max_slots = 2, num_pages = pages + 2, page_size = PS, bits_g = 4, bits_l = 8,
               max_pages_per_seq = pages, kv_rot = True, kv_ng = 4, kv_q16 = True, max_rows = a.chunk)
eng.page_table[1, :2] = 0
eng.page_table[0, :pages] = torch.arange(2, 2 + pages, dtype = torch.int32, device = dev)
ids = torch.randint(1000, 150000, (a.tokens,), generator = torch.Generator().manual_seed(1234)).to(dev)
outs = []
with torch.no_grad():
    for c0 in range(0, a.tokens, a.chunk):
        n = min(a.chunk, a.tokens - c0)
        B = Batch(torch.zeros(n, dtype = torch.int32, device = dev), torch.arange(c0, c0 + n, dtype = torch.int32, device = dev),
                  torch.arange(n, dtype = torch.int32, device = dev), True, nsplit_g = 8, nsplit_l = 4)
        z = eng.forward(ids[c0:c0 + n], B)
        outs.append(z[-16:].float().cpu())
logits = torch.cat(outs)
torch.save({"logits": logits, "model": a.model, "rank": a.rank}, a.out)
print(f"logits {tuple(logits.shape)}: finite {torch.isfinite(logits).all().item()}, checksum {logits.double().sum().item():.6f}",
      flush = True)
