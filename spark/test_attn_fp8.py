"""
Cost of FP8 attention projections (I2_ATTN_FP8) on the decode path, one GPU, first N layers.

Three runs over the same 1,500 tokens, all through the decode-sized (skinny GEMM) path except the first:
  bf16_512  BF16, 512-row chunks (prefill path)               -> reference
  bf16_64   BF16, 64-row chunks (decode path)                  -> noise floor of merely changing the batch shape
  fp8_64    FP8 attention projections, 64-row chunks           -> the change under test
Reports KL divergence / top-1 agreement of each against bf16_64 and bf16_512.

  python spark/test_attn_fp8.py --model M --layers 12
"""
import argparse, os, json
ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "/models/Inkling-2-Small-EXL3-3.48bpw")
ap.add_argument("--layers", type = int, default = 12)
ap.add_argument("--tokens", type = int, default = 1500)
a = ap.parse_args()
import torch


def main():
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    from exllamav3.modules.inkling2_engine import I2Engine, Batch
    from tokenizers import Tokenizer
    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg, tp_rank = 0, tp_size = 1)
    model.layers = model.layers[:a.layers]
    model.embed.load(dev)
    for L in model.layers:
        L.load(dev)
    model.head.load(dev)
    tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    text = open(os.path.join(a.model, "README.md"), encoding = "utf-8").read() * 8
    ids = torch.tensor(tk.encode(text).ids[:a.tokens], dtype = torch.long, device = dev)
    PS = 64; PP = (a.tokens + PS - 1) // PS + 2

    def run(fp8, chunk):
        os.environ["I2_ATTN_FP8"] = "1" if fp8 else "0"
        eng = I2Engine(model, dev, 0, 1, max_slots = 1, num_pages = PP + 4, page_size = PS, bits_g = 4, bits_l = 8,
                       max_rows = 512, kv_rot = True, kv_ng = 4, kv_q16 = True)
        for p in range(PP):
            eng.page_table[0, p] = p
        zs = []
        with torch.no_grad():
            for c0 in range(0, a.tokens, chunk):
                c = ids[c0:c0 + chunk]; m = c.shape[0]
                b = Batch(torch.zeros((m,), dtype = torch.int32, device = dev), torch.arange(c0, c0 + m, dtype = torch.int32, device = dev),
                          torch.arange(m, dtype = torch.int32, device = dev), True, nsplit_g = 8, nsplit_l = 4)
                zs.append(eng.forward(c, b).double().cpu())
        del eng; torch.cuda.empty_cache()
        return torch.cat(zs)[:, :cfg.i2.unpadded_vocab]

    def cmp(z1, z2):
        p1, p2 = torch.log_softmax(z1, -1), torch.log_softmax(z2, -1)
        kld = (p1.exp() * (p1 - p2)).sum(-1)
        return {"kld_mean": float(kld.mean()), "kld_p99": float(kld.quantile(0.99)), "kld_max": float(kld.max()),
                "top1": float((z1.argmax(-1) == z2.argmax(-1)).float().mean())}

    r512 = run(False, 512); r64 = run(False, 64); f64 = run(True, 64)
    res = {"layers": a.layers, "tokens": a.tokens,
           "bf16_64 vs bf16_512 (batch-shape noise)": cmp(r512, r64),
           "fp8_64 vs bf16_64 (FP8 cost)": cmp(r64, f64),
           "fp8_64 vs bf16_512": cmp(r512, f64)}
    print("ATTN-FP8", json.dumps(res, indent = 1), flush = True)


if __name__ == "__main__":
    main()
