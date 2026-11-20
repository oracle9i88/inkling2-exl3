"""
Stand-alone check of server/i2_vision.py (CPU is enough): load the reference vision tower from the checkpoint, preprocess
a synthetic image, encode it, and check shapes, finiteness, the rotary table and that placeholder expansion / cache keys
behave. Run inside the dev container:  python spark/test_vision.py --model /models/Inkling-2-Small-EXL3-3.48bpw
"""
import argparse, base64, io, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "/models/Inkling-2-Small-EXL3-3.48bpw")
ap.add_argument("--device", default = "cpu")
a = ap.parse_args()
import torch
from PIL import Image, ImageDraw
from i2_vision import VisionEncoder, expand_image_tokens, cache_key_ids, IMAGE_TOKEN_ID

t0 = time.time()
v = VisionEncoder(a.model, torch.device(a.device))
print(f"loaded {v.n_params/1e6:.1f}M params in {time.time()-t0:.1f}s on {a.device}")
ref_inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype = torch.float) / 32))
assert v.model.rotary_emb.inv_freq.dtype == torch.float32 and torch.allclose(v.model.rotary_emb.inv_freq.cpu(), ref_inv), "rotary table"

img = Image.new("RGB", (410, 170), "white")
d = ImageDraw.Draw(img); d.rectangle([10, 10, 400, 160], outline = "black", width = 3); d.text((40, 70), "INKLING 2 SEES 42", fill = "black")
buf = io.BytesIO(); img.save(buf, format = "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
im, dg = VisionEncoder.load_image(url)
pv, grid = v.prepare(im)
print("pixel_values", tuple(pv.shape), pv.dtype, "range", float(pv.min()), float(pv.max()), "grid", grid)
assert grid == (5, 11) and pv.shape == (55, 40, 40, 3)          # ceil(170/40)=5 rows, ceil(410/40)=11 columns
t0 = time.time()
f = v.encode(pv, [grid])
print("features", tuple(f.shape), f.dtype, f"in {time.time()-t0:.1f}s; mean |x| {f.float().abs().mean():.4f}, finite {bool(torch.isfinite(f.float()).all())}")
assert f.shape == (55, 4608) and torch.isfinite(f.float()).all()
f2 = v.encode(pv, [grid])
print("deterministic:", torch.equal(f, f2))

ids = [1, 2, IMAGE_TOKEN_ID, 3, IMAGE_TOKEN_ID, 4]
ex, pos = expand_image_tokens(ids, [3, 2])
assert ex == [1, 2] + [IMAGE_TOKEN_ID] * 3 + [3] + [IMAGE_TOKEN_ID] * 2 + [4] and pos == [2, 3, 4, 6, 7]
k1 = cache_key_ids(ex, pos, [3, 2], [b"a" * 32, b"b" * 32]); k2 = cache_key_ids(ex, pos, [3, 2], [b"a" * 32, b"c" * 32])
assert k1[:6] == k2[:6] and k1[6:8] != k2[6:8] and all(x < 0 for x in k1[2:5])
try:
    VisionEncoder.load_image("https://example.com/x.png"); raise SystemExit("remote URL should be refused")
except ValueError:
    pass
print("VISION-TEST PASS")
