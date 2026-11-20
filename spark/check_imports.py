"""Spark (GB10, sm_121) import + smoke check for exllamav3 v1.6.6 + the Inkling-2 overlay, without b12x.
Run inside the dev container with PYTHONPATH=/work/build/exllamav3."""
import os, sys, time
import torch

print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"GPU {torch.cuda.get_device_name()} cap {torch.cuda.get_device_capability()}")
from exllamav3.ext import exllamav3_ext as ext
print(f"exllamav3 extension: {ext.__file__}")
import exllamav3.modules.inkling2_engine, exllamav3.modules.inkling2_draft  # noqa: F401
import exllamav3.modules.inkling2_gemm, exllamav3.modules.inkling2_markov, exllamav3.modules.inkling2_ops  # noqa: F401
print("Inkling-2 overlay modules import (b12x not used on Spark)")
for f in ("/work/engine/server/i2_server.py", "/work/engine/bench/i2_sweep.py"):
    compile(open(f, encoding="utf-8").read(), f, "exec")
print("server and sweep compile")

# Triton on sm_121: the skinny GEMM the engine uses for decode-sized projections, against torch.mm
from exllamav3.modules.inkling2_gemm import skinny_mm
torch.manual_seed(0)
for M, N, K in ((1, 6144, 4608), (8, 4608, 6144), (64, 1024, 4608)):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
    ref = (x.float() @ w.float().t())
    t = time.time(); y = skinny_mm(x, w, out_dtype=torch.float32); torch.cuda.synchronize(); dt = time.time() - t
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"skinny_mm M={M:3d} N={N} K={K}: rel max err {err:.2e} (first call incl. JIT {dt:.1f}s)")
    assert err < 1e-2, err
print("OK")
