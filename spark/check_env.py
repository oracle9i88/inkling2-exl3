"""Environment probe for the Spark build: torch/CUDA/GPU and the Python packages exllamav3 + the Inkling-2 overlay import."""
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(), torch.cuda.get_device_name())
print("arch list", torch.cuda.get_arch_list())
free, total = torch.cuda.mem_get_info()
print("cuda mem free %.1f / %.1f GiB" % (free / 2**30, total / 2**30))
for m in ["triton", "tokenizers", "numpy", "rich", "typing_extensions", "safetensors", "ninja", "PIL", "yaml",
          "marisa_trie", "pydantic", "llguidance", "flash_attn", "transformers", "fastapi", "uvicorn"]:
    try:
        x = importlib.import_module(m)
        print(f"{m:18s}", getattr(x, "__version__", "ok"))
    except Exception as e:
        print(f"{m:18s} MISSING ({type(e).__name__})")
