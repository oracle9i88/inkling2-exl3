# inkling2-exl3

One-command serving and speed sweeps for [Inkling-2-Small EXL3 4.0 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw)
on **4x 96 GB Blackwell GPUs (RTX PRO 6000, SM120)**. It is an [exllamav3](https://github.com/turboderp-org/exllamav3) (v1.6.6) overlay that adds:
- the Inkling-2 architecture;
- a batched engine, with attention split across the 4 GPUs (TP4) and the routed experts in an **EP2xTP2** layout (2 expert groups x 2 intermediate halves, sharded straight from the checkpoint);
- a paged KV cache that knows about the sliding-window layers, with rotated, grouped 4-bit global-layer KV;
- CUDA-graphed **speculative decoding (DSpark/MTP), always on**, with a draft length per batch-size bucket, FP8 weight-only draft projections and a fused Markov-head step;
- every routed-expert call (verify and prefill, in the default layout) through the [b12x](https://github.com/local-inference-lab/b12x) fused Trellis MoE kernels from Local Inference Lab (now in FlashInfer `flashinfer/experimental/b12x`). This is on by default;
- Triton kernels for the decode-sized paths: a skinny GEMM for the verify projections, relative-position bias fused into attention, and a fused deterministic router. All-reduces run in fp16.

## Run

```bash
# OpenAI-compatible server on :8000 (16 slots x 380K context)
docker run --rm --gpus all --ipc=host -p 8000:8000 -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest serve

# speed sweep (c1/c2/c4/c8/c16 decode tok/s, acceptance length, prefill tok/s)
docker run --rm --gpus all --ipc=host -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest sweep

# import check, no GPU memory used: exllamav3 extension, overlay modules, b12x kernels
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= ghcr.io/0xsero/inkling2-exl3:latest check
```

The first run downloads the model (~313 GB, resumable) into `$HOME/models/Inkling-2-Small-EXL3-4bpw`.
- `sweep` prints a markdown table (median of 3 repeats) and writes JSON to `/models/results/`.
- `serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. `/health` also reports the KV token capacity (`kv_tokens`).
- On the first request of each batch size, the b12x kernels JIT-compile through the CUTLASS DSL, so the first requests are slow.
- Images are tagged by engine commit: `:latest` = `:v3-bfa11bed` (this snapshot); `:v3-40fafdc7` is the previous one.

| Env (pass with `-e`) | Default | |
|---|---|---|
| `PORT` | `8000` | server port |
| `I2_B12X` | `1` | routed experts through b12x; `0` = exllamav3's coop MoE kernel, in the EP4 layout (`I2_EP_LAYOUT` is then ignored) |
| `I2_EP_LAYOUT` | `ep2tp2` | b12x expert layout: `ep2tp2` (2 groups x 2 intermediate halves, every MoE call through b12x) or `ep4` (exllamav3 experts repacked; b12x for verify, coop for prefill) |
| `I2_DRAFT_FP8` | `1` | draft block projections FP8 weight-only (e4m3, per-row scales); `0` = bf16 |
| `CONCURRENCY` | `1,2,4,8,16` | sweep batch sizes |
| `SPEC_K` | `1:4,2:3,4:2,8:2,16:2` | sweep draft length per batch size (same as the server's `--spec-k`) |
| `REPEATS` / `STEPS` / `PROMPT` | `3` / `128` / `256` | sweep settings |
| `GRAPH` | `1` | CUDA graphs |
| `KV_ROT` / `KV_NG` / `KV_Q16` | `1` / `4` / `1` | sweep global-KV format (same as the server) |
| `I2_MOE_GROUP` | `64` | tokens per coop MoE call (the coop paths only: `I2_B12X=0`, or prefill with `ep4`) |
| `MODEL_DIR` / `HF_REPO` / `RESULTS_DIR` | `/models/...` | |

Server flags go after `serve`:

| Flag | Default | |
|---|---|---|
| `--max-model-len` | `380000` | |
| `--slots` | `16` | concurrent sequences |
| `--spec-k` | `1:4,2:3,4:2,8:2,16:2` | drafted tokens per step, by decode bucket (smallest bucket >= live rows). One number (1..7) sets all buckets |
| `--burst` | `1` | back-to-back speculative steps per scheduler round, with one host sync. Kept as an option; it barely helped (c1 13.8 -> 13.7 ms/step) |
| `--bits-g` / `--bits-l` | `4` / `8` | KV bits for global / sliding-window layers; `--bits-g 8` for quality, at about half the token capacity |
| `--kv-rot` / `--kv-ng` / `--kv-q16` | `1` / `4` / `1` | global 4-bit KV: Hadamard rotation along head_dim, 4 fp16 scale groups per 128-vector, 16-level symmetric grid |
| `--kv-gb` | auto | KV budget per GPU |
| `--b12x` | `$I2_B12X` | |

Speculative decoding has no off switch. It runs at every batch size.

### Without Docker
1. Make a Python 3.12 venv. Install `docker/requirements-engine.lock` with `--extra-index-url https://download.pytorch.org/whl/cu128`, then `docker/requirements-b12x.lock` with `--no-deps`.
   - The second lock deliberately overrides torch's `cuda-bindings` pin. This is the validated combination.
2. Run `scripts/install_overlay.sh <exllamav3 v1.6.6 checkout>`. Then run `pip install --no-build-isolation .` in that checkout with `TORCH_CUDA_ARCH_LIST=12.0`, against the CUDA 12.8 toolkit.
3. Fetch FlashInfer at commit `9c50996794564e1e3b2a4b0683e9aa2bcc528df8`. No build is needed.
4. Set:
   - `I2_B12X_PATH=<tree>`;
   - `PYTHONPATH=<tree>:<tree>/flashinfer/experimental/b12x/_compat`;
   - `I2_EP_LAYOUT=ep2tp2` and `I2_DRAFT_FP8=1` (the server defaults; the code alone falls back to `ep4` and a bf16 draft);
   - `EXL3_LOAD_ARENA=0`;
   - `NCCL_CUMEM_ENABLE=0`.
5. Run `scripts/sweep.sh` or `torchrun --nproc_per_node 4 server/i2_server.py --port 8000 --b12x 1`.

FlashInfer's own `b12x` extra asks for torch >= 2.12. The kernels import and run on torch 2.10 here, and that is the validated setup, so the image installs the extra's other dependencies at fixed versions and leaves out the torch floor.

## Requirements
- 4x SM120 GPUs with 96 GB each and NCCL P2P;
- an NVIDIA driver supporting CUDA 12.8 (>= 570). Tested on 595.84;
- ~313 GB free disk;
- Docker with the NVIDIA container toolkit.

## Status (work in progress)
Measured 2026-11-14 on 4x RTX PRO 6000 at the 300 W power cap. Decode is power-bound. All numbers use speculative decoding with CUDA graphs. There is no plain-decode path.

**Quant quality.** The model was quantized against the BF16 reference; full results are on the [model card](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw).

| | Value |
|---|---|
| KLD vs BF16 | 0.0213 |
| Top-1 agreement | 97.35% |

**Speed: served decode, current default** (ledger row for engine commit 74f1060e. The later 0daeb7f3 only adds the `--burst` option, off by default, and an opt-in profiler; a burst-4 run gave 192 / 286 / 389 / 488 / 660). Fixed realistic prompt set: 16 authored chat / code / agentic+tool prompts, at least 2K generated tokens per row. These are server tok/s with prefill excluded, from the server's `/health` counters.

| | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| temp 0, tok/s | 190 | 292 | 373 | 498 | 665 |
| temp 0.7, tok/s | 175 | 267 | 359 | 472 | 617 |

- Acceptance length (tokens per step, temp 0): 2.63 at c1, 2.18 at c16. The draft lengths are 4 / 3 / 2 / 2 / 2 tokens for c1 / c2 / c4 / c8 / c16. These came from a server sweep; shallow drafts win on real prompts.
- Step time: 13.8 ms at c1, 45.7 ms at c16. The engine alone takes 12.6 ms at c1; the ~1.2 ms difference is server host overhead between graph replays.
- The previous snapshot (40fafdc7) served 149 / 244 / 308 / 369 tok/s at c1 / c4 / c8 / c16 on the same prompts (temp 0, K = 7 at every size, EP4).
- What moved it: fp16 all-reduces and per-bucket draft length, the fused router, EP2xTP2 experts, the FP8 draft, the fused Markov head and relative bias, and the skinny verify GEMM. The step-by-step ledger is in the engine fork (`serving/SPEED_LOOP.md`).

**Ceiling.** All four GPUs sit at the 300 W cap in every phase.
- A streaming read reaches 1635 GB/s at 300 W.
- The b12x MoE draws the same 300 W at only ~0.9 TB/s, because the trellis decode and MMA work cost energy per byte. So the bytes/bandwidth floor is out of reach for the MoE share under the cap.
- Against the bandwidth floor (bytes per step / 1635 GB/s + all-reduce latency), the engine step ran at 57% (c1) / 46% (c16) in the last ceiling run, two kernel changes before this snapshot.
- Against a power-aware floor (MoE bytes at the ~0.9 TB/s b12x reaches at 300 W), that run was at 79% / 70%. The served default is at ~76% (c1) / ~69% (c16) of it.

**b12x vs coop MoE**, teacher-forced, 20K tokens, 16-bit KV, MoE at 128 rows:

| MoE kernel | KLD | Top-1 |
|---|---:|---:|
| b12x | 0.01489 | 97.08% |
| coop | 0.01480 | 96.92% |

The two are equivalent. Kernel vs fp32 truth (one layer, T = 1..128): b12x rel err ~8.6e-4, coop ~9.6e-4.

**KV cache: 16 x 380K.** The server default is:
- 4-bit global-layer KV, Hadamard-rotated, 16-level grid, 4 scale groups;
- 8-bit sliding-window KV;
- int8 draft KV.

That costs 2824 B/token/GPU. With the EP2xTP2 experts and the FP8 draft, more memory is left for KV: the pool holds **6,984,448 tokens** (it was 6,358,400 in the previous snapshot). 16 x 380K = 6.08M fits, with 14.9% to spare. Pages are shared, so any mix of lengths that totals 6.98M fits. The 40 sliding-window layers keep a 1536-token ring and cost no per-length memory.

KLD vs the BF16 teacher by KV format. These are eval subsets (short = 40K tokens, long = 3 x 32K long-context rows), so compare them only with each other:

| global / SWA KV | short | long | long, positions >= 16K | Top-1 short / long |
|---|---:|---:|---:|---:|
| 16 / 16 | 0.01231 | 0.01978 | 0.01494 | 97.33 / 97.81 |
| 8 / 8 | 0.01349 | 0.01952 | 0.01609 | 97.32 / 97.86 |
| 4 plain / 8 (older default) | 0.01841 | 0.02608 | 0.02111 | 96.35 / 97.24 |
| **4 rotated, q16, 4 groups / 8 (default)** | **0.01698** | **0.02277** | **0.01889** | **96.96 / 97.52** |

- The rotated format recovers about half of the 4-bit KV loss on long context, and about a quarter on short.
- `--bits-g 8` is closest to 16-bit, at 4744 B/token/GPU. That means fewer than 16 x 380K sessions.
- The same eval subsets after this snapshot's speed changes, as each was checked (fp16 all-reduce, EP2xTP2, fused relative bias, skinny GEMM): short 0.0157-0.0170, long 0.0219-0.0239. The fused Markov head is exact (identical greedy sequences). The FP8 draft only touches drafting: acceptance is unchanged at c2-c16, and 2.68 vs 2.73 at c1.

**Long context** (measured on the previous snapshot, 40fafdc7; not rerun on this one):

| Test | Result |
|---|---|
| Needle in a haystack, 375,802-token prompt (needle at 20% depth) | Found; 347 s |
| 16 concurrent sessions of ~24K tokens each (385K prompt tokens) | 16/16 needles correct |

**Next:**
- a pipelined server decode loop, to remove the ~1.2 ms/step host overhead;
- per-round verify depth;
- fewer bytes and less work per MoE byte, the lever under the power cap.

Expect these numbers to move.

**Verification:**
- **Docker GPU run, verified on image `v3-40fafdc7`** (`sha256:1a2b5d90…`, the previous snapshot): `/health` reported `kv_tokens` 6,358,400 with speculative decoding on; a chat request returned a correct answer; a 16-request burst completed with every request OK, at 435 tok/s end-to-end.
- **Image `v3-bfa11bed` (this snapshot, `:latest`): build and in-image import check only.** The check loads the exllamav3 extension and every overlay module (including the new skinny-GEMM and Markov-head modules), imports the b12x kernels from the pinned FlashInfer tree, and compiles the server and sweep. Its containerized GPU run is not verified yet: the GPUs were in use for engine work. The speed, KV and KLD numbers above come from the same code and the same pinned versions outside Docker.
- Refresh the engine code with `scripts/update_snapshot.sh <fork checkout>`. It also syncs `server/i2_server.py`. The current snapshot's fork commit is in `overlay/FORK_COMMIT`. Run `scripts/audit.sh` before pushing.

## Credits
- [Thinking Machines](https://thinkingmachines.ai): Inkling-2-Small (Apache-2.0).
- [turboderp](https://github.com/turboderp-org): exllamav3 (MIT).
- [Local Inference Lab](https://github.com/local-inference-lab):
  - the [b12x](https://github.com/local-inference-lab/b12x) fused Trellis MoE kernels, now in [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x) (Apache-2.0). This image runs them for every routed-expert call;
  - their EXL3 serving recipes. The 4.0 bpw quant's mcg codebook and uniform bit layout follow those recipes.

See `NOTICE.md`. This repo: MIT.
