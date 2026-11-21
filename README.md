# Inkling-2-Small on two or three DGX Sparks

Serving [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B parameters,
17B active per token, vision, tool calls) on NVIDIA DGX Spark: one GB10 per node, 128 GB of unified memory, 200G RoCE
between nodes.

- **Two Sparks**: our requantized build, [Evanguo12/Inkling-2-Small-EXL3-2.55bpw-2Spark](https://huggingface.co/Evanguo12/Inkling-2-Small-EXL3-2.55bpw-2Spark).
  The routed experts were requantized from the original checkpoint to 2.55 bits
  per weight on average, with a rate per expert; attention is EXL3 at 6 bits and the output head at 8 bits. Up to
  524,288 tokens per request.
- **Three Sparks**: [0xSero/Inkling-2-Small-EXL3-3.48bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-3.48bpw)
  as published.

This is a fork of [0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3), the Inkling-2 overlay for
[exllamav3](https://github.com/turboderp-org/exllamav3) v1.6.6 written for 4x RTX PRO 6000. Its README is in
[README-upstream.md](README-upstream.md); the top-level Dockerfile, `docker/`, `scripts/` and `bench/` are that 4-GPU
setup. Everything for the Sparks is in `spark/`, `server/` and `overlay/`.

## Known issue (2026-11-21)

The latest server commit `299eeaf` ("drop requests whose client hung up") can misfire under concurrent load: in our
test 3 of 19 requests were dropped while their clients were still connected, and those clients hang. Until it is
fixed, use the previous commit:

```
git checkout 3241926
```

It differs only by this change; the memory fixes (`malloc_trim`, glibc thresholds) are kept. A separate slow rank-0
memory growth under heavy agent-style traffic is still being investigated; restarting the server clears it.

## At a glance

Measured on our nodes, 2026-11-19 (two Sparks) and 2026-11-17 (three Sparks). Speeds are from the server's own
counters (`spark/eval/speed.py`): one request, temperature 0, no thinking, unless noted.

| | Two Sparks | Three Sparks |
|---|---|---|
| Routed experts | EXL3, 2.549 bpw on average, 1.5-4 bpw per expert, 187.8 GB | EXL3, 3.48 bpw, uniform within each layer's projection, 256 GB |
| Attention / output head | EXL3, 6 / 8 bpw | bf16 from the original checkpoint, as in 0xSero's quants |
| Embeddings, vision tower, MTP draft | bf16 from the original checkpoint | same |
| KL divergence from the original, 28,665 held-out tokens | **0.0118** (KV in bf16) / **0.0136** (as served: 4-bit global KV) | not measured on the same rows |
| Top-1 agreement with the original | 96.97% / 96.84% | |
| KV cache | 1.28M tokens; 4 slots of up to 524,288 tokens each | about 2.8M tokens; 4 slots of up to 262,144 |
| Chinese prose / legal text | 24.1 / 24.3 tokens/s | 23.3 / 23.4 tokens/s |
| English prose | 24.8 tokens/s | 23.8 tokens/s |
| Code (speculative steps) | 49.7 tokens/s | 52.4 tokens/s |
| Chinese, 2 / 4 concurrent requests (sum) | 35.9 / 54.7 tokens/s | 39.2 / 77.3 tokens/s |
| Reading a prompt: 63K cold / positions 49K-127K / positions 98K-300K | 646 / 604 / 491 tokens/s | not measured with the long-context kernels |
| Generating with two other requests running: at 6K / 127K context | 20.9 / 20.1 tokens/s | not measured |

**Hard-question exam, two Sparks:** 204 of 300 (68.0%) on JEC-QA case analysis and 100 of 100 on MATH levels 4-5. It
ran on v2: the same experts, with attention and head in bf16. Measured the same way (the quantization pipeline's
held-out check), v3's KL is within 0.0001 of v2's.

The exam (`spark/eval/exam.py`) is 300 case-analysis questions of the Chinese national judicial examination (JEC-QA,
multiple answers, all or nothing) and 100 MATH-500 problems of levels 4 and 5, with greedy decoding and thinking capped
at 24,000 tokens. Neither set was used for calibration.

## What this branch adds

| | 4x RTX PRO 6000 (upstream) | DGX Spark (this branch) |
|---|---|---|
| Tensor parallelism | attention heads / 4, KV heads / 4 | **whole KV groups per node**: 3 / 3 / 2 KV heads on three nodes, 4 / 4 on two. Every rank keeps the group size of 6 the attention kernels assume. |
| Experts | EP2xTP2 through the b12x kernels (uniform 4 bit) | split by expert over the nodes, through exllamav3's fused coop MoE kernels |
| Expert rates | one rate per projection in a layer | also **different rates inside a layer**. Rate classes run as modules over contiguous router ids for prefill. For decode-sized calls, one multi-rate fused kernel pair runs all of a node's experts, each at its own rate (`overlay/exllamav3/modules/inkling2_mixk.py`, kernel in `patches/`). Router columns are permuted at load; the checkpoint is unchanged. |
| Interconnect | PCIe P2P within one box | NCCL over RoCE between nodes, plus an optional one-exchange all-reduce for small tensors |
| Memory | 4x 96 GB dedicated | unified memory: KV sized from reclaimable memory, weight page cache dropped after loading |
| Decoding | speculative (MTP draft) on every step | **auto**: per step, speculative or plain, whichever yields more tokens/s at the acceptance being measured |
| Prompt reuse | none | **prefix cache**: shared KV pages plus snapshots of the per-slot sliding-window and convolution state |
| Thinking | `reasoning_effort` in the chat template | also a hard `thinking_budget` (tokens) enforced by the engine |
| Images | not served | **served**: the reference vision tower and image processor from transformers, on rank 0; prefix-cache keys include the image bytes |
| Attention and head weights | bf16 | also **EXL3**: attention projections and the output head, sliced per rank in 128-row blocks (`inkling2_attnq.py`) |
| Long contexts | one attention program per query row, also in 512-row prompt chunks: every row reads the whole KV cache | **far-key attention**: in prompt chunks and speculative-verify steps, the keys more than 1,024 positions behind a sequence's first row (no relative bias, no causal mask there) are read once for up to 10 rows x 6 heads (`_attn_far_kernel`); 2048-row prompt chunks. A global layer's call for 2048 rows at position 131,072: 93 ms instead of 426. |

## The two-Spark build

**Why requantize.** Two Sparks have about 99 GiB each for weights after the system, KV cache and state. That is room for
about 2.6 bits per expert weight, while the 3.48 bpw quant needs 256 GB of experts. Squeezing that quant further would
stack a second error on the first, so the experts were requantized from the original checkpoint.

**How.** `spark/quant/` follows 0xSero's recipe and reuses the overlay's reference ops:
- Each layer's experts are dequantized from the original (NVFP4 for layers 0-39, MXFP8 for 40-47; revision 2ad1157)
  and quantized to EXL3: trellis with LDLQ, mul1 codebook.
- Hessians are per expert, over the calibration tokens routed to it. Rarely-routed experts are shrunk toward the
  layer's input covariance (n0 = 1024).
- Each layer's input comes from the previous layers with quantized experts, so the error carries forward the way it
  will at inference.

**Calibration.** 491,520 tokens in 120 rows of 4,096, from six sources:

| Source | Rows |
|---|---|
| Wikipedia, Chinese and English | 30 |
| Code: transformers, exllamav3 | 24 |
| The model's own long reasoning, generated by the three-Spark server | 24 |
| Legal case descriptions (CAIL2018) | 18 |
| Tool calls (hermes-function-calling-v1) | 12 |
| Agent trajectories (SWE-agent) | 12 |

Seven rows are held out for the measurements; exam questions are excluded from all of it.

**Where the bits go.**

| Build | Allocation | KL | Top-1 | PPL (original 2.646) |
|---|---|---|---|---|
| v1 | 0xSero's per-layer allocation scaled down to 2.55 bpw | 0.02148 | 95.79% | 2.670 |
| v2 | budget per layer from measured KL contribution (`allocate_bits.py --measured`), then a rate per expert by routing share (`quantize_experts.py run --per-expert`) | **0.01204** | **96.90%** | **2.658** |
| v3 (published) | v2's experts; then attention at 6 bpw and the output head at 8 bpw (`quantize_attention.py`) | **0.01209** | **97.04%** | **2.658** |

- An ablation of v1, putting the original experts back six layers at a time, showed that the last 18 layers carried 62%
  of the KL. v2 gives the last 12 layers 2.61 bpw instead of 2.0.
- Within a layer, routing is skewed, so hot experts get more bits than cold ones. Rates run from 1.5 to 4 bpw in
  half-bit steps, for gate/up and down separately.
- Both builds are the same size. The numbers are over the 7 held-out rows, measured by the quantization pipeline.
- One full run took about 10.5 hours on one RTX 5090 (32 GB) with about 500 GB of RAM; v3's attention pass, 70 minutes.
- v3's attention pass quantizes each layer with Hessians from the same calibration, with the quantized experts in place.
  The sliding-window layers' query, gate, key and value projections form one EXL3 tensor; the global layers' query and
  gate form one, key and value one each. 3.4 GB in place of 9.1 GB of bf16; the head, padded to 201,216 rows, 0.9 GB in
  place of 1.9 GB.

**Checked on the Sparks** (`spark/eval/heldout_check.py`): the held-out rows through the engine on two nodes.
- v3 with the KV cache in bf16: KL 0.01184 (pipeline 0.01209). Engine and pipeline differ by KL 0.0036, numerical
  noise of the same order as the tensor-parallel split itself (0.0033 for v2).
- v3 as served, with the 4-bit global-layer KV: KL 0.01364.
- Prefill chunks (rate classes) and decode-sized chunks (multi-rate kernels) agree with each other.

## Build (every node)

Layout per node: `~/inkling2-spark/` with `engine/` (this repository), `build/exllamav3/` (exllamav3 v1.6.6 with the
patch and the overlay, extension built) and `wheels/`, plus the model under `~/models/`.

```bash
mkdir -p ~/inkling2-spark && cd ~/inkling2-spark
git clone https://github.com/oracle9i88/inkling2-exl3 engine
git clone --branch v1.6.6 --depth 1 https://github.com/turboderp-org/exllamav3 build/exllamav3
git -C build/exllamav3 apply ../../engine/patches/upstream-modifications.patch
cp -r engine/overlay/exllamav3/. build/exllamav3/exllamav3/

# pinned extra wheels (transformers 5.23, fastapi, uvicorn, marisa-trie), installed offline at container start
docker run --rm -v ~/inkling2-spark:/work nvcr.io/nvidia/pytorch:26.07-py3 \
  pip download --no-deps --require-hashes -r /work/engine/spark/requirements-spark.lock -d /work/wheels

# the CUDA extension for GB10 (compute capability 12.1): about 3 minutes; copy the .so to the other nodes or build there
docker run --rm --gpus all -v ~/inkling2-spark:/work nvcr.io/nvidia/pytorch:26.07-py3 bash -c \
  "cd /work/build/exllamav3 && TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=12 python setup.py build_ext --inplace"
```

Models go in `~/models/<name>` on every node (the folder names are set in `spark/cluster.env`).

## Run

Set the cluster up once on the machine you start from (any machine that can ssh to the nodes):

```bash
cp spark/cluster.env.example spark/cluster.env   # nodes, model folder names, image, LAN interface
```

**Free the display memory first** on headless nodes. The firmware sets aside 2 GB of the unified memory for a display.
From NVIDIA's October 2026 firmware (OTA2610, SoC firmware 2.158.15) the size is a setting, and `Auto` keeps 0.5 GB
when no display is connected. On each node:

```bash
sudo fwupdmgr refresh --force && sudo fwupdmgr upgrade      # firmware only; it asks to reboot
# after the reboot, request Auto (NVIDIA porting guide, "Display Reserved Memory"); applied at the next boot
printf '\x07\x00\x00\x00\x00' | sudo tee /sys/firmware/efi/efivars/NvidiaDispFrmSizeRequest-284e19b9-235f-4aab-ae4c-190456d1bd76 >/dev/null
sudo reboot
```

- Each node gains 1.5 GiB. A node still on the factory SoC firmware held another 2 GiB; the update returned it, 3.5 GiB
  in all on that node. Compare `MemTotal` before and after.
- The server sizes the KV cache from the memory free at start, so the gain goes to the KV cache with no flag to change.

**Two Sparks**, connected by one QSFP cable, with the image's stock NCCL (2.30.7):

```bash
spark/launch_2spark.sh start --slots 4 --max-model-len 524288 --default-max-tokens 32768 \
  --default-thinking-budget 60000 --served-name inkling-2-small --spec-mode auto \
  --chat-template /work/engine/server/chat_template-2ad1157.jinja --prefix-snapshots 24 --prefix-every 16384 \
  --chunk 2048
spark/launch_2spark.sh status          # also: stop, logs [rank]
```

- `--chunk` is the prompt chunk in rows: 2048 reads long prompts about 1.9x as fast as 512, and holds other requests
  for about 2.5 s per chunk.
- The published folder has no bf16 copies of attention and the output head, so keep `I2_ATTN_Q` and `I2_HEAD_Q` at
  their default (on).
- The long-context kernels are on by default. `EXTRA_ENV="-e I2_FAR=0 -e I2_FAR_V=0 -e I2_DEC_NSPLIT=8"` turns them
  off (prompt chunks, verify steps, plain decode).

**Three Sparks**, cabled as a triangle (each pair directly, no switch).
- NCCL has to send each peer's traffic over that peer's own link. We used a NCCL 2.29 build with subnet-aware IB
  routing (`NCCL_IB_SUBNET_AWARE_ROUTING=1`), set as `NCCL_DIR` in `spark/cluster.env`.
- That build is [zyang-dev/nccl](https://github.com/zyang-dev/nccl), branch `dgxspark-3node-ring` (commit `fab1850`,
  "Support 3 DGX Sparks in a ring topology via CX-7", NCCL 2.29.7), compiled unchanged for `sm_121`:
  `make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`. Point `NCCL_DIR` at its `build/lib`.
- The memory reserve can stay at 10 GiB: each node holds less of the model.

```bash
RESERVE_GB=10 spark/launch_3spark.sh start --slots 4 --max-model-len 262144 --default-max-tokens 32768 \
  --default-thinking-budget 60000 --served-name inkling-2-small --spec-mode auto \
  --chat-template /work/engine/server/chat_template-2ad1157.jinja
```

Rank 0 serves an OpenAI-compatible API on port 18100:
- `/v1/chat/completions` with streaming, reasoning content, tool calls (nested parameters and parallel calls) and
  images as `data:` URLs;
- `/v1/completions`, `/v1/models`;
- `/tokenize` (also `/v1/tokenize`): the token count of `{"prompt": ...}` or of `{"messages": [...], "tools": ...}` rendered
  as a chat request, images included;
- `/health`, with decode, prefill and prefix-cache counters.

Per request you can set `reasoning_effort` (chat template), `thinking_budget` (tokens), and `prefix_cache: false` to
bypass the cache.

## Checks and tools

| Script | What it does |
|---|---|
| `spark/eval/heldout_check.py` | held-out rows through the engine vs the quantization run's states and the original (KL, top-1, PPL, per MoE path) |
| `spark/eval/exam.py` | the hard-question exam above (resumable, keeps a slot free for other users) |
| `spark/eval/speed.py` | the speed numbers above |
| `spark/check_mixk_gpu.py`, `spark/make_mixed_layer.py` | multi-rate experts vs dense fp32, and timing of one layer's MoE, on one GPU |
| `spark/test_mixk_cpu.py` | rate classes, router permutation and the two-node layout (no GPU) |
| `spark/test_tp_equiv.py`, `spark/run_tp_controls.sh` | split vs unsplit logits, with run-to-run and 2/4-way controls |
| `spark/test_prefix.py`, `spark/test_prefix_bookkeeping.py` | prefix-cache restore is bit-exact; bookkeeping invariants |
| `spark/test_vision.py` | vision tower load, preprocessing, determinism, placeholder expansion, cache keys |
| `spark/run_kernel_tests.sh` | exllamav3's own kernel tests on this GPU (2,645 pass on GB10) |
| `spark/quant/test_i2q_cpu.py`, `spark/quant/test_allocate_bits.py` | the quantization pipeline on CPU: identity quantization reproduces the forward pass, resume is bit-exact |
| `spark/quant/test_attnq_cpu.py`, `spark/test_attnq_engine.py` | EXL3 attention and head: quantization on CPU; per-rank slicing, fused layouts and the reconstruct path on one GPU |
| `spark/quant/make_release.py` | the published folder from an assembled one: leaves out the bf16 attention and head that the EXL3 files replace, writes the index, config and quantization details, and checks every kept tensor byte for byte |
| `spark/release_load_check.py` | one rank's shard of a folder loaded the way the server does, on one GPU; two folders holding the same served tensors give bit-identical logits |
| `spark/test_far_attn.py` | far-key attention against the per-row kernel: output and time for prompt-chunk and verify shapes, on a synthetic KV cache in the served format |
| `spark/bench_prefill.py` | one rank's prefill: time per layer and per kernel; `--compare-far` and `--verify-check` run the far-key path on vs off through real layers |

## Credits

- [Thinking Machines](https://thinkingmachines.ai): Inkling-2-Small (Apache-2.0).
- [0xSero](https://github.com/0xSero/inkling2-exl3): the Inkling-2 exllamav3 overlay, server, draft and EXL3 quants,
  and the quantization recipe followed here (MIT).
- [turboderp](https://github.com/turboderp-org/exllamav3): exllamav3 (MIT).
- [transformers](https://github.com/huggingface/transformers): the reference implementation (`models/inkling2`),
  including the vision tower used as is (Apache-2.0).
- This branch was written by Claude (Anthropic's Claude Opus 5.5), working with
  [oracle9i88](https://github.com/oracle9i88) / [Evanguo12](https://huggingface.co/Evanguo12), who ran the machines and
  made the calls. It covers the DGX Spark port, the prefix cache, vision serving, the two-Spark quantization (experts,
  attention and head), the multi-rate kernels and the long-context attention kernels.

Code: MIT, see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Weights: Apache-2.0, as derivatives of Inkling-2-Small.
