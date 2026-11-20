# Third-party notices
- exllamav3 by turboderp (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp.
  The Docker build clones upstream v1.6.6 and applies the Inkling-2 overlay in `overlay/` and `patches/`; those
  files are derived from / extend exllamav3 and are likewise offered under MIT. Upstream's LICENSE ships in the cloned tree.
- b12x by Local Inference Lab (https://github.com/local-inference-lab/b12x), Apache License 2.0,
  Copyright (c) 2025 by the b12x authors. The fused Trellis W4A16 MoE kernels now live in FlashInfer
  (https://github.com/flashinfer-ai/flashinfer, `flashinfer/experimental/b12x`, Apache License 2.0). The Docker image
  fetches FlashInfer at commit 9c50996794564e1e3b2a4b0683e9aa2bcc528df8 into `/opt/flashinfer`, unmodified and with its
  LICENSE, and imports the kernels at runtime as `b12x`. They are not vendored in this repo.
  The driver call in `overlay/exllamav3/modules/inkling2_b12x.py` (weight preparation, buffers, compile and run of the
  w4a16 fused MoE) is adapted from b12x's `benchmarks/benchmark_mixed_trellis.py`, Apache License 2.0;
  the change is the in-memory repack of exllamav3's per-expert EXL3 tensors into b12x's trellis_t256 layout.

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance
  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
  an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
  specific language governing permissions and limitations under the License.
- NVIDIA CUTLASS DSL (`nvidia-cutlass-dsl`), installed from PyPI under NVIDIA's license; it compiles the b12x kernels.
- Inkling-2-Small model by Thinking Machines, Apache-2.0. The EXL3 quant weights are a derivative under the same license.
- `server/chat_template-2ad1157.jinja` is the chat template of Inkling-2-Small (Thinking Machines, revision 2ad1157),
  Apache-2.0, copied unmodified.
- The vision serving (`server/i2_vision.py`) imports the Inkling-2 vision tower and image processor from transformers
  5.23 (Apache-2.0) at runtime; they are not vendored here.
