"""
Image input for the Inkling-2 server.

The vision tower and the image processor are the reference implementation from transformers (models/inkling2,
v5.23.0): `Inkling2VisionModel` with the checkpoint's `model.visual.*` tensors (renamed exactly as transformers'
conversion mapping does) and `Inkling2ImageProcessor`. The image features are therefore the reference's. Images are
not resized (the reference does not resize either): each 40x40 pixel block becomes one language-model token, so a
1024x1024 image is 26 x 26 = 676 tokens. `max_tokens` (optional) downscales larger images, keeping the aspect ratio.

The encoder runs on rank 0; the server broadcasts the feature rows each prefill chunk needs to the other ranks.
Features replace the placeholder embeddings *after* the embedding norm, like the reference (`masked_scatter` into the
normalized `inputs_embeds`).
"""
from __future__ import annotations
import base64, hashlib, io, json, math, os, re

import torch
from safetensors import safe_open

IMAGE_TOKEN_ID = 200054          # <|unused_200054|>


def _rename(k: str) -> str:
    """checkpoint key -> Inkling2VisionModel state-dict key (transformers conversion_mapping 'inkling2')"""
    k = k.replace("model.visual.backbone.embeddings.", "embeddings.").replace("model.visual.backbone.ln_pre.", "pre_layernorm.")
    k = k.replace("model.visual.backbone.layers.", "layers.").replace("model.visual.proj_in.", "merger.linear_1.")
    k = k.replace("model.visual.proj_out.", "merger.linear_2.")
    if k == "embeddings.position_embedding":
        k = "embeddings.position_embedding.weight"
    k = re.sub(r"(layers\.\d+)\.ln_1\.", r"\1.layernorm_before.", k)
    k = re.sub(r"(layers\.\d+)\.ln_2\.", r"\1.layernorm_after.", k)
    k = re.sub(r"(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1.", k)
    k = re.sub(r"(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2.", k)
    k = re.sub(r"(layers\.\d+)\.ls_1\.gamma", r"\1.lambda_1", k)
    k = re.sub(r"(layers\.\d+)\.ls_2\.gamma", r"\1.lambda_2", k)
    return k


class VisionEncoder:
    def __init__(self, model_dir: str, device, dtype = torch.bfloat16, max_tokens: int = 0):
        from transformers import Inkling2Config
        from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
        from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
        cfg = Inkling2Config.from_pretrained(model_dir)
        vc = cfg.vision_config
        vc._attn_implementation = "sdpa"
        self.image_token_id = cfg.image_token_id
        assert self.image_token_id == IMAGE_TOKEN_ID
        self.device, self.dtype, self.max_tokens = device, dtype, max_tokens
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        keys = [k for k in idx if k.startswith("model.visual.")]
        sd = {}
        for f in sorted(set(idx[k] for k in keys)):
            with safe_open(os.path.join(model_dir, f), "pt") as fh:
                for k in keys:
                    if idx[k] == f:
                        sd[_rename(k)] = fh.get_tensor(k)
        with torch.device("meta"):
            model = Inkling2VisionModel(vc)
        model.load_state_dict(sd, strict = True, assign = True)       # every one of the 422 tensors, nothing extra
        inv = model.rotary_emb.compute_axial_rope_parameters(vc)[0]   # meta placeholder -> real tensor before .to()
        model.rotary_emb.inv_freq = inv; model.rotary_emb.original_inv_freq = inv.clone()
        self.model = model.to(device = device, dtype = dtype).eval()
        # the rotary table is a non-persistent buffer (not in the checkpoint) and stays fp32 in the reference: rebuild it
        # after the dtype cast, which would otherwise have rounded it to bf16
        inv = self.model.rotary_emb.compute_axial_rope_parameters(vc)[0].to(device)
        self.model.rotary_emb.inv_freq = inv
        self.model.rotary_emb.original_inv_freq = inv.clone()
        self.proc = Inkling2ImageProcessor()                         # defaults = the checkpoint's processor_config.json
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ---- input
    @staticmethod
    def load_image(url: str):
        """OpenAI-style image reference -> PIL image. Accepts data: URLs (base64); remote URLs are not fetched."""
        from PIL import Image
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ValueError("only data: URLs (base64-encoded images) are accepted")
        b64 = url.split(",", 1)[1]
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img, hashlib.sha256(raw).digest()

    def prepare(self, img):
        """PIL image -> (pixel_values [rows*cols, 40, 40, 3] float32, (rows, cols)); downscales above max_tokens"""
        w, h = img.size
        if self.max_tokens and math.ceil(w / 40) * math.ceil(h / 40) > self.max_tokens:
            s = math.sqrt(self.max_tokens / (math.ceil(w / 40) * math.ceil(h / 40)))
            while True:
                nw, nh = max(1, int(w * s)), max(1, int(h * s))
                if math.ceil(nw / 40) * math.ceil(nh / 40) <= self.max_tokens:
                    break
                s *= 0.98
            img = img.resize((nw, nh), resample = 3)                  # bicubic
        out = self.proc(images = [img], return_tensors = "pt")
        g = out["image_grid_thw"][0].tolist()
        return out["pixel_values"], (int(g[1]), int(g[2]))

    # ---- features
    @torch.no_grad()
    def encode(self, pixel_values, grids):
        """pixel_values of one or more images (concatenated), grids [(rows, cols)] -> features [sum rows*cols, 4608]"""
        thw = torch.tensor([[1, r, c] for r, c in grids], dtype = torch.long)
        out = self.model(pixel_values = pixel_values.to(self.device), image_grid_thw = thw)
        return out.pooler_output.to(torch.bfloat16)


def expand_image_tokens(ids, counts):
    """replace the i-th image placeholder token in `ids` by counts[i] copies; returns (ids, positions of image tokens)"""
    out, pos, i = [], [], 0
    for t in ids:
        if t == IMAGE_TOKEN_ID:
            if i >= len(counts):
                raise ValueError("more image placeholders than images")
            pos.extend(range(len(out), len(out) + counts[i]))
            out.extend([IMAGE_TOKEN_ID] * counts[i])
            i += 1
        else:
            out.append(t)
    if i != len(counts):
        raise ValueError(f"{len(counts)} images but {i} placeholders")
    return out, pos


def cache_key_ids(ids, positions, counts, digests):
    """token ids for prefix-cache matching: each image's placeholder run becomes ids derived from the image bytes, so
    prompts with different images never share a cached prefix past the first image"""
    key = list(ids)
    k = 0
    for n, d in zip(counts, digests):
        base = -(int.from_bytes(d[:6], "big") % (2 ** 40)) - 1
        for j in range(n):
            key[positions[k]] = base - j * 0          # same id for the whole run; the run's length is part of the key
            k += 1
    return key
