# VideoChat-Flash — OGA Export & Inference

Model: [OpenGVLab/VideoChat-Flash-Qwen2\_5-7B\_InternVideo2-1B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B)

---

## Architecture

```
Video frames / images
       ↓
InternVideo2-1B  (39-block ViT, 3D spatiotemporal attention)
  patch_embed → [cls_token + pos_embed + img_pos_embed] → blocks[0..38]
       ↓
mm_projector  (2-layer MLP, mm_projector_type = tome16_mlp_hd64)
  clip-level HiCo token compression → ~16 tokens / frame
       ↓
Embedding merger
  model.embed_tokens(input_ids) with image-pad positions replaced by visual tokens
       ↓
Qwen2.5-7B decoder  (28L, 28Q/4KV GQA, hidden=3584, rope_theta=1e6)
  optional in-LLM HiCo video compression at layers 8, 16, 24
  (llm_compress_layer_list, llm_compress_type=attention, mm_llm_compress=False by default)
       ↓
logits
```

Key config values:
| Field | Value |
|---|---|
| `mm_vision_tower` | `internvideo2` (39 transformer blocks) |
| `mm_projector_type` | `tome16_mlp_hd64` (2-layer MLP) |
| `mm_hidden_size` | 1408 (vision → 3584 LM projection) |
| Image token ID | 151646 (`<\|image_pad\|>`) |
| `llm_compress_layer_list` | `[8, 16, 24]` (disabled by default) |

---

## Phase 1 — Text Decoder (Done)

### What was implemented

| Component | File | Description |
|---|---|---|
| `VideoChatFlashQwenModel` | `src/python/py/models/builders/qwen.py` | Builder class inheriting `QwenModel`. Overrides `load_weights()` to load via `Qwen2ForCausalLM` (avoids video library imports from custom remote code) and `make_genai_config()` to bypass `AutoConfig` for the same reason. |
| Architecture mapping | `src/python/py/models/builder.py` | Maps `VideoChatFlashQwenForCausalLM` → `VideoChatFlashQwenModel`. Includes a config-bypass that peeks at `config.json` via `hf_hub_download` before invoking `AutoConfig`, preventing `av`/`cv2`/`decord` from being imported. |
| C++ model type | `src/models/model_type.h` | Registers `videochat_flash_qwen` in `IsVLM()`. Used when the full pipeline (vision + embedding + decoder) is present. |
| Export script | `examples/python/videochat-flash/builder.py` | `--text-only` exports the decoder with `input_ids` input and `type=qwen2` config (standalone mode). Full VLM export (`--no-text-only`) sets `type=videochat_flash_qwen` and `inputs_embeds` mode (Phase 2). |
| Inference script | `examples/python/videochat-flash/run.py` | Text-only inference using `generator.append_tokens()` and HF `AutoTokenizer` (OGA tokenizer backend is unsupported for this model). |

### Design decisions

- **`type=qwen2` in text-only mode**: `videochat_flash_qwen` triggers `MultiModalLanguageModel` in the OGA C++ runtime, which requires `vision.onnx` + `embedding.onnx`. Without them, model load fails. Text-only export uses `type=qwen2` (identical LM architecture) so it loads as a plain decoder.
- **`Qwen2ForCausalLM` weight loading**: The model's custom `modeling_videochat_flash.py` imports `av`, `cv2`, `decord`, `imageio`, and `timm` at module load time. These are video processing libraries not needed for the LM backbone. The builder bypasses them entirely by loading weights as `Qwen2ForCausalLM` directly.
- **Standard 2D RoPE**: Unlike Qwen2.5-VL which uses 3D MRoPE, VideoChat-Flash uses standard 2D RoPE (`rope_scaling=None`, `rope_theta=1e6`) — the decoder is reused unchanged from `QwenModel`.

### Usage

```bash
# Export text decoder (downloads from HuggingFace)
python builder.py --output ./vcf-oga-fp32 --text-only

# Export from a local PyTorch checkpoint
python builder.py --input ./pytorch_vcf --output ./vcf-oga-fp32 --text-only

# Run text inference
python run.py --model ./vcf-oga-fp32 --batch
python run.py --model ./vcf-oga-fp32 --prompt "What is the capital of France?"
```

### Validated

- Exported: `model.onnx` (228 weights, 28-layer Qwen2.5-7B, ~14 GB fp32)
- Inference: correct answers on QA prompts using `conda`-installed OGA build (Python 3.11, `onnxruntime-genai 0.13.0.dev0`)

---

## Phase 2 — Vision Encoder + Embedding Merger (TODO)

### Overview

The full VLM pipeline requires three ONNX models:

```
vision.onnx     — InternVideo2-1B + mm_projector MLP
embedding.onnx  — embed_tokens table + image-pad token replacement
model.onnx      — Qwen2.5-7B decoder (inputs_embeds mode, already exported)
```

### 2.1 Vision Encoder (`vcf-vision.onnx`)

**Source weights** (from `model.safetensors`):
- `model.vision_tower.vision_tower.*` — 39-block ViT (516 tensors)
- `model.mm_projector.mlp.*` — 2-layer MLP projector (4 tensors)

**Export wrapper** (following `examples/python/qwen3-vl/builder.py` pattern):

```python
class VisionExportWrapper(nn.Module):
    def forward(self, pixel_values, num_frames):
        # pixel_values: [T*N_patches, C*patch_size*patch_size]
        # InternVideo2 encodes frames with 3D spatiotemporal attention
        visual_tokens = self.vision_tower(pixel_values, num_frames)
        # mm_projector: clip-level HiCo compression → ~16 tokens/frame
        return self.mm_projector(visual_tokens)

torch.onnx.export(
    wrapper,
    (pixel_values, num_frames),
    "vcf-vision.onnx",
    input_names=["pixel_values", "num_frames"],
    output_names=["visual_tokens"],
    dynamic_axes={
        "pixel_values": {0: "total_patches"},
        "visual_tokens": {0: "num_visual_tokens"},
    },
    opset_version=17,
)
```

**Key challenges:**
- InternVideo2 uses **3D spatiotemporal attention** (temporal + spatial patch tokens). Verify ONNX opset 17 supports the `einops.rearrange` and `timm` attention ops used. May require `einops` decomposition or custom ONNX ops.
- `img_pos_embed` (temporal position embedding) is dynamic based on `num_frames` — must be handled as a runtime input or computed inside the wrapper.
- The HiCo clip-level compression ratio (`tome16`) means the projector outputs ~16 tokens per frame regardless of input resolution. Verify this is deterministic (no graph-capture issues).
- `mm_projector_type = tome16_mlp_hd64`: ToMe (Token Merging) may involve non-trivial dynamic gather/scatter ops that need ONNX-compatible implementations.

### 2.2 Embedding Merger (`vcf-embedding.onnx`)

**Source weights**: `model.embed_tokens` (vocab embedding table, 152064 × 3584)

**Export wrapper** (identical pattern to Qwen3-VL):

```python
class EmbeddingWrapper(nn.Module):
    IMAGE_TOKEN_ID = 151646  # <|image_pad|>

    def forward(self, input_ids, visual_tokens):
        # input_ids: [1, seq_len]
        # visual_tokens: [num_visual_tokens, 3584]  (from vision.onnx)
        inputs_embeds = self.embed_tokens(input_ids)  # [1, seq_len, 3584]
        vision_mask = (input_ids.view(-1) == self.IMAGE_TOKEN_ID)
        inputs_embeds[0, vision_mask] = visual_tokens
        return inputs_embeds  # [1, seq_len, 3584]

torch.onnx.export(
    wrapper,
    (input_ids, visual_tokens),
    "vcf-embedding.onnx",
    input_names=["input_ids", "visual_tokens"],
    output_names=["inputs_embeds"],
    dynamic_axes={
        "input_ids": {1: "seq_len"},
        "visual_tokens": {0: "num_visual_tokens"},
    },
    opset_version=17,
)
```

### 2.3 Re-export text decoder in VLM mode

```bash
python builder.py --output ./vcf-oga-vlm --precision fp32
# (omit --text-only → exclude_embeds=true, type=videochat_flash_qwen)
```

### 2.4 `genai_config.json` additions

```json
"vision": {
    "filename": "vcf-vision.onnx",
    "inputs": { "pixel_values": "pixel_values", "num_frames": "num_frames" },
    "outputs": { "image_features": "visual_tokens" }
},
"embedding": {
    "filename": "vcf-embedding.onnx",
    "inputs": { "input_ids": "input_ids", "image_features": "visual_tokens" },
    "outputs": { "inputs_embeds": "inputs_embeds" }
}
```

### 2.5 Video preprocessing pipeline

Before the vision encoder, frames must be extracted and preprocessed. The model uses:
- `frame_aspect_ratio = square` (video frames are resized to squares)
- `image_aspect_ratio = anyres_nopad` (images use any-resolution tiling, no padding)
- `mm_spatial_pool_mode = bilinear`
- `mm_pos_num_frames = 8` (temporal position embeddings support up to 8 frames)

A `vision_processor.json` (following `qwen3-vl` pattern) or a Python preprocessing script will be needed.

### 2.6 In-LLM HiCo video compression (optional, advanced)

When `mm_llm_compress = True`, layers `[8, 16, 24]` apply additional token compression inside the decoder using cross-attention with `mm_num_compress_latents = 128` learnable query tokens. This is disabled by default (`mm_llm_compress = False`) and can be ignored for Phase 2.

If enabled in future, the compression layers would need to be exported as part of `model.onnx` or as separate side-car ONNX models.

---

## File Map

```
examples/python/videochat-flash/
├── builder.py              # Export script (Phase 1: --text-only; Phase 2: full pipeline)
├── run.py                  # Text-only inference test
└── README.md               # This file

src/python/py/models/
├── builder.py              # create_model() dispatcher (VideoChatFlashQwenForCausalLM mapping)
└── builders/
    ├── qwen.py             # VideoChatFlashQwenModel class
    └── __init__.py         # Export registration

src/models/
└── model_type.h            # IsVLM(): videochat_flash_qwen registered
```

---

## Reference

- Qwen3-VL export (closest working example): `examples/python/qwen3-vl/builder.py`
- OGA MultiModalLanguageModel runtime: `src/models/multi_modal.cpp`
- VLM model type dispatch: `src/models/model.cpp` (line ~1294)
- InternVideo2 paper: [https://arxiv.org/abs/2312.07514](https://arxiv.org/abs/2312.07514)
- VideoChat-Flash HF model: [https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B)
