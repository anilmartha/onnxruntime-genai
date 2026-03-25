# -------------------------------------------------------------------------
# Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Text-only inference test for VideoChat-Flash (OpenGVLab) exported via OGA.

Usage:
  # After exporting with builder.py --text-only:
  python run.py --model ./vcf-oga-fp32-standalone

  # Custom prompt:
  python run.py --model ./vcf-oga-fp32-standalone --prompt "Explain ONNX in one sentence."

Notes:
  - Requires the model to be exported WITHOUT exclude_embeds (i.e. with
    `--extra_options exclude_embeds=false` or the standalone export mode).
  - The installed OGA binary may not recognise 'videochat_flash_qwen' as a
    model type. If you see an unsupported-model-type error, patch
    genai_config.json to set "type": "qwen2" — the LM backbone is identical.
  - Uses HF AutoTokenizer directly (og.Tokenizer may fail for this model).
"""

import argparse
import os

import numpy as np

HF_MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"

# Chat-ML template tokens (Qwen-style)
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def build_prompt(user_text: str) -> str:
    return f"{_IM_START}user\n{user_text}{_IM_END}\n{_IM_START}assistant\n"


def run_inference(model_dir: str, prompt: str, max_length: int = 256) -> str:
    from onnxruntime_genai import onnxruntime_genai as og  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    print(f"[1/3] Loading tokenizer from HuggingFace ({HF_MODEL_ID})...")
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=False)

    print(f"[2/3] Loading OGA model from {model_dir}...")
    model = og.Model(model_dir)

    formatted = build_prompt(prompt)
    input_ids = np.array(tok.encode(formatted), dtype=np.int32)
    print(f"[3/3] Generating (input={len(input_ids)} tokens, max_length={max_length})...")

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    output_tokens = []
    while not generator.is_done():
        generator.generate_next_token()
        tok_id = int(generator.get_next_tokens()[0])
        output_tokens.append(tok_id)

    return tok.decode(output_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Text-only inference test for VideoChat-Flash OGA export"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to exported OGA model directory (e.g. ./vcf-oga-fp32-standalone)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France? Give a short answer.",
        help="User prompt (plain text, chat template applied automatically)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run a small batch of built-in test prompts instead of --prompt",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model)

    if args.batch:
        test_prompts = [
            "What is the capital of France? Give a short answer.",
            "What is 7 * 8?",
            "Explain what ONNX Runtime is in 2 sentences.",
            "Name three primary colors.",
        ]
    else:
        test_prompts = [args.prompt]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(test_prompts)}] Prompt: {prompt}")
        print("=" * 60)
        response = run_inference(model_dir, prompt, args.max_length)
        print(f"Response: {response}")

    print("\n[OK] Inference complete.")


if __name__ == "__main__":
    main()
