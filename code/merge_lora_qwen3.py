#!/usr/bin/env python3
"""E5: merge a LoRA adapter back into the Qwen3-VL-32B base model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=None, help="comma-separated GPU indices")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("[merge] loading base ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print("[merge] loading adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merging ...")
    model = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    processor.save_pretrained(str(out))
    print(f"[merge] saved to {out}")


if __name__ == "__main__":
    main()
