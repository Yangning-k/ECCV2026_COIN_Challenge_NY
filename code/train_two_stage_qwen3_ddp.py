#!/usr/bin/env python3
"""E5: two-stage LoRA SFT for Qwen3-VL-32B -- DDP + QLoRA variant.

Run with: torchrun --nproc_per_node=N data_expansion/train_two_stage_qwen3_ddp.py ...

Each rank holds its own 4-bit copy of the base model (QLoRA) and trains with
true data parallelism (batch=1 per rank, grad_acc global). This is ~4x faster
than the bf16 device_map='auto' sharded path for the same global batch.

Stage 1: R+S only. Stage 2: continue with ask samples mixed at 1:10.
Tokenized items are precached to disk (torch.save/load) so re-runs skip
re-tokenization; rank 0 builds the cache, all ranks load it after a barrier.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "/shared_disk/models/huggingface/Qwen3-VL-32B-Instruct"
DEFAULT_STAGE1 = ROOT / "data_expansion" / "data" / "e5_stage_1.jsonl"
DEFAULT_STAGE2 = ROOT / "data_expansion" / "data" / "e5_stage_2.jsonl"
DEFAULT_OUT = ROOT / "checkpoints" / "Qwen3-VL-32B-LoRA-v1"


def rank_info():
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def load_rows(path: str):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def tokenize_row(r, processor, img_size=0):
    import torch

    from PIL import Image

    image = Image.open(r["image"]).convert("RGB")
    if img_size and img_size > 0:
        image = image.resize((img_size, img_size), Image.LANCZOS)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": r["prompt"]},
            ],
        },
        {"role": "assistant", "content": r["response"]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_kwargs = {}
    if img_size and img_size > 0:
        image_kwargs = {
            "min_pixels": img_size * img_size,
            "max_pixels": img_size * img_size,
        }
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        **image_kwargs,
    )
    labels = inputs["input_ids"].clone()
    labels[:] = -100
    tok = inputs["input_ids"][0]
    im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    asst_ids = processor.tokenizer.convert_tokens_to_ids("assistant")
    starts = [
        i
        for i in range(len(tok) - 1)
        if tok[i] == im_start and tok[i + 1] == asst_ids
    ]
    if not starts:
        raise ValueError(f"no assistant turn in sample {r['id']}")
    labels[0, starts[-1]:] = inputs["input_ids"][0, starts[-1]:]
    item = {k: v.squeeze(0) for k, v in inputs.items()}
    item["labels"] = labels.squeeze(0)
    # bf16 pixel_values: the model casts to visual dtype anyway; halves cache RAM.
    item["pixel_values"] = item["pixel_values"].to(torch.bfloat16)
    return item


def precache(rows, processor, cache_path, rank, img_size=0):
    import torch

    cache_path = Path(cache_path)
    if cache_path.exists():
        if rank == 0:
            print(f"[precache] loading {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)
    if rank != 0:
        # rank 0 builds the cache; others wait for it.
        return None
    items = []
    for i, r in enumerate(rows):
        items.append(tokenize_row(r, processor, img_size))
        if (i + 1) % 200 == 0:
            print(f"[precache] {i + 1}/{len(rows)} tokenized", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, cache_path)
    print(f"[precache] saved {len(items)} items -> {cache_path}", flush=True)
    return items


def build_dataset(items):
    import torch

    class DS(torch.utils.data.Dataset):
        def __init__(self, items_):
            self.items = items_

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    return DS(items)


def collate(batch):
    import torch

    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.nn.utils.rnn.pad_sequence(
            [b[k] for b in batch],
            batch_first=True,
            padding_value=0 if k != "labels" else -100,
        )
    if out["labels"].shape[0] == 1:
        labeled = torch.nonzero(out["labels"][0] != -100).flatten()
        if labeled.numel():
            start = int(labeled[0])
            start = max(0, start - 1)
            out["labels"] = out["labels"][:, start:]
            out["logits_to_keep"] = torch.arange(
                start,
                out["input_ids"].shape[1],
                dtype=torch.long,
            )
    return out


def train_stage(rows, model, processor, out_dir, args, tag: str, rank):
    import torch
    import torch.distributed as dist
    from transformers import Trainer, TrainingArguments

    cache_path = Path(args.precache_dir) / f"{tag}.pt"
    items = precache(rows, processor, cache_path, rank, args.img_size)
    if rank != 0:
        dist.barrier()
        items = precache(rows, processor, cache_path, rank, args.img_size)  # now cached
    else:
        dist.barrier()
    ds = build_dataset(items)
    train_args = TrainingArguments(
        output_dir=str(out_dir / tag),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=999999,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        local_rank=rank,
    )
    sampling_weights = [
        row.get("sampling_weight")
        for row in rows
    ] if tag == "stage1" else None

    class WeightedDistributedSampler(torch.utils.data.Sampler):
        def __init__(self, weights, num_replicas, rank, seed=0):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.num_replicas = num_replicas
            self.rank = rank
            self.seed = seed
            self.epoch = 0
            self.num_samples = (
                len(self.weights) + num_replicas - 1
            ) // num_replicas
            self.total_size = self.num_samples * num_replicas

        def __iter__(self):
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=generator,
            ).tolist()
            return iter(indices[self.rank:self.total_size:self.num_replicas])

        def __len__(self):
            return self.num_samples

        def set_epoch(self, epoch):
            self.epoch = epoch

    class WeightedTrainer(Trainer):
        def __init__(self, *trainer_args, sampler_weights=None, **kwargs):
            super().__init__(*trainer_args, **kwargs)
            self.sampler_weights = sampler_weights

        def _get_train_sampler(self, train_dataset=None):
            if self.sampler_weights is None:
                return super()._get_train_sampler(train_dataset)
            return WeightedDistributedSampler(
                self.sampler_weights,
                1,
                0,
                seed=args.seed,
            )

    use_weights = (
        sampling_weights
        and all(weight is not None for weight in sampling_weights)
    )
    trainer_class = WeightedTrainer if use_weights else Trainer
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": ds,
        "data_collator": collate,
    }
    if use_weights:
        trainer_kwargs["sampler_weights"] = sampling_weights
    trainer = trainer_class(
        **trainer_kwargs,
    )
    trainer.train()
    if rank == 0:
        print(f"[{tag}] trained {len(rows)} samples", flush=True)
    del trainer, ds, items
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--stage1", default=str(DEFAULT_STAGE1))
    ap.add_argument("--stage2", default=str(DEFAULT_STAGE2))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--precache-dir", default="")
    ap.add_argument("--img-size", type=int, default=0,
                    help="resize training images to NxN before tokenizing "
                         "(0 = keep native resolution)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--stage1-only", action="store_true")
    mode.add_argument("--stage2-only", action="store_true")
    ap.add_argument(
        "--init-adapter",
        default="",
        help="adapter directory to load before stage 2-only training",
    )
    args = ap.parse_args()
    if args.stage2_only and not args.init_adapter:
        ap.error("--stage2-only requires --init-adapter")
    if args.stage1_only and args.init_adapter:
        ap.error("--stage1-only cannot use --init-adapter")

    rank, world = rank_info()
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from transformers.integrations import tensor_parallel

    if not hasattr(tensor_parallel, "EmbeddingParallel"):
        tensor_parallel.EmbeddingParallel = tensor_parallel.RowwiseParallel

    if world > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
        )
    if rank == 0:
        print(f"[train] DDP world={world} rank={rank} base={args.base}", flush=True)
    processor = AutoProcessor.from_pretrained(str(args.base), trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.base),
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        device_map={"": rank},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.precache_dir = args.precache_dir or str(out / "precache")
    rows1 = load_rows(args.stage1)
    if not args.stage2_only:
        train_stage(rows1, model, processor, out, args, "stage1", rank)
    if args.stage1_only:
        if rank == 0:
            model.save_pretrained(str(out))
            processor.save_pretrained(str(out))
            print(f"[train] saved stage1 adapter to {out}", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    rows2 = load_rows(args.stage2)
    train_stage(rows2, model, processor, out, args, "stage2", rank)

    if rank == 0:
        model.save_pretrained(str(out))
        processor.save_pretrained(str(out))
        print(f"[train] saved adapter to {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
