# CoIN Challenge 2026 — Questioner Submission

**Author:** Ning Yang

A structured questioner for Collaborative Instance Navigation (question-asking protocol).
The submitted agent is **Qwen3-VL-32B-Instruct** fine-tuned with two-stage QLoRA
(Full-FT). Inference uses a frozen **Structured Attribute Prompt (SAP)** and a
**category-only** question-deduplication rule.

Selection metric order: **FR > SR > NQ**. All reported numbers use temperature 0.

Technical report: [`report/report.pdf`](report/report.pdf).

---

## Hugging Face weights

Fill these URLs after the adapters are uploaded. Git does **not** contain
`.safetensors` files (the merged 32B checkpoint is ~63 GB).

| Artifact | Hugging Face URL |
| --- | --- |
| **Submitted LoRA adapter (Full-FT / v3f)** | `TODO: https://huggingface.co/<org>/<adapter-repo>` |
| Optional merged bf16 weights | `TODO: https://huggingface.co/<org>/<merged-repo>` |
| Mix-FT adapter (clean holdout numbers, not the submission weight) | `TODO` |
| Base model | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |

Local copies used during development (do not push):

- adapter: `weights/adapter_v3f/`
- merged: `weights/merged_v3f/`
- Mix-FT backup: `weights/adapter_v3o/`

---

## Repository layout

```
.
├── README.md
├── report/
│   └── report.pdf          # technical report
├── code/                   # questioner + eval + training helpers
│   ├── Questioner.py       # SAP prompt + hybrid policy
│   ├── eval_model.py       # official eval loop
│   ├── env.py / Oracle.py / utils.py
│   ├── episodes_train.jsonl
│   ├── stats_results.py
│   ├── train_two_stage_qwen3_ddp.py
│   ├── merge_lora_qwen3.py
│   ├── build_official_*.py
│   └── official_split.json
├── scripts/
│   ├── install.sh
│   └── install_vllm.sh
└── weights/                # optional local cache; not for git
```

This directory is meant to be the **GitHub repository root**. It is a
self-contained fork of the official starter
([e-zorzi/coin_challenge](https://github.com/e-zorzi/coin_challenge)) plus our
questioner, training helpers, and report.

The development workspace (logs, checkpoints, synthetic image factory, etc.)
is **not** this package and should not be pushed.

---

## Setup

```bash
source scripts/install.sh
source scripts/install_vllm.sh   # second env, for serving the VLM
```

Download official images (required by `eval_model.py`):

```bash
mkdir -p images
hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images
```

Download the submitted adapter from Hugging Face into e.g. `weights/adapter_v3f`.

---

## Inference (reproduces the submitted system)

Serve the questioner with vLLM (LoRA on the bf16 base). Replace the adapter
path with the Hugging Face snapshot once uploaded.

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 --port 8001 --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 6000 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --mm-encoder-tp-mode data --mm-encoder-attn-backend TORCH_SDPA \
  --mm-processor-cache-gb 0 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules coin_final=<path-to>/weights/adapter_v3f
```

Alternatively serve the merged bf16 checkpoint and omit `--enable-lora`.

Oracle: any local VLM that implements `OracleInterface` (we used
Qwen3-VL-32B-Instruct on port 8000 during development). See the report, §Oracle.

From `code/`:

```bash
cd code
export QUESTIONER_MODEL_ID=coin_final QUESTIONER_VLLM_PORT=8001
export ORACLE_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct ORACLE_VLLM_PORT=8000

# category: category-only dedup + forced decision on repeats
python eval_model.py 0 167 --local 1 --description-type category \
  --prompt-variant our_prompt_v3 --policy dedup_force_decide --temperature 0

# all other description types: raw model (no extra policy)
python eval_model.py 0 167 --local 1 --description-type <type> \
  --prompt-variant our_prompt_v3 --policy baseline --temperature 0
```

`--policy dedup_category_only` applies the same split in a single run:
dedup only when `description_type == category`, otherwise baseline.

SAP is `PROMPT_VARIANTS["our_prompt_v3"]` in `code/Questioner.py`. The
category-only rule is `_is_duplicate_question` / `_force_decide`.

Aggregate metrics:

```bash
python stats_results.py results/<run>/*.json
```

---

## Training (outline)

Full numbers, ablations, and data-construction details are in the report.
A short recipe:

1. **Synthetic pairs** labeled under SAP (Gemini Flash/Pro adjudication), then
   a 5:1 ask-weighted Stage-2 mixture (Syn-FT).
2. **Official episodes**: leakage-free split by shared-image connected
   components (`code/build_official_pairs.py`, split file
   `code/official_split.json`); Flash labels validated against ground truth
   (`code/build_official_sft.py`).
3. **Mix-FT**: continue Stage 2 on official train-split pairs.
4. **Full-FT (submitted)**: same recipe after absorbing the 47 holdout
   episodes. QLoRA rank 16, lr 5e-5, 1 epoch, image size 224.

```bash
torchrun --nproc_per_node=8 code/train_two_stage_qwen3_ddp.py \
  --base Qwen/Qwen3-VL-32B-Instruct \
  --stage1 <stage1.jsonl> --stage2 <stage2.jsonl> \
  --out <adapter-dir>
python code/merge_lora_qwen3.py --base Qwen/Qwen3-VL-32B-Instruct \
  --adapter <adapter-dir> --out <merged-dir>
```

The Habitat/Gemini **image-generation** factory used for synthetic pairs is
documented in the report and is not required to run the submitted questioner.

---

## Held-out development numbers (Mix-FT, 47 episodes, never trained on)

| | SR | FR | NQ/obs |
| --- | ---: | ---: | ---: |
| Mix-FT (clean holdout) | 0.801 | 0.713 | 0.67 |
| Full-FT sanity (after absorbing holdout) | 0.768 | 0.678 | 0.61 |

Submitted weight: **Full-FT**. Mix-FT is the selection checkpoint.

---

## License

Official starter files (`env.py`, `Oracle.py`, `utils.py`, eval harness)
follow [e-zorzi/coin_challenge](https://github.com/e-zorzi/coin_challenge)
(Apache 2.0). Our questioner, training scripts, and report are part of this
challenge submission.
