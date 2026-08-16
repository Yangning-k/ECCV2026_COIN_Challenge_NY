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

https://huggingface.co/Njoker/CoIN_Challenge_NY

| Artifact | Location in that repo |
| --- | --- |
| **Submitted LoRA adapter (Full-FT)** | `adapter/` (files may still be at repo root until moved) |
| Merged bf16 32B weights | `merged/` |
| Base model | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |

---

## Repository layout

```
.
├── README.md
├── eval_model.py           # wrapper; real eval lives in code/
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

---

## Inference (submitted system)

**Defaults in `eval_model.py` / `QuestionerLocalVLM` are the submission config:**
`--prompt-variant our_prompt_v3`, `--policy dedup_category_only`, `--temperature 0`.
`dedup_category_only` applies question-dedup + forced decision only on `category`; other description types are raw model output.

Serve the merged questioner (no LoRA flag). The served name must match `QUESTIONER_MODEL_ID`:

```bash
hf download Njoker/CoIN_Challenge_NY --include "merged/*" --local-dir weights/hf_merged
vllm serve weights/hf_merged/merged \
  --host 0.0.0.0 --port 8001 --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 6000 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --served-model-name Njoker/CoIN_Challenge_NY
```

Oracle: a local VLM implementing `OracleInterface` (development used Qwen3-VL-32B-Instruct on port 8000).

```bash
export QUESTIONER_MODEL_ID=Njoker/CoIN_Challenge_NY QUESTIONER_VLLM_PORT=8001
export ORACLE_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct ORACLE_VLLM_PORT=8000

# From repo root or from code/. Images: mkdir images && hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images
python eval_model.py 0 <N> --local 1 --description-type all
```

Hidden test: put `episodes_test.jsonl` in the repo root or `code/`, then:

```bash
python eval_model.py 0 <N> --local 1 --description-type all --run-type test
```

Or `export EPISODES_JSONL_OVERRIDE=/path/to/episodes_test.jsonl`.

SAP is `PROMPT_VARIANTS["our_prompt_v3"]` in `code/Questioner.py`. The category-only rule is `_is_duplicate_question` / `_force_decide`.

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
