# CoIN Challenge 2026 — Questioner Submission

**Author:** Ning Yang

| | URL |
| --- | --- |
| **Code** | https://github.com/Yangning-k/ECCV2026_COIN_Challenge_NY |
| **Weights** | https://huggingface.co/Njoker/CoIN_Challenge_NY |
| **Report** | [`report/report.pdf`](report/report.pdf) |

Submitted agent: **Qwen3-VL-32B-Instruct** + Full-FT LoRA (merged bf16).
Inference: frozen Structured Attribute Prompt (SAP = `our_prompt_v3`) and
`dedup_category_only` (dedup only on `category`; other types = raw model).
Temperature **0**. Metric order: FR > SR > NQ.

---

## For organizers: download and run the hidden test

**Do not pass `--prompt-variant` or `--policy`.** Defaults in this repo already
are the submitted system (`our_prompt_v3` + `dedup_category_only`, temperature 0).

### A. Clone the code

```bash
git clone https://github.com/Yangning-k/ECCV2026_COIN_Challenge_NY.git
cd ECCV2026_COIN_Challenge_NY
source scripts/install.sh
source scripts/install_vllm.sh
```

Pick **one** of the two weight options below. They are the same submitted
system. Then go to C. Eval (`QUESTIONER_MODEL_ID`) is identical for both.

| | Option 1 — merged 32B | Option 2 — LoRA only |
| --- | --- | --- |
| Download | `merged/` (~63 GB) | `adapter/` (~168 MB) |
| Extra | none | public base [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Serve | merged checkpoint, no LoRA flag | base + `--enable-lora` |

### B1. Option 1 — download and serve the merged weights

```bash
hf download Njoker/CoIN_Challenge_NY --include "merged/*" --local-dir weights/hf
# Need: weights/hf/merged/config.json
```

`--served-model-name` must equal `QUESTIONER_MODEL_ID` in step D.

```bash
vllm serve weights/hf/merged \
  --host 0.0.0.0 --port 8001 --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 6000 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --served-model-name Njoker/CoIN_Challenge_NY
```

### B2. Option 2 — download and serve the LoRA adapter

vLLM will pull the public Qwen3-VL-32B base if it is not cached. Do **not**
set `--served-model-name` here: that name would alias the **unadapted** base.
Requests must use the LoRA module name `Njoker/CoIN_Challenge_NY`.

```bash
hf download Njoker/CoIN_Challenge_NY --include "adapter/*" --local-dir weights/hf
# Need: weights/hf/adapter/adapter_config.json
# If adapter/ is missing, LoRA files may still be at the Hub repo root:
#   hf download Njoker/CoIN_Challenge_NY --local-dir weights/hf
#   then use weights/hf instead of weights/hf/adapter below.
```

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 --port 8001 --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 6000 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --enable-lora --max-lora-rank 16 \
  --lora-modules Njoker/CoIN_Challenge_NY=weights/hf/adapter
```

### C. Hidden-test files and oracle

Keep your official eval harness, or use `eval_model.py` in this repo.

1. Put the hidden-test episode file where the script can see it, **same schema
   as** `code/episodes_train.jsonl`, named `episodes_test.jsonl` (repo root or
   `code/`), **or** set `EPISODES_JSONL_OVERRIDE=/absolute/path/to/your.jsonl`.
2. Put test images where the JSONL `path` fields point (usually `images/...`
   relative to the process working directory, or next to the JSONL).
3. Serve your Oracle VLM on port 8000 (`OracleInterface` / OpenAI-compatible
   vLLM). Development used Qwen3-VL-32B-Instruct.

### D. Run (this repo's eval)

From the **repository root**. Replace `<N>` with the number of test episodes.

```bash
export QUESTIONER_MODEL_ID=Njoker/CoIN_Challenge_NY
export QUESTIONER_VLLM_PORT=8001
export ORACLE_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct   # or your oracle served name
export ORACLE_VLLM_PORT=8000

python eval_model.py 0 <N> --local 1 --description-type all --run-type test
```

If the JSONL is not named `episodes_test.jsonl`:

```bash
export EPISODES_JSONL_OVERRIDE=/path/to/hidden_test.jsonl
python eval_model.py 0 <N> --local 1 --description-type all
```

### E. If you keep the official starter and only drop in our Questioner

Copy `code/Questioner.py` over the starter `Questioner.py` (keep starter
`eval_model.py` / `env.py`). Wire the questioner as:

```python
from Questioner import QuestionerLocalVLM
questioner = QuestionerLocalVLM(
    info,
    model_id=os.environ["QUESTIONER_MODEL_ID"],
    port=int(os.environ.get("QUESTIONER_VLLM_PORT", 8001)),
    prompt_variant="our_prompt_v3",
    policy="dedup_category_only",
    description_type=task_type,  # current episode description type
    temperature=0.0,
)
```

Set the same `QUESTIONER_*` environment variables as in step D, and serve
with **B1 or B2**. Pass `description_type` through from the eval loop so
category-only dedup is applied only on category episodes.

---

## Hugging Face weights

https://huggingface.co/Njoker/CoIN_Challenge_NY

| Artifact | Location in that repo |
| --- | --- |
| Option 1: merged bf16 32B (~63 GB) | `merged/` |
| Option 2: LoRA adapter (~168 MB) | `adapter/` (files may still be at the Hub repo root until moved) |
| Base for Option 2 | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |

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

See **For organizers** above. Same defaults: `our_prompt_v3`,
`dedup_category_only`, temperature 0. Serve with B1 (merged) or B2 (LoRA).

SAP is `PROMPT_VARIANTS["our_prompt_v3"]` in `code/Questioner.py`.
Category-only dedup is `_is_duplicate_question` / `_force_decide`.

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
