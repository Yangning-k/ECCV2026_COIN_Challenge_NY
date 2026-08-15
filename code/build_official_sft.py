#!/usr/bin/env python3
"""Build stage-2 SFT rows from GT-validated official Flash labels.

Keep rules:
- conclude (match/reject) kept only when Flash label agrees with GT
- ask kept when parse ok and a question is present (valid depth-0 behavior)
- conflicts / invalid rows are dropped and reported
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Questioner import PROMPT_VARIANTS  # noqa: E402

NO_CONTEXT = "There are no previous questions or answers."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default=str(ROOT / "data_expansion/data/official_flash_raw.jsonl"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "data_expansion/data/official_sft_stage2.jsonl"),
    )
    args = ap.parse_args()

    template = PROMPT_VARIANTS["our_prompt_v3"]
    rows = [json.loads(l) for l in Path(args.raw).open() if l.strip()]

    kept, drops = [], Counter()
    for r in rows:
        label = r.get("label")
        if r.get("parse_status") != "ok" or label == "invalid":
            drops["invalid_or_parse"] += 1
            continue
        gt = "match" if r["is_match"] else "reject"
        if label in ("match", "reject"):
            if label != gt:
                drops[f"gt_conflict_{label}_vs_{gt}"] += 1
                continue
        elif label == "ask":
            q = (r.get("question") or "").strip()
            if not q or q.lower() == "none" or r.get("score") != 1:
                drops["bad_ask"] += 1
                continue
        else:
            drops[f"unknown_label_{label}"] += 1
            continue

        reasoning = r.get("reasoning") or ""
        if not reasoning.strip():
            drops["empty_reasoning"] += 1
            continue
        q = r.get("question") if label == "ask" else None
        response = (
            f"<motivation>{reasoning}</motivation>"
            f"<score>{r['score']}</score>"
            f"<question>{q if q else 'None'}</question>"
        )
        kept.append(
            dict(
                id=r["id"],
                image=r["image"],
                prompt=template.format(
                    USER_TASK=r["description"], CONTEXT=NO_CONTEXT
                ),
                response=response,
                label=label,
                desc_type=r["desc_type"],
                score=r["score"],
                match=(label == "match"),
                instance_id=r["instance_id"],
                source="official_train",
            )
        )

    out = Path(args.out)
    with out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_ct = Counter(r["label"] for r in kept)
    print(f"kept {len(kept)} / {len(rows)}  labels={dict(label_ct)}")
    print(f"drops={dict(drops)}")


if __name__ == "__main__":
    main()
