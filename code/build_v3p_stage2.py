#!/usr/bin/env python3
"""Assemble v3p stage-2 data: base 5:1 + rebalanced official + ask chains.

Rebalance: official plain reject rows are downsampled per desc_type to
1.2x the match count, so confident-reject behavior does not swamp the
ask behavior for attribute-rich description types.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_expansion"))

from label_e5 import format_context  # noqa: E402
from Questioner import PROMPT_VARIANTS  # noqa: E402

NO_CONTEXT = "There are no previous questions or answers."


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=str(ROOT / "data_expansion/data/e5_v3d_train_5to1__2.jsonl"),
    )
    ap.add_argument(
        "--official",
        default=str(ROOT / "data_expansion/data/official_sft_stage2.jsonl"),
    )
    ap.add_argument(
        "--chains",
        default=str(ROOT / "data_expansion/data/official_ask_chains.jsonl"),
    )
    ap.add_argument("--reject-ratio", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument(
        "--out",
        default=str(ROOT / "data_expansion/data/e5_v3p_train_5to1__2.jsonl"),
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    template = PROMPT_VARIANTS["our_prompt_v3"]

    base = load_jsonl(args.base)
    official = load_jsonl(args.official)
    chains = load_jsonl(args.chains)

    # rebalance official conclude rows
    by_type = defaultdict(lambda: defaultdict(list))
    for r in official:
        by_type[r["desc_type"]][r["label"]].append(r)
    kept_official = []
    for t, groups in by_type.items():
        kept_official += groups["ask"] + groups["match"]
        rejects = groups["reject"]
        cap = round(args.reject_ratio * len(groups["match"]))
        if len(groups["match"]) == 0:
            cap = len(rejects)
        if len(rejects) > cap:
            rejects = rng.sample(rejects, cap)
        kept_official += rejects

    # convert chain rows to SFT format
    chain_rows = []
    for r in chains:
        ctx = r.get("context")
        prompt = template.format(
            USER_TASK=r["description"],
            CONTEXT=format_context(ctx) if ctx else NO_CONTEXT,
        )
        q = r.get("question") if r["label"] == "ask" else None
        response = (
            f"<motivation>{r['reasoning']}</motivation>"
            f"<score>{r['score']}</score>"
            f"<question>{q if q else 'None'}</question>"
        )
        row = dict(
            id=r["id"], image=r["image"], prompt=prompt, response=response,
            label=r["label"], desc_type=r["desc_type"], score=r["score"],
            match=(r["label"] == "match"), instance_id=r["instance_id"],
            source=r["source"],
        )
        if ctx:
            row["context"] = ctx
        chain_rows.append(row)

    merged = base + kept_official + chain_rows
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in merged data")

    with Path(args.out).open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"base={len(base)} official_kept={len(kept_official)} "
          f"chains={len(chain_rows)} total={len(merged)}")
    ct = Counter((r["desc_type"], r["label"]) for r in kept_official + chain_rows)
    for t in sorted({k[0] for k in ct}):
        print(f"{t:24s} ask={ct[(t, 'ask')]:4d} match={ct[(t, 'match')]:4d} "
              f"reject={ct[(t, 'reject')]:4d}")


if __name__ == "__main__":
    main()
