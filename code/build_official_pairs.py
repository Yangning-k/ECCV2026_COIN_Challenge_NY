#!/usr/bin/env python3
"""Build V3 pair records from official episodes_train.jsonl.

Episodes sharing candidate images are grouped into connected components;
whole components are assigned to train or holdout so no image leaks
across the split. Only train-split pairs are emitted for labeling.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(ROOT / "episodes_train.jsonl"))
    ap.add_argument("--holdout-frac", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument(
        "--out-pairs",
        default=str(ROOT / "data_expansion/data/official_pairs_v3.jsonl"),
    )
    ap.add_argument(
        "--out-split",
        default=str(ROOT / "data_expansion/data/official_split.json"),
    )
    args = ap.parse_args()

    eps = [
        json.loads(line)
        for line in Path(args.episodes).open()
        if line.strip()
    ]

    # union-find over episodes via shared candidate images
    parent = {e["id"]: e["id"] for e in eps}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    img2ep = defaultdict(list)
    for e in eps:
        for d in e["distractors"]:
            img2ep[d["path"]].append(e["id"])
    for ids in img2ep.values():
        for other in ids[1:]:
            union(ids[0], other)

    comps = defaultdict(list)
    for e in eps:
        comps[find(e["id"])].append(e)

    comp_list = sorted(comps.values(), key=lambda c: c[0]["id"])
    rng = random.Random(args.seed)
    rng.shuffle(comp_list)

    target_holdout = round(len(eps) * args.holdout_frac)
    holdout_eps: list[dict] = []
    train_eps: list[dict] = []
    for comp in comp_list:
        if len(holdout_eps) + len(comp) <= target_holdout:
            holdout_eps.extend(comp)
        else:
            train_eps.extend(comp)

    train_imgs = {
        d["path"] for e in train_eps for d in e["distractors"]
    }
    holdout_imgs = {
        d["path"] for e in holdout_eps for d in e["distractors"]
    }
    assert not (train_imgs & holdout_imgs), "image leakage across split"

    pairs = []
    for e in train_eps:
        for desc_type, description in e["tasks"].items():
            for i, d in enumerate(e["distractors"]):
                pairs.append(
                    {
                        "id": f"off_{e['id']}_{desc_type}_{i}",
                        "split": "official_train",
                        "ep_id": e["id"],
                        "instance_id": f"off_{e['id']}",
                        "category": e["category"],
                        "desc_type": desc_type,
                        "description": description,
                        "image": d["path"],
                        "target_reference_image": e["path"],
                        "is_match": bool(d.get("match")),
                        "trust_oracle": True,
                        "source": "official_train",
                    }
                )

    out_pairs = Path(args.out_pairs)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    split = {
        "seed": args.seed,
        "train_episodes": sorted(e["id"] for e in train_eps),
        "holdout_episodes": sorted(e["id"] for e in holdout_eps),
        "n_train": len(train_eps),
        "n_holdout": len(holdout_eps),
        "n_pairs": len(pairs),
        "n_match": sum(1 for p in pairs if p["is_match"]),
        "n_components": len(comp_list),
    }
    Path(args.out_split).write_text(json.dumps(split, indent=2))
    print(json.dumps({k: v for k, v in split.items() if k != "train_episodes" and k != "holdout_episodes"}, indent=2))


if __name__ == "__main__":
    main()
