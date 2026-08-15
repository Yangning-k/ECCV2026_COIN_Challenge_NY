#!/usr/bin/env python3
"""Compute CoIN-style SR / FR / NQ from eval result JSON + episodes_train.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_episodes(path: Path):
    eps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    return eps


def metrics_for_result(result_path: Path, episodes):
    with result_path.open() as f:
        data = json.load(f)

    id_to_ep = {e["id"]: e for e in episodes}
    n = len(data["id"])
    if n == 0:
        return {"n": 0}

    sr_list = []
    fr_list = []
    nq_list = []
    n_q_total = 0
    n_obs_total = 0
    fp_match = 0  # concluded match incorrectly / failed early after wrong match
    # Approximate error taxonomy from incomplete trajectories:
    # if n_successes < n_distractors, episode failed.
    failed = 0
    asked_eps = 0

    for i in range(n):
        eid = data["id"][i]
        ep = id_to_ep.get(eid)
        n_dist = len(ep["distractors"]) if ep else None
        n_succ = data["n_successes"][i]
        n_q = data["n_questions"][i]
        obs = data["observations"][i]
        n_obs = len(obs) if isinstance(obs, list) else 1
        n_obs = max(n_obs, 1)

        if n_dist is None:
            continue
        # decisions attempted ≈ successes if FR else successes+1 (failed on next)
        n_dec = n_succ if n_succ >= n_dist else n_succ + 1
        n_dec = max(n_dec, 1)
        sr_list.append(n_succ / n_dec)
        fr = 1.0 if n_succ >= n_dist else 0.0
        fr_list.append(fr)
        if fr < 1:
            failed += 1
        nq_list.append(n_q / n_obs)
        n_q_total += n_q
        n_obs_total += n_obs
        if n_q > 0:
            asked_eps += 1

    return {
        "file": str(result_path),
        "n": len(sr_list),
        "SR": sum(sr_list) / len(sr_list) if sr_list else None,
        "FR": sum(fr_list) / len(fr_list) if fr_list else None,
        "NQ_per_obs": sum(nq_list) / len(nq_list) if nq_list else None,
        "NQ_total_mean": n_q_total / len(sr_list) if sr_list else None,
        "ask_rate": asked_eps / len(sr_list) if sr_list else None,
        "failed": failed,
        "mean_time": (
            sum(data["time_required"]) / n if data.get("time_required") else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result json files")
    ap.add_argument(
        "--episodes",
        default="episodes_train.jsonl",
        help="episodes jsonl for distractor counts",
    )
    args = ap.parse_args()
    episodes = load_episodes(Path(args.episodes))
    rows = []
    for r in args.results:
        m = metrics_for_result(Path(r), episodes)
        rows.append(m)
        print(
            f"{Path(r).name}: n={m.get('n')} SR={m.get('SR'):.4f} FR={m.get('FR'):.4f} "
            f"NQ/obs={m.get('NQ_per_obs'):.4f} ask_rate={m.get('ask_rate'):.3f} "
            f"failed={m.get('failed')} time={m.get('mean_time')}"
            if m.get("n")
            else f"{Path(r).name}: EMPTY"
        )
    out = Path("overnight/results/stats_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
