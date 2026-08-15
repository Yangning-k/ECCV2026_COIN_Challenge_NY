#!/usr/bin/env python3
"""Build 1-round ask chains from official data via 3 Gemini batch waves.

Wave A (flash): force a discriminative question for GT-conflict pairs
    (flash said match, GT says reject -> genuinely ambiguous pairs).
Wave B (pro):   answer every question from the target reference image
    (A questions + questions of official depth-0 ask samples).
Wave C (flash): re-judge each pair with the 1-QA context; keep chains
    whose conclusion agrees with GT.

Outputs SFT-ready rows:
  - forced ask rows (depth 0, score 1 + question) for conflict pairs
  - depth-1 conclude rows (context = [{question, answer}]) for all chains
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_expansion"))

from data_expansion.gemini_label_batch_v3 import (  # noqa: E402
    FAILED_STATES,
    SUCCESS_STATES,
    download_result,
    get_job,
    make_client,
    parse_response,
    resolve_image,
    response_text,
    state_name,
)
from label_e5 import format_context  # noqa: E402
from Questioner import OUR_PROMPT_V3  # noqa: E402

FLASH = "gemini-3-flash-preview"
PRO = "gemini-3.1-pro-preview"
ANSWER_PROMPT = (
    "You are a faithful assistant. Answer the following question based on "
    "the target image. Be concise (under 15 words): {QUESTION}"
)
FORCE_ASK_SUFFIX = (
    "\n\nIMPORTANT: For this candidate, the description alone is NOT "
    "sufficient to decide reliably whether it is the exact target instance "
    "(similar objects may also fit the description). You MUST output score 1 "
    "together with ONE discriminative question about the target object that "
    "would best distinguish it from visually similar objects. Do not ask "
    "about attributes the description already states."
)


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def image_part(path: str) -> dict:
    data = base64.b64encode(resolve_image(path).read_bytes()).decode()
    return {"inline_data": {"mime_type": "image/png", "data": data}}


def gen_config(model: str, max_tokens: int) -> dict:
    thinking = (
        {"thinking_level": "LOW"} if "3.1-pro" in model
        else {"thinking_budget": 0}
    )
    return {
        "response_mime_type": "text/plain",
        "temperature": 0.0,
        "max_output_tokens": max_tokens,
        "thinking_config": thinking,
    }


def run_wave(work_dir: Path, name: str, model: str, requests: list[dict],
             batch_size: int = 50, poll_interval: int = 60) -> dict:
    """Submit request dicts ({key, request}) as batches, poll, return
    {key: result_dict}. Resumable via {name}_jobs.json."""
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = work_dir / f"{name}_jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    client = make_client()

    chunks = [
        requests[i:i + batch_size]
        for i in range(0, len(requests), batch_size)
    ]
    for no, chunk in enumerate(chunks):
        stem = f"{name}_{no:03d}"
        req_path = work_dir / f"{stem}.jsonl"
        if stem not in jobs:
            with req_path.open("w") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for attempt in range(5):
                try:
                    uploaded = client.files.upload(
                        file=str(req_path),
                        config={"mime_type": "text/plain"},
                    )
                    job = client.batches.create(
                        model=model, src=uploaded.name,
                        config={"display_name": f"official_chain_{stem}"},
                    )
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    print(f"submit {stem} retry {attempt}: {exc}", flush=True)
                    time.sleep(15 * (attempt + 1))
                    client = make_client()
            jobs[stem] = {"name": job.name, "state": state_name(job.state)}
            jobs_path.write_text(json.dumps(jobs, indent=1))
            print(f"submitted {stem}: {job.name}", flush=True)

    results: dict[str, dict] = {}
    pending = set(jobs)
    while pending:
        done_now = set()
        for stem in sorted(pending):
            info = jobs[stem]
            if info.get("result_file"):
                done_now.add(stem)
                continue
            try:
                job = get_job(client, info["name"])
            except Exception:
                client = make_client()
                continue
            if job is None:
                client = make_client()
                continue
            state = state_name(job.state)
            info["state"] = state
            if state in SUCCESS_STATES:
                dest = getattr(job, "dest", None)
                fname = getattr(dest, "file_name", None) if dest else None
                blob = download_result(client, fname) if fname else None
                if blob is None:
                    raise SystemExit(f"{stem}: no result file")
                out_path = work_dir / f"{stem}.results.jsonl"
                out_path.write_bytes(blob)
                info["result_file"] = str(out_path)
                done_now.add(stem)
                print(f"{stem}: succeeded", flush=True)
            elif state in FAILED_STATES:
                raise SystemExit(f"{stem}: {state}")
        jobs_path.write_text(json.dumps(jobs, indent=1))
        pending -= done_now
        if pending:
            time.sleep(poll_interval)

    for stem, info in jobs.items():
        for line in Path(info["result_file"]).open(
            encoding="utf-8", errors="replace"
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                results[key] = row
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default=str(ROOT / "data_expansion/data/official_flash_raw.jsonl"),
    )
    ap.add_argument(
        "--work-dir",
        default=str(ROOT / "data_expansion/gemini_official_chains"),
    )
    ap.add_argument(
        "--out",
        default=str(
            ROOT / "data_expansion/data/official_ask_chains.jsonl"
        ),
    )
    args = ap.parse_args()
    work_dir = Path(args.work_dir)

    rows = {r["id"]: r for r in load_jsonl(args.raw)}
    conflicts = [
        r for r in rows.values()
        if r.get("label") in ("match", "reject")
        and r["label"] != ("match" if r["is_match"] else "reject")
    ]
    asks = [
        r for r in rows.values()
        if r.get("label") == "ask" and (r.get("question") or "").strip()
    ]
    print(f"conflict pairs: {len(conflicts)}, ask pairs: {len(asks)}",
          flush=True)

    # ---- Wave A: force questions for conflict pairs (flash) ----
    reqs = []
    for r in conflicts:
        prompt = OUR_PROMPT_V3.format(
            USER_TASK=r["description"],
            CONTEXT="There are no previous questions or answers.",
        ) + FORCE_ASK_SUFFIX
        reqs.append({
            "key": r["id"],
            "request": {
                "contents": [{
                    "parts": [image_part(r["image"]), {"text": prompt}]
                }],
                "generation_config": gen_config(FLASH, 2048),
            },
        })
    a_results = run_wave(work_dir, "waveA", FLASH, reqs)

    forced_ask = {}
    a_drops = Counter()
    for r in conflicts:
        res = a_results.get(r["id"])
        text, err = response_text(res["response"] if res else {})
        if err or not text:
            a_drops["no_text"] += 1
            continue
        try:
            parsed = parse_response(text)
        except ValueError:
            a_drops["parse_fail"] += 1
            continue
        q = (parsed.get("question") or "").strip()
        if parsed["score"] != 1 or not q or q.lower() == "none":
            a_drops["not_ask"] += 1
            continue
        forced_ask[r["id"]] = {
            "reasoning": parsed["reasoning"], "question": q,
        }
    print(f"waveA kept {len(forced_ask)} drops={dict(a_drops)}", flush=True)

    # ---- Wave B: pro answers from target reference image ----
    chain_specs = []
    for r in conflicts:
        if r["id"] in forced_ask:
            chain_specs.append((r, forced_ask[r["id"]]["question"], "conflict"))
    for r in asks:
        chain_specs.append((r, r["question"].strip(), "official_ask"))

    reqs = []
    for r, q, _kind in chain_specs:
        reqs.append({
            "key": r["id"],
            "request": {
                "contents": [{
                    "parts": [
                        image_part(r["target_reference_image"]),
                        {"text": ANSWER_PROMPT.format(QUESTION=q)},
                    ]
                }],
                "generation_config": gen_config(PRO, 512),
            },
        })
    b_results = run_wave(work_dir, "waveB", PRO, reqs)

    answers = {}
    b_drops = Counter()
    for r, q, _kind in chain_specs:
        res = b_results.get(r["id"])
        text, err = response_text(res["response"] if res else {})
        text = (text or "").strip()
        if err or not text:
            b_drops["no_answer"] += 1
            continue
        answers[r["id"]] = text
    print(f"waveB kept {len(answers)} drops={dict(b_drops)}", flush=True)

    # ---- Wave C: flash re-judge with 1-QA context ----
    reqs = []
    for r, q, _kind in chain_specs:
        if r["id"] not in answers:
            continue
        ctx = [{"question": q, "answer": answers[r["id"]]}]
        prompt = OUR_PROMPT_V3.format(
            USER_TASK=r["description"], CONTEXT=format_context(ctx),
        )
        reqs.append({
            "key": r["id"],
            "request": {
                "contents": [{
                    "parts": [image_part(r["image"]), {"text": prompt}]
                }],
                "generation_config": gen_config(FLASH, 2048),
            },
        })
    c_results = run_wave(work_dir, "waveC", FLASH, reqs)

    out_rows = []
    c_stats = Counter()
    for r, q, kind in chain_specs:
        if r["id"] not in answers:
            continue
        res = c_results.get(r["id"])
        text, err = response_text(res["response"] if res else {})
        if err or not text:
            c_stats["no_text"] += 1
            continue
        try:
            parsed = parse_response(text)
        except ValueError:
            c_stats["parse_fail"] += 1
            continue
        gt = "match" if r["is_match"] else "reject"
        got = {2: "match", 0: "reject", 1: "ask"}[parsed["score"]]
        if got != gt:
            c_stats[f"rejudge_{got}_vs_{gt}"] += 1
            continue
        ctx = [{"question": q, "answer": answers[r["id"]]}]
        base = dict(
            split="official_train", ep_id=r["ep_id"],
            instance_id=r["instance_id"], category=r["category"],
            desc_type=r["desc_type"], description=r["description"],
            image=r["image"],
            target_reference_image=r["target_reference_image"],
            is_match=r["is_match"], source=f"official_chain_{kind}",
        )
        out_rows.append(dict(
            base, id=f"{r['id']}_d1", context=ctx, label=gt,
            score=2 if gt == "match" else 0,
            reasoning=parsed["reasoning"], question=None,
        ))
        c_stats[f"kept_{gt}"] += 1
        if kind == "conflict":
            fa = forced_ask[r["id"]]
            out_rows.append(dict(
                base, id=f"{r['id']}_d0ask", label="ask", score=1,
                reasoning=fa["reasoning"], question=fa["question"],
            ))
            c_stats["kept_forced_ask"] += 1

    with Path(args.out).open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"waveC stats={dict(c_stats)}", flush=True)
    print(f"wrote {len(out_rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
