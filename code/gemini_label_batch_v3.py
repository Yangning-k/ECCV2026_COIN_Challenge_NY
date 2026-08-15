#!/usr/bin/env python3
"""Build, submit, and parse Gemini labels for V3 pair records."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Questioner import OUR_PROMPT_V3  # noqa: E402

NO_CONTEXT = "There are no previous questions and answers."
SCORE_RE = re.compile(r"<score>\s*([012])\s*</score>", re.I | re.S)
MOTIVATION_RE = re.compile(r"<motivation>(.*?)</motivation>", re.I | re.S)
QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.I | re.S)
NONE_VALUES = {"", "none", "null", "n/a", "na", "''", '""'}
ARTIFACT_RE = re.compile(r"\b(artifact|distortion|blur)\w*\b", re.I)

STATE_ALIASES = {
    "SUCCEEDED": "JOB_STATE_SUCCEEDED",
    "PARTIALLY_SUCCEEDED": "JOB_STATE_PARTIALLY_SUCCEEDED",
    "FAILED": "JOB_STATE_FAILED",
    "CANCELLED": "JOB_STATE_CANCELLED",
    "EXPIRED": "JOB_STATE_EXPIRED",
}
SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
FAILED_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def resolve_image(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def state_name(state) -> str:
    name = getattr(state, "name", str(state))
    return STATE_ALIASES.get(name, name)


def make_client():
    load_dotenv(Path.home() / ".env.ml")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    http_client = httpx.Client(
        proxy=proxy,
        trust_env=False,
        timeout=300,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            httpxClient=http_client,
            timeout=300000,
        ),
    )


def make_request(row: dict, model: str, max_output_tokens: int) -> dict:
    image = resolve_image(row["image"])
    image_b64 = base64.b64encode(image.read_bytes()).decode()
    prompt = OUR_PROMPT_V3.format(
        USER_TASK=row["description"],
        CONTEXT=NO_CONTEXT,
    )
    thinking_config = (
        {"thinking_level": "LOW"}
        if "3.1-pro" in model
        else {"thinking_budget": 0}
    )
    return {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
                {"text": prompt},
            ]
        }],
        "generation_config": {
            "response_mime_type": "text/plain",
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
            "thinking_config": thinking_config,
        },
    }


def build_batches(args):
    rows = load_jsonl(Path(args.pairs))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input pair IDs are not unique")
    for row in rows:
        if not resolve_image(row["image"]).is_file():
            raise FileNotFoundError(row["image"])

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    for batch_no in range((len(rows) + batch_size - 1) // batch_size):
        chunk = rows[batch_no * batch_size:(batch_no + 1) * batch_size]
        request_path = work_dir / f"batch_{batch_no:03d}.jsonl"
        meta_path = work_dir / f"batch_{batch_no:03d}.meta.jsonl"
        with request_path.open("w") as request_file, meta_path.open("w") as meta_file:
            for row in chunk:
                key = f"v3|{row['id']}"
                request_file.write(json.dumps({
                    "key": key,
                    "request": make_request(
                        row,
                        args.model,
                        args.max_output_tokens,
                    ),
                }, ensure_ascii=False) + "\n")
                meta_file.write(json.dumps({
                    "key": key,
                    "pair": row,
                }, ensure_ascii=False) + "\n")
        print(f"{request_path.name}: {len(chunk)} requests", flush=True)
    print(f"total {len(rows)} requests", flush=True)


def load_jobs(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_jobs(path: Path, jobs: dict):
    path.write_text(json.dumps(jobs, indent=1, ensure_ascii=False))


def submit_batches(args):
    work_dir = Path(args.work_dir)
    jobs_path = work_dir / "batch_jobs.json"
    jobs = load_jobs(jobs_path)
    batch_files = sorted(
        path for path in work_dir.glob("batch_*.jsonl")
        if not path.name.endswith(".meta.jsonl")
    )
    if not batch_files:
        raise SystemExit(f"no batch files in {work_dir}; run --build first")
    client = make_client()
    for batch_file in batch_files:
        stem = batch_file.stem
        if stem in jobs:
            continue
        uploaded = client.files.upload(
            file=str(batch_file),
            config={"mime_type": "text/plain"},
        )
        job = client.batches.create(
            model=args.model,
            src=uploaded.name,
            config={"display_name": f"gemini_v3_labels_{stem}"},
        )
        jobs[stem] = {
            "name": job.name,
            "state": state_name(job.state),
            "meta": str(work_dir / f"{stem}.meta.jsonl"),
        }
        save_jobs(jobs_path, jobs)
        print(f"submitted {stem}: {job.name} ({job.state})", flush=True)
    save_jobs(jobs_path, jobs)


def response_text(result: dict) -> tuple[str, str | None]:
    if result.get("error"):
        return "", str(result["error"])
    response = result.get("response") or {}
    if response.get("text"):
        return response["text"], None
    for candidate in response.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("text"):
                return part["text"], None
    return "", "response contains no text"


def usage_values(result: dict) -> dict:
    response = result.get("response") or {}
    usage = (
        response.get("usageMetadata")
        or response.get("usage_metadata")
        or result.get("usageMetadata")
        or result.get("usage_metadata")
        or {}
    )
    return {
        "prompt_tokens": usage.get("promptTokenCount", usage.get("prompt_token_count")),
        "output_tokens": usage.get(
            "candidatesTokenCount",
            usage.get("candidates_token_count"),
        ),
        "total_tokens": usage.get("totalTokenCount", usage.get("total_token_count")),
        "thought_tokens": usage.get(
            "thoughtsTokenCount",
            usage.get("thoughts_token_count"),
        ),
    }


def normalize_question(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value.lower() in NONE_VALUES else value


def has_positive_unverifiable(reasoning: str) -> bool:
    step2_match = re.search(
        r"\bStep\s*2\s*(?:-|:)\s*(.*?)(?=\bStep\s*3\s*(?:-|:)|$)",
        reasoning,
        re.I | re.S,
    )
    step2 = step2_match.group(1) if step2_match else ""
    for match in re.finditer(r"\bunverifiable\b", step2, re.I):
        sentence_start = max(
            step2.rfind(".", 0, match.start()),
            step2.rfind("!", 0, match.start()),
            step2.rfind("?", 0, match.start()),
            step2.rfind(";", 0, match.start()),
            step2.rfind(",", 0, match.start()),
            step2.rfind("\n", 0, match.start()),
        )
        prefix = step2[sentence_start + 1:match.start()]
        if re.search(
            r"\b(?:no|none|not|without|never|neither)\b"
            r"[^.!?;\n,]{0,80}$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def parse_response(text: str) -> dict:
    score_match = SCORE_RE.search(text)
    motivation_match = MOTIVATION_RE.search(text)
    question_match = QUESTION_RE.search(text)
    if not score_match or not motivation_match or not question_match:
        raise ValueError("missing required output tags")
    score = int(score_match.group(1))
    reasoning = motivation_match.group(1).strip()
    question = normalize_question(question_match.group(1))
    flags = []
    if not all(f"Step {index}:" in reasoning for index in (1, 2, 3)):
        flags.append("missing_steps")
    if len(reasoning.split()) > 120:
        flags.append("reasoning_over_120_words")
    if score != 1 and question is not None:
        flags.append("question_with_non_ask_score")
    if score == 1 and question is None:
        flags.append("missing_question")
    if question is not None and len(question.split()) < 5:
        flags.append("question_too_short")
    if ARTIFACT_RE.search(reasoning) or (question and ARTIFACT_RE.search(question)):
        flags.append("artifact_mention")
    if score == 2 and has_positive_unverifiable(reasoning):
        flags.append("score2_unverifiable")
    return {
        "score": score,
        "reasoning": reasoning,
        "question": question,
        "validation_flags": flags,
    }


def question_valid(question: str | None, description: str) -> bool:
    if not question or len(question.split()) < 5:
        return False
    if ARTIFACT_RE.search(question):
        return False
    if question.strip().strip("?.").lower() == description.strip().lower():
        return False
    return True


def route_label(row: dict, score: int, question: str | None) -> str:
    if row.get("is_match") is None or row.get("trust_oracle"):
        if score == 2:
            return "match"
        if score == 0:
            return "reject"
        return "ask" if question_valid(question, row["description"]) else "ask_invalid"
    if row["is_match"] is True and score == 2:
        return "match"
    if row["is_match"] is False and score == 0:
        return "reject"
    if score == 1:
        return "ask" if question_valid(question, row["description"]) else "ask_invalid"
    return "hard_negative"


def parse_batch_result(result_path: Path, meta_path: Path, model: str) -> list[dict]:
    meta = {
        row["key"]: row["pair"]
        for row in load_jsonl(meta_path)
    }
    output = []
    for line in result_path.open():
        if not line.strip():
            continue
        result = json.loads(line)
        key = result.get("key", "")
        row = dict(meta.get(key, {"id": key}))
        text, error = response_text(result)
        row.update({
            "label_model": model,
            "response": text,
            **usage_values(result),
            "parse_status": "ok",
            "parse_error": error,
            "score": None,
            "reasoning": "",
            "question": None,
            "validation_flags": [],
            "label": "invalid",
            "gt_conflict": False,
        })
        if error:
            row["parse_status"] = "failed"
            row["parse_error"] = error
            output.append(row)
            continue
        try:
            parsed = parse_response(text)
            row.update(parsed)
            row["label"] = route_label(
                row,
                parsed["score"],
                parsed["question"],
            )
            row["gt_conflict"] = (
                row.get("is_match") is True and parsed["score"] == 0
            ) or (
                row.get("is_match") is False and parsed["score"] == 2
            )
            if parsed["validation_flags"]:
                row["parse_status"] = "flagged"
        except Exception as exc:
            row["parse_status"] = "failed"
            row["parse_error"] = str(exc)
        output.append(row)
    return output


def download_result(client, file_name: str) -> bytes | None:
    for attempt in range(3):
        try:
            return client.files.download(file=file_name)
        except Exception as exc:
            if attempt == 2:
                print(f"download failed: {file_name}: {exc}", flush=True)
            else:
                time.sleep(5 * (attempt + 1))
    return None


def get_job(client, name: str):
    for attempt in range(3):
        try:
            return client.batches.get(name=name)
        except Exception as exc:
            if attempt == 2:
                print(f"get job failed: {name}: {exc}", flush=True)
            else:
                time.sleep(5 * (attempt + 1))
    return None


def consolidate(work_dir: Path, out_path: Path):
    rows = []
    for path in sorted(work_dir.glob("parsed_batch_*.jsonl")):
        rows.extend(load_jsonl(path))
    rows.sort(key=lambda row: row.get("id", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"consolidated {len(rows)} rows -> {out_path}; "
        f"labels={dict(Counter(row.get('label') for row in rows))}",
        flush=True,
    )


def poll_batches(args):
    work_dir = Path(args.work_dir)
    jobs_path = work_dir / "batch_jobs.json"
    jobs = load_jobs(jobs_path)
    if not jobs:
        raise SystemExit(f"no submitted jobs in {jobs_path}")
    client = make_client()
    while True:
        all_terminal = True
        jobs = load_jobs(jobs_path)
        for stem, info in jobs.items():
            if info.get("processed"):
                continue
            state = state_name(info.get("state"))
            if state in SUCCESS_STATES:
                result_file = info.get("result_file")
                if not result_file:
                    job = get_job(client, info["name"])
                    if job is None:
                        client = make_client()
                        all_terminal = False
                        continue
                    result_file = getattr(job.dest, "file_name", None)
                result_path = work_dir / f"{stem}.results.jsonl"
                if not result_path.exists() and result_file:
                    data = download_result(client, result_file)
                    if data is None:
                        client = make_client()
                        all_terminal = False
                        continue
                    result_path.write_bytes(data)
                parsed_path = work_dir / f"parsed_{stem}.jsonl"
                if not parsed_path.exists() and result_path.exists():
                    rows = parse_batch_result(
                        result_path,
                        Path(info["meta"]),
                        args.model,
                    )
                    with parsed_path.open("w") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                info["result_file"] = result_file
                info["processed"] = parsed_path.exists()
                save_jobs(jobs_path, jobs)
                print(f"{stem}: processed={info['processed']}", flush=True)
            elif state in FAILED_STATES:
                info["processed"] = True
                save_jobs(jobs_path, jobs)
                print(f"{stem}: failed ({state})", flush=True)
            else:
                all_terminal = False
                job = get_job(client, info["name"])
                if job is None:
                    client = make_client()
                    continue
                info["state"] = state_name(job.state)
                save_jobs(jobs_path, jobs)
                print(f"{stem}: {info['state']}", flush=True)
        out_path = Path(args.out)
        if any((work_dir / f"parsed_{stem}.jsonl").exists() for stem in jobs):
            consolidate(work_dir, out_path)
        if all_terminal:
            break
        time.sleep(args.poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_output_tokens <= 0 or args.poll_interval <= 0:
        parser.error("batch-size, max-output-tokens, and poll-interval must be positive")
    if args.build and not args.pairs:
        parser.error("--pairs is required with --build")
    if args.build:
        build_batches(args)
    if args.submit:
        submit_batches(args)
    if args.poll:
        poll_batches(args)
    if not (args.build or args.submit or args.poll):
        parser.error("at least one of --build, --submit, or --poll is required")


if __name__ == "__main__":
    main()
