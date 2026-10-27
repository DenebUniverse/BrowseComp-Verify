"""Configuration and bookkeeping shared by the released research harnesses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .data import load_dataset, read_jsonl
from .prompts import VERIFY_PROMPT, VERIFY_PROMPT_strict


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_agent_arguments(parser):
    parser.add_argument("--output_dir", default="results/run")
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--search_url", default=os.getenv("SEARCH_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model_url", default=os.getenv("MODEL_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--summary_url", default=os.getenv("SUMMARY_URL", "http://127.0.0.1:8002/v1/chat/completions"))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--num_workers", type=positive_int, default=1)
    parser.add_argument("--max_turns", type=positive_int, default=100)
    parser.add_argument("--reasoning_effort", choices=("low", "medium", "high", "max"), default="high")
    parser.add_argument("--prompt", choices=("strict", "relaxed"), default="strict")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--debug", action="store_true", help="run one task in the current process")


def parse_agent_args(task):
    parser = argparse.ArgumentParser()
    add_agent_arguments(parser)
    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name or MODEL_NAME is required")
    for name in ("search_url", "model_url"):
        value = getattr(args, name)
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            parser.error(f"{name} must be an HTTP(S) URL without embedded credentials")
        setattr(args, name, value.rstrip("/"))
    args.task = task
    return args


def get_api_key(provider=None):
    specific = {"deepseek": "DEEPSEEK_API_KEY", "glm": "ZHIPU_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(provider)
    names = [specific] if specific else []
    names += ["OPENAI_API_KEY", "GATEWAY_API_KEY", "OPENROUTER_API_KEY"]
    return next((os.environ[n] for n in names if os.environ.get(n)), "EMPTY")


def chat_completions_url(base_url):
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def verification_prompt(args, question, candidate):
    template = VERIFY_PROMPT_strict if args.prompt == "strict" else VERIFY_PROMPT
    return template.format(question=question, prediction=candidate)


def completed_ids(path):
    if not Path(path).exists():
        return set()
    latest = {}
    for record in read_jsonl(path):
        key = record.get("qid", record.get("id"))
        if key is not None:
            latest[str(key)] = record
    return {key for key, record in latest.items()
            if record.get("end_flag") is True and not record.get("error")}


def prepare_run(args):
    records = load_dataset(args.data_path, task=args.task, limit=args.limit)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = [r.get("label") for r in records]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task, "model_name": args.model_name,
        "harness": getattr(args, "harness", None),
        "model_url": args.model_url, "search_url": args.search_url,
        "summary_url": getattr(args, "summary_url", None),
        "prompt": args.prompt if args.task == "verify" else None,
        "max_turns": args.max_turns, "num_workers": args.num_workers,
        "reasoning_effort": args.reasoning_effort,
        "data_sha256": hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest(),
        "samples": len(records), "true_labels": labels.count(True), "false_labels": labels.count(False),
        "sample_ids": [r["qid"] for r in records],
        "retrieval": {key: os.getenv(key) for key in (
            "SEARCHER_TYPE", "CORPUS_PARQUET_PATH", "DENSE_INDEX_PATH",
            "DENSE_MODEL_NAME", "LUCENE_INDEX_DIR", "SNIPPET_MODE", "MAX_SNIPPET_LEN")},
    }
    target = output / "run_manifest.json"
    if target.exists():
        previous = json.loads(target.read_text())
        # Runtime timestamps and concurrency can change safely during a resume.
        for key in ("task", "harness", "model_name", "model_url", "search_url", "summary_url", "prompt", "max_turns", "reasoning_effort", "data_sha256", "sample_ids", "retrieval"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Existing output has a different {key}; use a new --output_dir")
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records
