#!/usr/bin/env python3
"""Convert the authors' selected DIAMOND result records into minimal tasks."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from browsecomp_verify.data import TEXT_FIELDS, obfuscate, pair_id, read_jsonl


def convert_one(record):
    if not isinstance(record.get("correct"), bool) or record.get("parse_error"):
        raise ValueError("DIAMOND source needs a boolean correct field and no parse errors")
    question, candidate = record.get("question"), record.get("response")
    if not isinstance(question, str) or not question.strip() or not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("Source record needs nonempty question and response")
    task = {"id": pair_id(question, candidate), "question_id": record["id"], "question": question,
            "candidate_answer": candidate, "label": record["correct"]}
    if isinstance(record.get("answer"), str):
        task["answer"] = record["answer"]
    fields = [field for field in TEXT_FIELDS if field in task]
    for field in fields:
        task[field] = obfuscate(task[field])
    task["_obfuscated_fields"] = fields
    return task


def convert(records, workers=1):
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if workers == 1:
        tasks = map(convert_one, records)
        return validate_unique(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return validate_unique(pool.map(convert_one, records, chunksize=8))


def validate_unique(tasks):
    output = []
    seen = set()
    for task in tasks:
        identity = task["id"]
        if identity in seen:
            raise ValueError("Duplicate verification pair")
        seen.add(identity)
        output.append(task)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    tasks = convert(read_jsonl(source), workers=args.workers)
    if len(tasks) != 207:
        raise ValueError(f"Expected 207 DIAMOND tasks, got {len(tasks)}")
    if output.exists():
        raise FileExistsError("Choose a new output path instead of overwriting released data")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    manifest = {"name": "DIAMOND", "records": len(tasks),
                "true_labels": sum(task["label"] for task in tasks),
                "false_labels": sum(not task["label"] for task in tasks),
                "source_file": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "mapping": {"question": "question", "candidate_answer": "response", "label": "correct", "question_id": "id", "answer": "answer"},
                "label_provenance": "Boolean correct fields preserved from the author-designated source file; labels are not recomputed.",
                "text_encoding": "SHA256-keyed XOR plus Base64; reversible contamination deterrent"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("name", "records", "true_labels", "false_labels")}, indent=2))


if __name__ == "__main__":
    main()
