"""Deterministic verification scoring with explicit failure denominators."""

import argparse
import json
from pathlib import Path
import re

from .data import load_dataset, read_jsonl


def message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def final_response(record):
    """Read only an assistant answer, never the stored candidate in response."""
    messages = record.get("messages", [])
    assistants = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        author = message.get("author", {})
        role = message.get("role") or (author.get("role") if isinstance(author, dict) else None)
        if role == "assistant" and message.get("channel") not in ("analysis", "analysis code") and not message.get("recipient"):
            assistants.append(message)
    if assistants:
        return message_text(assistants[-1])
    return record.get("verifier_response", "")


def parse_verdict(text):
    if not isinstance(text, str):
        return None
    if len(re.findall(r"<think>", text, re.I)) != len(re.findall(r"</think>", text, re.I)):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if answers:
        text = answers[-1]
    verdicts = re.findall(r"^\s*(?:\*\*)?verdict\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(yes|no)\b", text, re.M | re.I)
    values = {v.lower() for v in verdicts}
    if len(values) == 1:
        return next(iter(values)) == "yes"
    if not verdicts and text.strip().lower() in ("yes", "no"):
        return text.strip().lower() == "yes"
    return None


def score(records, predictions):
    expected = {str(r["qid"]): r for r in records}
    if not expected:
        raise ValueError("Cannot score an empty dataset")
    if any(not isinstance(r.get("label"), bool) for r in records):
        raise ValueError("Scoring requires a boolean label for every task")
    latest = {}
    duplicate_rows = 0
    for prediction in predictions:
        key = prediction.get("qid", prediction.get("id"))
        if key is None:
            raise ValueError("Every prediction requires a pair qid or id")
        key = str(key)
        if key not in expected:
            raise ValueError("Prediction IDs do not match the dataset; use IDs emitted by the release loader")
        duplicate_rows += key in latest
        latest[key] = prediction
    details = []
    for key, record in expected.items():
        prediction = latest.get(key)
        complete = bool(prediction and prediction.get("end_flag") is True and not prediction.get("error"))
        verdict = parse_verdict(final_response(prediction)) if complete else None
        details.append({"id": key, "label": record["label"], "prediction": verdict,
                        "present": prediction is not None, "completed": complete,
                        "valid": verdict is not None,
                        "correct": verdict is not None and verdict == record["label"]})
    total = len(details)
    correct = sum(r["correct"] for r in details)
    valid = sum(r["valid"] for r in details)
    completed = sum(r["completed"] for r in details)
    summary = {
        "total": total, "prediction_rows_after_deduplication": len(latest),
        "duplicate_rows_superseded": duplicate_rows,
        "missing": sum(not r["present"] for r in details),
        "completed": completed, "valid_verdicts": valid,
        "completed_without_valid_verdict": completed - valid,
        "correct": correct, "overall_accuracy": correct / total,
        "completion_rate": completed / total, "coverage": valid / total,
        "accuracy_on_valid_verdicts": correct / valid if valid else None,
    }
    for label, name in ((True, "true"), (False, "false")):
        subset = [r for r in details if r["label"] is label]
        summary[name + "_total"] = len(subset)
        summary[name + "_accuracy"] = sum(r["correct"] for r in subset) / len(subset) if subset else None
    return {"summary": summary, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--predictions", required=True, help="data_stream.jsonl from a verification run")
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument("--limit", type=int, help="use the same limit as the inference run")
    args = parser.parse_args()
    result = score(load_dataset(args.data_path, limit=args.limit), read_jsonl(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
