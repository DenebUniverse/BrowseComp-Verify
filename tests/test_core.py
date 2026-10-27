"""Offline protocol and scoring checks; examples here are synthetic."""

import ast
import asyncio
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from browsecomp_verify.data import decode_record, deobfuscate, load_dataset, normalize_record, obfuscate, pair_id
from browsecomp_verify.evaluate import final_response, parse_verdict, score
from browsecomp_verify.runtime import completed_ids, prepare_run, verification_prompt

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_unicode_obfuscation_roundtrip(self):
        for text in ("", "A synthetic example", "测试 — café 🧪"):
            self.assertEqual(deobfuscate(obfuscate(text)), text)

    def test_duplicate_question_distinct_candidates(self):
        first = normalize_record({"id": 7, "question": "Toy question?", "candidate_answer": "A", "label": True})
        second = normalize_record({"id": 7, "question": "Toy question?", "candidate_answer": "B", "label": False})
        self.assertNotEqual(first["qid"], second["qid"])
        self.assertEqual(first["question_id"], second["question_id"])

    def test_label_strings_rejected(self):
        with self.assertRaises(ValueError):
            normalize_record({"question": "Toy question?", "candidate_answer": "A", "label": "false"})

    def test_solver_file_cannot_silently_become_verification_tasks(self):
        with self.assertRaises(ValueError):
            normalize_record({"qid": 7, "question": "Toy question?", "answer": "A"})

    def test_bundled_distribution_integrity(self):
        data = ROOT / "data/diamond.jsonl"
        rows = load_dataset(data)
        manifest = json.loads(data.with_suffix(".manifest.json").read_text())
        self.assertEqual(len(rows), 207)
        self.assertEqual(sum(row["label"] for row in rows), 103)
        self.assertEqual(len({row["qid"] for row in rows}), 207)
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), manifest["output_sha256"])
        encoded = json.loads(data.read_text().splitlines()[0])
        self.assertNotEqual(encoded["question"], rows[0]["question"])
        self.assertEqual(set(encoded["_obfuscated_fields"]), {"question", "candidate_answer", "answer"})


class ScoringTests(unittest.TestCase):
    def records(self):
        return [normalize_record({"question": "Toy question?", "candidate_answer": str(i), "label": label})
                for i, label in enumerate((True, False, True))]

    def prediction(self, row, text, **extra):
        return {"qid": row["qid"], "end_flag": True,
                "messages": [{"role": "assistant", "content": text}], **extra}

    def test_coverage_and_accuracy_include_missing(self):
        rows = self.records()
        predictions = [self.prediction(rows[0], "verdict: yes"), self.prediction(rows[1], "Not enough information")]
        summary = score(rows, predictions)["summary"]
        self.assertEqual(summary["missing"], 1)
        self.assertAlmostEqual(summary["overall_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["coverage"], 1 / 3)
        self.assertEqual(summary["true_accuracy"], 0.5)
        self.assertEqual(summary["false_accuracy"], 0)

    def test_failed_but_parseable_answer_still_counts_as_failure(self):
        row = self.records()[0]
        result = score([row], [self.prediction(row, "verdict: yes", error="budget exceeded")])
        self.assertEqual(result["summary"]["overall_accuracy"], 0)
        self.assertEqual(result["summary"]["coverage"], 0)

    def test_candidate_is_never_used_as_verifier_answer(self):
        row = self.records()[0]
        prediction = {"qid": row["qid"], "end_flag": True, "response": "verdict: yes"}
        self.assertEqual(score([row], [prediction])["summary"]["valid_verdicts"], 0)

    def test_analysis_channel_is_not_final_answer(self):
        self.assertEqual(final_response({"messages": [
            {"role": "assistant", "channel": "final", "content": [{"type": "text", "text": "verdict: no"}]},
            {"role": "assistant", "channel": "analysis", "content": "verdict: yes"},
        ]}), "verdict: no")

    def test_verdict_parser_handles_reasoning_and_ambiguity(self):
        self.assertTrue(parse_verdict("<think>verdict: no</think><answer>verdict: yes</answer>"))
        self.assertFalse(parse_verdict("**verdict:** no"))
        self.assertIsNone(parse_verdict("verdict: yes\nverdict: no"))
        self.assertIsNone(parse_verdict("<think>\nverdict: yes"))

    def test_latest_attempt_supersedes_failure(self):
        row = self.records()[0]
        result = score([row], [self.prediction(row, "verdict: no", end_flag=False), self.prediction(row, "verdict: yes")])
        self.assertEqual(result["summary"]["overall_accuracy"], 1)
        self.assertEqual(result["summary"]["duplicate_rows_superseded"], 1)

    def test_foreign_ids_rejected(self):
        with self.assertRaises(ValueError):
            score(self.records(), [{"qid": "unrelated"}])


class RuntimeTests(unittest.TestCase):
    def test_resume_retries_failures_and_keeps_candidate_ids_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            path.write_text('\n'.join(json.dumps(r) for r in [
                {"qid": "pair-a", "end_flag": True},
                {"qid": "pair-b", "end_flag": False},
                {"qid": "pair-c", "end_flag": True, "error": "timeout"},
            ]))
            self.assertEqual(completed_ids(path), {"pair-a"})

    def test_manifest_prevents_mixing_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "input.jsonl"
            data.write_text(json.dumps({"question": "Toy question?", "candidate_answer": "A", "label": True}) + '\n')
            args = SimpleNamespace(task="verify", data_path=str(data), limit=None, output_dir=temp,
                                   model_name="toy-model", model_url="http://localhost:1/v1",
                                   search_url="http://localhost:2", prompt="strict", max_turns=100,
                                   num_workers=1, reasoning_effort="high")
            prepare_run(args)
            args.prompt = "relaxed"
            with self.assertRaises(ValueError):
                prepare_run(args)

    def test_all_verifier_prompt_boundaries_exclude_gold(self):
        # Execute the actual adapter process_item functions with a fake model
        # boundary, without importing model packages or calling any service.
        for path in sorted((ROOT / "browsecomp_verify/agents").glob("*_verify.py")):
            with self.subTest(adapter=path.name), tempfile.TemporaryDirectory() as temp:
                captured = []

                async def fake_worker(args, qid, prompt):
                    captured.append(prompt)
                    return {"end_flag": True, "messages": [{"role": "assistant", "content": "verdict: yes"}]}

                tree = ast.parse(path.read_text())
                function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_item")
                namespace = {"worker_entry": fake_worker, "verification_prompt": verification_prompt,
                             "FileLock": lambda _: nullcontext(), "json": json}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
                item = normalize_record({"id": 7, "question": "Toy question?", "candidate_answer": "Toy candidate",
                                         "answer": "GOLD_SENTINEL_NEVER_TO_MODEL", "label": False})
                output = Path(temp) / "data_stream.jsonl"
                asyncio.run(namespace["process_item"](SimpleNamespace(prompt="strict"), item, str(output)))
                self.assertEqual(len(captured), 1)
                self.assertIn("Toy candidate", captured[0])
                self.assertNotIn("GOLD_SENTINEL_NEVER_TO_MODEL", captured[0])
                saved = json.loads(output.read_text())
                self.assertEqual(saved["qid"], item["qid"])
                self.assertIs(saved["label"], False)


if __name__ == "__main__":
    unittest.main()
