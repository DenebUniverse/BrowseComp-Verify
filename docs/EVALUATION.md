# Verification evaluation

The verification target is candidate correctness: `true` means the verifier should accept the candidate; `false` means it should reject it. Reference answers and labels are retained for offline analysis but are not inserted into model prompts.

## Final-answer parsing

The scorer reads the final non-tool assistant message, excluding Harmony analysis messages. For Tongyi-style output, it reads the final `<answer>...</answer>` region after removing `<think>...</think>`. A verdict must be an explicit `verdict: yes` / `verdict: no`, or a bare `yes` / `no`. Conflicting verdict lines are unparseable. Unclosed reasoning tags are not treated as final answers.

The saved `response` field is the candidate supplied to the verifier; it must never be scored as the verifier's judgment. `messages` (or the explicit `verifier_response` field) supplies that judgment.

## Denominators

| Metric | Definition |
| --- | --- |
| Overall accuracy | Correct verdicts / all expected tasks |
| True accuracy | Correct accepts / all True-labeled tasks |
| False accuracy | Correct rejects / all False-labeled tasks |
| Completion rate | Successful completed agent runs / all expected tasks |
| Coverage | Completed runs with a valid verdict / all expected tasks |
| Accuracy on valid verdicts | Correct verdicts / valid verdicts |

Failures, missing outputs, and invalid verdicts contribute zero correct predictions to overall and class accuracy. Missing classes return `null` instead of a fabricated zero. Metrics use proportions in `[0, 1]`. Overall accuracy is computed from counts, not the unweighted mean of the two class accuracies.

The latest record for a pair ID supersedes previous appended attempts. Duplicate input rows are reported. Unknown prediction IDs fail explicitly instead of silently changing the denominator. When inference uses `--limit N`, pass the same `--limit N` to the scorer.

The deterministic scorer is a release utility with explicit parsing. It should not be conflated with the optional LLM-based solving-answer grader or treated as proof of reproducing published numbers under another evaluator.

## Resuming runs

`run_manifest.json` records the data checksum, pair IDs, model endpoint/name, prompt, reasoning effort, and budget. Reusing an output directory with incompatible settings fails. Concurrency may change during a resume. Completed pairs are skipped; failed pairs can be retried.

Each DIAMOND task has a question/candidate hash ID. Resume logic uses that ID rather than question text, so two candidates for the same question remain distinct tasks. Keep output directories and manifests with experiment records, but do not publish logs containing decoded benchmark text.

