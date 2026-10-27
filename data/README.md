# DIAMOND

DIAMOND contains **207 verification tasks**: 103 correct candidates and 104 incorrect candidates. Each task pairs a question with one candidate answer and a Boolean correctness label.

## Files

- `diamond.jsonl`: public distribution, with question/candidate/reference text obfuscated.
- `diamond.manifest.json`: record counts, source/output checksums, and field mapping.

The authors designated `eval_result_diamond.jsonl` as the source of this distribution. The conversion preserves the source's `correct` Boolean as `label` and `response` text as `candidate_answer`; it does not rerun a grader, relabel tasks, or distribute source trajectories. The manifest records this provenance without claiming additional annotation history.

## Schema

| Field | Meaning |
| --- | --- |
| `id` | Stable pair ID derived from the decoded question and candidate |
| `question_id` | Source question ID |
| `question` | Obfuscated question text |
| `candidate_answer` | Obfuscated candidate text, preserving the supplied response |
| `label` | JSON Boolean: whether the candidate is correct |
| `answer` | Obfuscated reference answer, for offline use only |
| `_obfuscated_fields` | Explicit list of encoded fields |

The `id` hash uses UTF-8 JSON serialization of `[question, candidate_answer]`, with `ensure_ascii=False` and separators `(',', ':')`; the public ID is `bcv-` plus the first 24 hexadecimal SHA-256 characters. It is not just a question ID.

```python
from browsecomp_verify.data import load_dataset

tasks = load_dataset("data/diamond.jsonl")
assert len(tasks) == 207
assert sum(task["label"] for task in tasks) == 103
# Pass only task["question"] and task["candidate_answer"] to the verifier.
```

`load_dataset` decodes text in memory and sets the internal `qid` to the stable pair ID. Neither `answer` nor `label` is inserted into the verification prompt. Please do not publish decoded benchmark text, screenshots, or run logs containing it. See [BrowseComp's contamination request](https://openai.com/index/browsecomp/) and the upstream [BrowseComp-Plus decoding script](https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_build_index/decrypt_dataset.py).

## Preparing the distribution

With the author-designated source file available locally and the package installed:

```bash
python scripts/prepare_diamond.py \
  --source /path/to/eval_result_diamond.jsonl \
  --output data/local/diamond.jsonl \
  --workers 8
```

The converter allows 1–8 processes, limits numerical-library threads to one, preserves input order, and refuses to overwrite an existing output. The obfuscation is the upstream SHA256-keyed XOR + Base64 convention with the BrowseComp canary. It is reversible and does not imply encryption for confidentiality.

## Rights and use

Questions/reference answers originate from BrowseComp; retain the applicable upstream terms and notices in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The project code license does not override rights in upstream dataset content, model-generated text, or source web pages. The retrieval corpus and indexes are downloaded separately from BrowseComp-Plus.

