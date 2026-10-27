# BrowseComp-Verify

English | [简体中文](README.zh-CN.md)

**[BrowseComp-Verify: Benchmarking Self-derived Verification Agents in Web Search](https://aclanthology.org/2026.emnlp-main.1889/)**  
Guanzhen Li and Shijie Li · EMNLP 2026

[Paper](https://aclanthology.org/2026.emnlp-main.1889/) · [DIAMOND](data/README.md) · [Setup](docs/SETUP.md) · [Agent interfaces](docs/AGENTS.md) · [Website source](website/README.md) · [Citation](#citation)

BrowseComp-Verify studies **self-derived verification**: given a question and a candidate answer, an agent derives the question's constraints, retrieves evidence, and decides whether to accept the candidate. The agent receives neither a reference answer, a solver trajectory, nor a human-written rubric.

This repository provides the **207-task DIAMOND dataset**, native agent harnesses, a local retrieval service, and evaluation tools. It supports comparisons between strict and relaxed verification prompts. The fixed retrieval corpus and indexes come from [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus).

## DIAMOND

| Property | Value |
| --- | --- |
| Verification tasks | 207 |
| Correct candidates (`label: true`) | 103 |
| Incorrect candidates (`label: false`) | 104 |
| Distribution format | JSONL with obfuscated text |
| Runtime input | Question + candidate answer |
| Offline scoring target | Boolean candidate correctness |

The loader deobfuscates text in memory. Please follow the [BrowseComp request against posting benchmark examples in plaintext or images online](https://openai.com/index/browsecomp/). Obfuscation reduces accidental contamination; it is not a confidentiality mechanism. See the [data card](data/README.md) for the schema and provenance.

## Quick start

Run commands from the repository root. Use Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Dataset and configuration checks require no model, GPU, or retrieval service.
python -m browsecomp_verify.data --data_path data/diamond.jsonl
python -m browsecomp_verify.run \
  --agent gptoss --task verify --data_path data/diamond.jsonl --dry_run
```

### 1. Install agent and retrieval dependencies

```bash
bash scripts/0_init_env.sh client
# In the retrieval environment:
bash scripts/0_init_env.sh retrieval
```

The recipes follow the supplied environment scripts and the three upstream projects cited below. Dense retrieval requires [Tevatron](https://github.com/texttron/tevatron) and a CUDA GPU. Lucene snippets/BM25 require Java 21 and matching Anserini/Lucene jars; see the [setup guide](docs/SETUP.md). Model serving runs separately from the client; `bash scripts/0_init_env.sh server` preserves the original `vllm==0.10.2` / CUDA 12.8 installation recipe for a Linux CUDA environment.

### 2. Download retrieval resources and start the service

```bash
python -m pip install huggingface_hub
bash scripts/download_retrieval_data.sh qwen3-embedding-8b

cp .env.example .env
# Edit .env with your model endpoint, model name, API key, and retrieval paths.
set -a
source .env
set +a
bash scripts/1_search_engine_setup.sh
```

The service exposes `POST /search` and `POST /get_content`. It searches the configured fixed corpus rather than the live web. Keep it running while evaluating agents.

### 3. Run verification

With a compatible model endpoint and the retrieval service available:

```bash
python -m browsecomp_verify.run \
  --agent gptoss --task verify \
  --model_name gpt-oss-120b \
  --model_url http://127.0.0.1:8001/v1 \
  --search_url http://127.0.0.1:8000 \
  --data_path data/diamond.jsonl \
  --output_dir results/diamond-gptoss-strict \
  --prompt strict --reasoning_effort high \
  --max_turns 100 --num_workers 1
```

Use `--limit 2 --debug` for a single-task diagnostic run (`--debug` runs only the first pending task). Use `--prompt relaxed` and a **new output directory** for the prompt comparison. Completed pairs are skipped when resuming the same configuration. Pair IDs distinguish different candidates for the same question.

Available verifier interfaces: `gptoss`, `gpt_fc`, `deepseek`, `glm`, `miro`, `tongyi`, `claude`, and `qwen3_5`. The explicit `deepseek_v4_flash` / `deepseek_v4_pro` aliases reuse the DeepSeek interface; `deepseek_v4_flash_visit` is the separate tool-interface ablation. These preserve different tool protocols; selecting an interface does not automatically make a model compatible with it. See [AGENTS.md](docs/AGENTS.md) for endpoint requirements and implementation coverage, and [model configurations](docs/MODEL_CONFIGS.md) for the server-source mappings and additional editable examples.

[`scripts/2_run.sh`](scripts/2_run.sh) is an **editable example**, with the same model, endpoint, budget, and concurrency options shown above. Run it only after configuring your own services and credentials. You can use the Python CLI directly instead; the example is not a required entry point.

### 4. Score verification results

```bash
python -m browsecomp_verify.evaluate \
  --data_path data/diamond.jsonl \
  --predictions results/diamond-gptoss-strict/data_stream.jsonl \
  --output results/diamond-gptoss-strict/metrics.json
```

The scorer reads the verifier's final assistant answer and extracts `verdict: yes/no`. It reports overall accuracy, class-conditional accuracy, completion rate, and valid-verdict coverage. Missing tasks, failed runs, and unparseable verdicts count as incorrect in overall/class accuracy. It never grades the candidate text stored in `response` as if that were the verifier's answer.

This release's deterministic scorer makes its parsing and denominators explicit. It is not a claim that changing the scoring procedure reproduces a paper table. See [evaluation details](docs/EVALUATION.md).

## Repository layout

```text
browsecomp_verify/
  agents/              Model-specific solving and verification harnesses
  retrieval/           Fixed-corpus search and document fetch service
  browser.py           Browser-tool adaptation
  prompts.py           Research prompts
  data.py              Obfuscation, normalization, and pair IDs
  run.py               Agent selection and offline preflight
  evaluate.py          Deterministic verification scoring
  evaluate_solver.py   Optional LLM-based solving-answer grader
data/                  Obfuscated DIAMOND tasks and data card
docs/                  Setup, interfaces, scoring, and release notes
scripts/               Environment setup, retrieval downloads, and run example
examples/              Additional editable model configuration examples
tests/                 Offline tests using synthetic examples
third_party/           Upstream license copies
```

## Acknowledgments and references

- **[BrowseComp](https://openai.com/index/browsecomp/)** by OpenAI provides the underlying questions and the benchmark contamination-deterrence convention.
- **[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)** provides the fixed retrieval corpus and prebuilt indexes used by this project. Its [upstream instructions](https://github.com/texttron/BrowseComp-Plus#-using-browsecomp-plus) are the reference for those resources.
- **[GPT-OSS-BrowseCompPlus-Eval](https://github.com/Hannibal046/GPT-OSS-BrowseCompPlus-Eval)** provides the native GPT-OSS browsing implementation on which parts of the browser/retrieval integration are based.
- We also use [GPT-OSS](https://github.com/openai/gpt-oss), [OpenAI Harmony](https://github.com/openai/harmony), [Tevatron](https://github.com/texttron/tevatron), and [Pyserini](https://github.com/castorini/pyserini).

Original project contributions use **Apache-2.0**. Upstream material retains its applicable terms and notices. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{li-li-2026-browsecomp,
  title = {BrowseComp-Verify: Benchmarking Self-derived Verification Agents in Web Search},
  author = {Li, Guanzhen and Li, Shijie},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  url = {https://aclanthology.org/2026.emnlp-main.1889/},
  pages = {43046--43075}
}
```

The official ACL Anthology BibTeX and upstream references are in [references.bib](references.bib). Please also cite BrowseComp and BrowseComp-Plus when using their questions or retrieval resources.

