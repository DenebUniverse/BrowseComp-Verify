# Environment and retrieval setup

## Python environments

Use Python **3.12+** for the current `gpt-oss` dependency. A minimal install supports dataset validation, offline preflight, and deterministic scoring without any third-party Python packages:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The environment scripts are the main setup entry points. They separate the client, retrieval, and inference-server environments:

```bash
bash scripts/0_init_env.sh client
bash scripts/0_init_env.sh retrieval
# Separate Linux CUDA environment, only if hosting your own model:
bash scripts/0_init_env.sh server
```

The recipes retain `vllm==0.10.2` with the CUDA 12.8 package index and `qwen-agent==0.0.26` from the supplied `scripts/0_init_env.sh`. The client package uses `gpt-oss==0.0.9`, whose upstream package metadata requires Python 3.12+. Heavy GPU serving dependencies are optional for HTTP clients. These scripts only prepare environments; they never start evaluation.

The reference environments and instructions are [BrowseComp](https://openai.com/index/browsecomp/), [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), and [GPT-OSS-BrowseCompPlus-Eval](https://github.com/Hannibal046/GPT-OSS-BrowseCompPlus-Eval). Hardware-dependent PyTorch/CUDA installation remains specific to your host. No global pip mirror settings or system package changes are made automatically.

## Retrieval resources

We use the corpus and indexes supplied by **[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)**. Download resources from the official Hugging Face repositories:

```bash
python -m pip install huggingface_hub
bash scripts/download_retrieval_data.sh qwen3-embedding-8b
# Or: bash scripts/download_retrieval_data.sh bm25
```

Sources:

- [Corpus: Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus), with `docid`, `url`, and `text` columns.
- [Indexes: Tevatron/browsecomp-plus-indexes](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes).
- [Upstream index download script](https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_build_index/download_indexes.sh).

No corpus, index shards, or model weights are bundled here. Dense index shards are Python pickle files from the upstream distribution; use only trusted index artifacts.

## Tevatron and CUDA

Dense retrieval uses `tevatron.retriever` and a CUDA encoder. Install PyTorch for your hardware, then install the compatible Tevatron checkout:

```bash
git clone https://github.com/texttron/tevatron.git external/tevatron
# Select the experiment-compatible revision if available.
python -m pip install -e external/tevatron
```

The setup script follows the upstream editable-install approach. Set `TEVATRON_DIR` to reuse your working checkout or `TEVATRON_REV` to select a specific revision. See [Tevatron's installation instructions](https://github.com/texttron/tevatron#installation).

## Java and snippet highlighting

BM25 and the default Lucene snippet highlighter require **Java 21** and Pyserini. Configure `JAVA_HOME` for your installation before starting the service. See [Pyserini's installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md).

`LUCENE_EXTRA_DIR` may contain additional `lucene-highlighter`, `lucene-queries`, and `lucene-memory` jars. **Match their versions to the installed Anserini/Lucene version.** The native-tool upstream example uses 9.9.1. To avoid mixing it with a newer Pyserini, the helper reads the installed fatjar metadata and downloads matching jars:

```bash
python scripts/prepare_lucene_jars.py
# If Maven version metadata is absent, identify the installed version first:
# python scripts/prepare_lucene_jars.py --version 9.9.1
```

Use the explicit version only when it matches your installed environment. The helper downloads jars; it does not start a retrieval service or model.

The default `SNIPPET_MODE=lucene` retains the original highlighting path and the 180-character snippet limit. If a Lucene class/highlighter call fails, the original prefix fallback is retained. Setting `SNIPPET_MODE=prefix` explicitly bypasses the highlighter; this changes the snippet protocol and must be disclosed in comparisons. A dense-only prefix setup can run without Java.

## Configuration and startup

```bash
cp .env.example .env
# Edit values before loading them; keep .env private.
set -a
source .env
set +a
bash scripts/1_search_engine_setup.sh
```

| Variable | Purpose |
| --- | --- |
| `CORPUS_PARQUET_PATH` | Corpus parquet path/glob |
| `SEARCHER_TYPE` | `dense` or `bm25` |
| `DENSE_INDEX_PATH` | Dense pickle shard glob |
| `DENSE_MODEL_NAME` | Encoder compatible with the index |
| `GPU_IDS` | Comma-separated CUDA device IDs |
| `LUCENE_INDEX_DIR` | BM25 index directory |
| `LUCENE_EXTRA_DIR` | Optional matching Lucene jars |
| `SNIPPET_MODE` | `lucene` (default) or explicit `prefix` alternative |
| `MAX_SNIPPET_LEN` | Snippet character limit, default 180 |
| `MODEL_URL`, `MODEL_NAME` | Model inference endpoint and served identifier |
| `OPENAI_API_KEY` | Model credential; use `EMPTY` only for unauthenticated local servers |

The server binds to `127.0.0.1:8000` by default and uses one Uvicorn worker to avoid duplicating the corpus/encoder in memory. `GET /` reports the active searcher and snippet mode. `/search` and `/get_content` use POST, matching [the browser integration](../browsecomp_verify/browser.py).

Corpus loading currently builds in-memory document mappings. Provision RAM for the corpus in addition to GPU memory for the encoder/model. No model or retrieval evaluation is run by package installation or dataset preparation.

