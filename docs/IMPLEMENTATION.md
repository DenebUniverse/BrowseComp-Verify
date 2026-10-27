# Release preparation notes

The selected research sources were copied into a standalone Python package. The original workspace is not required at runtime. Paper assets, local recordings, slides, temporary files, credentials, and raw model trajectories are excluded.

## Changes from the supplied research scripts

- Package-relative imports replace loose imports from the working directory.
- A shared loader maps `candidate_answer` to the legacy `response` interface, decodes only explicitly marked text fields, and creates stable pair IDs.
- Verification prompts receive only question/candidate text; labels and reference answers are stored only after inference.
- Strict and relaxed prompt bodies are preserved, with `strict` as the explicit CLI default. The supplied GPT-OSS verifier selected the relaxed template while other adapters selected strict; callers now select the protocol explicitly.
- Shared configuration exposes the turn budget, data limit, and process count. Previously unbounded loops now enforce the selected turn budget.
- Resume logic tracks completed pairs by ID, retries failed records, and checks configuration/data compatibility through a manifest.
- API keys come from environment variables; URLs and paths are configurable. Empty-response diagnostics no longer reference an undefined exception variable.
- Dense-specific libraries are imported only by the dense path. Corpus SQL uses a parameter for the path, index shards are ordered, and missing FAISS hit indices are excluded.
- Default Lucene snippets and their prefix fallback are retained; an explicit prefix-only alternative is documented separately.
- Verification scoring is deterministic and reports failure denominators. The original LLM-based solving grader remains a separate optional utility with an explicit judge-model argument.

These changes support using DIAMOND and the available interfaces. They do not establish numerical equivalence to an archived experiment environment.

## Upstream relationship

[BrowseComp](https://openai.com/index/browsecomp/) supplies source questions and the contamination-deterrence convention. [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) supplies retrieval resources. Parts of the native browser and retrieval bridge are based on [GPT-OSS-BrowseCompPlus-Eval](https://github.com/Hannibal046/GPT-OSS-BrowseCompPlus-Eval); its source attribution is retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Dataset conversion uses up to 8 local processes and one numerical-library thread per process. It does not call inference services or modify the source result file.


## Server-source model coverage

The model clients and shared browser/retrieval sources were refreshed from the experiment server. Claude solving/verification and Qwen3.5 verification are included, together with its summary HTTP sampler and extractor prompt. DeepSeek-v4 Flash/Pro reuse the main DeepSeek client; the Flash search/visit ablation has a separate adapter. Model mappings and the source CLI-versus-payload distinction for `max` effort are documented in [MODEL_CONFIGS.md](MODEL_CONFIGS.md).

Only packaging, DIAMOND data boundaries, environment credentials, and explicit resume/budget handling are adapted. Standalone demo blocks containing sample conversations are omitted from helper modules. Server synchronization and private deployment utilities are not part of the published code.
