# Included agent interfaces

The release keeps model-specific tool formatting instead of assuming one tool schema works for every model. These are the harnesses found in the supplied research code; filenames preserve their original implementation labels. They do not imply that every model in the paper has a released native client.

| `--agent` | Included tasks | Model-side protocol |
| --- | --- | --- |
| `gptoss` | `verify`, `solve` | GPT-OSS Harmony serialization; `/v1/completions` endpoint |
| `gpt_fc` | `verify` | OpenAI-compatible chat completions with function calling |
| `deepseek` | `verify`, `solve` | DeepSeek-style chat completions, thinking payload, function calling |
| `glm` | `verify`, `solve` | GLM-style HTTP chat completions and tool calls |
| `miro` | `verify`, `solve` | MiroThinker XML/MCP-style tool calls |
| `tongyi` | `verify`, `solve` | Tongyi XML tool calls and tokenizer chat template; `/v1/completions` |
| `claude` | `verify`, `solve` | Claude via the experiment's OpenAI-compatible gateway and function calls |
| `qwen3_5` | `verify` | Qwen3.5 streaming function calls; webpage extraction uses a summary endpoint |
| `deepseek_v4_flash` | `verify`, `solve` | Reuses the DeepSeek client with `--model_name deepseek-v4-flash` |
| `deepseek_v4_pro` | `verify`, `solve` | Reuses the DeepSeek client with `--model_name deepseek-v4-pro` |
| `deepseek_v4_flash_visit` | `verify` | DeepSeek-v4-Flash with the Qwen3.5 search/visit harness and summary endpoint |

`gptoss` builds its Harmony prompt locally and requires the model server to preserve relevant special tokens. `tongyi` loads a tokenizer using `--model_name`; use a Hugging Face name or local path that the tokenizer can resolve. Arbitrary provider aliases may not work for these adapters.

All clients use the local browser backend for retrieval. BrowseComp-Plus supplies the fixed corpus and indexes; this code does not query a live web search engine.

## Coverage of the paper's main-table models

The included source files and reuse mappings cover all eleven main-table model configurations: GPT-OSS-20B/120B, GLM-v4.7, DeepSeek-v3.2, GPT-v5.2-Thinking, Miro-v1.5-30B-A3B, Tongyi-30B-A3B, Claude-v4.6-Opus, Qwen3.5-397B-A17B, and DeepSeek-v4-Flash/Pro-Preview. The two GPT-OSS sizes share one interface. DeepSeek-v4 Flash/Pro share the existing DeepSeek implementation, as specified by the experiment launch scripts.

The DeepSeek Flash search/visit variant is a separate tool-interface ablation. It must not replace the search/open/find main interface in comparisons.

Claude uses the same OpenAI-compatible gateway protocol as the source experiment; it is not a direct Anthropic Messages API adapter. The source gateway's model alias is `claude-opus-4-6-cc`; configure a compatible alias for your deployment rather than assuming it is an official Anthropic model ID.

Qwen3.5 uses `--model_name Qwen3.5-397B-A17B` in the source launch script. Both Qwen3.5 and the DeepSeek search/visit variant require `--summary_url` for a separately hosted webpage-summary model. The included sampler is an HTTP client; installing a package named SGLang is not required for that client.

The source launch script requests `--reasoning_effort max` for DeepSeek-v4. The shared main DeepSeek client does not forward this CLI field in its API payload, so the release records that distinction and preserves the source behavior. The separate search/visit adapter does forward the field. A requested CLI value alone does not establish the effective provider-side setting.

`gpt_fc`, `qwen3_5`, and `deepseek_v4_flash_visit` currently have verification code only. The other interfaces have solving clients. Gemini does not appear among the paper's eleven main-table models. Examples and source configuration details are in [MODEL_CONFIGS.md](MODEL_CONFIGS.md).

## Arguments

Run `python -m browsecomp_verify.run --help`. Common options:

- `--prompt strict|relaxed`: verification prompt; default `strict`.
- `--max_turns 100`: agent-turn budget. A turn may include multiple tool operations or provider-specific requests; it is not a universal token/cost cap.
- `--num_workers`: process concurrency; default 1. Use at most 8 for the local preparation environment.
- `--limit N`: select the first N normalized tasks for a smoke run.
- `--debug`: run only the first pending task in the current process.
- `--reasoning_effort low|medium|high|max`: `max` is reserved for DeepSeek interfaces. Forwarding remains adapter-specific, as noted above.
- `--summary_url`: summary model's chat-completions URL for Qwen3.5 and the DeepSeek search/visit variant.
- `--dry_run`: read/validate the dataset and chosen interface without importing model packages or making network calls.

API credentials are read from environment variables. `DEEPSEEK_API_KEY`, `ZHIPU_API_KEY`, and `ANTHROPIC_API_KEY` take precedence for their respective adapters, followed by `OPENAI_API_KEY`, the legacy `GATEWAY_API_KEY`, and `OPENROUTER_API_KEY`. Credentials are not CLI arguments or manifest fields.

For solving on DIAMOND questions, use `--task solve`; the loader runs once per `question_id`. Solving receives only the question. The optional grader is separate:

```bash
python -m browsecomp_verify.evaluate_solver \
  --input_dir results/diamond-solver \
  --model_url http://127.0.0.1:8001/v1 \
  --judge_model YOUR_JUDGE_MODEL --num_workers 1
```

This LLM-based answer grader incurs model calls. It is not the deterministic verifier scorer.

