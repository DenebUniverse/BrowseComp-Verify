# Model configuration examples

These mappings follow the experiment launch commands. All examples use DIAMOND, environment-supplied credentials, and deployment URLs supplied by the user. No internal proxy, machine address, cache path, or credential is distributed.

| Model | Release interface | Source client | Source launch script |
| --- | --- | --- | --- |
| Claude-v4.6-Opus | `claude` | `agent_client_claude_v4_6_fc_verify.py`; solving: `agent_client_claude_v4_6_fc.py` | `2_dev_strict_claude.sh` |
| Qwen3.5-397B-A17B | `qwen3_5` | `agent_client_qwen3_5_verify.py` | `2_dev_strict_qwen3_5.sh` |
| DeepSeek-v4-Flash-Preview | `deepseek_v4_flash` | `agent_client_deepseek_v3_2_verify.py` | `2_dev_strict_dpsk_v4.sh` |
| DeepSeek-v4-Pro-Preview | `deepseek_v4_pro` | `agent_client_deepseek_v3_2_verify.py` | `2_dev_strict_dpsk_v4.sh` |
| DeepSeek-v4-Flash, search/visit ablation | `deepseek_v4_flash_visit` | `agent_client_deepseek_v4_flash_qwen3_5_verify.py` | `2_dev_strict_dpsk_v4_search_visit.sh` |

The native DeepSeek filenames retain their original v3.2 names because the launch scripts reuse them for v4. For solving, the same source script selects `agent_client_deepseek_v3_2.py` with the corresponding v4 model name.

## Example commands

Read and configure [`examples/run_models.sh`](../examples/run_models.sh) before running it. Like `scripts/2_run.sh`, it is an editable example, not an automatically executed workflow. It runs one selected model at a time and defaults to one process. These examples call your configured model services when executed.

```bash
# Configure MODEL_URL, SEARCH_URL, and the appropriate API key in your environment.
bash examples/run_models.sh claude
bash examples/run_models.sh qwen3_5
bash examples/run_models.sh deepseek_v4_flash
bash examples/run_models.sh deepseek_v4_pro
bash examples/run_models.sh deepseek_v4_flash_visit
```

For Qwen3.5 and the search/visit ablation, also configure `SUMMARY_URL`. It points to the separate webpage extraction/summary model, not to the corpus retrieval service. The source summary sampler uses a server-selected `default` model identifier.

## Preserve configuration distinctions

- Main interfaces use the source agent's search/open/find tools; Qwen3.5 and the ablation expose search/visit-style extraction tools.
- The Claude launch script uses the gateway alias `claude-opus-4-6-cc`. Override `MODEL_NAME` for your own compatible gateway.
- Native DeepSeek launch commands request `max` effort, but the shared source client does not include that field in the outgoing payload. This is preserved rather than silently changing the experiment code. The separate search/visit client includes it.
- The source search/visit ablation command specifies **500 turns**, whereas the general/main example uses **100**. The example script exposes this difference and allows `MAX_TURNS` overrides; these settings must not be treated as matched budgets.
- Recorded source commands used higher process counts for some deployments. Public examples default to one; process concurrency is not evidence for a model's accuracy or its per-task budget.

These are code/configuration mappings, not new evaluation runs or a claim that current provider aliases still expose the exact historical Preview checkpoints.
