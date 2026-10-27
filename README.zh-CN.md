# BrowseComp-Verify

[English](README.md) | 简体中文

**[BrowseComp-Verify: Benchmarking Self-derived Verification Agents in Web Search](https://aclanthology.org/2026.emnlp-main.1889/)**  
Guanzhen Li 和 Shijie Li · EMNLP 2026

[论文](https://aclanthology.org/2026.emnlp-main.1889/) · [DIAMOND 数据集](data/README.md) · [环境配置](docs/SETUP.md) · [Agent 接口](docs/AGENTS.md) · [网站源码](website/README.md) · [引用方式](#引用方式)

BrowseComp-Verify 研究 **self-derived verification（自主推导验证）**：给定问题和候选答案，agent 自行推导问题中的约束，检索证据，并决定是否接受该候选答案。验证过程中不向 agent 提供参考答案、solver 的求解轨迹或人工编写的评判准则。

本仓库提供包含 **207 条验证任务的 DIAMOND 数据集**、保留模型工具协议的 agent 实现、本地检索服务和评测工具，支持比较严格与宽松两种验证提示。固定检索语料和索引来自 [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)。

## DIAMOND

| 项目 | 数值或说明 |
| --- | --- |
| 验证任务数量 | 207 |
| 正确候选答案（`label: true`） | 103 |
| 错误候选答案（`label: false`） | 104 |
| 分发格式 | 文本经过混淆处理的 JSONL |
| 验证时的输入 | 问题 + 候选答案 |
| 离线评分目标 | 表示候选答案正确与否的布尔标签 |

加载器会在内存中自动还原混淆文本。请遵循 [BrowseComp 关于不要在网上以明文或图片形式公开基准样例的请求](https://openai.com/index/browsecomp/)。混淆处理用于减少意外的数据污染，不是保密机制。字段定义和数据来源见[数据说明](data/README.md)。

## 快速开始

以下命令均在仓库根目录执行，使用 Python 3.12 或更新版本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 仅检查数据和配置，不需要模型、GPU 或检索服务。
python -m browsecomp_verify.data --data_path data/diamond.jsonl
python -m browsecomp_verify.run \
  --agent gptoss --task verify --data_path data/diamond.jsonl --dry_run
```

### 1. 安装 agent 和检索服务依赖

```bash
bash scripts/0_init_env.sh client
# 在检索服务环境中执行：
bash scripts/0_init_env.sh retrieval
```

环境配置以项目原有脚本和下方引用的三个上游项目为依据。稠密检索需要 [Tevatron](https://github.com/texttron/tevatron) 和 CUDA GPU；Lucene 摘要生成与 BM25 检索需要 Java 21，以及版本匹配的 Anserini/Lucene jar，详见[环境配置说明](docs/SETUP.md)。

模型服务与客户端分开部署。对于 Linux CUDA 环境，`bash scripts/0_init_env.sh server` 保留了原脚本中 `vllm==0.10.2` 与 CUDA 12.8 软件源的安装方式。

### 2. 下载检索资源并启动服务

```bash
python -m pip install huggingface_hub
bash scripts/download_retrieval_data.sh qwen3-embedding-8b

cp .env.example .env
# 修改 .env，填写模型服务地址、模型名称、API key 和检索资源路径。
set -a
source .env
set +a
bash scripts/1_search_engine_setup.sh
```

检索服务提供 `POST /search` 和 `POST /get_content` 接口，在配置的固定语料中检索，不访问实时互联网搜索。在运行 agent 时，需保持检索服务运行。

### 3. 运行验证

准备好兼容的模型服务和检索服务后，可执行：

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

需要单条任务诊断时，可增加 `--limit 2 --debug`；其中 `--debug` 只运行第一条尚未完成的任务。比较宽松提示时，使用 `--prompt relaxed`，并指定**新的输出目录**。以相同配置恢复运行时，会跳过已完成的问题与候选答案对；同一道问题的不同候选通过独立的 pair ID 区分。

目前提供的 verifier 接口包括：`gptoss`、`gpt_fc`、`deepseek`、`glm`、`miro`、`tongyi`、`claude` 和 `qwen3_5`。`deepseek_v4_flash` / `deepseek_v4_pro` 明确复用 DeepSeek 接口；`deepseek_v4_flash_visit` 是单独的工具接口消融版本。这些接口保留各自的工具协议；选择某个接口并不代表任意模型都与它兼容。服务接口要求及代码覆盖范围见 [Agent 接口说明](docs/AGENTS.md)；服务器代码与实验配置的对应关系，以及更多可修改示例见[模型配置说明](docs/MODEL_CONFIGS.md)。

[`scripts/2_run.sh`](scripts/2_run.sh) 是一个**可修改的运行示例**，展示上述模型、服务地址、预算和并发参数。请先配置自己的服务和凭据，再运行示例。也可以直接使用 Python 命令行；该示例不是必经入口。

### 4. 对验证结果评分

```bash
python -m browsecomp_verify.evaluate \
  --data_path data/diamond.jsonl \
  --predictions results/diamond-gptoss-strict/data_stream.jsonl \
  --output results/diamond-gptoss-strict/metrics.json
```

评分器读取 verifier 的最终 assistant 回答，提取 `verdict: yes/no`，并报告总体准确率、按候选正确性分类的准确率、任务完成率和有效判定覆盖率。计算总体和分类准确率时，缺失任务、运行失败和无法解析的判定均计为错误。评分器不会将 `response` 字段中保存的候选答案误当作 verifier 的判定。

本版本的确定性评分器明确规定了解析方式和统计分母。采用这一评分流程不代表能够直接复现论文表格中的数值；详见[评测说明](docs/EVALUATION.md)。

## 仓库结构

```text
browsecomp_verify/
  agents/              各模型的求解与验证实现
  retrieval/           固定语料检索与文档获取服务
  browser.py           浏览工具适配
  prompts.py           实验提示词
  data.py              文本混淆、数据规范化和 pair ID
  run.py               Agent 选择与离线配置检查
  evaluate.py          确定性验证评分
  evaluate_solver.py   可选的 LLM 求解答案评分器
data/                  混淆后的 DIAMOND 数据及数据说明
docs/                  环境、接口、评测和发布说明
scripts/               环境安装、检索下载、服务启动和运行示例
examples/              其他模型的可修改配置示例
tests/                 使用合成样例的离线测试
third_party/           上游许可证副本
```

## 致谢与相关项目

- OpenAI 的 **[BrowseComp](https://openai.com/index/browsecomp/)** 提供底层问题，以及减少基准数据污染的分发约定。
- **[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)** 提供本项目使用的固定检索语料与预构建索引。相关资源的获取和使用方式以其[上游说明](https://github.com/texttron/BrowseComp-Plus#-using-browsecomp-plus)为准。
- **[GPT-OSS-BrowseCompPlus-Eval](https://github.com/Hannibal046/GPT-OSS-BrowseCompPlus-Eval)** 提供原生 GPT-OSS 浏览工具实现，本项目部分浏览与检索集成基于该实现。
- 本项目还使用了 [GPT-OSS](https://github.com/openai/gpt-oss)、[OpenAI Harmony](https://github.com/openai/harmony)、[Tevatron](https://github.com/texttron/tevatron) 和 [Pyserini](https://github.com/castorini/pyserini)。

项目原创贡献采用 **Apache-2.0**。上游材料保留其适用条款和声明，详见 [LICENSE](LICENSE)、[第三方来源与许可说明](THIRD_PARTY_NOTICES.md)。

## 引用方式

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

ACL Anthology 的官方 BibTeX 和上游项目的引用条目见 [references.bib](references.bib)。使用相关问题或检索资源时，也请引用 BrowseComp 和 BrowseComp-Plus。

本页提供中文版项目介绍与操作说明；链接到的专题文档目前主要为英文。
