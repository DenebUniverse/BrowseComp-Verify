# Third-party sources and permissions

The Apache-2.0 project license covers the authors' original contributions. It does not replace upstream copyright notices, licenses, dataset terms, or permissions for adapted code.

| Source | Use in this project | Terms / notice |
| --- | --- | --- |
| [OpenAI BrowseComp](https://openai.com/index/browsecomp/) / [simple-evals](https://github.com/openai/simple-evals) | Underlying questions/reference answers and canary convention | MIT notice retained in [BrowseComp.LICENSE](third_party/BrowseComp.LICENSE); follow the benchmark's request against online plaintext examples |
| [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | Fixed retrieval corpus/index resources and the compatible obfuscation convention | MIT notice retained in [BrowseComp-Plus.LICENSE](third_party/BrowseComp-Plus.LICENSE); resources are downloaded separately |
| [GPT-OSS-BrowseCompPlus-Eval](https://github.com/Hannibal046/GPT-OSS-BrowseCompPlus-Eval) | Adapted native browser and retrieval integration, especially `browser.py`, `retrieval/backend.py`, and `retrieval/service.py` | The project authors confirm permission to redistribute the relevant integrated code. No repository-wide LICENSE was found upstream at preparation time. This does not assert an Apache-2.0 license for the upstream repository; retain any applicable permission conditions |
| [OpenAI GPT-OSS](https://github.com/openai/gpt-oss) | Imported browser tools and Harmony-oriented inference interfaces | Apache-2.0; copy in [gpt-oss.LICENSE](third_party/gpt-oss.LICENSE) |
| [Tevatron](https://github.com/texttron/tevatron), [Pyserini](https://github.com/castorini/pyserini), [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), model/tokenizer packages | Runtime dependencies and native interface conventions | Installed separately; retain their respective licenses and model terms |

The author-designated DIAMOND source supplies the candidate text and labels. Reference questions retain upstream rights; distributing obfuscated text does not change those rights. No retrieval corpus, index, model weights, or third-party source checkout is redistributed in this repository.

Permission evidence/specific conditions for the Hannibal046 integration should be retained by the maintainers. Add a public permission statement if the author has approved publishing it; do not invent license wording on their behalf.

