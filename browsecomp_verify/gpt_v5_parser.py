# Synced from the server; standalone demo blocks omitted.
from typing import Any, Dict, Optional, List
import datetime
import textwrap

DEVELOPER_CONTENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.
Your response should be in the following format:
Explanation: {your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
""".strip()

## tools
GPT_OSS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches for information related to `query` and displays `topn` results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "topn": {
                        "type": "number",
                        "default": 10
                    },
                    "source": {
                        "type": "string"
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\nValid link ids are displayed with the formatting: `【{id}†.*】`.\nIf `cursor` is not provided, the most recent page is implied.\nIf `id` is a string, it is treated as a fully qualified URL associated with `source`.\nIf `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\nUse this function without `id` to scroll to a new location of an opened page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": [
                            "number",
                            "string"
                        ],
                        "default": -1
                    },
                    "cursor": {
                        "type": "number",
                        "default": -1
                    },
                    "loc": {
                        "type": "number",
                        "default": -1
                    },
                    "num_lines": {
                        "type": "number",
                        "default": -1
                    },
                    "view_source": {
                        "type": "boolean",
                        "default": False
                    },
                    "source": {
                        "type": "string"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string"
                    },
                    "cursor": {
                        "type": "number",
                        "default": -1
                    }
                },
                "required": [
                    "pattern"
                ]
            }
        }
    }
]

## systems
HACK_SYSTEM_CONTENT = """You are **Deep Research Agent** - a web agent specialized in evidence-based research and information retrieval.

<TOOLS>
<<<TOOLS>>>
</TOOLS>

**MANDATORY XML FORMAT - NO EXCEPTIONS:**
You MUST respond using XML format with <think> and <tool_call> tags. Failure to use this exact XML format will result in system errors. 

<think>
Write your inner monolog here before tool call. 
Consider: What information gaps exist? Have I searched comprehensively? Am I ready for a definitive answer? 
The thinking must be as detailed as possible, longer than **3,000 tokens**!!!
THINK comprehensively before search action. Write detailed inner monologue analyzing all information gaps, search comprehensiveness, and readiness for definitive answers. 
Your thinking process must exceed **3,000 tokens**!!!
Start by examining clues and search results individually, explore all possibilities, generate and test candidate hypotheses against all evidence, apply cross-domain reasoning when information isn't directly relevant, systematically eliminate incorrect options with clear explanations, then synthesize all validated clues into the most probable search action with complete justification.
</think>

<tool_call>
{"name": "<function-name>", "arguments": { ... }}
</tool_call>

**CRITICAL:** Always use <think> before ANY <tool_call>. No exceptions."""

def format_openai_system_prompt(
    tool_config: Any,
    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI.",
    knowledge_cutoff: str = "2024-06",
    current_date: Optional[str] = None,
    reasoning_effort: str = "high",
    valid_channels: Optional[List[str]] = None,
) -> str:
    """
    Convert ToolNamespaceConfig(name, description, tools=[ToolDescription...]) to a
    plain-text system prompt compatible with OpenAI Chat Completions.

    This mimics the 'gpt_oss_prompt' system section:
      - identity + metadata (cutoff/date/reasoning)
      - "# Tools" block
      - "namespace <name> { ... }" tool signatures
      - channel config line

    Notes:
      - OpenAI Chat Completions does NOT accept Harmony 'system_content' objects.
      - This returns ONLY the system text content (for messages[0].content).
    """
    if current_date is None:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if valid_channels is None:
        valid_channels = ["analysis", "commentary", "final"]

    ns = getattr(tool_config, "name", None) or "tools"
    ns_desc = getattr(tool_config, "description", "") or ""

    # Render namespace description as comment lines.
    desc_lines: List[str] = []
    if ns_desc.strip():
        for line in ns_desc.splitlines():
            line = line.rstrip()
            if line:
                desc_lines.append(f"// {line}")
            else:
                desc_lines.append("//")
    ns_desc_block = "\n".join(desc_lines)

    def _fmt_default(v: Any) -> str:
        # Match gpt_oss_prompt style: JSON booleans are lowercase
        if v is True:
            return "true"
        if v is False:
            return "false"
        return str(v)

    # Render each tool signature
    tool_blocks: List[str] = []
    for td in getattr(tool_config, "tools", []) or []:
        tname = getattr(td, "name", "")
        tdesc = getattr(td, "description", "") or ""
        tparams = getattr(td, "parameters", None) or {"type": "object", "properties": {}}

        # Description -> comment block
        tdesc_lines: List[str] = []
        for line in tdesc.splitlines():
            line = line.rstrip()
            if line:
                tdesc_lines.append(f"// {line}")
            else:
                tdesc_lines.append("//")
        tdesc_block = "\n".join(tdesc_lines)

        # Generate a signature by reading properties + required + defaults.
        props: Dict[str, Any] = (tparams.get("properties") or {}) if isinstance(tparams, dict) else {}
        required = set(tparams.get("required", []) or []) if isinstance(tparams, dict) else set()

        param_lines: List[str] = ["_: {"]
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any") if isinstance(pschema, dict) else "any"
            default = pschema.get("default", None) if isinstance(pschema, dict) else None

            # Normalize type field to TS-ish
            if isinstance(ptype, list):
                ptype_str = " | ".join(str(x) for x in ptype)
            else:
                ptype_str = str(ptype)

            opt = "" if pname in required else "?"
            line = f"{pname}{opt}: {ptype_str},"
            if default is not None:
                line += f" // default: {_fmt_default(default)}"
            param_lines.append(line)

        param_lines.append("}) => any;")

        tool_blocks.append(
            "\n".join(
                [
                    tdesc_block,
                    f"type {tname} = (" + "\n".join(param_lines),
                    "",
                ]
            ).rstrip()
        )

    tools_block = "\n\n".join(tool_blocks).rstrip()
    
    sys = f"""\
{model_identity}
Knowledge cutoff: {knowledge_cutoff}
Current date: {current_date}

Reasoning: {reasoning_effort}

# Tools

## {ns}

{ns_desc_block}
namespace {ns} {{

{tools_block}

}} // namespace {ns}

# Valid channels: {", ".join(valid_channels)}. Channel must be included for every message."""
    return textwrap.dedent(sys).rstrip()

def get_tools_from_tool_config(tool_config):
    tools = [{
        "type": "function",
        "function": {
            "name": config.name,
            "description": config.description,
            "parameters": config.parameters,
            } 
        }
        for config in tool_config.tools]
    return tools

    # prompt=format_openai_system_prompt(tool_config)

    # assert prompt==gpt_oss_prompt, prompt
    
