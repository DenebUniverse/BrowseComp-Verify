"""Select a native research harness without importing model dependencies for --help."""

import argparse
import importlib
import json
import sys

from .data import load_dataset
from .runtime import add_agent_arguments

AGENTS = {
    "gptoss": {"verify": "agent_client_gptoss_verify", "solve": "agent_client_gptoss"},
    "gpt_fc": {"verify": "agent_client_gpt_v5_fc_verify"},
    "deepseek": {"verify": "agent_client_deepseek_v3_2_verify", "solve": "agent_client_deepseek_v3_2"},
    "glm": {"verify": "agent_client_glm_v4_7_verify", "solve": "agent_client_glm_v4_7"},
    "miro": {"verify": "agent_client_miro_thinker_verify", "solve": "agent_client_miro_thinker"},
    "tongyi": {"verify": "agent_client_tongyi_verify", "solve": "agent_client_tongyi_dev"},
    "claude": {"verify": "agent_client_claude_v4_6_fc_verify", "solve": "agent_client_claude_v4_6_fc"},
    "qwen3_5": {"verify": "agent_client_qwen3_5_verify"},
    "deepseek_v4_flash": {"verify": "agent_client_deepseek_v3_2_verify", "solve": "agent_client_deepseek_v3_2"},
    "deepseek_v4_pro": {"verify": "agent_client_deepseek_v3_2_verify", "solve": "agent_client_deepseek_v3_2"},
    "deepseek_v4_flash_visit": {"verify": "agent_client_deepseek_v4_flash_qwen3_5_verify"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, choices=sorted(AGENTS))
    parser.add_argument("--task", choices=("verify", "solve"), default="verify")
    parser.add_argument("--dry_run", action="store_true", help="validate data and configuration without model or retrieval calls")
    add_agent_arguments(parser)
    args = parser.parse_args()
    if args.task not in AGENTS[args.agent]:
        parser.error(f"No {args.task} harness is included for {args.agent}")
    if args.reasoning_effort == "max" and args.agent not in ("deepseek", "deepseek_v4_flash", "deepseek_v4_pro", "deepseek_v4_flash_visit"):
        parser.error("max reasoning effort is reserved for the DeepSeek interfaces")
    if args.dry_run:
        rows = load_dataset(args.data_path, args.task, args.limit)
        print(json.dumps({"agent": args.agent, "task": args.task, "samples": len(rows),
                          "prompt": args.prompt, "max_turns": args.max_turns,
                          "network_calls": 0}, indent=2))
        return
    # Strip wrapper-only options; each harness uses the same argument parser.
    forwarded = sys.argv[1:]
    for option in ("--agent", "--task"):
        if option in forwarded:
            index = forwarded.index(option)
            del forwarded[index:index + 2]
        forwarded = [arg for arg in forwarded if not arg.startswith(option + "=")]
    sys.argv = [sys.argv[0]] + forwarded
    module = importlib.import_module("browsecomp_verify.agents." + AGENTS[args.agent][args.task])
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    module.main()


if __name__ == "__main__":
    main()
