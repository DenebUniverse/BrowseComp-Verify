# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import multiprocessing as mp
import tqdm
from filelock import FileLock
from openai import APIConnectionError, OpenAI, RateLimitError

from browsecomp_verify.agents.agent_client_qwen3_5_verify import (
    BrowserPool,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS_CAP,
    MODEL_TIMEOUT,
    Qwen_3_5_SearchAgent,
    TOP_P,
)
from browsecomp_verify.prompts import VERIFY_PROMPT_strict
from browsecomp_verify.utils import load_bcplus_data

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = True):
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            stream=sys.stderr,
            format="%(asctime)s.%(msecs)03d %(levelname)s [PID=%(process)d] %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        root.setLevel(level)

    for noisy in ("urllib3", "httpx", "openai", "httpcore", "httpcore.connection", "httpcore.http11"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class DeepSeekV4FlashQwen35SearchAgent(Qwen_3_5_SearchAgent):
    """Qwen3.5 verifier agent loop with DeepSeek official API as the model backend."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_turns: int = 100,
        browser_pool: BrowserPool = None,
        args=None,
        temperature=1,
    ):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_turns=max_turns,
            browser_pool=browser_pool,
            args=args,
            temperature=temperature,
        )
        self.api_key = api_key
        self.reasoning_effort = getattr(args, "reasoning_effort", "max") if args is not None else "max"

        model_server = self.base_url.rstrip("/")
        if model_server.endswith("/chat/completions"):
            model_server = model_server[: -len("/chat/completions")]
        model_server = model_server.rstrip("/")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=model_server,
        )

    def _single_turn_inference_sync(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        retry_count: int = 32,
    ) -> Dict[str, Any]:
        extra_body = {
            "thinking": {"type": "enabled"},
        }
        if self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": TOP_P,
            "timeout": MODEL_TIMEOUT,
            "extra_body": extra_body,
        }

        for retry in range(retry_count):
            try:
                response = self.client.chat.completions.create(**payload)
                choice = response.choices[0]
                message = choice.message

                message_dict = message.model_dump()
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content is not None:
                    message_dict["reasoning_content"] = reasoning_content

                result = {
                    "choices": [
                        {
                            "message": message_dict,
                            "finish_reason": choice.finish_reason,
                        }
                    ],
                    "model": getattr(response, "model", None),
                }
                if getattr(response, "usage", None):
                    result["usage"] = response.usage.model_dump()
                return result

            except (APIConnectionError, RateLimitError) as e:
                logger.warning("Retry %s/%s failed: %s", retry + 1, retry_count, e, exc_info=True)
                time.sleep(2)
            except Exception as e:
                logger.warning("Retry %s/%s failed: %s", retry + 1, retry_count, e, exc_info=True)
                time.sleep(2)

        return {"error": "Max retries exceeded"}


async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)
    search_agent = DeepSeekV4FlashQwen35SearchAgent(
        api_key=get_api_key('deepseek'),
        base_url=args.model_url,
        model_name=args.model_name,
        max_turns=args.max_turns,
        browser_pool=browser_pool,
        args=args,
    )

    max_retry = 5
    attempt = 0
    t0 = time.time()

    while attempt < max_retry:
        attempt += 1
        try:
            search_answer = await search_agent.search_loop(qid, question)
            return {
                "latency_s": time.time() - t0,
                **search_answer,
            }
        except Exception as e:
            traceback.print_exc()
            error_msg = traceback.format_exc()
            print(f"[qid {qid} attempt {attempt}/{max_retry}] failed: {e}")
            return {
                "latency_s": 0.0,
                "error": error_msg,
                "attempts": attempt,
                "status": "fail",
            }


async def process_item(args, item_data: dict, data_save_path: str):
    if "error" in item_data:
        return

    qid = item_data["qid"]
    question = item_data["question"]
    prediction = item_data["response"]
    prompt = verification_prompt(args, question, prediction)

    result = await worker_entry(args, qid, prompt)
    result['answer'] = item_data.get('answer', '')
    result['qid'] = qid
    result['id'] = qid
    result['question_id'] = item_data.get('question_id')
    if 'label' in item_data:
        result['label'] = item_data['label']
    result["question"] = question
    result["response"] = prediction

    lock_path = data_save_path.replace(".jsonl", ".lock")
    lock = FileLock(lock_path)
    with lock:
        with open(data_save_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def process_item_worker(args_and_item):
    t0 = time.perf_counter()
    args, item, data_save_path = args_and_item
    setup_logging()
    res = asyncio.run(process_item(args, item, data_save_path))
    t1 = time.perf_counter()
    print(f"[PID={os.getpid()} qid={item['qid']}] total worker time={t1-t0:.4f}s", flush=True)
    return res


def main():
    args = parse_agent_args('verify')
    args.harness = os.path.basename(__file__)

    setup_logging()
    logger.info(args)

    os.makedirs(args.output_dir, exist_ok=True)
    source_data = prepare_run(args)
    data_save_path = os.path.join(args.output_dir, "data_stream.jsonl")

    processed = completed_ids(data_save_path)
    data = [item for item in source_data if str(item['qid']) not in processed]
    print(f'Remaining tasks: {len(data)}')
    if not data:
        return

    if args.debug:
        asyncio.run(process_item(args, data[0], data_save_path))
    else:
        tasks = [(args, item, data_save_path) for item in data]
        with mp.Pool(processes=args.num_workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(process_item_worker, tasks, chunksize=1),
                total=len(tasks),
                desc="Processing data",
            ):
                pass

    print("All workers finished. Script done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
