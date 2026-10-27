# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
# -*- coding: utf-8 -*-
import argparse
import asyncio
import os
import json
import random
import sys
import time
import traceback
from typing import List, Dict, Any
from filelock import FileLock
from openai import OpenAI
import requests
import logging
import multiprocessing as mp
from openai_harmony import Message, Role
import tiktoken
import tqdm
from browsecomp_verify.prompts import VERIFY_PROMPT_strict
from browsecomp_verify.utils import load_bcplus_data

from browsecomp_verify.browser import BrowserTool, LocalServiceBrowserBackend

from browsecomp_verify.glm_v4_7_parser import TOOLS

enc = tiktoken.get_encoding("o200k_base")

logger = logging.getLogger(__name__)

def setup_logging(debug: bool = True):
    level = logging.DEBUG if debug else logging.INFO

    # 如果已配置过 handler，避免重复 basicConfig 导致重复输出
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

    # 可选：压掉一些第三方库的噪音
    for noisy in ("urllib3", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

class BrowserPool:
    def __init__(self, search_url):
        self.search_url = search_url
        self.sessions: Dict[Any, BrowserTool] = {}

    def init_session(self, qid: Any) -> dict:
        tool = BrowserTool(backend=LocalServiceBrowserBackend(base_url=self.search_url))
        self.sessions[qid] = tool
        return tool.tool_config

    async def call(self, qid: Any, tool_call_msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool = self.sessions[qid]
        tool_call_msg = Message.from_dict(tool_call_msg_dict)
        results = []
        async for msg in tool.process(tool_call_msg):
            results.append(msg)
            # print(msg.to_dict()['content'][0][:30], flush=True)
        return [m.to_dict() for m in results]

    def cleanup(self, qid: Any):
        if qid in self.sessions:
            del self.sessions[qid]

class GLM_SearchAgent():
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_turns: int = 100,
        browser_pool: BrowserPool = None,
        args = None,
        temperature = 1,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_turns = max_turns
        self.messages = []
        self.info = {}
        self.browser_pool = browser_pool
        self.temperature = temperature
        
        self.info = {
            "all_prompts": [],
            "all_responses": [],
        }
        self.api_key = self.api_key if self.api_key else os.getenv("OPENAI_API_KEY")
        self.tools = [{"type": "function", "function": tool} for tool in TOOLS]
        # self.client = OpenAI(
        #     api_key=self.api_key,
        #     base_url=self.base_url,
        # )
        
    async def chat_completion(self, messages, tool_choice = "auto"):
        max_retries = 32
        
        for retry in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 1.0,
                    "thinking": { "type": "enabled" },
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "stream": False,
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                response = await asyncio.to_thread(
                    requests.post,
                    chat_completions_url(self.base_url),
                    json=payload,
                    headers=headers,
                    timeout=600,
                )
                if response is None:
                    print(f"Retry {retry+1}/{max_retries} failed: empty response")
                    print(json.dumps(messages, indent=2))
                    print(f"sleeping for {1 + 2 * retry} seconds")
                    time.sleep(1 + 2 * retry)
                    continue
                
                return response
            
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning("Retry %s/%s failed: %s", retry + 1, max_retries, e, exc_info=True)
                    traceback.print_exc()
                    logger.warning("messages: %s", messages)
                    sleep_s = 1 + 2 * retry
                    logger.debug("sleeping for %.1f seconds", sleep_s)
                    await asyncio.sleep(sleep_s)
                else:
                    logger.error("All %s retries failed, last error: %s", max_retries, e)
                    error_msg = traceback.format_exc()
                    return {
                        "error": f"All {max_retries} retries failed, last error: {error_msg}",
                    }

        return {
            "error": "Request retries exhausted",
        }
        

    async def search_loop(self, qid: Any, question: str):
        tool_config = self.browser_pool.init_session(qid)
        messages = [
            {"role":"user","content": "Question: " + question},
        ]

        turn = 0
        end_flag = True
        success_flag = True
        while True:
            turn += 1
            if turn > self.max_turns:
                return {"messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in messages],
                        "end_flag": False, "error": "max_turns exceeded", "info": self.info}
            logger.debug("# Turn: %s", turn)

            # step1 chat_completion
            t1 = time.perf_counter()
            response = await self.chat_completion(messages)
            response = response.json() if hasattr(response, "json") else response
            
            t2 = time.perf_counter()
            self.info["all_responses"].append(response)
            logger.debug("[qid=%s turn=%s] chat_completion=%.4fs response_type=%s", qid, turn, t2 - t1, type(response))
            
            if response is None:
                return {
                    "messages": messages,
                    "end_flag": False,
                    "is_success": False,
                    "error": "response is None",
                    "info": self.info,
                }
            elif isinstance(response, dict) and "error" in response:
                return {
                    "messages": messages,
                    "end_flag": False,
                    "is_success": False,
                    "error": response['error'],
                    "info": self.info,
                }
            
            reasoning_content = response['choices'][0]['message'].get('reasoning_content')
            content = response['choices'][0]['message']['content']
            tool_calls = response['choices'][0]['message'].get('tool_calls', None)
            msg={
                'role': 'assistant',
                'content': content,
                'reasoning_content': reasoning_content,
            }
            if tool_calls:
                msg['tool_calls'] = tool_calls
            messages.append(msg)
            
            logger.debug(f"[qid=%s turn=%s] {reasoning_content=}\n{content=}\n{tool_calls=}")
            
            # step2 tool parse
            if not tool_calls:
                break
            elif "finish" in [tool['function']['name'] for tool in tool_calls]:
                break
            
            # step3: tool calls
            t3 = time.perf_counter()
            temp_observations = []
            for tool_call in tool_calls:
                function = tool_call['function']
                tool_name = function['name']
                recipient = f"browser.{tool_name}"
                
                tool_call_msg_dict = {
                    "role": Role.ASSISTANT,
                    "name": tool_name,
                    "content": [
                        {
                            "type": "text",
                            "text": function['arguments'],
                        }
                    ],
                    "channel": "analysis",
                    "recipient": recipient,
                    "content_type": "code",
                }
                result_msgs_dict = await self.browser_pool.call(
                    qid, tool_call_msg_dict
                )
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": result_msgs_dict[0]["content"][0]["text"]
                }
                messages.append(tool_message)
                temp_observations.append(tool_message['content'])
                
            t4 = time.perf_counter()
            logger.debug("[qid=%s turn=%s] func_call=%.4fs observations(%d)=%s", qid, turn, t4-t3, len(temp_observations) ,temp_observations)

        return {
            "messages": messages,
            "end_flag": end_flag,
            "is_success": success_flag,
            "info": self.info,
        }


async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)
    t0 = time.time()

    search_agent = GLM_SearchAgent(
        api_key=get_api_key('glm'),
        base_url=args.model_url,
        model_name=args.model_name,
        browser_pool=browser_pool,
        max_turns=args.max_turns,
        args=args,
    )

    MAX_RETRY = 5
    attempt = 0
    error_msg = None
    t0 = time.time()

    while attempt < MAX_RETRY:
        attempt += 1
        try:
            search_answer = await search_agent.search_loop(
                qid,
                question,
            )
            dt = time.time() - t0
            return {
                "latency_s": dt,
                **search_answer,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = traceback.format_exc()
            print(
                f"[qid {qid} attempt {attempt}/{MAX_RETRY} failed: {e}"
            )
            return {
                "latency_s": 0.0,
                "error": error_msg,
                "attempts": attempt,
                "status": "fail",
            }


async def process_item(args, item_data: dict, data_save_path: str):
    if 'error' in item_data:
        return
    qid = item_data['qid']
    question = item_data['question']
    prediction = item_data['response']
    # prompt = verification_prompt(args, question, prediction)
    prompt = verification_prompt(args, question, prediction)
    result = await worker_entry(args, qid, prompt)
    result['answer'] = item_data.get('answer', '')
    result['qid'] = qid
    result['id'] = qid
    result['question_id'] = item_data.get('question_id')
    if 'label' in item_data:
        result['label'] = item_data['label']
    result['question'] = question
    result['response'] = prediction

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
        # 把需要的参数打包成列表，方便传给 worker
        tasks = [(args, item, data_save_path) for item in data]

        with mp.Pool(processes=args.num_workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(process_item_worker, tasks, chunksize=1),
                total=len(tasks),
                desc="Processing data"
            ):
                pass
    
    print("All workers finished. Script done.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
