# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
# -*- coding: utf-8 -*-
import functools

import requests
import argparse
import os, json, time, argparse, math, traceback
from typing import List, Dict, Any
import multiprocessing as mp
import asyncio
import datetime
import uuid
from filelock import FileLock
import tiktoken
import tqdm
from openai_harmony import (
    Message, Conversation, SystemContent, Role, ReasoningEffort,
    StreamableParser, load_harmony_encoding, HarmonyEncodingName
)
from multiprocessing import Pool

from browsecomp_verify.browser import BrowserTool,LocalServiceBrowserBackend
from browsecomp_verify.utils import load_bcplus_data, DEVELOPER_CONTENT
from browsecomp_verify.prompts import VERIFY_PROMPT, VERIFY_PROMPT_strict

enc = tiktoken.get_encoding("o200k_base")

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

class GPT_OSS_SearchAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_turns: int = 100,
        browser_pool: BrowserPool = None,
        args = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_turns = max_turns
        self.messages = []

        self.info = {
            "all_prompts": [],
            "all_responses": [],
            "all_search_res": [],
        }
        self.init = False
        self.browser_pool = browser_pool
        t0 = time.perf_counter()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = getattr(args, "reasoning_effort", None)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.stop_tokens = [self.encoding.decode([stop_token_id]) for stop_token_id in self.stop_token_ids]
        t1 = time.perf_counter()
        print(f"[PID={os.getpid()}] init encoding+stop_tokens took {t1-t0:.4f}s", flush=True)

    async def chat_completion(self, messages):
        max_retries = 32
        # max_retries = 16
        loop = asyncio.get_running_loop()

        for retry in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                }
                if self.api_key:
                    headers['Authorization'] = f"Bearer {self.api_key}"
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 1.0,
                    # "top_p": 0.95,
                    "max_tokens": 16384,
                    # "max_tokens": 8192,
                    "skip_special_tokens": False,
                    # "tools": BROWSE_TOOLS,
                    "thinking": {"type": "enabled"},
                    "stop": self.stop_tokens,
                }

                if "openrouter" in self.base_url:
                    payload["provider"] = {
                        "only": ["deepinfra/fp4"],
                        "allow_fallbacks": False
                    }

                # print("model_name: ", self.model_name)
                # print(messages)

                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        requests.post,
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        # timeout=180,
                        # timeout=360,
                        timeout=1800,
                    ),
                )

                resp_json = response.json()

                if (
                    response.status_code != 200
                    or resp_json is None
                    or "error" in resp_json
                    or "code" in resp_json
                ):
                    # print("response.status_code: ", response.status_code)
                    # print("response.text: ", response.text)
                    # print("resp_json:", resp_json)
                    print(f"Retry {retry + 1}/{max_retries} failed (status={response.status_code})", flush=True)
                    sleep_s = 1 + 0.5 * retry

                    if "maximum context" in str(resp_json):
                        return {
                            "error": "maximum context",
                        }

                    if retry == max_retries - 1:
                        return {
                            "error": f"All {max_retries} retries failed, last error: {resp_json}",
                        }

                    print(f"sleeping for {sleep_s} seconds")
                    await asyncio.sleep(sleep_s)
                    continue

                return resp_json

            except Exception as e:
                if retry < max_retries - 1:
                    import traceback
                    traceback.print_exc()
                    print(f"Retry {retry + 1}/{max_retries} failed: {e}")
                    sleep_s = 1 + 0.5 * retry
                    print(f"sleeping for {sleep_s} seconds")
                    print(response.text)
                    await asyncio.sleep(sleep_s)
                else:
                    print(f"All {max_retries} retries failed, last error: {e}")
                    error_msg = traceback.format_exc()
                    return {
                        "error": f"All {max_retries} retries failed, last error: {error_msg}",
                    }

        return {
            "error": "Request retries exhausted",
        }
        
        
    async def completion(self, prompt):
        max_retries = 32
        loop = asyncio.get_running_loop()

        for retry in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                }
                if self.api_key:
                    headers['Authorization'] = f"Bearer {self.api_key}"
                
                payload = {
                    "model": self.model_name,
                    # "messages": messages,
                    "prompt": prompt,
                    "temperature": 1.0,
                    # "top_p": 0.95,
                    "max_tokens": 16384,
                    # "max_tokens": 8192,
                    "skip_special_tokens": False,
                    "thinking": {"type": "enabled"},
                    "stop": self.stop_tokens,
                    # "tools": BROWSE_TOOLS,
                }


                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        requests.post,
                        f"{self.base_url}/completions",
                        headers=headers,
                        json=payload,
                        # timeout=180,
                        # timeout=360,
                        timeout=1800,
                    ),
                )

                resp_json = response.json()

                if (
                    response.status_code != 200
                    or resp_json is None
                    or "error" in resp_json
                    or "code" in resp_json
                ):
                    # print("response.status_code: ", response.status_code)
                    # print("resp_json:", resp_json)
                    print(f"Retry {retry + 1}/{max_retries} failed (status={response.status_code})", flush=True)
                    sleep_s = 1 + 0.5 * retry

                    if "maximum context" in str(resp_json):
                        return {
                            "error": "maximum context",
                        }

                    if retry == max_retries - 1:
                        return {
                            "error": f"All {max_retries} retries failed, last error: {resp_json}",
                        }

                    print(f"sleeping for {sleep_s} seconds")
                    await asyncio.sleep(sleep_s)
                    continue

                return resp_json

            except Exception as e:
                if retry < max_retries - 1:
                    import traceback
                    traceback.print_exc()
                    print(f"Retry {retry + 1}/{max_retries} failed: {e}")
                    sleep_s = 1 + 0.5 * retry
                    print(f"sleeping for {sleep_s} seconds")
                    print(traceback.format_exc())
                    await asyncio.sleep(sleep_s)
                else:
                    print(f"All {max_retries} retries failed, last error: {e}")
                    error_msg = traceback.format_exc()
                    return {
                        "error": f"All {max_retries} retries failed, last error: {error_msg}",
                    }

        return {
            "error": "Request retries exhausted",
        }
        
    async def search_loop(self, qid: Any, question: str):
        tool_config = self.browser_pool.init_session(qid)

        system_message_content = (
            SystemContent.new()
            .with_reasoning_effort({
                    "high": ReasoningEffort.HIGH,
                    "medium": ReasoningEffort.MEDIUM,
                    "low": ReasoningEffort.LOW,
                }[self.reasoning_effort])
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
            .with_tools(tool_config)
        )
        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_message_content),
            Message.from_role_and_content(Role.DEVELOPER, DEVELOPER_CONTENT),
            Message.from_role_and_content(Role.USER, "Question: " + question),
        ]
        turn = 0
        prompt, response = None, None
        try:
            while True:
                last_message = messages[-1]

                if getattr(last_message, "recipient", None) and str(last_message.recipient).startswith("browser."):
                    tool_call_msg_dict = last_message.to_dict()
                    t_browser_0 = time.perf_counter()
                    result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
                    t_browser_1 = time.perf_counter()
                    print(f"[qid={qid} turn={turn}] browser={t_browser_1-t_browser_0:.4f}s", flush=True)
                    result_msgs = [Message.from_dict(m) for m in result_msgs_dict]
                    messages += result_msgs
                    continue
                
                turn += 1
                if turn > self.max_turns:
                    return {"messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in messages],
                            "end_flag": False, "error": "max_turns exceeded", "info": self.info}
                print("# Turn: ", turn, flush=True)
                conversation = Conversation.from_messages(messages)

                if last_message.author.role == Role.ASSISTANT and getattr(last_message, "channel", None) == "final":
                    break
                elif last_message.author.role == Role.ASSISTANT and getattr(last_message, "channel", None) in ['analysis','analysis code']:
                    pass
                elif last_message.author.role == Role.ASSISTANT:
                    channel = getattr(last_message, "channel", None)
                    if prompt:
                        self.info["all_prompts"].append(prompt)
                    if response:
                        self.info["all_responses"].append(response)
                    print(f"[Warning] ASSISTANT channel {channel}. no tool call and not finished message: ", str(last_message)[-50:], flush=True)
                    # return {
                    #     "messages": [m.to_dict() for m in conversation.messages],
                    #     "end_flag": False,
                    #     "error": "No function calls content",
                    #     "info": self.info,
                    # }
                t0 = time.perf_counter()
                tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                t1 = time.perf_counter()
                prompt = self.encoding.decode(tokens)
                t2 = time.perf_counter()
                
                print(f"[qid={qid} turn={turn}] render={t1-t0:.4f}s decode={t2-t1:.4f}s len_tokens={len(tokens)}", flush=True)
                
                self.info["turn_num"]=turn
                # self.info["all_prompts"].append(prompt)
                response = await self.completion(prompt)
                t3 = time.perf_counter()
                # self.info["all_responses"].append(response)
                print(f"[qid={qid} turn={turn}] http+server={t3-t2:.4f}s", flush=True)
                
                if 'error' in response:
                    return {
                        "messages": [m.to_dict() for m in conversation.messages],
                        "end_flag": False,
                        "error": response['error'],
                        "info": self.info,
                    }
                response_text = response.get('choices', [{}])[0].get('text', '')
                if not response_text:
                    return {
                        "messages": [m.to_dict() for m in conversation.messages],
                        "end_flag": False,
                        "error": "Empty response content",
                        "info": self.info,
                    }
                
                
                completion_tokens = self.encoding.encode(response_text, allowed_special="all")
                t4 = time.perf_counter()
                new_messages = self.encoding.parse_messages_from_completion_tokens(completion_tokens, role=Role.ASSISTANT)
                t5 = time.perf_counter()
                print(f"[qid={qid} turn={turn}] new_messages encode={t4-t3:.4f}s decode={t5-t4:.4f}s len_tokens={len(tokens)}", flush=True)
                
                messages += new_messages
                if messages and messages[-1].author.role == Role.ASSISTANT and getattr(messages[-1], "channel", None) == "final":
                    break

            conv = Conversation.from_messages(messages)
            return {
                "messages": [m.to_dict() for m in conv.messages],
                "end_flag": True,
                "info": self.info,
            }
        finally:
            self.browser_pool.cleanup(qid)


async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)
    t0 = time.time()

    search_agent = GPT_OSS_SearchAgent(
        api_key=get_api_key(None),
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
    prompt = verification_prompt(args, question, prediction)
    # prompt = verification_prompt(args, question, prediction)
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
    res = asyncio.run(process_item(args, item, data_save_path))
    t1 = time.perf_counter()
    print(f"[PID={os.getpid()} qid={item['qid']}] total worker time={t1-t0:.4f}s", flush=True)
    return res


def main():
    args = parse_agent_args('verify')
    args.harness = os.path.basename(__file__)
    print(args)

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

        with Pool(processes=args.num_workers) as pool:
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
