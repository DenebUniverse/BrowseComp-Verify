# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
# -*- coding: utf-8 -*-
import functools

import json5
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
from openai_harmony import Role, Message
from qwen_agent.llm.schema import Message as Qwen_Message
from transformers import AutoTokenizer
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
enc = tiktoken.get_encoding("o200k_base")

from browsecomp_verify.prompts import VERIFY_PROMPT, VERIFY_PROMPT_strict

from browsecomp_verify.browser import BrowserTool, LocalServiceBrowserBackend
from browsecomp_verify.prompts import TONGYI_SYSTEM_PROMPT_SEARCH_OPEN_ONLY, today_date
from browsecomp_verify.utils import load_bcplus_data, DEVELOPER_CONTENT
MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))
SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
DEFAULT_SYSTEM_MESSAGE = ''


def get_tongyi_system_prompt_with_visit() -> str:
    return TONGYI_SYSTEM_PROMPT_SEARCH_OPEN_ONLY + str(today_date())


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


class Tongyi_SearchAgent_verify:
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
            "termination": None,
            "all_prompts": [],
            "all_responses": [],
            "all_search_res": [],
        }
        self.init = False
        self.browser_pool = browser_pool
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.stop_tokens = ["\n<tool_response>", "<tool_response>"]
        self.search_tool = None

    async def chat_completion(self, messages):
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
                    "messages": messages,
                    "temperature": 1.0,
                    "max_tokens": 16384,
                    "skip_special_tokens": False,
                    "thinking": {"type": "enabled"},
                    "stop": self.stop_tokens,
                }

                if "openrouter" in self.base_url:
                    payload["provider"] = {
                        "only": ["deepinfra/fp4"],
                        "allow_fallbacks": False
                    }

                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        requests.post,
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
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
                    "prompt": prompt,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 10000,
                    "skip_special_tokens": False,
                    "presence_penalty": 1.1,
                    "stop": self.stop_tokens,
                }

                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        requests.post,
                        f"{self.base_url}/completions",
                        headers=headers,
                        json=payload,
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

    def custom_format_tool_call(self, tool_name, tool_args):
        tool_calls = []
        if tool_name.lower().strip() == 'search':
            queries = tool_args['query']
            for query in queries:
                tool_call_msg_dict = {
                    'role': Role.ASSISTANT, 'name': None,
                    'content': [
                        {'type': 'text',
                         'text': json.dumps({"query": query, "topn": 10}, ensure_ascii=False, indent=1)}
                    ],
                    'channel': 'analysis',
                    'recipient': 'browser.search',
                    'content_type': 'code'}
                tool_calls.append(tool_call_msg_dict)
        elif tool_name.lower().strip() == 'visit':
            urls = tool_args.get('url')
            if isinstance(urls, str):
                urls = [urls]
            for url in urls:
                tool_call_msg_dict = {
                    'role': Role.ASSISTANT, 'name': None,
                    'content': [
                        {'type': 'text',
                         'text': json.dumps({"id": url}, ensure_ascii=False, indent=1)}
                    ],
                    'channel': 'analysis',
                    'recipient': 'browser.open',
                    'content_type': 'code'}
                tool_calls.append(tool_call_msg_dict)

        return tool_calls

    def custom_format_tool_response(self, tool_name, tool_responses):
        if tool_name.lower().strip() == 'search':
            responses = []
            for tool_response in tool_responses:
                responses.append(tool_response[0]['content'][0]['text'])
            return "\n=======\n".join(responses)
        if tool_name.lower().strip() == 'visit':
            responses = []
            for tool_response in tool_responses:
                responses.append(tool_response[0]['content'][0]['text'])
            return "\n=======\n".join(responses)
        return f"Error: Tool {tool_name} not found"

    async def custom_call_tool(self, qid, tool_name: str, tool_args: dict, **kwargs):
        tool_calls = self.custom_format_tool_call(tool_name, tool_args)

        result_msgs = []
        for tool_call_msg_dict in tool_calls:
            result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
            result_msgs.append(result_msgs_dict)

        if result_msgs:
            return self.custom_format_tool_response(tool_name, result_msgs)
        else:
            return f"Error: Tool {tool_name} not found"

    def count_tokens(self, messages, model="gpt-4o"):
        full_message = [Qwen_Message(**x) for x in messages]
        full_prompt = self.build_text_completion_prompt(full_message, allow_special=True)
        return len(self.tokenizer.encode(full_prompt))

    def build_text_completion_prompt(
        self,
        messages: List[Qwen_Message],
        allow_special: bool = False,
        default_system: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> str:
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'

        if messages and messages[0].role == SYSTEM:
            sys = messages[0].content
            assert isinstance(sys, str)
            prompt = f'{im_start}{SYSTEM}\n{sys}{im_end}'
            messages = messages[1:]
        elif default_system:
            prompt = f'{im_start}{SYSTEM}\n{default_system}{im_end}'
        else:
            prompt = ''

        if messages[-1].role != ASSISTANT:
            messages = messages + [Qwen_Message(ASSISTANT, '')]

        for msg in messages:
            assert isinstance(msg.content, str)
            content = msg.content
            if allow_special:
                assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)
                if msg.function_call:
                    assert msg.role == ASSISTANT
                    tool_call = msg.function_call.arguments
                    try:
                        tool_call = {'name': msg.function_call.name, 'arguments': json.loads(tool_call)}
                        tool_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
                    except json.decoder.JSONDecodeError:
                        tool_call = '{"name": "' + msg.function_call.name + '", "arguments": ' + tool_call + '}'
                    if content:
                        content += '\n'
                    content += f'<tool_call>\n{tool_call}\n</tool_call>'
            else:
                assert msg.role in (USER, ASSISTANT)
                assert msg.function_call is None
            if prompt:
                prompt += '\n'
            prompt += f'{im_start}{msg.role}\n{content}{im_end}'

        assert prompt.endswith(im_end)
        prompt = prompt[:-len(im_end)]
        return prompt

    async def search_loop(self, qid: Any, question: str):
        tool_config = self.browser_pool.init_session(qid)
        start_time = time.time()
        system_prompt = get_tongyi_system_prompt_with_visit()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = self.max_turns
        turn = 0
        prompt, response = None, None
        try:
            while num_llm_calls_available > 0:
                if time.time() - start_time > 150 * 60:
                    termination = 'No answer found after 2h30mins'
                    self.info['termination']=termination
                    result = {
                        "messages": messages,
                        "end_flag": False,
                        "error": termination,
                        "info": self.info,
                    }
                    return result
                turn += 1
                if turn > self.max_turns:
                    return {"messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in messages],
                            "end_flag": False, "error": "max_turns exceeded", "info": self.info}
                num_llm_calls_available -= 1
                self.info["turn_num"] = turn
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                prompt += "<|im_start|>assistant\n<think>\n"

                t0 = time.perf_counter()
                response = await self.completion(prompt)
                content = response['choices'][0]['text']
                content = "<think>\n" + content
                t1 = time.perf_counter()
                logger.info("[qid=%s turn=%s] call_model=%.4fs", qid, turn, t1 - t0)
                self.info['all_prompts'].append(prompt)
                self.info['all_responses'].append(content)
                
                if '<tool_response>' in content:
                    pos = content.find('<tool_response>')
                    content = content[:pos]
                messages.append({"role": "assistant", "content": content})
                if '<tool_call>' in content and '</tool_call>' in content:
                    tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    try:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = await self.custom_call_tool(qid, tool_name, tool_args)

                    except Exception:
                        import traceback
                        traceback.print_exc()
                        result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                    result = "<tool_response>\n" + result + "\n</tool_response>"

                    messages.append({"role": "user", "content": result})
                if '<answer>' in content and '</answer>' in content:
                    termination = 'answer'
                    self.info['termination']=termination
                    break
                if num_llm_calls_available <= 0 and '<answer>' not in content:
                    messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'
                t2 = time.perf_counter()
                logger.info("[qid=%s turn=%s] call_tool=%.4fs", qid, turn, t2 - t1)

                max_tokens = 110 * 1024
                token_count = self.count_tokens(messages)
                t3 = time.perf_counter()

                logger.info("[qid=%s turn=%s] count_tokens=%.4fs token_count=%d", qid, turn, t3 - t2, token_count)

                if token_count > max_tokens:
                    print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                    messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                    response = await self.completion(prompt)
                    content = response['choices'][0]['text']
                    messages.append({"role": "assistant", "content": content.strip()})
                    if '<answer>' in content and '</answer>' in content:
                        prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                        termination = 'generate an answer as token limit reached'
                    else:
                        prediction = messages[-1]['content']
                        termination = 'format error: generate an answer as token limit reached'
                    self.info['termination']=termination
                    result = {
                        "messages": messages,
                        "end_flag": True,
                        "info": self.info,
                    }
                    return result
                
                if messages[-1]['content'] == 'Sorry, the number of llm calls exceeds the limit.':
                    result = {
                        "messages": messages,
                        "end_flag": False,
                        "info": self.info,
                    }
                    return result
                
            result = {
                "messages": messages,
                "end_flag": True,
                "info": self.info,
            }
            return result
        finally:
            self.browser_pool.cleanup(qid)


async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)

    search_agent = Tongyi_SearchAgent_verify(
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
        logger.info("[qid=%s attempt=%s] start", qid, attempt)
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
    res = asyncio.run(process_item(args, item, data_save_path))
    t1 = time.perf_counter()
    print(f"[PID={os.getpid()} qid={item['qid']}] total worker time={t1-t0:.4f}s", flush=True)
    return res


def main():
    args = parse_agent_args('verify')
    args.harness = os.path.basename(__file__)
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    data = prepare_run(args)

    data_save_path = os.path.join(args.output_dir, "data_stream.jsonl")

    processed = completed_ids(data_save_path)
    data = [item for item in data if str(item['qid']) not in processed]
    print(f'Remaining tasks: {len(data)}')
    if not data:
        return

    if args.debug:
        asyncio.run(process_item(args, data[0], data_save_path))
    else:
        tasks = [(args, item, data_save_path) for item in data]
        logger.info(f"num_workers = {args.num_workers}, Go go go")
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