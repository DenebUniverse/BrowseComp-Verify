# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
# -*- coding: utf-8 -*-
import copy
import functools
import argparse
import os
import json
import time
import math
import traceback
import logging
from typing import List, Dict, Any
import asyncio
import datetime
import uuid
import requests
import tiktoken
from openai import OpenAI
from openai_harmony import (
    Message, Conversation, SystemContent, Role, ReasoningEffort,
    load_harmony_encoding, HarmonyEncodingName
)
import multiprocessing as mp
from filelock import FileLock
import tqdm
from openai_harmony import Role, Message
from multiprocessing import Pool

import sys
from browsecomp_verify.browser import BrowserTool, LocalServiceBrowserBackend
import logging


from browsecomp_verify.gpt_v5_parser import GPT_OSS_TOOLS, format_openai_system_prompt, DEVELOPER_CONTENT
from browsecomp_verify.prompts import VERIFY_PROMPT_strict
from browsecomp_verify.utils import load_bcplus_data

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


class Claude_V4_6_SearchAgent():
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
            "all_search_res": [],
            "token_usage_per_turn": [],
            "token_usage_total": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            },
        }
        self.api_key = self.api_key if self.api_key else os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.init = False
        
        t0 = time.perf_counter()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = getattr(args, "reasoning_effort", "high")
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.stop_tokens = [self.encoding.decode([stop_token_id]) for stop_token_id in self.stop_token_ids]
        t1 = time.perf_counter()
        logger.info("init encoding+stop_tokens took %.4fs", t1 - t0)

    async def chat_completion(self, messages, tool_choice = "auto"):
        max_retries = 32
        # max_retries = 16
        
        for retry in range(max_retries):
            try:
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 1.0,
                    # "reasoning": {"enabled": True}
                    "extra_body": {"reasoning": {"enabled": True}},
                }
                if tool_choice in ["required", "auto"]:
                    payload["tools"] = GPT_OSS_TOOLS
                    payload["tool_choice"] = tool_choice
                else:
                    pass
                
                # response = self.client.chat.completions.create(
                #     model="openai/gpt-5.2",
                #     messages=messages,
                #     extra_body=payload
                # )
                response = await asyncio.to_thread(self.client.chat.completions.create, **payload)
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
            {
                "role": "user",
                "content": f"{DEVELOPER_CONTENT}\n\nQuestion: {question}",
            },
        ]
        
        turn = 0
        prompt, response = None, None
        while True:
            turn += 1
            logger.debug("# Turn: %s", turn)
            
            self.info["turn_num"]=turn
            
            # step1 chat_completion
            t1 = time.perf_counter()
            force_final = turn >= self.max_turns
            if force_final:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool turns. "
                        "Do not call any more tools. Based only on the evidence gathered so far, "
                        "produce the final verify answer in the required format, including verdict and confidence."
                    ),
                })
            tool_choice = False if force_final else "auto"
            response = await self.chat_completion(messages, tool_choice=tool_choice)
            
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
            if isinstance(response,dict):
                logger.debug("[qid=%s turn=%s] error response_type_dict response=%s", qid, turn, response)
                continue
            
            response.choices[0].message
            res_json = json.loads(response.model_dump_json()) if not isinstance(response, dict) else response
            t2 = time.perf_counter()
            self.info["all_responses"].append(res_json)
            logger.debug("[qid=%s turn=%s] chat_completion=%.4fs response_type=%s", qid, turn, t2 - t1, type(response))
            
            # step2 tool parse
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            reasoning_details = getattr(response.choices[0].message, "reasoning_details", None)
            
            if not response.choices[0].message.tool_calls: # TODO
                msg={
                    "role": "assistant",
                    "content": content,
                }
                if reasoning_details:
                    msg["reasoning_details"]= reasoning_details
                messages.append(msg)
                logger.debug("[qid=%s turn=%s] response=%s finish_reason=%s tool_calls(%d)=%s", qid, turn, response, finish_reason, 0, [])
                self._record_usage_tokens(qid, turn, response)
                break
            
            tool_calls = response.choices[0].message.tool_calls
            tool_calls_dict = [json.loads(tool_call.model_dump_json()) for tool_call in tool_calls]
            content_dict = {"content": content if content else ""}
            msg={
                    "role": "assistant",
                    **content_dict,
                    "tool_calls": tool_calls_dict,
                }
            if reasoning_details:
                msg["reasoning_details"]= reasoning_details
            messages.append(msg)
            
            logger.debug("[qid=%s turn=%s] response=%s finish_reason=%s tool_calls(%d)=%s", qid, turn, response, finish_reason, len(tool_calls_dict), tool_calls_dict)
            self._record_usage_tokens(qid, turn, response)
            
            if tool_calls[0].function.name == "finish": # TODO
                break
            
            # step3: tool calls
            t3 = time.perf_counter()
            temp_observations = []
            for tool_call in tool_calls:
                function = tool_call.function
                tool_name = function.name
                function_arguments = function.arguments
                if tool_name == "open":
                    try:
                        args_obj = json.loads(function_arguments)
                        if isinstance(args_obj.get("id"), str) and args_obj["id"].isdigit():
                            args_obj["id"] = int(args_obj["id"])
                            function_arguments = json.dumps(args_obj, ensure_ascii=False)
                    except Exception:
                        pass
                recipient = f"browser.{tool_name}"
                
                tool_call_msg_dict = {
                    "role": Role.ASSISTANT,
                    "name": tool_name,
                    "content": [
                        {
                            "type": "text",
                            "text": function_arguments,
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
                    "tool_call_id": tool_call.id,
                    "content": result_msgs_dict[0]["content"][0]["text"]
                }
                messages.append(tool_message)
                temp_observations.append(tool_message['content'])
                
            t4 = time.perf_counter()
            logger.debug("[qid=%s turn=%s] func_call=%.4fs observations(%d)=%s", qid, turn, t4-t3, len(temp_observations) ,temp_observations)
        
        return {
            "messages": messages,
            "end_flag": True,
            "is_success": True,
            "info": self.info,
        }
    
    async def wrinting(self, qid: Any, messages: str, last_message=None, search_model = "gptoss"):
        writing_messages = copy.deepcopy(messages)
        if writing_messages and writing_messages[0].get("role") == "system":
            writing_messages.pop(0)
        while writing_messages and writing_messages[-1].get("role") == "assistant":
            writing_messages.pop()
        try: 
            if search_model == "gptoss":
                writing_messages = self.format_gptoss_messages(writing_messages)
            elif search_model == "miro":
                writing_messages = self.format_miro_messages(writing_messages)
            else:
                raise ValueError(f"NotImplemented search_model {search_model}")
        except Exception as e:
            import traceback
            return {
                "messages": writing_messages,
                "search_messages": messages,
                "writing_info": {
                    "end_flag": False,
                    "is_success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "error_type": type(e).__name__,
                }
            }
        if last_message is not None:
            writing_messages.append(last_message)
        turn = sum([1 for msg in writing_messages if msg.get("role")=='assistant'])
        print(f"messages length {len(messages)} -> {len(writing_messages)} Turn {turn}")
        
        # writing
        response = None
        t1 = time.perf_counter()
        response = await self.chat_completion(writing_messages, tool_choice=False)
        # time.sleep(120)
        res_json = json.loads(response.model_dump_json()) if response else response
        t2 = time.perf_counter()
        turn+=1
        logger.debug("[qid=%s turn=%s] chat_completion=%.4fs response_type=%s", qid, turn, t2 - t1, type(response))
        
        if response is None:
            return {
                "messages": writing_messages,
                "search_messages": messages,
                "writing_info": {
                    "end_flag": False,
                    "is_success": False,
                    "response": res_json,
                    "error": "response is None",
                }
            }
        
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content
        
        if response.choices[0].message.tool_calls:
            print(f"[Warning] wrinting response w tool_calls: {response}")
        if not content:
            print(f"[Warning] wrinting response w/o content: {content}")

        writing_messages.append({
            "role": "assistant",
            "content": content if content else ""
        })
        
        logger.debug("[qid=%s turn=%s] response=%s finish_reason=%s", qid, turn, content, finish_reason)
        
        return {"messages": writing_messages, 
                "search_messages": messages,
                "writing_info": {
                    "end_flag": True,
                    "is_success": True,
                    "response": res_json}
                }
    
    def format_gptoss_messages(self, messages):
        clean_messages = []
        for msg in messages:
            if msg["role"] in ['user', 'developer']:
                assert isinstance(msg['content'],List) and len(msg['content'])==1 and 'text' in msg['content'][0]
                clean_msg = {"role": msg["role"], "content": msg['content'][0]['text']}
                clean_messages.append(clean_msg)
            elif msg["role"] in ['assistant']:
                # --- gpt-oss tool invocation -> standard chat tool_calls ---
                recipient = msg.get("recipient")
                if isinstance(recipient, str) and recipient.startswith("browser."):
                    tool_name = recipient.split(".", 1)[1]  # browser.search -> search

                    assert isinstance(msg.get("content"), List) and len(msg["content"]) == 1 and "text" in msg["content"][0]
                    raw_args = (msg["content"][0]["text"] or "").strip()

                    # arguments should be a JSON string in OpenAI tool_calls
                    try:
                        args_obj = json.loads(raw_args)
                        args_str = json.dumps(args_obj, ensure_ascii=False)
                    except Exception:
                        args_str = raw_args  # fallback: keep raw text

                    call_id = f"call_{uuid.uuid4().hex[:24]}"

                    clean_msg = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": args_str,
                                },
                            }
                        ],
                    }
                    # Merge consecutive assistant:
                    # If previous is plain assistant text, merge it into this tool_call message content.
                    while clean_messages and clean_messages[-1].get("role") == "assistant":
                        last = clean_messages[-1]
                        assert last.get("role") == "assistant" and "tool_calls" not in last, last
                        if last.get("role") == "assistant" and "tool_calls" not in last:
                            prev_text = last.get("content", "")
                            if prev_text:
                                clean_msg["content"] = prev_text + clean_msg["content"]
                            clean_messages.pop()
                    
                    assert clean_messages[-1].get("role") != "assistant", last
                    clean_messages.append(clean_msg)
                else:
                    assert isinstance(msg.get("content"), List) and len(msg["content"]) == 1 and "text" in msg["content"][0]
                    if not msg["content"][0]["text"]:
                        continue
                    clean_msg = {"role": "assistant", "content": msg["content"][0]["text"]}
                    clean_messages.append(clean_msg)
            elif msg["role"] in ['tool']:
                if "assistant<|channel|>" in msg['name']:
                    if clean_messages:
                        last = clean_messages[-1]
                        if last.get("role") == "assistant":
                            clean_messages.pop()
                    continue
                assert isinstance(msg['content'],List) and len(msg['content'])==1 and 'text' in msg['content'][0]
                last = clean_messages[-1]
                assert last.get("role") == "assistant" and "tool_calls" in last and last['tool_calls'] and last['tool_calls'][0].get('id') is not None, last
                tool_call_id = last['tool_calls'][0].get('id')
                clean_msg = {"role": msg["role"], "content": msg['content'][0]['text'], "tool_call_id": tool_call_id}
                clean_messages.append(clean_msg)
            else:
                raise ValueError(f'unsupported role value {msg["role"]}')
        
        return clean_messages
        
    def format_miro_messages(self, messages):
        clean_messages = []
        for msg in messages:
            if msg["role"] in ['user']:
                clean_messages.append(msg)
            elif msg["role"] in ['assistant']:
                chat_msg = {"role": msg["role"], "content": ""}
                if "reasoning_content" in msg:
                    chat_msg["content"]+=f'<think>\n{msg["reasoning_content"]}\n</think>'
                if "content" in msg:
                    chat_msg["content"]+=f'\n\n{msg["content"]}'
                    chat_msg["content"] = chat_msg["content"].strip()
                if "tool_calls" in msg:
                    chat_msg["tool_calls"]=[]
                    assert len(msg["tool_calls"])==1, msg
                    for tool_call in msg["tool_calls"]:
                        call_id = f"call_{uuid.uuid4().hex[:24]}"
                        chat_msg["tool_calls"].append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": tool_call
                            }
                        )
                assert "reasoning_content" not in chat_msg
                clean_messages.append(chat_msg)
            elif msg["role"] in ['tool']:
                last = clean_messages[-1]
                assert last.get("role") == "assistant" and "tool_calls" in last and last['tool_calls'] and last['tool_calls'][0].get('id') is not None, last
                tool_call_id = last['tool_calls'][0].get('id')
                chat_msg = {"role": msg["role"], "content": msg['content'][0]['output'], "tool_call_id": tool_call_id}
                clean_messages.append(chat_msg)
            else:
                raise ValueError(f'unsupported role value {msg["role"]}')
        
        return clean_messages
        
    def _extract_usage_tokens(self, response):
        """
        Return 5-token fields:
        prompt_tokens, completion_tokens, total_tokens, reasoning_tokens, cached_tokens
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            }

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
        cached_tokens = cached_tokens or 0

        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
        reasoning_tokens = reasoning_tokens or 0

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        }

    def _record_usage_tokens(self, qid, turn, response):
        usage_tokens = self._extract_usage_tokens(response)

        # per-turn
        self.info["token_usage_per_turn"].append({
            "qid": str(qid),
            "turn": turn,
            **usage_tokens,
        })

        # totals
        tot = self.info["token_usage_total"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens"):
            tot[k] += usage_tokens[k]

        # log
        logger.debug("[qid=%s turn=%s] tokens prompt=%d completion=%d total=%d reasoning=%d cached=%d", qid, turn,usage_tokens["prompt_tokens"], usage_tokens["completion_tokens"], usage_tokens["total_tokens"], usage_tokens["reasoning_tokens"], usage_tokens["cached_tokens"])
        return usage_tokens


async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)
    t0 = time.time()

    search_agent = Claude_V4_6_SearchAgent(
        api_key=get_api_key("claude"),
        base_url=args.model_url,
        model_name=args.model_name,
        max_turns=args.max_turns,
        browser_pool=browser_pool,
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
