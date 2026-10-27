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
import uuid
from typing import List, Dict, Any
from filelock import FileLock
from openai import OpenAI, APIConnectionError, RateLimitError
import requests
import logging
import multiprocessing as mp
from openai_harmony import Message, Role
import tiktoken
import tqdm
from browsecomp_verify.prompts import VERIFY_PROMPT_strict, EXTRACTOR_PROMPT
from browsecomp_verify.sglang_sampler import SglangSampler
from browsecomp_verify.utils import load_bcplus_data

from browsecomp_verify.browser import BrowserTool, LocalServiceBrowserBackend

from browsecomp_verify.glm_v4_7_parser import TOOLS
try:
    from qwen_agent.llm import get_chat_model
except ModuleNotFoundError:
    get_chat_model = None

ADD_SP = "Search intensity is set to high. Please conduct thorough, multi-source research and provide comprehensive, well-cited results."
MAX_CONTEXT_TOKENS = 255 * 1024
MAX_OUTPUT_TOKENS_CAP = 65500
MODEL_TIMEOUT = 3600
TOP_P = 1.0
DEFAULT_SUMMARY_URL = "127.0.0.1:9999"


QWEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Python code sandbox, which can be used to execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"description": "The python code.", "type": "string"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for information from the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of search queries.",
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_extractor",
            "description": "Crawl webpage content, and if given a goal, further summarize the relevant content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The webpage urls.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit.",
                    },
                },
                "required": ["urls", "goal"],
            },
        },
    },
]

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
    for noisy in ("urllib3", "httpx", "openai", "httpcore", "httpcore.connection", "httpcore.http11"):
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

class Qwen_3_5_SearchAgent():
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
        self.summary_url = self._normalize_summary_url(
            getattr(args, "summary_url", DEFAULT_SUMMARY_URL) if args is not None else DEFAULT_SUMMARY_URL
        )
        self.summary_sampler = SglangSampler(
            url=self.summary_url,
            model="default",
            temperature=0.7,
            max_tokens=16384,
        )

        self.info = {
            "all_prompts": [],
            "all_responses": [],
        }
        self.api_key = self.api_key if self.api_key else os.getenv("OPENAI_API_KEY")

        self.tools = QWEN_TOOLS

        model_server = self.base_url
        if model_server.endswith("/chat/completions"):
            model_server = model_server[: -len("/chat/completions")]
        model_server = model_server.rstrip("/")

        self.client = OpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=model_server,
        )

    @staticmethod
    def _normalize_summary_url(summary_url: str | None) -> str | None:
        if not summary_url:
            return None
        summary_url = summary_url.strip()
        if not summary_url:
            return None
        if "://" not in summary_url:
            summary_url = "http://" + summary_url
        summary_url = summary_url.rstrip("/")
        if summary_url.endswith("/v1/chat/completions") or summary_url.endswith("/chat/completions"):
            return summary_url
        if summary_url.endswith("/v1"):
            return summary_url + "/chat/completions"
        return summary_url + "/v1/chat/completions"

    @staticmethod
    def _parse_function_args(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"raw_input": arguments}
        return {}

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    @staticmethod
    def _extract_browser_text(result_msgs_dict: List[Dict[str, Any]]) -> str:
        if not result_msgs_dict:
            return ""
        try:
            return result_msgs_dict[0]["content"][0]["text"]
        except Exception:
            return json.dumps(result_msgs_dict, ensure_ascii=False)

    @staticmethod
    def _parse_summary_response(summary_content: str) -> Dict[str, Any]:
        if "</think>" in summary_content:
            summary_content = summary_content.split("</think>", 1)[-1]
        summary_content = summary_content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(summary_content)
        if not isinstance(parsed, dict):
            raise ValueError("summary output is not a JSON object")
        return parsed

    def _build_browser_tool_call_msg(
        self,
        recipient: str,
        arguments: Dict[str, Any],
        tool_name: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "role": Role.ASSISTANT,
            "name": tool_name,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(arguments, ensure_ascii=False),
                }
            ],
            "channel": "analysis",
            "recipient": recipient,
            "content_type": "code",
        }

    async def _call_browser_compatible_tool(
        self,
        qid: Any,
        tool_name: str,
        arguments: Any,
    ) -> str:
        tool_args = self._parse_function_args(arguments)

        if tool_name in {"search", "web_search"}:
            queries = tool_args.get("queries")
            if queries is None:
                query = tool_args.get("query") or tool_args.get("raw_input")
                queries = [query] if query else []
            elif isinstance(queries, str):
                queries = [queries]

            results = []
            for query in queries:
                if not query:
                    continue
                tool_call_msg_dict = self._build_browser_tool_call_msg(
                    recipient="browser.search",
                    arguments={"query": query, "topn": 10},
                    tool_name=tool_name,
                )
                result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
                if result_msgs_dict:
                    try:
                        results.append(result_msgs_dict[0]["content"][0]["text"])
                    except Exception:
                        results.append(json.dumps(result_msgs_dict, ensure_ascii=False))
            return "\n=======\n".join(results)

        if tool_name in {"open", "web_extractor"}:
            urls = tool_args.get("urls")
            if urls is None:
                url = tool_args.get("url") or tool_args.get("id") or tool_args.get("raw_input")
                urls = [url] if url else []
            elif isinstance(urls, str):
                urls = [urls]

            results = []
            goal = tool_args.get("goal")
            if goal:
                results.append(f"Goal: {goal}")

            for url in urls:
                if not url:
                    continue
                tool_call_msg_dict = self._build_browser_tool_call_msg(
                    recipient="browser.open",
                    arguments={"id": url},
                    tool_name=tool_name,
                )
                result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
                if tool_name == "open":
                    if result_msgs_dict:
                        results.append(self._extract_browser_text(result_msgs_dict))
                    continue

                useful_information = ""
                content = self._extract_browser_text(result_msgs_dict)
                if content and not content.startswith("[visit] Failed to read page."):
                    content = self._truncate_to_tokens(content, max_tokens=95000)
                    if self.summary_sampler is None:
                        useful_information += f"URL: {url}\n"
                        useful_information += "Evidence in page: \n" + content[:8000] + "\n\n"
                        useful_information += "Summary: \nSummary sampler is unavailable, returning raw extracted content.\n\n"
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": EXTRACTOR_PROMPT.format(
                                    webpage_content=content,
                                    goal=goal or "Extract the information most relevant to the webpage.",
                                ),
                            }
                        ]
                        for _ in range(3):
                            try:
                                summary_content = self.summary_sampler(messages)
                                raw = self._parse_summary_response(summary_content)
                                useful_information += f"URL: {url}\n"
                                useful_information += "Evidence in page: \n" + str(raw.get("evidence", content[:8000])) + "\n\n"
                                useful_information += "Summary: \n" + str(raw.get("summary", "")) + "\n\n"
                                break
                            except Exception:
                                continue

                if not useful_information:
                    useful_information = (
                        f"URL: {url}\n"
                        "Evidence in page: \n"
                        + (
                            content[:8000]
                            if content
                            else "The provided webpage content could not be accessed. Please check the URL or file format."
                        )
                        + "\n\nSummary: \n"
                        + (
                            "Summary generation failed, returning raw extracted content."
                            if content
                            else "The webpage content could not be processed, and therefore, no information is available."
                        )
                        + "\n\n"
                    )
                results.append(useful_information)
            return "\n=======\n".join(results)

        if tool_name == "find":
            pattern = tool_args.get("pattern") or tool_args.get("raw_input")
            if not pattern:
                return "Error: Missing pattern for find"
            tool_call_msg_dict = self._build_browser_tool_call_msg(
                recipient="browser.find",
                arguments={"pattern": pattern},
                tool_name=tool_name,
            )
            result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
            if result_msgs_dict:
                try:
                    return result_msgs_dict[0]["content"][0]["text"]
                except Exception:
                    return json.dumps(result_msgs_dict, ensure_ascii=False)
            return ""

        if tool_name == "code_interpreter":
            return "Error: code_interpreter is not supported in this verify client."

        return f"Error: Unsupported tool '{tool_name}'."

    @staticmethod
    def _build_tool_call(function_call: Dict[str, Any]) -> Dict[str, Any]:
        arguments = function_call.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)

        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": function_call["name"],
                "arguments": arguments,
            },
        }

    def format_saved_messages(self, raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        pending_assistant: Dict[str, Any] | None = None

        def ensure_pending_assistant() -> Dict[str, Any]:
            nonlocal pending_assistant
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": "",
                }
            return pending_assistant

        def flush_pending_assistant() -> None:
            nonlocal pending_assistant
            if pending_assistant is None:
                return

            if (
                pending_assistant.get("content")
                or pending_assistant.get("reasoning_content") is not None
                or pending_assistant.get("tool_calls")
            ):
                messages.append(pending_assistant)

            pending_assistant = None

        for msg in raw_messages:
            role = msg.get("role")

            if role == "assistant":
                assistant_msg = ensure_pending_assistant()

                content = msg.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    if content:
                        assistant_msg["content"] += content

                reasoning_content = msg.get("reasoning_content")
                if reasoning_content is not None:
                    if assistant_msg.get("reasoning_content"):
                        assistant_msg["reasoning_content"] += reasoning_content
                    else:
                        assistant_msg["reasoning_content"] = reasoning_content

                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    assistant_msg.setdefault("tool_calls", []).extend(tool_calls)

                function_call = msg.get("function_call")
                if function_call is not None:
                    assistant_msg.setdefault("tool_calls", []).append(
                        self._build_tool_call(function_call)
                    )
                continue

            flush_pending_assistant()

            if role == "function":
                last_assistant = messages[-1] if messages else None
                tool_calls = (last_assistant or {}).get("tool_calls") or []
                tool_call_id = None

                if len(tool_calls) == 1:
                    tool_call_id = tool_calls[0]["id"]
                else:
                    tool_name = msg.get("name")
                    for tool_call in reversed(tool_calls):
                        if tool_call.get("function", {}).get("name") == tool_name:
                            tool_call_id = tool_call["id"]
                            break

                tool_msg = {
                    "role": "tool",
                    "content": msg.get("content", ""),
                }
                if tool_call_id is not None:
                    tool_msg["tool_call_id"] = tool_call_id
                messages.append(tool_msg)
                continue

            messages.append(msg)

        flush_pending_assistant()
        return messages

    @staticmethod
    def _count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += len(enc.encode(content))
        return total

    def _single_turn_inference_sync(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        retry_count: int = 32,
    ) -> Dict[str, Any]:
        for retry in range(retry_count):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=TOP_P,
                    timeout=MODEL_TIMEOUT,
                    stream=True,
                    extra_body={
                        "enable_thinking": True,
                    },
                )

                reasoning_content = ""
                content = ""
                tool_calls_dict: Dict[int, Dict[str, Any]] = {}
                finish_reason = None
                model_name = ""
                usage = None

                for chunk in stream:
                    model_name = getattr(chunk, "model", None) or model_name
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage.model_dump()

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta
                    if delta is None:
                        continue

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content

                    if delta.content:
                        content += delta.content

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.id:
                                tool_calls_dict[idx]["id"] = tc_delta.id
                            if tc_delta.type:
                                tool_calls_dict[idx]["type"] = tc_delta.type
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_dict[idx]["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_dict[idx]["function"]["arguments"] += tc_delta.function.arguments

                tool_calls_list = None
                if tool_calls_dict:
                    tool_calls_list = [tool_calls_dict[k] for k in sorted(tool_calls_dict.keys())]

                message_dict = {
                    "role": "assistant",
                    "content": content or None,
                }
                if reasoning_content:
                    message_dict["reasoning_content"] = reasoning_content
                if tool_calls_list:
                    message_dict["tool_calls"] = tool_calls_list

                result = {
                    "choices": [{
                        "message": message_dict,
                        "finish_reason": finish_reason,
                    }],
                    "model": model_name,
                }
                if usage:
                    result["usage"] = usage
                return result

            except (APIConnectionError, RateLimitError) as e:
                logger.warning("Retry %s/%s failed: %s", retry + 1, retry_count, e, exc_info=True)
                time.sleep(2)
            except Exception as e:
                logger.warning("Retry %s/%s failed: %s", retry + 1, retry_count, e, exc_info=True)
                time.sleep(2)

        return {"error": "Max retries exceeded"}

    async def single_turn_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._single_turn_inference_sync,
            messages,
            max_tokens,
        )

    async def search_loop(self, qid: Any, question: str):
        tool_config = self.browser_pool.init_session(qid)
        raw_messages = [
            {"role": "system", "content": ADD_SP},
            {"role": "user", "content": "Question: " + question},
        ]

        turn = 0
        end_flag = True
        success_flag = True

        while True:
            turn += 1
            logger.debug("# Turn: %s", turn)

            t1 = time.perf_counter()
            prompt_tokens = self._count_messages_tokens(raw_messages)
            dynamic_max_tokens = max(1, min(MAX_CONTEXT_TOKENS - prompt_tokens, MAX_OUTPUT_TOKENS_CAP))

            response = await self.single_turn_inference(raw_messages, max_tokens=dynamic_max_tokens)
            t2 = time.perf_counter()
            logger.debug(
                "[qid=%s turn=%s] qwen3_5_chat=%.4fs response_type=%s prompt_tokens=%s dynamic_max_tokens=%s",
                qid, turn, t2 - t1, type(response), prompt_tokens, dynamic_max_tokens,
            )

            if "error" in response:
                return {
                    "raw_messages": raw_messages,
                    "messages": self.format_saved_messages(raw_messages),
                    "end_flag": False,
                    "is_success": False,
                    "error": response["error"],
                    "info": self.info,
                }
            if not response.get("choices"):
                return {
                    "raw_messages": raw_messages,
                    "messages": self.format_saved_messages(raw_messages),
                    "end_flag": False,
                    "is_success": False,
                    "error": "No choices in response",
                    "info": self.info,
                }

            message_data = response["choices"][0]["message"]
            finish_reason = response["choices"][0]["finish_reason"]

            if message_data.get("content") is None:
                message_data["content"] = ""

            if message_data.get("tool_calls"):
                for tool_call in message_data["tool_calls"]:
                    if not tool_call.get("id"):
                        tool_call["id"] = f"call_{uuid.uuid4().hex[:24]}"
                    function = tool_call["function"]
                    raw_args = function["arguments"]
                    if isinstance(raw_args, dict):
                        raw_args = json.dumps(raw_args, ensure_ascii=False)
                        function["arguments"] = raw_args

                    try:
                        json.loads(raw_args)
                    except json.JSONDecodeError:
                        fixed_args = "{}"
                        if function["name"] == "code_interpreter":
                            fixed_args = json.dumps({"code": raw_args}, ensure_ascii=False)
                        elif function["name"] == "web_search":
                            fixed_args = json.dumps({"queries": [raw_args]}, ensure_ascii=False)
                        function["arguments"] = fixed_args

            raw_messages.append(message_data)
            logger.debug(
                "[qid=%s turn=%s] finish_reason=%s reasoning_content=%s content=%s tool_calls=%s",
                qid,
                turn,
                finish_reason,
                message_data.get("reasoning_content", "").strip()[:50],
                message_data.get("content", "").strip()[:50],
                message_data.get("tool_calls"),
            )

            if finish_reason != "tool_calls":
                break

            t3 = time.perf_counter()
            temp_observations = []
            for tool_call in message_data.get("tool_calls", []):
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = function["arguments"]
                tool_content = await self._call_browser_compatible_tool(
                    qid=qid,
                    tool_name=tool_name,
                    arguments=arguments,
                )

                tool_message = {
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call["id"],
                }
                raw_messages.append(tool_message)
                temp_observations.append(tool_message["content"])

            t4 = time.perf_counter()
            logger.debug(
                "[qid=%s turn=%s] func_call=%.4fs observations(%d)=%s",
                qid, turn, t4 - t3, len(temp_observations), [str(obs)[:50] for obs in temp_observations]
            )

            if turn >= self.max_turns:
                logger.warning("[qid=%s] reached max_turns=%s", qid, self.max_turns)
                end_flag = False
                success_flag = False
                break

        return {
            "raw_messages": raw_messages,
            "messages": self.format_saved_messages(raw_messages),
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

    search_agent = Qwen_3_5_SearchAgent(
        api_key=get_api_key(None),
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
