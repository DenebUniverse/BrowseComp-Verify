# Synced from server sources; credentials sanitized before packaging.
# Release adaptations: package imports, DIAMOND loader, pair IDs, CLI and resume.
from browsecomp_verify.runtime import (
    parse_agent_args, prepare_run, completed_ids, verification_prompt,
    get_api_key, chat_completions_url,
)
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime
import json
import logging
from multiprocessing import Pool
import multiprocessing as mp
import os
import inspect
import re
import sys
import time
import traceback
from typing import List, Dict, Any, Optional
from filelock import FileLock
from openai import OpenAI
from json_repair import repair_json
from openai_harmony import Message, Role
import tiktoken
import logging

import tqdm
from browsecomp_verify.browser import BrowserTool, LocalServiceBrowserBackend
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


MIRO_SYSTEM_PROMPT_TEMPLATE_PREFIX = """You are MiroThinker, an advanced AI assistant developed by MiroMind.

In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {today_date}"""

MIRO_SYSTEM_PROMPT_TEMPLATE_SUFFIX_no_sandbox = """

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description:
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{
  "param1": "value1",
  "param2": "value2 \"escaped string\""
}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:

## Server name: search_and_scrape_webpage
### Tool name: google_search
Description:
    Tool to perform web searches via Serper API and retrieve rich results.

    It is able to retrieve organic search results, people also ask,
    related searches, and knowledge graph.

    Args:
        q: Search query string
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query

    Returns:
        Dictionary containing search results and metadata.

Input JSON schema: {'properties': {'q': {'title': 'Q', 'type': 'string'}, 'gl': {'default': 'us', 'title': 'Gl', 'type': 'string'}, 'hl': {'default': 'en', 'title': 'Hl', 'type': 'string'}, 'location': {'default': None, 'title': 'Location', 'type': 'string'}, 'num': {'default': None, 'title': 'Num', 'type': 'integer'}, 'tbs': {'default': None, 'title': 'Tbs', 'type': 'string'}, 'page': {'default': None, 'title': 'Page', 'type': 'integer'}, 'autocorrect': {'default': None, 'title': 'Autocorrect', 'type': 'boolean'}}, 'required': ['q'], 'title': 'google_searchArguments', 'type': 'object'}

## Server name: jina_scrape_llm_summary
### Tool name: scrape_and_extract_info
Description:
    Scrape content from a URL and extract specific types of information using LLM.

    Args:
        url (str): The URL to scrape content from
        info_to_extract (str): The specific types of information to extract (usually a question)
        custom_headers (Dict[str, str]): Additional headers to include in the scraping request

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - url (str): The original URL
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - scrape_stats (Dict): Statistics about the scraped content
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)

Input JSON schema: {'properties': {'url': {'title': 'Url', 'type': 'string'}, 'info_to_extract': {'title': 'Info To Extract', 'type': 'string'}, 'custom_headers': {'additionalProperties': {'type': 'string'}, 'default': None, 'title': 'Custom Headers', 'type': 'object'}}, 'required': ['url', 'info_to_extract'], 'title': 'scrape_and_extract_infoArguments', 'type': 'object'}

# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."""


def generate_miro_system_prompt(date: Optional[str] = None) -> str:
    """Get Miro Thinker system prompt with fixed tools."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    # return MIRO_SYSTEM_PROMPT_TEMPLATE_PREFIX.format(today_date=date) + MIRO_SYSTEM_PROMPT_TEMPLATE_SUFFIX
    return MIRO_SYSTEM_PROMPT_TEMPLATE_PREFIX.format(today_date=date) + MIRO_SYSTEM_PROMPT_TEMPLATE_SUFFIX_no_sandbox

def parse_mcp_tool_call(response_text: str):
    """Parse MCP-style tool call from model response. Returns first tool call or None."""
    match = re.search(r'<use_mcp_tool>(.*?)</use_mcp_tool>', response_text, re.DOTALL)
    if not match:
        return None
    content = match.group(1)
    server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
    tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
    args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)
    server_name = server_match.group(1).strip() if server_match else None
    tool_name = tool_match.group(1).strip() if tool_match else None
    if args_match:
        try:
            arguments = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Failed to parse arguments JSON: {e}, attempting to repair...")
            try:
                repaired = repair_json(args_match.group(1).strip())
                arguments = json.loads(repaired)
                print(f"✅  Successfully repaired JSON")
            except Exception as repair_error:
                print(f"❌  Failed to repair JSON: {repair_error}")
                arguments = {}
    else:
        arguments = {}
    if server_name and tool_name:
        return {"server_name": server_name, "tool_name": tool_name, "arguments": arguments}
    return None

class MiroThinker_SearchAgent():
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_turns: int = 600,
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
        
        self.client = OpenAI(
            api_key="REDACTED",
            base_url=self.base_url,
        )
        
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
                    # "extra_body": { "thinking": { "type": "enabled" } }
                }
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
        """Run a complete conversation with tool calling"""
        tool_config = self.browser_pool.init_session(qid)
        system_prompt = generate_miro_system_prompt()
        
        messages = [
            {'role': 'system', 'content': system_prompt}, {"role":"user","content": "Question: " + question},
            ]
        
        turn = 0
        prompt, response = None, None
        while True:
            turn += 1
            if turn > self.max_turns:
                return {"messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in messages],
                        "end_flag": False, "error": "max_turns exceeded", "info": self.info}
            logger.debug("# Turn: %s", turn)

            # step1 chat_completion
            t1 = time.perf_counter()
            response = await self.chat_completion(messages)
            res_json = json.loads(response.model_dump_json()) if not isinstance(response, dict) else response
            t2 = time.perf_counter()
            self.info["all_responses"].append(res_json)
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
            
            response_message = response.choices[0].message
            response_content = response_message.content
            tool_call = parse_mcp_tool_call(response_content)
            
            logger.debug(f"[qid=%s turn=%s] {response_content=}\n{tool_call=}")
            
            messages.append({"role":"assistant", "content": response_content})
            
            # step2 tool parse
            if not tool_call:
                break
            
            # step3: tool calls
            server_name = tool_call["server_name"]
            tool_name = tool_call["tool_name"]
            function_args = tool_call["arguments"]
            logger.debug(f"\n🔧 Model decided to call tool:\n  - Server: {server_name}\n    Tool: {tool_name}\n    Args: {json.dumps(function_args, ensure_ascii=False)}")
            
            format_tool_calls = self.custom_format_tool_call(tool_name, function_args)
            
            t3 = time.perf_counter()
            observations = []
            for tool_call_msg_dict in format_tool_calls:
                function_response = await self.browser_pool.call(qid, tool_call_msg_dict)
                function_response_content = function_response[0]['content'][0]['text']
                # json.dumps({function_response_content}, ensure_ascii=False)
                messages.append({"role": "user", "content": function_response_content})
                observations.append(function_response_content)
            
            t4 = time.perf_counter()
            logger.debug("[qid=%s turn=%s] func_call=%.4fs observations(%d)=%s", qid, turn, t4-t3, len(observations) ,observations)

        return {
            "messages": messages,
            "end_flag": True,
            "is_success": True,
            "info": self.info,
        }
    
    def custom_format_tool_call(self, tool_name, tool_args):
        tool_calls = []
        if tool_name.lower().strip() == 'google_search':
            tool_call_msg_dict = {
                'role': Role.ASSISTANT, 'name': None,
                'content': [
                    {'type': 'text',
                    'text': json.dumps({"query": tool_args['q'], "topn": tool_args['num'] if 'num' in tool_args else 10}, ensure_ascii=False, indent=1)}
                ],
                'channel': 'analysis',
                'recipient': 'browser.search',
                'content_type': 'code'}
            tool_calls.append(tool_call_msg_dict)
        elif tool_name.lower().strip() == 'scrape_and_extract_info':
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

async def worker_entry(
    args,
    qid: Any,
    question: str,
):
    browser_pool = BrowserPool(args.search_url)
    t0 = time.time()

    search_agent = MiroThinker_SearchAgent(
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
