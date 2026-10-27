# Optional server-sourced solving grader; never used by deterministic verification scoring.
from browsecomp_verify.runtime import get_api_key, positive_int
from openai import OpenAI
from collections import defaultdict
from typing import Dict, Any, List
import re
import os
import argparse
import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import json
import openai
import re
import json
import glob
from prettytable import PrettyTable

GRADER_TEMPLATE_BCplus = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

GRADER_TEMPLATE = """
Compare the extracted_final_answer from [response] with the [correct_answer].  
Treat harmless surface differences—case, spacing, punctuation, leading articles, middle names/initials and common company suffixes—as irrelevant.  
If, once normalized, both clearly point to the same entity, concept, or (within tiny rounding) number, mark them correct; otherwise mark them incorrect.

[question]: {question}

[response]: {response}

extracted_final_answer: The final exact answer from the [response]; write "None" if none is given.

[correct_answer]: {correct_answer}

reasoning: Briefly state why the normalized answers match or don't.

correct: yes / no

confidence: Copy the confidence score from the [response], or 100 if none.
""".strip()


class ThreadRateLimiter:
    def __init__(self, qps: float):
        self.capacity = max(1, int(qps))
        self.tokens = self.capacity
        self.refill_rate = qps
        self.timestamp = time.perf_counter()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.perf_counter()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens +
                              elapsed * self.refill_rate)
            self.timestamp = now
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.refill_rate
                pass_wait = wait_time
            else:
                pass_wait = 0.0
        if pass_wait > 0:
            time.sleep(pass_wait)
        with self.lock:
            now2 = time.perf_counter()
            elapsed2 = now2 - self.timestamp
            self.tokens = min(self.capacity, self.tokens +
                              elapsed2 * self.refill_rate)
            self.timestamp = now2
            self.tokens -= 1


class OpenAISampler:
    """
    Sample from OpenAI's completion API
    """

    def __init__(
        self,
        url: str = "https://one-api.glm.ai/v1",
        model: str = "gpt-4-turbo-2024-04-09",
        api_key: str = 'None',
        system_message: str = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 256,
        request_config: Dict[str, Any] = None,
    ):
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=url, timeout=1800)
        self.top_p = top_p
        self.request_config = request_config or {}

    def get_resp(self, message_list, tools=None):
        for i in range(5):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens,
                    **self.request_config
                )
                choice = chat_completion.choices[0]
                return chat_completion if tools is not None else choice.message.content
            except Exception as e:
                print(f"Openai Error: {e}", flush=True)
                if "Your request exceeded model token limit: 262144" in str(e):
                    return "Your request exceeded model token limit: 262144"
                time.sleep(1)
                continue
        return "" if tools is not None else None

    def __call__(self, message_list: List[Dict], tools: List = None) -> str:
        if self.system_message:
            message_list = [
                {"role": "system", "content": self.system_message}] + message_list
        return self.get_resp(message_list, tools)


def judge_answer(
    sampler,
    question: str,
    answer: str,
    response: str
):
    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=answer,
        response=response,
    )

    grader_res = sampler(
        [dict(content=grader_prompt, role="user")],
    )

    judge_match = re.search(r"correct: (yes|no)", grader_res)
    extract_judge = judge_match.group(1) if judge_match else "no"
    is_correct = (extract_judge == "yes")

    return {
        "grader_res": grader_res,
        "score":  100.0 * int(is_correct)
    }


class LLMJudge:
    def __init__(
        self,
        sampler,
        qps: int = 128,
        max_retries: int = 5,
        max_workers: int = 128,
    ):
        self.sampler = sampler
        self.rate_limiter = ThreadRateLimiter(qps)
        self.qps = qps
        self.max_workers = max_workers
        self.max_retries = max_retries

    def judge(self, data):
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._judge, d) for d in data]
            for f in tqdm(as_completed(futures), total=len(futures)):
                output.append(f.result())
        return output

    def _judge(self, data):
        question = data['question']
        content = data['messages'][-1]['content']
        
        if isinstance(content, list):
            gen_output = content[0]['text']
        else:
            gen_output = content

        if 'label' in data:
            answer = data['label']
        else:
            answer = data['answer']

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                result = judge_answer(
                    sampler=self.sampler,
                    question=question,
                    answer=answer,
                    response=gen_output
                )
                result["content"] = result["grader_res"]
                # Parse correct field for consistency with original
                judge_match = re.search(
                    r"correct: (yes|no)", result["content"]
                )
                result["correct"] = judge_match.group(
                    1) == "yes" if judge_match else False
                result["parse_error"] = judge_match is None
                # print("result: ", result, flush=True)
                result["question"] = question
                result['messages'] = data['messages']
                if 'info' in data:
                    result['all_responses'] = data['info']['all_responses']
                result['response'] = gen_output
                result['answer'] = answer
                result['id'] = data['qid']
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    return {"correct": False, "error": str(e), "score": 0.0, "id": data['qid']}
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)
        result["parse_error"] = True
        result['id'] = data['qid']
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_url", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--num_workers", type=positive_int, default=1)
    args = parser.parse_args()
    files = glob.glob(args.input_dir + "/*stream.jsonl")

    import os
    from tqdm import tqdm
    
    output_summary_file = os.path.join(args.input_dir, "eval_result.json")
    output_file = os.path.join(args.input_dir, "eval_result.jsonl")

    def load_jsonl_with_progress(filename):
        file_size = os.path.getsize(filename)  # 获取文件大小
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Loading {os.path.basename(filename)}") as pbar:
                for line in f:
                    data.append(json.loads(line))
                    pbar.update(len(line.encode('utf-8')))  # 根据读取字节数更新进度条
        return data

    data = []
    for file in files:
        data.extend(load_jsonl_with_progress(file))

    print("len(data): ", len(data))

    clean_data = []
    error_data = []
    for d in data:
        if d.get('end_flag') is True and not d.get('error'):
            clean_data.append(d)
        else:
            error_data.append(d)

    print("len(clean_data): ", len(clean_data))

    # open-source judge
    sampler = OpenAISampler(
        url=args.model_url,
        model=args.judge_model,
        api_key=get_api_key(),
        temperature=0.1,
        max_tokens=16384,
    )
    judger = LLMJudge(sampler=sampler, qps=args.num_workers, max_workers=args.num_workers)

    output = judger.judge(clean_data)
    

    parsed_output = [x for x in output if not x.get('parse_error', True)]
    correct_cnt_list = [x for x in parsed_output if x.get('correct') is True]

    total_samples = len(data)
    success_samples = len(clean_data)
    error_samples = len(error_data)

    judged_samples = len(output)
    parsed_ok_samples = len(parsed_output)
    parse_error_samples = judged_samples - parsed_ok_samples

    correct_samples = len(correct_cnt_list)

    success_rate = (success_samples /
                    total_samples) if total_samples > 0 else 0
    parse_error_rate = (parse_error_samples /
                        judged_samples) if judged_samples > 0 else 0
    judged_accuracy = (correct_samples /
                       parsed_ok_samples) if parsed_ok_samples > 0 else 0
    overall_accuracy = (
        correct_samples / total_samples) if total_samples > 0 else 0

    table = PrettyTable()
    table.title = "Evaluation Results Summary"
    table.field_names = ["Metric", "Count", "Percentage"]
    table.align = "l"
    table.align["Count"] = "r"
    table.align["Percentage"] = "r"

    table.add_row(["Total Samples", total_samples, f"{100:.2f}%"])
    table.add_row(
        ["  - Success Status", success_samples, f"{success_rate:.2%}"])
    table.add_row(["  - Error Status", error_samples,
                  f"{(1-success_rate):.2%}"])
    table.add_row(["-"*25, "-"*10, "-"*12], divider=True)
    table.add_row(["Judged Samples (Success Status)",
                  judged_samples, f"{100:.2f}% of Success"])
    table.add_row(["  - Parsed OK", parsed_ok_samples,
                  f"{(1-parse_error_rate):.2%}"])
    table.add_row(["  - Parse Error", parse_error_samples,
                  f"{parse_error_rate:.2%}"])
    table.add_row(["-"*25, "-"*10, "-"*12], divider=True)
    table.add_row(["Correct Predictions", correct_samples, ""])
    table.add_row(["Judged Accuracy (Correct/Parsed OK)",
                  "", f"{judged_accuracy:.2%}"])
    table.add_row(["Overall Accuracy (Correct/Total)",
                  "", f"{overall_accuracy:.2%}"])

    print(table)

    # Save results to eval_result.json
    eval_results = {
        "summary": {
            "total_samples": total_samples,
            "success_samples": success_samples,
            "error_samples": error_samples,
            "judged_samples": judged_samples,
            "parsed_ok_samples": parsed_ok_samples,
            "parse_error_samples": parse_error_samples,
            "correct_samples": correct_samples,
            "success_rate": success_rate,
            "parse_error_rate": parse_error_rate,
            "judged_accuracy": judged_accuracy,
            "overall_accuracy": overall_accuracy
        },
        "detailed_results": output
    }

    save_results = eval_results["detailed_results"]

    # Process error_data to include in detailed results
    for d in error_data:
        question = d.get('question', '')
        answer = d.get('answer', '')
        gen_output = d.get('response', '')
        result = {
            "id": d.get('id', ''),
            "correct": False,
            "score": 0.0,
            "parse_error": True
        }
        result["question"] = question
        result['messages'] = d.get('messages', [])
        result['all_responses'] = d.get('info', {}).get('all_responses', [])
        result['response'] = gen_output
        result['answer'] = answer
        save_results.append(result)

    with open(output_summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results["summary"], f, indent=2, ensure_ascii=False)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in eval_results["detailed_results"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")
