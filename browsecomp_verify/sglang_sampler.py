# Synced from the server; standalone demo blocks omitted.
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import time, re
import requests
import traceback
import pdb
from browsecomp_verify.base import SamplerBase

class SglangSampler(SamplerBase):
    """
    Sample from TGI's completion API
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8002/v1/chat/completions",
        model: str = 'default',
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        request_config: Optional[Dict[str, Any]] = {},
    ):
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = url
        self.model = model
        self.api_key=api_key
        self.request_config = request_config or {}
    
    def get_chat_completion_resp(self, message_list, top_p=-1, temperature=-1, max_tokens=-1):
        assert self.url.endswith("/chat/completions"), f"Sampler URL {self.url} is not support '/chat/completions' mode."
        temperature = temperature if temperature > 0 else self.temperature
        top_p = top_p if top_p > 0 else 1.0
        max_tokens = max_tokens if max_tokens > 0 else self.max_tokens
        request_data = {
            "model": self.model if self.model else "default",
            "messages": message_list,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "skip_special_tokens": False,
            # "presence_penalty": 1.1, # if model is tongyi
            "stop": ['<|user|>', '<|endoftext|>', '<|im_end|>', '<|observation|>', "<|im_start|>", "<|assistant|>"],
        }
        request_data.update(self.request_config)
        
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp_json = None
        # print(f"request_data: {request_data}")
        for i in range(3):
            try:
                response = requests.post(
                    self.url,
                    json=request_data,
                    headers=headers,
                    timeout=1200,
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    break
                else:
                    print(f"Error in get_resp: {response.status_code}")
                    print(f"Error in get_resp: {response.text}")
                    continue
            except Exception as e:
                traceback.print_exc()
                print(f"Error in get_resp: {e}")
                time.sleep(1)
                continue
            
        if resp_json is None:
            print("failed in get_resp for 3 times")
            return f"Error fetching response from sglang sampler"
        
        content = resp_json['choices'][0]['message']
        reasoning_keys = ['reasoning_content', 'reasoning'] # 'reasoning_content' for GLM; 'reasoning' for PlantForm api
        reasoning = None
        output = content.get("content", "")
        if output is None:
            output = ""
        else:
            output = output.strip()
            
        for key in reasoning_keys:
            if key in content and content[key] is not None:
                reasoning = content[key]
                break
        if reasoning:
            content = f'<think>{reasoning}</think>\n\n{output}'
        else:
            content = output
        return content

    def get_completions_resp(self, prompt, top_p=-1, temperature=-1, max_tokens=-1):
        temperature = temperature if temperature > 0 else self.temperature
        top_p = top_p if top_p > 0 else 1
        max_tokens = max_tokens if max_tokens > 0 else self.max_tokens
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # "presence_penalty": 1.1, # if model is tongyi
            "stop": ['<|im_end|>', '<|observation|>', '<|user|>', "<|assistant|>", "<|im_start|>", "<|endoftext|>"],
            "skip_special_tokens": False
        }
        request_data.update(self.request_config)
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for i in range(5):
            try:
                response = requests.post(
                    self.url,
                    json=request_data,
                    headers=headers,
                    timeout=1800
                )
                resp_json = response.json()
                if "choices" in resp_json and len(resp_json["choices"]) > 0:
                    text = resp_json["choices"][0]["text"].strip()
                    return text

            except Exception as e:
                print(f"Error in get_resp_completion: {e}")
                traceback.print_exc()
                time.sleep(1)
                continue
        return f"Error fetching response in sglang sampler"

    def get_generate_resp(self, prompt, top_p=-1, temperature=-1, max_tokens=-1):
        temperature = temperature if temperature > 0 else self.temperature
        top_p = top_p if top_p > 0 else 1.0
        max_tokens = max_tokens if max_tokens > 0 else self.max_tokens
        request_data = {
            "model": self.model,
            "text": prompt,
            "sampling_params": {
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "stop": ['<|im_end|>', '<|observation|>', '<|user|>', "<|assistant|>", "<|im_start|>"],
                "skip_special_tokens": False,
            },
            "skip_special_tokens": False
        }
        request_data['sampling_params'].update(self.request_config)
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for i in range(3):
            try:
                response = requests.post(
                    self.url,
                    json=request_data,
                    headers=headers,
                    timeout=1000
                )
                resp_json = response.json()
                if "text" not in resp_json:
                    if 'error' in resp_json and 'message' in resp_json and "is longer than the model's context" in resp_json['error']['message']:
                        return resp_json['error']['message']
                    print(" Error resp_json: ", resp_json)
                text = resp_json["text"].strip()
                resp_json.pop("text")
                return text

            except Exception as e:
                print(f"Error in get_resp_completion: {e}")
                traceback.print_exc()
                time.sleep(1)
                continue
        return ''
    
    def get_resp_from_base(self, message_list, top_p=-1, temperature=-1, system_message="base", max_tokens=-1):        
        if system_message == "base":
            prefix = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it in a step by step manner. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question} Assistant:"
        else:
            raise ValueError("Invalid system message")
        
        if message_list[0]["role"] == "user":
            prompt = prefix.format(question=message_list[0]["content"])
        else:
            prompt = prefix.format(question=message_list[1]["content"])

        temperature = temperature if temperature > 0 else self.temperature
        top_p = top_p if top_p > 0 else 0.95

        if top_p == 0:
            top_p = 0.7

        request_data = {
            "model": "test",
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": top_p,
                "stop": ['<|user|>', '<|endoftext|>', '</answer>', '<|im_end|>']
            }
        }
        request_data['sampling_params'].update(self.request_config)
        if self.url.endswith("v1") or self.url.endswith("v1/"):
            url = self.url.strip("v1/") + "/generate"
        elif self.url.endswith("v1/chat/completions"):
            url = self.url.replace("v1/chat/completions", "generate")
        else:
            url = self.url + "/generate"
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            url,
            json=request_data,
            headers=headers
        )
        resp_json = response.json()
        # print(resp_json)
        content = resp_json["text"].strip()
        return content

    def __call__(self, message_list, top_p=-1, temperature=-1, max_tokens=-1) -> str:
        if isinstance(message_list, str):
            if self.url.endswith("/generate"):
                return self.get_generate_resp(message_list, top_p, temperature, max_tokens)
            elif self.url.endswith("/completions"):
                return self.get_completions_resp(message_list, top_p, temperature, max_tokens)
        elif isinstance(message_list, list):
            if self.system_message:
                if self.system_message.startswith("base"):
                    return self.get_resp_from_base(message_list, top_p, temperature, self.system_message, max_tokens)
                
                message_list = [{"role": "system", "content": self.system_message}] + message_list
            else:
                return self.get_chat_completion_resp(message_list, top_p, temperature, max_tokens)
        else:
            raise ValueError("Type of message_list is not string or List!")
        

