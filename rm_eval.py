import json
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from judger import PREFERENCE_ATTRIBUTES


class PersonalizationRMEvaluator:

    def __init__(self, attribute: str, device: str, rm_batch_size: int = 128):
        self.attribute = attribute
        self.device = device
        self.rm_batch_size = rm_batch_size
        rm_path = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
        self.rmodel = AutoModelForSequenceClassification.from_pretrained(rm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_path, use_fast=True)

    def load_data(self, file_path):
        responses = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            if content.startswith('[') and content.endswith(']'):
                json_arrays = content.split(']\n[')
                for i, json_str in enumerate(json_arrays):
                    if i == 0:
                        json_str = json_str + ']' if not json_str.endswith(']') else json_str
                    elif i == len(json_arrays) - 1:
                        json_str = '[' + json_str if not json_str.startswith('[') else json_str
                    else:
                        json_str = '[' + json_str + ']'

                    try:
                        array_data = json.loads(json_str)
                        responses.extend(array_data)
                    except json.JSONDecodeError:
                        continue
            else:
                # Handle JSONL format
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            responses.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return responses
    
    def batch_score(self, messages: List[List[Dict[str, str]]]) -> List[float]:
        batch_input = []
        for message in messages:
            message_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            batch_input.append(message_text)

        batch_inputs = self.tokenizer(batch_input, padding=True, truncation=True, add_special_tokens=True, max_length=1024, return_tensors="pt")

        input_ids, attention_mask = batch_inputs.input_ids.to(self.device), batch_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            output = self.rmodel(input_ids=input_ids, attention_mask=attention_mask)
            multi_obj_rewards = output.rewards.cpu().float()
            instruction_following_scores = multi_obj_rewards[:, 6]  # 6th dimension is instruction-following

        return instruction_following_scores.tolist()
    
    def divide_data_batch(self, dataset):
        total_size = len(dataset)
        num_processes = total_size // self.rm_batch_size
        subsets = [dataset[i * self.rm_batch_size : (i + 1) * self.rm_batch_size if i != num_processes - 1 else total_size + 1] for i in range(num_processes)]

        return subsets
    
    def get_rm_eval(self, response_file: str, response_key: str = "ours"):

        responses = self.load_data(response_file)
        all_batch_rwd = []
        batch_datas = self.divide_data_batch(responses)

        for data_subset in tqdm(batch_datas):
            preference_text = PREFERENCE_ATTRIBUTES.get(self.attribute, f"Your answer should be {self.attribute} as much as possible.")
            messages = []

            for data in data_subset:
                if 'query' in data and response_key in data:
                    user_content = data['query'] + ' ' + preference_text
                    assistant_content = data[response_key]

                    messages.append([
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ])

            if messages:
                method_rwd = self.batch_score(messages)
                all_batch_rwd.extend(method_rwd)

        avg_rwd = np.array(all_batch_rwd).mean()
        return avg_rwd