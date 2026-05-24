"""
Data Builder: 自动构造偏好对数据集
利用稳定性分析自动标记幻觉token，无需GPT-4V标注

流程:
1. 用基座模型对VG图像生成描述
2. 用稳定性分析标记每个token的不稳定分数
3. 结合VG ground truth对象标注构造偏好对
"""

import json
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from torch.utils.data import Dataset


IGNORE_INDEX = -100


class DTPDPODataset(Dataset):
    """DTP-DPO偏好对数据集

    Qwen2.5-VL要求input_ids中包含特殊的image token (<|vision_start|>, <|image_pad|>, <|vision_end|>)，
    所以必须通过processor.apply_chat_template + processor()来编码图像+文本。

    InternVL2.5-2B使用tile-based动态分辨率，需要单独处理图像为[num_tiles, 3, 448, 448]，
    并通过conversation template构建input_ids（含<IMG_CONTEXT>占位符）。

    关键设计：分别编码prompt和full(prompt+response)，通过prompt长度确定response起始位置，
    将labels中的prompt部分设为IGNORE_INDEX(-100)，只在response tokens上计算logprobs。
    """

    def __init__(self, data_path: str, tokenizer, processor,
                 max_length: int = 2048, model_path: str = "",
                 min_pixels: int = 256*28*28, max_pixels: int = 512*28*28):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.model_path = model_path
        self.is_qwen = "qwen" in model_path.lower()
        self.is_internvl = "internvl" in model_path.lower()
        # 限制图像分辨率以控制显存（默认max_pixels=512*28*28≈401K像素）
        if self.is_qwen and hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.min_pixels = min_pixels
            self.processor.image_processor.max_pixels = max_pixels

    def __len__(self):
        return len(self.data)

    def _build_inputs_with_labels(self, image, query, response,
                                   pixel_values=None, num_patches=None):
        """构建完整模型输入，并生成正确mask了prompt的labels。

        步骤:
        1. 构建prompt-only输入(add_generation_prompt=True)，获取prompt token长度
        2. 构建full输入(prompt+response，add_generation_prompt=False)
        3. labels = full_input_ids.clone()，将前prompt_len个token设为IGNORE_INDEX
        """
        if self.is_qwen:
            # Step 1: prompt-only (含assistant header)
            messages_prompt = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': query},
            ]}]
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors='pt',
                padding=False, truncation=True, max_length=self.max_length)
            prompt_len = prompt_inputs['input_ids'].shape[1]

            # Step 2: full input (prompt + response)
            messages_full = [
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': query},
                ]},
                {'role': 'assistant', 'content': response},
            ]
            full_text = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False)
            full_inputs = self.processor(
                text=[full_text], images=[image], return_tensors='pt',
                padding=False, truncation=True, max_length=self.max_length)

            # Step 3: labels with prompt masked
            labels = full_inputs['input_ids'].clone()
            labels[0, :prompt_len] = IGNORE_INDEX

            full_inputs['labels'] = labels
            full_inputs['prompt_len'] = prompt_len
            return full_inputs

        elif self.is_internvl:
            from src.internvl_utils import build_internvl_input_ids
            result = build_internvl_input_ids(
                self.tokenizer, query, response=response,
                num_patches=num_patches, num_image_token=256,
                model_path=self.model_path, max_length=self.max_length)
            # Attach pixel_values (already processed)
            result['pixel_values'] = pixel_values
            result['num_patches'] = num_patches
            return result
        else:
            inputs = self.processor(
                text=query + " " + response, images=image,
                return_tensors='pt', padding=True, truncation=True,
                max_length=self.max_length)
            inputs['labels'] = inputs['input_ids'].clone()
            inputs['prompt_len'] = 0
            return inputs

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')

        query = item['query']
        chosen = item['chosen']
        rejected = item['rejected']

        # InternVL: pre-process image once, share pixel_values for chosen/rejected
        pixel_values = None
        num_patches = None
        if self.is_internvl:
            from src.internvl_utils import load_image_for_internvl
            pixel_values, num_patches = load_image_for_internvl(image, max_num=4)

        # 分别构建chosen和rejected的完整输入（含正确的labels masking）
        chosen_inputs = self._build_inputs_with_labels(
            image, query, chosen, pixel_values=pixel_values,
            num_patches=num_patches)
        rejected_inputs = self._build_inputs_with_labels(
            image, query, rejected, pixel_values=pixel_values,
            num_patches=num_patches)

        instability_scores = torch.tensor(
            item.get('instability_scores', []), dtype=torch.float32)
        his_sem = torch.tensor(
            item.get('his_sem', item.get('instability_scores', [])), dtype=torch.float32)
        his_vis = torch.tensor(
            item.get('his_vis', item.get('instability_scores', [])), dtype=torch.float32)
        rejected_his_sem = torch.tensor(
            item.get('rejected_his_sem', item.get('instability_scores', [])), dtype=torch.float32)
        hsa_severity = float(item.get('hsa_severity', 1.0))

        return {
            'chosen_inputs': chosen_inputs,
            'rejected_inputs': rejected_inputs,
            'instability_scores': instability_scores,
            'his_sem': his_sem,
            'his_vis': his_vis,
            'rejected_his_sem': rejected_his_sem,
            'hsa_severity': torch.tensor(hsa_severity, dtype=torch.float32),
        }
