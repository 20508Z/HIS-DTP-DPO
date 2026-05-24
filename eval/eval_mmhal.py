"""
Generate MMHal-Bench responses for Qwen2.5-VL + optional LoRA.

MMHal-Bench uses an LLM judge for final scoring. This script writes the
response JSON expected by the official evaluator.
"""

import argparse
import json
import os

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--input_file", default="data/MMHal-Bench/test.jsonl")
    p.add_argument("--output_file", default="outputs/mmhal_responses.json")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_image_pixels", type=int, default=1048576)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, processor, tokenizer


def load_items(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resize_if_needed(image, max_pixels):
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    resized = image.copy()
    max_side = int(max_pixels ** 0.5)
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


@torch.no_grad()
def generate_answer(model, processor, image, question, device, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    return processor.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    model, processor, _ = load_model(args.model_path, args.lora_path)
    items = load_items(args.input_file)

    results = []
    for item in tqdm(items, desc="MMHal-Bench"):
        row = dict(item)
        image_path = row.get("image_path")
        if not image_path and row.get("image_src"):
            image_path = os.path.join(
                os.path.dirname(args.input_file), "repo", "images",
                os.path.basename(row["image_src"]),
            )
        image = Image.open(image_path).convert("RGB")
        image = resize_if_needed(image, args.max_image_pixels)
        row["model_answer"] = generate_answer(
            model, processor, image, row["question"], args.device,
            args.max_new_tokens,
        )
        results.append(row)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} MMHal-Bench responses to {args.output_file}")
    print("Final MMHal-Bench scoring requires the official GPT-4 judge script.")


if __name__ == "__main__":
    main()
