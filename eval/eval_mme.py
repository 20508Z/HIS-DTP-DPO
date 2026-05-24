"""
Evaluate MME yes/no subsets for Qwen2.5-VL + optional LoRA.

For each MME category, this reports:
- accuracy: per-question yes/no accuracy
- accuracy_plus: image-level accuracy where both paired questions are correct
- score: accuracy + accuracy_plus, matching the common MME category score
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--lora_path", default=None)
    p.add_argument(
        "--mme_dir",
        default="data/MME/MME_Benchmark_release_version/MME_Benchmark",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["existence", "count", "position", "color"],
    )
    p.add_argument("--output_file", default="outputs/mme_hallucination.json")
    p.add_argument("--max_new_tokens", type=int, default=16)
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


def find_image(txt_path):
    stem = txt_path.with_suffix("")
    for ext in IMAGE_EXTS:
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {txt_path}")


def parse_answer(text):
    normalized = text.lower().strip()
    if normalized.startswith("yes") or " yes" in normalized[:24]:
        return "yes"
    if normalized.startswith("no") or " no" in normalized[:24]:
        return "no"
    if "yes" in normalized and "no" not in normalized:
        return "yes"
    if "no" in normalized and "yes" not in normalized:
        return "no"
    return "unknown"


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


def read_questions(txt_path):
    pairs = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            question, answer = line.rsplit("\t", 1)
            pairs.append((question, answer.lower()))
    return pairs


def evaluate_subset(model, processor, subset_dir, device, max_new_tokens, max_image_pixels):
    records = []
    total = correct = 0
    pair_total = pair_correct = 0

    for txt_path in tqdm(sorted(subset_dir.glob("*.txt")), desc=subset_dir.name):
        image_path = find_image(txt_path)
        image = Image.open(image_path).convert("RGB")
        image = resize_if_needed(image, max_image_pixels)
        image_records = []
        all_correct = True
        for question, gt in read_questions(txt_path):
            raw = generate_answer(
                model, processor, image, question, device, max_new_tokens
            )
            pred = parse_answer(raw)
            ok = pred == gt
            total += 1
            correct += int(ok)
            all_correct = all_correct and ok
            image_records.append(
                {
                    "question": question,
                    "gt": gt,
                    "prediction": pred,
                    "raw_answer": raw,
                    "correct": ok,
                }
            )
        pair_total += 1
        pair_correct += int(all_correct)
        records.append(
            {
                "image": str(image_path),
                "qa": image_records,
                "all_correct": all_correct,
            }
        )

    accuracy = 100.0 * correct / total if total else 0.0
    accuracy_plus = 100.0 * pair_correct / pair_total if pair_total else 0.0
    return {
        "accuracy": accuracy,
        "accuracy_plus": accuracy_plus,
        "score": accuracy + accuracy_plus,
        "correct": correct,
        "total": total,
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "records": records,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    model, processor, _ = load_model(args.model_path, args.lora_path)

    mme_dir = Path(args.mme_dir)
    results = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "subsets": {},
    }
    for subset in args.subsets:
        subset_result = evaluate_subset(
            model, processor, mme_dir / subset, args.device, args.max_new_tokens,
            args.max_image_pixels,
        )
        results["subsets"][subset] = subset_result
        print(
            f"{subset}: acc={subset_result['accuracy']:.2f}, "
            f"acc+={subset_result['accuracy_plus']:.2f}, "
            f"score={subset_result['score']:.2f}"
        )

    results["hallucination_score"] = sum(
        item["score"] for item in results["subsets"].values()
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Total hallucination subset score: {results['hallucination_score']:.2f}")
    print(f"Saved MME results to {args.output_file}")


if __name__ == "__main__":
    main()
