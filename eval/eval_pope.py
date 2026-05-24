"""
eval_pope.py: POPE基准评估
参考: VCD-master/experiments/eval/eval_pope.py

评估指标: Precision, Recall, F1, Accuracy, Yes Proportion
"""

import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--pope_file", type=str, required=True,
                   help="POPE问题文件，如 coco_pope_popular.json")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output_file", type=str, default="pope_results.jsonl")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_model(args):
    """加载模型，支持LoRA"""
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    is_internvl = "internvl" in args.model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)

    if "qwen2_5_vl" in model_type:
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    elif is_internvl:
        processor = tokenizer
        try:
            import flash_attn  # noqa: F401
            use_fa = True
        except ImportError:
            use_fa = False
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=use_fa).cuda()
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.img_context_token_id = img_ctx_id
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto")

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        if is_internvl:
            # Re-set img_context_token_id after LoRA wrapping
            targets = [model]
            if hasattr(model, 'base_model'):
                targets.append(model.base_model)
                if hasattr(model.base_model, 'model'):
                    targets.append(model.base_model.model)
            for t in targets:
                if hasattr(t, 'img_context_token_id'):
                    t.img_context_token_id = img_ctx_id
    model.eval()
    return model, processor


def build_chat_input(processor, question, image):
    """用chat template构建Qwen2.5-VL输入"""
    prompt = question + " Please answer yes or no."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


def build_internvl_chat_input(model, tokenizer, question, image, device):
    """Build InternVL chat input for a single question."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.internvl_utils import load_image_for_internvl, build_internvl_input_ids
    prompt = question + " Please answer yes or no."
    pv, num_patches = load_image_for_internvl(image, max_num=4)
    pv = pv.to(device).to(torch.bfloat16)
    result = build_internvl_input_ids(
        tokenizer, prompt, response=None,
        num_patches=num_patches, num_image_token=model.num_image_token
            if hasattr(model, 'num_image_token') else 256,
        model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'InternVL2_5-2B'))
    return {
        'input_ids': result['input_ids'].to(device),
        'attention_mask': result['attention_mask'].to(device),
        'pixel_values': pv,
    }


@torch.no_grad()
def generate_answers(model, processor, pope_data, image_dir, device):
    """对POPE问题生成回答"""
    is_internvl = hasattr(model, 'img_context_token_id') or (
        hasattr(model, 'base_model') and hasattr(model.base_model, 'model')
        and hasattr(model.base_model.model, 'img_context_token_id'))

    results = []
    for item in tqdm(pope_data, desc="Generating"):
        img_file = item.get("image", "")
        img_path = os.path.join(image_dir, img_file)
        question = item["text"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            results.append({"question_id": item["question_id"], "text": "no"})
            continue

        if is_internvl:
            inputs = build_internvl_chat_input(model, processor, question,
                                               image, device)
            output_ids = model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=16, do_sample=False)
            # InternVL generate may return only new tokens
            if output_ids.shape[1] > inputs['input_ids'].shape[1]:
                answer = processor.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True).strip()
            else:
                answer = processor.decode(
                    output_ids[0], skip_special_tokens=True).strip()
            answer = answer.split('<|im_end|>')[0].strip()
        else:
            inputs = build_chat_input(processor, question, image).to(device)
            output_ids = model.generate(**inputs, max_new_tokens=16,
                                        do_sample=False)
            answer = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
        results.append({"question_id": item["question_id"], "text": answer})
    return results


def evaluate_pope(gt_data, gen_data):
    """
    POPE评估指标计算
    参考: VCD-master/experiments/eval/eval_pope.py line 18-67
    """
    tp = tn = fp = fn = yes_cnt = 0
    total = len(gt_data)

    for i, gt_item in enumerate(gt_data):
        gt_ans = gt_item["label"].lower().strip()
        gen_ans = gen_data[i]["text"].lower().strip()

        if gt_ans == "yes":
            if "yes" in gen_ans:
                tp += 1; yes_cnt += 1
            else:
                fn += 1
        elif gt_ans == "no":
            if "no" in gen_ans:
                tn += 1
            else:
                fp += 1; yes_cnt += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "yes_proportion": yes_cnt / total if total > 0 else 0,
    }


def main():
    args = parse_args()
    model, processor = load_model(args)

    pope_data = [json.loads(q) for q in open(args.pope_file, "r")]
    gen_data = generate_answers(
        model, processor, pope_data,
        args.image_dir, args.device)

    # 保存生成结果
    with open(args.output_file, "w") as f:
        for item in gen_data:
            f.write(json.dumps(item) + "\n")

    # 评估
    metrics = evaluate_pope(pope_data, gen_data)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
