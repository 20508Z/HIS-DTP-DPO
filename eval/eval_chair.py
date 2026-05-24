"""
eval_chair.py: CHAIR基准评估
CHAIR_s: 包含幻觉的响应比例
CHAIR_i: 所有对象提及中幻觉对象的比例
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
    p.add_argument("--coco_ann_file", type=str, required=True,
                   help="COCO instances annotation json")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for image sampling (default: None=sequential)")
    p.add_argument("--shard_id", type=int, default=0,
                   help="0-based shard id for parallel CHAIR generation")
    p.add_argument("--num_shards", type=int, default=1,
                   help="number of shards for parallel CHAIR generation")
    p.add_argument("--generations_file", type=str, default=None,
                   help="optional file to save raw generations for later merging")
    p.add_argument("--output_file", type=str, default="chair_results.json")
    return p.parse_args()


# COCO 80类对象名称
COCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_gt_objects(coco_ann, image_id):
    """从COCO标注获取图像中的GT对象"""
    cat_map = {c['id']: c['name'] for c in coco_ann['categories']}
    anns = [a for a in coco_ann['annotations'] if a['image_id'] == image_id]
    return set(cat_map[a['category_id']].lower() for a in anns)


def extract_objects(text):
    """从生成文本中提取COCO对象"""
    text_lower = text.lower()
    found = []
    for obj in COCO_OBJECTS:
        if obj in text_lower:
            found.append(obj)
    return found


def compute_chair(generations, coco_ann):
    """
    计算CHAIR_s和CHAIR_i
    CHAIR_s = 包含幻觉对象的句子数 / 总句子数
    CHAIR_i = 幻觉对象数 / 总提及对象数
    """
    total_sents = 0
    hal_sents = 0
    total_objs = 0
    hal_objs = 0

    for gen in generations:
        gt_objs = get_gt_objects(coco_ann, gen['image_id'])
        sentences = gen['text'].split('.')

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            total_sents += 1
            mentioned = extract_objects(sent)
            sent_has_hal = False
            for obj in mentioned:
                total_objs += 1
                if obj not in gt_objs:
                    hal_objs += 1
                    sent_has_hal = True
            if sent_has_hal:
                hal_sents += 1

    chair_s = hal_sents / total_sents * 100 if total_sents > 0 else 0
    chair_i = hal_objs / total_objs * 100 if total_objs > 0 else 0
    return {"CHAIR_s": chair_s, "CHAIR_i": chair_i}


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    # 自动检测模型类型
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")

    is_internvl = "internvl" in args.model_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if "qwen2_5_vl" in model_type:
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    elif is_internvl:
        processor = tokenizer
        try:
            import flash_attn
            use_fa = True
        except ImportError:
            use_fa = False
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=use_fa).cuda()
        img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_context_token_id = img_ctx_id
    else:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto")

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        if is_internvl:
            for t in [model] + ([model.base_model] if hasattr(model, "base_model") else []) + ([model.base_model.model] if hasattr(model, "base_model") and hasattr(model.base_model, "model") else []):
                if hasattr(t, "img_context_token_id"):
                    t.img_context_token_id = img_ctx_id
    model.eval()

    with open(args.coco_ann_file, 'r') as f:
        coco_ann = json.load(f)

    # 选取图像（支持随机采样）
    import random
    if args.seed is not None:
        random.seed(args.seed)
        images_info = random.sample(coco_ann['images'], min(args.num_samples, len(coco_ann['images'])))
        print(f"Random sampling with seed={args.seed}, {len(images_info)} images")
    else:
        images_info = coco_ann['images'][:args.num_samples]
        print(f"Sequential sampling, {len(images_info)} images")
    if args.num_shards > 1:
        total_images = len(images_info)
        images_info = images_info[args.shard_id::args.num_shards]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"{len(images_info)} of {total_images} images"
        )
    prompt = "Describe this image in detail."

    if is_internvl:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.internvl_utils import load_image_for_internvl, build_internvl_input_ids

    # 断点续传：加载已有中间结果
    cache_file = args.output_file.replace('.json', '_cache.jsonl')
    generations = []
    done_ids = set()
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            for line in f:
                item = json.loads(line)
                generations.append(item)
                done_ids.add(item['image_id'])
        print(f"Resumed from cache: {len(done_ids)} images already done")

    cache_fp = open(cache_file, 'a')
    device = "cuda"
    for img_info in tqdm(images_info, desc="Generating"):
        if img_info['id'] in done_ids:
            continue
        img_path = os.path.join(args.image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        if is_internvl:
            pv, num_patches = load_image_for_internvl(image, max_num=4)
            pv = pv.to(device).to(torch.bfloat16)
            nim = model.num_image_token if hasattr(model, 'num_image_token') else 256
            result = build_internvl_input_ids(tokenizer, prompt, response=None,
                num_patches=num_patches, num_image_token=nim,
                model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'InternVL2_5-2B'))
            with torch.no_grad():
                ids = model.generate(pixel_values=pv,
                    input_ids=result['input_ids'].to(device),
                    attention_mask=result['attention_mask'].to(device),
                    max_new_tokens=512, do_sample=False)
            if ids.shape[1] > result['input_ids'].shape[1]:
                text = tokenizer.decode(ids[0][result['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            text = text.split('<|im_end|>')[0].strip()
        else:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image],
                               return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False)
            text = processor.decode(ids[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        item = {'image_id': img_info['id'], 'text': text}
        generations.append(item)
        cache_fp.write(json.dumps(item) + '\n')
        cache_fp.flush()

    cache_fp.close()
    metrics = compute_chair(generations, coco_ann)
    print(f"CHAIR_s: {metrics['CHAIR_s']:.2f}")
    print(f"CHAIR_i: {metrics['CHAIR_i']:.2f}")

    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    if args.generations_file:
        with open(args.generations_file, 'w') as f:
            json.dump(generations, f, indent=2)
    if os.path.exists(cache_file):
        os.remove(cache_file)


if __name__ == "__main__":
    main()
