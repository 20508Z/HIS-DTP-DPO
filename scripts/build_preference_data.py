"""
build_preference_data.py: 自动构造偏好对数据集 (多GPU并行版)

流程:
1. 加载基座模型和VG数据
2. 对每张图像生成描述，同时记录hidden_states和attentions
3. 用稳定性分析标记每个token的不稳定分数 (CLSS + CTSS)
4. 结合VG ground truth对象标注，判断哪些token是幻觉
5. 构造 (chosen, rejected, instability_scores) 三元组

支持4卡并行: --gpu_id 指定卡号, --num_gpus 指定总卡数，自动分片
"""

import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel

# 确保从项目根目录导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.stability_analyzer import compute_stability_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="基座模型路径")
    parser.add_argument("--vg_image_dir", type=str, required=True,
                        help="Visual Genome图像目录")
    parser.add_argument("--vg_objects_file", type=str, required=True,
                        help="VG objects.json路径")
    parser.add_argument("--output_path", type=str, default="data/preference_data.json")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # 多GPU并行参数
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="当前使用的GPU编号")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="总GPU数量，用于数据分片")
    return parser.parse_args()


def load_model(model_path, device):
    """加载模型，自动适配Qwen2.5-VL和InternVL"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if "qwen" in model_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # 需要eager才能输出attentions
            device_map=device)
    elif "internvl" in model_path.lower():
        processor = tokenizer  # InternVL: tokenizer acts as processor
        try:
            import flash_attn  # noqa: F401
            use_fa = True
        except ImportError:
            use_fa = False
        # For stability analysis, need eager attention to output attentions
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=False).to(device)
        # Set img_context_token_id
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.img_context_token_id = img_ctx_id
    else:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=device)
    model.eval()
    return model, processor, tokenizer


def load_vg_objects(vg_objects_file: str, num_samples: int):
    """加载VG对象标注，返回 [(image_id, object_names)] 列表"""
    with open(vg_objects_file, 'r') as f:
        vg_data = json.load(f)
    result = []
    for item in vg_data[:num_samples]:
        img_id = item['image_id']
        obj_names = set()
        for obj in item.get('objects', []):
            for name in obj.get('names', []):
                obj_names.add(name.lower().strip())
        result.append((img_id, obj_names))
    return result


def check_hallucination(generated_text: str, gt_objects: set) -> list:
    """
    检查生成文本中的名词是否在VG ground truth中存在。
    返回每个句子的幻觉标记 (sentence, is_hallucinated)
    """
    import re
    sentences = re.split(r'[.!?]', generated_text)
    results = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = set(sent.lower().split())
        # 如果句子中的名词都不在gt_objects中，标记为幻觉
        overlap = words & gt_objects
        is_hal = len(overlap) == 0 and len(words) > 3
        results.append((sent, is_hal))
    return results


@torch.no_grad()
def generate_with_stability(model, processor, tokenizer, image, prompt,
                            max_new_tokens, device, is_qwen=False,
                            is_internvl=False, model_path=None):
    """两次前向：第一次快速generate拿文本，第二次teacher-forcing拿HIS。"""
    if is_qwen:
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                          padding=True).to(device)
    elif is_internvl:
        from src.internvl_utils import (load_image_for_internvl,
                                        build_internvl_input_ids)
        pixel_values, num_patches = load_image_for_internvl(image, max_num=4)
        pixel_values = pixel_values.to(device).to(torch.bfloat16)
        result = build_internvl_input_ids(
            tokenizer, prompt, response=None,
            num_patches=num_patches, num_image_token=model.num_image_token,
            model_path=model_path, max_length=2048)
        inputs = {
            'input_ids': result['input_ids'].to(device),
            'attention_mask': result['attention_mask'].to(device),
            'pixel_values': pixel_values,
        }
    else:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    if is_internvl:
        num_visual_tokens = inputs['pixel_values'].shape[0] * model.num_image_token
    else:
        num_visual_tokens = inputs.get('pixel_values', torch.empty(0)).shape[0]
    prefix_len = inputs['input_ids'].shape[1]

    # === Pass 1: fast generate, NO hidden_states/attentions ===
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      return_dict_in_generate=True)
    if is_internvl:
        gen_out = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **gen_kwargs)
    else:
        gen_out = model.generate(**inputs, **gen_kwargs)

    seq_len = gen_out.sequences.shape[1]
    if is_internvl:
        generated_ids = gen_out.sequences[0][prefix_len:] if seq_len > prefix_len \
            else gen_out.sequences[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.split('<|im_end|>')[0].strip()
    else:
        generated_ids = gen_out.sequences[0][prefix_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_gen_tokens = len(generated_ids)
    if num_gen_tokens == 0:
        return generated_text, [1.0], [1.0], [1.0]

    # === Pass 2: teacher-forcing forward on full sequence for HIS ===
    full_ids = gen_out.sequences  # (1, prefix+gen)
    attn_mask = torch.ones_like(full_ids)

    if is_internvl:
        fwd_out = model(
            pixel_values=inputs['pixel_values'],
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
    else:
        fwd_out = model(
            **{k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')},
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    # Extract only the generated-token slice from hidden states
    hidden_states = tuple(
        h[:, prefix_len:, :] for h in fwd_out.hidden_states
    )  # each: (1, gen_len, hidden)

    # lm_head weight
    if hasattr(model, 'lm_head'):
        lm_head_weight = model.lm_head.weight
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'output'):
        lm_head_weight = model.language_model.output.weight
    else:
        lm_head_weight = model.get_output_embeddings().weight

    # Visual attention for CTSS: last layer, generated-token rows, visual-token cols
    try:
        last_attn = fwd_out.attentions[-1]  # (1, heads, full_len, full_len)
        vis_attn = last_attn[:, :, prefix_len:, :num_visual_tokens]  # (1,H,gen,vis)
        vis_attn = vis_attn / (vis_attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn_tuple = (vis_attn.unsqueeze(0),)  # wrap so compute_ctss can index [-1]
        # Reformat to match compute_ctss expected shape: (batch, heads, seq, seq)
        # We pass attentions=None and compute VGC manually below
        attn_for_vgc = vis_attn  # (1, heads, gen_len, vis)
    except Exception:
        attn_for_vgc = None

    scores = compute_stability_scores(
        hidden_states=hidden_states,
        attentions=None,
        lm_head_weight=lm_head_weight,
        num_visual_tokens=num_visual_tokens,
        lambda_ctss=0.0,
    )
    scr = scores['kss']  # (1, gen_len), already normalized SCR

    # Compute VGC from attn_for_vgc
    if attn_for_vgc is not None and num_visual_tokens > 0:
        from torch.nn.functional import kl_div as _kl
        p = attn_for_vgc[:, :, 1:, :].clamp(1e-8)
        q = attn_for_vgc[:, :, :-1, :].clamp(1e-8)
        m = 0.5 * (p + q)
        vfd = 0.5 * (_kl(m.log(), p, reduction='none').sum(-1) +
                     _kl(m.log(), q, reduction='none').sum(-1))
        vfd = vfd.mean(1)  # (1, gen_len-1)
        pad = torch.zeros(1, 1, device=vfd.device)
        vfd = torch.cat([pad, vfd], dim=1)
        vgc = 1.0 - vfd  # (1, gen_len)
        from src.stability_analyzer import _normalize
        vgc = _normalize(vgc)
    else:
        vgc = scr.clone()

    min_len = min(scr.shape[1], vgc.shape[1])
    scr = scr[:, :min_len]
    vgc = vgc[:, :min_len]

    his_combined = (1.0 - (scr ** 0.5) * (vgc ** 0.5))[0].cpu().tolist()
    his_sem = (1.0 - scr)[0].cpu().tolist()
    his_vis = (1.0 - vgc)[0].cpu().tolist()

    return generated_text, his_combined, his_sem, his_vis

def build_chosen_response(sentences_with_hal, gt_objects):
    """用VG ground truth对象构造chosen响应：移除幻觉句子"""
    chosen_sents = [s for s, is_hal in sentences_with_hal if not is_hal]
    if not chosen_sents:
        # 如果全是幻觉，用gt对象构造简单描述
        objs = list(gt_objects)[:5]
        return "The image contains " + ", ".join(objs) + "."
    return ". ".join(chosen_sents) + "."


def find_image(img_id, vg_image_dirs):
    """在多个VG子目录中查找图像"""
    for d in vg_image_dirs:
        p = os.path.join(d, f"{img_id}.jpg")
        if os.path.exists(p):
            return p
    return None


def main():
    args = parse_args()
    # 当使用CUDA_VISIBLE_DEVICES隔离时，进程内只看到cuda:0
    device = "cuda:0"
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    print(f"[GPU {args.gpu_id}] Loading model to {device}...")
    model, processor, tokenizer = load_model(args.model_path, device)

    # 加载VG数据并按GPU分片
    all_vg = load_vg_objects(args.vg_objects_file, args.num_samples)
    # 分片: GPU i 处理 [i::num_gpus] 的数据
    my_vg = all_vg[args.gpu_id::args.num_gpus]
    print(f"[GPU {args.gpu_id}] Processing {len(my_vg)}/{len(all_vg)} samples")

    prompt = "Describe this image in detail."
    is_qwen = "qwen" in args.model_path.lower()
    is_internvl = "internvl" in args.model_path.lower()

    # VG图像子目录
    vg_image_dirs = []
    for subdir in ['VG_100K', 'VG_100K_2', 'images/VG_100K', 'images/VG_100K_2', '']:
        d = os.path.join(args.vg_image_dir, subdir) if subdir else args.vg_image_dir
        if os.path.isdir(d):
            vg_image_dirs.append(d)

    preference_data = []
    for img_id, gt_objs in tqdm(my_vg, desc=f"GPU{args.gpu_id}"):
        img_path = find_image(img_id, vg_image_dirs)
        if img_path is None:
            continue
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        try:
            gen_text, instab, h_sem, h_vis = generate_with_stability(
                model, processor, tokenizer, image, prompt,
                args.max_new_tokens, device, is_qwen=is_qwen,
                is_internvl=is_internvl, model_path=args.model_path)
        except Exception as e:
            print(f"[GPU {args.gpu_id}] Error on image {img_id}: {e}")
            continue

        # 检查幻觉
        hal_results = check_hallucination(gen_text, gt_objs)
        has_hal = any(is_hal for _, is_hal in hal_results)
        if not has_hal:
            continue

        chosen = build_chosen_response(hal_results, gt_objs)

        # rejected_his_sem复用chosen的HIS（rejected=原始生成，HIS已算好）
        preference_data.append({
            'image_path': img_path,
            'query': prompt,
            'chosen': chosen,
            'rejected': gen_text,
            'instability_scores': instab,
            'his_sem': h_sem,
            'his_vis': h_vis,
            'rejected_his_sem': h_sem,
        })

    # 保存本GPU的分片结果
    with open(args.output_path, 'w') as f:
        json.dump(preference_data, f, indent=2)
    print(f"[GPU {args.gpu_id}] Built {len(preference_data)} preference pairs -> {args.output_path}")


if __name__ == "__main__":
    main()
