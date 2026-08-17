"""
train.py: DTP-DPO主训练脚本
使用LoRA微调，支持Qwen2.5-VL-3B/7B和InternVL2.5-2B
支持多GPU训练 (accelerate / torchrun)

超参数:
- beta=0.1, lr=1e-6: DPO标准超参
- LoRA rank=64, alpha=128: 论文实验使用的LoRA配置
- gamma_visual=0.2: 视觉锚定正则化权重
- gamma_anchor=0.1: 偏好锚定损失权重
- mask_ratio=0.3: 图像掩码比例
"""

import argparse
import json
import os
import random
from collections import deque

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer, AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from src.dtp_dpo_trainer import DTPDPOTrainer
from src.data_builder import DTPDPODataset
from src.stability_analyzer import compute_stability_scores


@torch.no_grad()
def recompute_his_for_subset(model, dataset, data_json, indices, device,
                              is_qwen, processor, tokenizer, model_path):
    """改进C: 用当前policy重算部分样本的HIS，更新data_json in-place。"""
    from transformers import AutoProcessor
    lm_head = None
    # get lm_head weight
    for name, module in model.named_modules():
        if name.endswith('lm_head') or name == 'lm_head':
            lm_head = module.weight.detach()
            break
    if lm_head is None:
        return  # skip if can't find lm_head

    model.eval()
    for idx in indices:
        item = data_json[idx]
        try:
            from PIL import Image
            image = Image.open(item['image_path']).convert('RGB')
            if is_qwen:
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': item['query']},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image],
                                   return_tensors='pt').to(device)
                num_vis = inputs.get('pixel_values', torch.empty(0)).shape[0]
                outputs = model(**inputs, output_hidden_states=True,
                                output_attentions=True)
                hidden = outputs.hidden_states
                attns = outputs.attentions
            else:
                continue  # only implemented for Qwen

            scores = compute_stability_scores(
                hidden_states=hidden, attentions=attns,
                lm_head_weight=lm_head, num_visual_tokens=num_vis)

            # update only the last (response) tokens
            seq_len = scores['instability'].shape[1]
            item['instability_scores'] = scores['instability'][0].cpu().tolist()[:seq_len]
            item['his_sem'] = scores['his_sem'][0].cpu().tolist()[:seq_len]
            item['his_vis'] = scores['his_vis'][0].cpu().tolist()[:seq_len]
        except Exception:
            continue  # keep old scores on failure
    model.train()


def is_qwen_model(model_path):
    return "qwen" in model_path.lower()


def is_internvl_model(model_path):
    return "internvl" in model_path.lower()


def load_model_and_ref(model_path, device):
    """加载训练模型(bf16)和参考模型(int8量化)到指定device"""
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if is_qwen_model(model_path):
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(device)
        ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=bnb_config, device_map={"": device})
    elif is_internvl_model(model_path):
        from transformers import AutoModel
        try:
            import flash_attn  # noqa: F401
            use_fa = True
        except ImportError:
            use_fa = False
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=use_fa).to(device)
        ref_model = AutoModel.from_pretrained(
            model_path, quantization_config=bnb_config, device_map={"": device},
            trust_remote_code=True, use_flash_attn=use_fa)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(device)
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map={"": device},
            trust_remote_code=True)
    return model, ref_model


def get_lora_target_modules(model_path):
    """根据模型架构返回LoRA目标模块"""
    if is_qwen_model(model_path):
        return ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif is_internvl_model(model_path):
        return ["wqkv", "wo", "w1", "w2", "w3"]
    else:
        return ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    # 模型
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/dtp-dpo")
    # 训练
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    # DPO
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--source_method", type=str, default="dtp_dpo",
                   choices=["dtp_dpo", "hsa_source", "opa_source", "standard_dpo"],
                   help="Training loss: DTP-DPO or source-adapted comparison baselines")
    p.add_argument("--hsa_alpha", type=float, default=1.0,
                   help="HSA-source response severity multiplier: weight=1+alpha*severity")
    p.add_argument("--opa_copo_coef", type=float, default=0.2,
                   help="OPA-source CoPO auxiliary coefficient")
    p.add_argument("--opa_anchor_coef", type=float, default=0.1,
                   help="OPA-source AncPO auxiliary coefficient")
    p.add_argument("--gamma_visual", type=float, default=0.2)
    p.add_argument("--gamma_anchor", type=float, default=0.1)
    p.add_argument("--vis_threshold", type=float, default=0.594,
                   help="Hard gate threshold for L_prevent (p75 of instability)")
    p.add_argument("--stab_threshold", type=float, default=0.454,
                   help="Hard gate threshold for L_stable (p50 of instability)")
    p.add_argument("--anchor_value", type=float, default=0.0)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--mask_method", type=str, default="random")
    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # 断点续传 & 定期保存
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint目录路径，用于断点续传")
    p.add_argument("--save_steps", type=int, default=200,
                   help="每隔多少个优化步保存一次checkpoint")
    p.add_argument("--dynamic_his", action="store_true",
                   help="改进C: 每epoch结束后对部分样本重算HIS")
    p.add_argument("--dynamic_his_ratio", type=float, default=0.2,
                   help="每epoch重算的样本比例")
    p.add_argument("--log_every", type=int, default=100,
                   help="每隔多少个step输出一次滚动统计日志")
    # 消融实验开关
    p.add_argument("--disable_stability_weights", action="store_true",
                   help="消融实验：将instability权重全设为1（等权DPO）")
    return p.parse_args()


def collate_fn(batch, tokenizer, processor, model_path, max_length=2048):
    """将batch中的样本处理为模型输入格式。

    关键：使用dataset返回的labels（已将prompt部分mask为IGNORE_INDEX=-100），
    而不是直接用input_ids.clone()。这确保logprobs只在response tokens上计算。

    支持两种pixel_values格式：
    - Qwen2.5-VL: (num_patches, 1176), 可直接cat
    - InternVL2.5: (num_tiles, 3, 448, 448), 需cat + 构建image_flags
    """
    IGNORE_INDEX = -100
    is_internvl = "internvl" in model_path.lower()
    instab_scores = []
    his_sem_list = []
    his_vis_list = []
    rejected_his_sem_list = []
    hsa_severity_list = []

    chosen_ids_list = []
    chosen_mask_list = []
    chosen_labels_list = []
    chosen_pv_list = []
    chosen_grid_list = []
    rejected_ids_list = []
    rejected_mask_list = []
    rejected_labels_list = []
    rejected_pv_list = []
    rejected_grid_list = []
    # InternVL: track num_patches per sample for image_flags construction
    chosen_num_patches_list = []
    rejected_num_patches_list = []

    for item in batch:
        ci = item['chosen_inputs']
        ri = item['rejected_inputs']
        chosen_ids_list.append(ci['input_ids'].squeeze(0))
        chosen_mask_list.append(ci['attention_mask'].squeeze(0))
        chosen_labels_list.append(ci['labels'].squeeze(0))
        chosen_pv_list.append(ci['pixel_values'])
        if 'image_grid_thw' in ci:
            chosen_grid_list.append(ci['image_grid_thw'])
        if is_internvl and 'num_patches' in ci:
            chosen_num_patches_list.append(ci['num_patches'])
        rejected_ids_list.append(ri['input_ids'].squeeze(0))
        rejected_mask_list.append(ri['attention_mask'].squeeze(0))
        rejected_labels_list.append(ri['labels'].squeeze(0))
        rejected_pv_list.append(ri['pixel_values'])
        if 'image_grid_thw' in ri:
            rejected_grid_list.append(ri['image_grid_thw'])
        if is_internvl and 'num_patches' in ri:
            rejected_num_patches_list.append(ri['num_patches'])
        instab_scores.append(item['instability_scores'])
        his_sem_list.append(item.get('his_sem', item['instability_scores']))
        his_vis_list.append(item.get('his_vis', item['instability_scores']))
        rejected_his_sem_list.append(item.get('rejected_his_sem', item['instability_scores']))
        hsa_severity_list.append(item.get('hsa_severity', torch.tensor(1.0)))

    # Pad input_ids和attention_mask
    chosen_ids = torch.nn.utils.rnn.pad_sequence(
        chosen_ids_list, batch_first=True,
        padding_value=tokenizer.pad_token_id or 0)
    chosen_mask = torch.nn.utils.rnn.pad_sequence(
        chosen_mask_list, batch_first=True, padding_value=0)
    # labels用IGNORE_INDEX填充（pad位置也不参与loss计算）
    chosen_labels = torch.nn.utils.rnn.pad_sequence(
        chosen_labels_list, batch_first=True,
        padding_value=IGNORE_INDEX)

    rejected_ids = torch.nn.utils.rnn.pad_sequence(
        rejected_ids_list, batch_first=True,
        padding_value=tokenizer.pad_token_id or 0)
    rejected_mask = torch.nn.utils.rnn.pad_sequence(
        rejected_mask_list, batch_first=True, padding_value=0)
    rejected_labels = torch.nn.utils.rnn.pad_sequence(
        rejected_labels_list, batch_first=True,
        padding_value=IGNORE_INDEX)

    # Cat pixel_values
    chosen_pv = torch.cat(chosen_pv_list, dim=0)
    rejected_pv = torch.cat(rejected_pv_list, dim=0)

    # pad instability scores
    def _pad_scores(score_list):
        max_len = max(len(s) for s in score_list) if score_list[0].numel() > 0 else 1
        padded = []
        for s in score_list:
            if s.numel() == 0:
                padded.append(torch.ones(max_len))
            elif len(s) < max_len:
                padded.append(torch.cat([s, torch.ones(max_len - len(s))]))
            else:
                padded.append(s[:max_len])
        return torch.stack(padded)

    result = {
        'chosen_pixel_values': chosen_pv,
        'chosen_input_ids': chosen_ids,
        'chosen_attention_mask': chosen_mask,
        'chosen_labels': chosen_labels,
        'rejected_pixel_values': rejected_pv,
        'rejected_input_ids': rejected_ids,
        'rejected_attention_mask': rejected_mask,
        'rejected_labels': rejected_labels,
        'instability_scores': _pad_scores(instab_scores),
        'his_sem': _pad_scores(his_sem_list),
        'his_vis': _pad_scores(his_vis_list),
        'rejected_his_sem': _pad_scores(rejected_his_sem_list),
        'hsa_severity': torch.stack(hsa_severity_list),
    }
    if chosen_grid_list:
        result['chosen_image_grid_thw'] = torch.cat(chosen_grid_list, dim=0)
        result['rejected_image_grid_thw'] = torch.cat(rejected_grid_list, dim=0)
    if is_internvl:
        # InternVL: image_flags = all 1s, shape [total_tiles, 1]
        # chosen and rejected share same pixel_values per sample,
        # but we cat them separately for chosen/rejected forward passes
        result['chosen_image_flags'] = torch.ones(
            chosen_pv.shape[0], 1, dtype=torch.long)
        result['rejected_image_flags'] = torch.ones(
            rejected_pv.shape[0], 1, dtype=torch.long)
    return result


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"World size: {world_size}, Device: {device}")

    # 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    if is_qwen_model(args.model_path):
        processor = AutoProcessor.from_pretrained(args.model_path)
    elif is_internvl_model(args.model_path):
        # InternVL: AutoProcessor returns the tokenizer itself, no separate processor
        processor = tokenizer
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True)

    # 加载模型和参考模型到当前GPU
    if is_main:
        print("Loading models...")
    model, ref_model = load_model_and_ref(args.model_path, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # LoRA配置
    target_modules = get_lora_target_modules(args.model_path)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM" if not is_internvl_model(args.model_path) else None,
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    if is_main:
        model.print_trainable_parameters()

    # 多GPU: DDP包装训练模型 (ref_model不需要DDP，只做推理)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=True)

    # 数据集
    with open(args.data_path) as f:
        data_json = json.load(f)
    dataset = DTPDPODataset(args.data_path, tokenizer, processor,
                              model_path=args.model_path)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                      rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer, processor,
                                         args.model_path),
        num_workers=2, pin_memory=True)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps)

    # 训练器 (传入底层model，不是DDP wrapper)
    raw_model = model.module if world_size > 1 else model
    trainer = DTPDPOTrainer(raw_model, ref_model, tokenizer, processor, args)

    # 轻量统计：用于判断偏好信号是否退化
    margin_hist = deque(maxlen=args.log_every)
    target_logit_hist = deque(maxlen=args.log_every)
    vis_gate_hist = deque(maxlen=args.log_every)
    stab_gate_hist = deque(maxlen=args.log_every)
    instab_mean_hist = deque(maxlen=args.log_every)
    instab_var_hist = deque(maxlen=args.log_every)

    # === 断点续传：加载checkpoint ===
    start_epoch = 0
    start_step = 0  # dataloader内的step偏移
    global_step = 0
    if args.resume_from and os.path.isdir(args.resume_from):
        from peft import PeftModel
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=device)
            start_epoch = state['epoch']
            start_step = state['step']
            global_step = state['global_step']
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            if is_main:
                print(f"Resumed from epoch={start_epoch}, step={start_step}, "
                      f"global_step={global_step}")
        # 加载LoRA权重
        lora_path = os.path.join(args.resume_from, "adapter_model.safetensors")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(args.resume_from, "adapter_model.bin")
        if os.path.exists(lora_path):
            from safetensors.torch import load_file
            if lora_path.endswith(".safetensors"):
                lora_weights = load_file(lora_path)
            else:
                lora_weights = torch.load(lora_path, map_location=device)
            raw_model.load_state_dict(lora_weights, strict=False)
            if is_main:
                print(f"Loaded LoRA weights from {args.resume_from}")

    def save_checkpoint(save_dir, epoch, step, global_step):
        """保存完整checkpoint（LoRA权重 + 训练状态）"""
        os.makedirs(save_dir, exist_ok=True)
        raw_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        torch.save({
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(save_dir, "training_state.pt"))

    import shutil
    prev_step_ckpt = [None]  # 用list包装以便在闭包中修改

    def save_step_checkpoint(epoch, step, global_step):
        """保存中间step checkpoint，只保留最新一个（节省磁盘）"""
        ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
        save_checkpoint(ckpt_dir, epoch, step, global_step)
        # 删除上一个step checkpoint（epoch checkpoint不删）
        if prev_step_ckpt[0] and os.path.isdir(prev_step_ckpt[0]):
            shutil.rmtree(prev_step_ckpt[0], ignore_errors=True)
        prev_step_ckpt[0] = ckpt_dir

    # === 训练循环 ===
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not is_main)
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(pbar):
            # 断点续传：跳过已完成的step
            if epoch == start_epoch and step < start_step:
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            # InternVL: pixel_values must match model dtype (bfloat16)
            if is_internvl_model(args.model_path):
                for key in ['chosen_pixel_values', 'rejected_pixel_values']:
                    if key in batch:
                        batch[key] = batch[key].to(torch.bfloat16)

            # 消融实验：禁用稳定性权重时，将instability_scores全设为1
            if args.disable_stability_weights:
                batch['instability_scores'] = torch.ones_like(batch['instability_scores'])
                batch['his_sem'] = torch.ones_like(batch['his_sem'])
                batch['his_vis'] = torch.ones_like(batch['his_vis'])

            try:
                metrics = trainer.compute_loss(batch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if is_main:
                        print(f"\n[WARNING] OOM at step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue
                raise

            loss = metrics['loss'] / args.grad_accum
            loss.backward()

            # 滚动统计（仅主进程）
            if is_main:
                margin_hist.append(metrics['reward_margin'])
                target_logit_hist.append(metrics.get('target_logit', metrics['reward_margin']))
                instab = batch['instability_scores']
                instab_mean_hist.append(instab.mean().item())
                instab_var_hist.append(instab.var(unbiased=False).item())
                if 'vis_gate' in metrics:
                    vis_gate_hist.append(metrics['vis_gate'])
                if 'stab_gate' in metrics:
                    stab_gate_hist.append(metrics['stab_gate'])

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 定期保存checkpoint（只保留最新一个step checkpoint）
                if is_main and global_step % args.save_steps == 0:
                    save_step_checkpoint(epoch, step + 1, global_step)
                    print(f"\n[Checkpoint] Saved at global_step={global_step}")

            epoch_loss += metrics['loss'].item()
            epoch_steps += 1

            if is_main:
                pbar.set_postfix(
                    loss=f"{metrics['loss'].item():.4f}",
                    l_tgt=f"{metrics['l_target']:.4f}",
                    l_vis=f"{metrics['l_visual']:.4f}",
                    logit=f"{metrics.get('target_logit', metrics['reward_margin']):.4f}",
                    margin=f"{metrics['reward_margin']:.4f}")

                if (step + 1) % args.log_every == 0:
                    avg_margin = sum(margin_hist) / max(len(margin_hist), 1)
                    avg_target_logit = sum(target_logit_hist) / max(len(target_logit_hist), 1)
                    avg_vis_gate = sum(vis_gate_hist) / max(len(vis_gate_hist), 1)
                    avg_stab_gate = sum(stab_gate_hist) / max(len(stab_gate_hist), 1)
                    avg_instab_mean = sum(instab_mean_hist) / max(len(instab_mean_hist), 1)
                    avg_instab_var = sum(instab_var_hist) / max(len(instab_var_hist), 1)
                    print(
                        f"\n[Stats] step={step+1} avg_target_logit={avg_target_logit:.6f} "
                        f"avg_raw_margin={avg_margin:.6f} "
                        f"avg_vis_gate={avg_vis_gate:.6f} avg_stab_gate={avg_stab_gate:.6f} "
                        f"instab_mean={avg_instab_mean:.6f} instab_var={avg_instab_var:.6f}"
                    )

        # epoch结束后重置step偏移
        start_step = 0

        if is_main:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

            # 改进C: 动态HIS重算（用当前policy重算部分样本）
            if args.dynamic_his and epoch < args.epochs - 1:
                n_resample = max(1, int(len(data_json) * args.dynamic_his_ratio))
                indices = random.sample(range(len(data_json)), n_resample)
                print(f"[DynamicHIS] Recomputing HIS for {n_resample} samples...")
                recompute_his_for_subset(
                    model.module if hasattr(model, 'module') else model,
                    dataset, data_json, indices, device,
                    is_qwen=is_qwen_model(args.model_path),
                    processor=processor, tokenizer=tokenizer,
                    model_path=args.model_path)
                # 重建dataset使新HIS生效
                dataset.data = data_json
                print(f"[DynamicHIS] Done.")

            # 保存epoch checkpoint
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            save_checkpoint(save_path, epoch + 1, 0, global_step)
            print(f"Saved checkpoint to {save_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
