"""
vcd_inference.py: VCD推理增强模块
在推理时结合VCD对比解码进一步降低幻觉

核心公式 (来自VCD代码 vcd_sample.py line 150-153):
  cutoff = log(cd_beta) + logits.max()
  diffs = (1+cd_alpha)*logits_orig - cd_alpha*logits_noisy
  cd_logits = diffs.masked_fill(logits_orig < cutoff, -inf)

超参数 (来自VCD论文):
  cd_alpha=1.0, cd_beta=0.1, noise_step=500
"""

import torch
import torch.nn.functional as F
from src.visual_contrastive import add_diffusion_noise


@torch.no_grad()
def vcd_generate(model, processor, tokenizer, image, prompt,
                 cd_alpha=1.0, cd_beta=0.1, noise_step=500,
                 max_new_tokens=512, device="cuda"):
    """
    VCD对比解码生成

    参考: VCD-master/vcd_utils/vcd_sample.py line 130-161
    """
    inputs = processor(text=prompt, images=image,
                       return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    # 生成加噪图像
    noisy_pixels = add_diffusion_noise(pixel_values, noise_step)
    noisy_pixels = noisy_pixels.to(pixel_values.dtype).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 原始图像前向
        out_orig = model(input_ids=generated,
                         attention_mask=torch.ones_like(generated),
                         pixel_values=pixel_values)
        logits_orig = out_orig.logits[:, -1, :]

        # 加噪图像前向
        out_noisy = model(input_ids=generated,
                          attention_mask=torch.ones_like(generated),
                          pixel_values=noisy_pixels)
        logits_noisy = out_noisy.logits[:, -1, :]

        # VCD对比解码 (vcd_sample.py line 150-153)
        cutoff = (torch.log(torch.tensor(cd_beta))
                  + logits_orig.max(dim=-1, keepdim=True).values)
        diffs = (1 + cd_alpha) * logits_orig - cd_alpha * logits_noisy
        cd_logits = diffs.masked_fill(logits_orig < cutoff, -float("inf"))

        probs = F.softmax(cd_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0][input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
