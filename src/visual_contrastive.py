"""
Visual Contrastive: 图像扰动模块
- add_diffusion_noise: 扩散噪声（用于VCD推理增强）
- mask_single_image: 图像掩码（用于DTP-DPO Phase III视觉通路干预）
"""

import copy
import torch


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int) -> torch.Tensor:
    """
    扩散噪声添加函数，用于VCD推理时对比解码。
    """
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    return q_x(image_tensor.clone(), noise_step)


def mask_single_image(base_image: torch.Tensor, mask_percentage: float,
                      mask_method: str = 'random') -> torch.Tensor:
    """
    图像掩码函数，用于DTP-DPO Phase III视觉通路干预（L_prevent）。
    随机遮蔽指定比例的像素区域，替换为均值。
    """
    image = copy.deepcopy(base_image)
    mean_value = image.mean()
    _, C, H, W = image.shape
    if mask_method == 'random':
        total_pixels = H * W
        mask_pixels = int(total_pixels * mask_percentage)
        mask_indices = torch.randperm(total_pixels)[:mask_pixels]
        flat_image = image.view(C, -1)
        flat_image[:, mask_indices] = mean_value
    elif mask_method == 'blockwise':
        block_size = 14
        H_blocks = H // block_size
        W_blocks = W // block_size
        total_blocks = H_blocks * W_blocks
        mask_blocks = int(total_blocks * mask_percentage)
        mask_indices = torch.randperm(total_blocks)[:mask_blocks]
        flat_image = image.view(C, H_blocks, block_size, W_blocks, block_size)
        for idx in mask_indices:
            h = idx // W_blocks
            w = idx % W_blocks
            flat_image[:, h, :, w, :] = mean_value
    else:
        raise NotImplementedError(f"Unknown mask method: {mask_method}")
    return flat_image.view(1, C, H, W)
