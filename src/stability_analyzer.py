"""
Stability Analyzer: DTP-DPO Phase I diagnostic module
用于自动定位模型生成中不稳定（易幻觉）的token

核心指标：
1. CLSS (Cross-Layer Stability Score): 跨层稳定性，基于相邻层logit分布的JSD
2. CTSS (Cross-Token Stability Score): 跨token稳定性，基于相邻token视觉注意力的JSD
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def jsd(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Jensen-Shannon Divergence between two probability distributions."""
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction='none').sum(dim=dim) +
                  F.kl_div(m.log(), q, reduction='none').sum(dim=dim))


def compute_clss(hidden_states: Tuple[torch.Tensor, ...],
                 lm_head_weight: torch.Tensor,
                 layer_range: Optional[Tuple[int, int]] = None,
                 topk: int = 50) -> torch.Tensor:
    """
    Semantic Convergence Rate via Top-K entropy reduction.

    Instead of full-vocab JSD (O(V)), project each layer to logit space,
    take top-K logits, compute entropy over the restricted distribution,
    and measure entropy reduction rate across a sliding window.
    Top-K entropy is more sensitive to prediction crystallization than
    full-vocab JSD, and ~3000x faster for V=151936, K=50.
    """
    num_layers = len(hidden_states)
    if layer_range is None:
        start = num_layers // 2
        end = num_layers
    else:
        start, end = layer_range

    selected = hidden_states[start:end]
    entropies = []
    for layer in selected:
        logits = F.linear(layer.float(), lm_head_weight.float())  # (B, T, V)
        topk_logits = torch.topk(logits, k=topk, dim=-1).values   # (B, T, K)
        probs = F.softmax(topk_logits, dim=-1)
        H = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)     # (B, T)
        entropies.append(H)

    entropies = torch.stack(entropies, dim=0)  # (L, B, T)

    # SCR: entropy reduction rate over sliding window κ=4
    kappa = min(4, len(entropies) - 1)
    scr_scores = []
    for i in range(kappa, len(entropies)):
        h_prev = entropies[i - kappa]
        h_curr = entropies[i]
        rate = F.relu((h_prev - h_curr) / (h_prev + 1e-8))
        scr_scores.append(rate)

    clss = torch.stack(scr_scores, dim=0).mean(dim=0)  # (B, T)
    return clss


def compute_ctss(attentions: Tuple[torch.Tensor, ...],
                 num_visual_tokens: int,
                 layer_idx: int = -1) -> torch.Tensor:
    """
    Cross-Token Stability Score (CTSS) - Phase I diagnostic: cross-token visual attention consistency

    相邻token应关注相似的视觉区域，如果视觉注意力焦点突然分散，
    说明模型对当前token的视觉依据不稳定。

    VFD_t^l = (1/|H|) * sum_h JSD(A_t^{l,h}, A_{t-1}^{l,h})
    CTSS_t^l = 1 - VFD_t^l

    Args:
        attentions: 模型各层的attention weights
        num_visual_tokens: 视觉token的数量
        layer_idx: 使用哪一层的attention
    Returns:
        ctss: (batch, seq_len) 每个token的跨token视觉注意力稳定性
    """
    attn = attentions[layer_idx]  # (batch, num_heads, seq_len, seq_len)
    # 只取对视觉token的注意力
    visual_attn = attn[:, :, :, :num_visual_tokens]  # (batch, heads, seq, vis_tokens)
    # 归一化
    visual_attn = visual_attn / (visual_attn.sum(dim=-1, keepdim=True) + 1e-8)

    # 计算相邻token间的JSD
    attn_curr = visual_attn[:, :, 1:, :]  # (batch, heads, seq-1, vis)
    attn_prev = visual_attn[:, :, :-1, :]

    vfd = jsd(attn_curr, attn_prev, dim=-1)  # (batch, heads, seq-1)
    vfd = vfd.mean(dim=1)  # 对所有head取平均 -> (batch, seq-1)

    # padding第一个token
    pad = torch.zeros(vfd.shape[0], 1, device=vfd.device)
    vfd = torch.cat([pad, vfd], dim=1)  # (batch, seq)

    ctss = 1.0 - vfd
    return ctss


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


def compute_stability_scores(
    hidden_states: Tuple[torch.Tensor, ...],
    attentions: Tuple[torch.Tensor, ...],
    lm_head_weight: torch.Tensor,
    num_visual_tokens: int,
    lambda_clss: float = 1.0,
    lambda_ctss: float = 1.0,
    layer_range: Optional[Tuple[int, int]] = None,
    attn_layer_idx: int = -1,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Phase I diagnostic: compute per-token HIS and its semantic/visual components.

    Returns:
        dict with keys:
          'instability'   — combined HIS (geometric mean), shape (batch, seq)
          'his_sem'       — semantic instability 1-SCR, high = knowledge unstable
          'his_vis'       — visual instability 1-VGC, high = visual neglect
          'clss', 'ctss'  — raw stability scores (kept for compatibility)
    """
    clss = compute_clss(hidden_states, lm_head_weight, layer_range)
    scr = _normalize(clss)          # Semantic Convergence Rate ∈ [0,1]

    if attentions is not None and lambda_ctss > 0:
        ctss = compute_ctss(attentions, num_visual_tokens, attn_layer_idx)
        min_len = min(scr.shape[1], ctss.shape[1])
        scr = scr[:, :min_len]
        ctss = ctss[:, :min_len]
        vgc = _normalize(ctss)      # Visual Grounding Coherence ∈ [0,1]
    else:
        ctss = torch.zeros_like(clss)
        vgc = torch.zeros_like(scr)

    # Geometric-mean HIS (paper Eq. for combined instability)
    his_combined = 1.0 - (scr ** alpha) * (vgc ** (1.0 - alpha))

    # Type-decomposed components (used for cause-specific loss routing)
    his_sem = 1.0 - scr   # semantic instability: knowledge non-convergence
    his_vis = 1.0 - vgc   # visual instability: attention grounding loss

    return {
        'instability': his_combined,
        'his_sem': his_sem,
        'his_vis': his_vis,
        'clss': clss,
        'ctss': ctss,
        # legacy key
        'kss': scr,
    }
