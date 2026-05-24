"""
DTP-DPO Trainer: Diagnostic-Therapeutic Preference Direct Preference Optimization

Core loss function consists of three phases (inspired by clinical trial methodology):

1. L_treat (Phase I - Treatment): HIS-weighted DPO loss — uses Hallucination
   Instability Score (HIS) to weight per-token DPO loss, applying stronger
   preference optimization signal on tokens with high hallucination instability.

2. L_prevent (Phase II - Prevention): Visual pathway intervention
   (Visual Grounding Regularization) — contrasts response probabilities under
   original vs. masked images, penalizing the model's tendency to ignore visual
   information, reducing hallucination instability at its source.

3. L_stable (Phase III - Stabilization): Optimization stability constraint
   (Preference Anchoring Loss) — prevents preferred response probability from
   dropping during DPO training, stabilizing the optimization process.

Key design:
- Per-token DPO: each response token independently computes logratio and DPO loss,
  then weighted-averaged
- Instability weighting: loss_t multiplied by instability_weight_t, so unstable
  tokens contribute more to the loss
- labels中prompt部分设为IGNORE_INDEX(-100)，只在response tokens上计算logprobs

修复记录:
- v1: labels未mask prompt → margin≈0, l_vis=0
- v2: 修复prompt masking，但用sum-then-sigmoid → reward信号太弱，收敛慢
- v3(当前): 改为per-token DPO，instability作为token loss权重
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

IGNORE_INDEX = -100


def compute_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算每个token的log probability，返回logprobs和response mask。

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)，prompt部分为IGNORE_INDEX(-100)

    Returns:
        per_token_logps: (batch, seq_len) 每个token的logprob（prompt部分为0）
        resp_mask: (batch, seq_len) response token的mask（1=response, 0=prompt/pad）
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # 为gather准备index：IGNORE_INDEX位置用0代替（gather不接受负数index）
    gather_labels = labels.clone()
    gather_labels[labels == IGNORE_INDEX] = 0
    per_token_logps = torch.gather(log_probs, dim=-1,
                                   index=gather_labels.unsqueeze(-1)).squeeze(-1)
    # mask: response tokens = 1, prompt/IGNORE_INDEX tokens = 0
    resp_mask = (labels != IGNORE_INDEX).float()
    return per_token_logps * resp_mask, resp_mask


def _temperature_weights(scores: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Temperature-scaled normalization: sum = T (preserves DPO gradient scale)."""
    e = torch.exp(scores / tau)
    return e / (e.mean(dim=-1, keepdim=True) + 1e-8)


def stability_weighted_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    instability_weights: torch.Tensor,
    rejected_instability_weights: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_treat: HIS-weighted per-token DPO.

    chosen: weighted by his_sem (semantic instability drives treatment).
    rejected: weighted by rejected HIS if available, else uniform.
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    weighted_chosen = chosen_logratios * instability_weights * chosen_mask
    chosen_rewards = beta * weighted_chosen.sum(dim=-1)

    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    if rejected_instability_weights is not None:
        rej_len = rejected_mask.shape[1]
        rw = rejected_instability_weights
        if rw.shape[1] >= rej_len:
            rw = rw[:, :rej_len]
        else:
            rw = F.pad(rw, (0, rej_len - rw.shape[1]), value=1.0)
        weighted_rejected = rejected_logratios * rw * rejected_mask
    else:
        weighted_rejected = rejected_logratios * rejected_mask
    rejected_rewards = beta * weighted_rejected.sum(dim=-1)

    n_chosen = chosen_mask.sum(dim=-1).clamp(min=1.0)
    n_rejected = rejected_mask.sum(dim=-1).clamp(min=1.0)
    # Length-balanced sum:
    #   1. Compare average token reward so a 106-token rejected answer does not
    #      win simply because it has more terms than a 50-token chosen answer.
    #   2. Multiply by an effective length so the DPO gradient keeps the same
    #      order of magnitude as standard sum-DPO.
    #
    # Pure mean-DPO made the 7B signal collapse around logit ~= 0; pure sum-DPO
    # over-rewarded longer rejected responses. sqrt(n_c*n_r) is symmetric and
    # reduces to the standard sum scale when n_c == n_r.
    chosen_avg_rewards = chosen_rewards / n_chosen
    rejected_avg_rewards = rejected_rewards / n_rejected
    effective_len = torch.sqrt(n_chosen * n_rejected)
    logits = (chosen_avg_rewards - rejected_avg_rewards) * effective_len
    losses = (
        -F.logsigmoid(logits) * (1 - label_smoothing)
        - F.logsigmoid(-logits) * label_smoothing
    )
    return losses, chosen_rewards, rejected_rewards, logits, effective_len


def visual_contrastive_loss(
    policy_logps_orig: torch.Tensor,
    policy_logps_masked: torch.Tensor,
    ref_logps_orig: torch.Tensor,
    ref_logps_masked: torch.Tensor,
    resp_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DTP-DPO Phase II: Visual Pathway Intervention Loss (L_prevent)

    Contrasts response probabilities under original vs. masked images,
    forcing the model to rely on visual information rather than language priors.

    Pair: (original image, chosen) >> (masked image, chosen)

    Per-token logratio sum, no normalization.
    """
    orig_logratios = (policy_logps_orig - ref_logps_orig) * resp_mask
    masked_logratios = (policy_logps_masked - ref_logps_masked) * resp_mask
    logits = beta * (orig_logratios.sum(dim=-1) - masked_logratios.sum(dim=-1))
    return -F.logsigmoid(logits)


def standard_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Length-balanced response-level DPO used by source-adapted baselines."""
    chosen_logratios = (policy_chosen_logps - ref_chosen_logps) * chosen_mask
    rejected_logratios = (policy_rejected_logps - ref_rejected_logps) * rejected_mask
    chosen_rewards = beta * chosen_logratios.sum(dim=-1)
    rejected_rewards = beta * rejected_logratios.sum(dim=-1)

    n_chosen = chosen_mask.sum(dim=-1).clamp(min=1.0)
    n_rejected = rejected_mask.sum(dim=-1).clamp(min=1.0)
    effective_len = torch.sqrt(n_chosen * n_rejected)
    logits = ((chosen_rewards / n_chosen) - (rejected_rewards / n_rejected)) * effective_len
    losses = -F.logsigmoid(logits)
    return losses, chosen_rewards, rejected_rewards, logits, effective_len


def mask_pixel_values(pixel_values, mask_ratio=0.3, is_internvl=False):
    """对pixel_values进行掩码，用于L_prevent的Phase II损失计算。

    Qwen2.5-VL: (num_patches, patch_dim) — patch级别masking
    InternVL2.5: (num_tiles, 3, 448, 448) — tile级别空间masking
    """
    masked = pixel_values.clone()
    if is_internvl:
        # InternVL: mask random spatial patches within each tile
        # pixel_values shape: (num_tiles, 3, H, W) where H=W=448
        _, C, H, W = masked.shape
        patch_size = 32  # mask in 32x32 blocks
        num_h = H // patch_size
        num_w = W // patch_size
        num_patches_per_tile = num_h * num_w
        num_mask = int(num_patches_per_tile * mask_ratio)
        for t in range(masked.shape[0]):
            mask_indices = torch.randperm(num_patches_per_tile,
                                          device=masked.device)[:num_mask]
            for idx in mask_indices:
                r = idx // num_w
                c_idx = idx % num_w
                masked[t, :,
                       r * patch_size:(r + 1) * patch_size,
                       c_idx * patch_size:(c_idx + 1) * patch_size] = 0.0
    else:
        # Qwen: patch-level masking
        num_patches = masked.shape[0]
        num_mask = int(num_patches * mask_ratio)
        mask_indices = torch.randperm(num_patches, device=masked.device)[:num_mask]
        masked[mask_indices] = masked.mean()
    return masked


class DTPDPOTrainer:
    """
    DTP-DPO训练器

    总损失: L = L_treat + gamma1 * L_prevent + gamma2 * L_stable
    - Phase I  (L_treat):   HIS-weighted DPO for targeted hallucination correction
    - Phase II (L_prevent): Visual pathway intervention to reduce instability at source
    - Phase III(L_stable):  Optimization stability constraint
    """

    def __init__(self, model, ref_model, tokenizer, processor, args):
        self.args = args
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.beta = args.beta
        self.gamma1 = args.gamma_visual
        self.gamma2 = args.gamma_anchor
        self.anchor_value = args.anchor_value
        self.vis_threshold = getattr(args, 'vis_threshold', 0.5)   # hard gate for L_prevent
        self.stab_threshold = getattr(args, 'stab_threshold', 0.4) # hard gate for L_stable
        self.aux_gate_temp = getattr(args, 'aux_gate_temp', 0.05)
        self.aux_loss_cap = getattr(args, 'aux_loss_cap', 2.0)
        self.mask_ratio = args.mask_ratio  # 0.3
        self.mask_method = args.mask_method  # 'random'
        self.is_internvl = "internvl" in args.model_path.lower()
        # InternVL: set img_context_token_id on both models
        if self.is_internvl:
            img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            self._set_img_context_token_id(model, img_ctx_id)
            self._set_img_context_token_id(ref_model, img_ctx_id)

    def _set_img_context_token_id(self, model, token_id):
        """Set img_context_token_id on InternVL model (handles LoRA wrapper)."""
        # For PeftModel: model.base_model.model is the InternVLChatModel
        # For raw model: model itself is InternVLChatModel
        targets = [model]
        if hasattr(model, 'base_model'):
            targets.append(model.base_model)
            if hasattr(model.base_model, 'model'):
                targets.append(model.base_model.model)
        for t in targets:
            if hasattr(t, 'img_context_token_id'):
                t.img_context_token_id = token_id

    def get_logprobs(self, model, input_ids, attention_mask, labels,
                     pixel_values=None, image_grid_thw=None,
                     image_flags=None, **kwargs):
        """获取模型对给定输入的token级logprobs。

        返回的logprobs只包含response部分（prompt部分通过labels中的
        IGNORE_INDEX自动mask为0）。
        """
        if self.is_internvl:
            # InternVL forward requires: pixel_values, input_ids, attention_mask, image_flags
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            )
        else:
            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            # Qwen2.5-VL需要image_grid_thw
            if image_grid_thw is not None:
                forward_kwargs['image_grid_thw'] = image_grid_thw
            outputs = model(**forward_kwargs)

        logits = outputs.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        logps, resp_mask = compute_logprobs(logits, labels_shifted)
        return logps, resp_mask, outputs

    @torch.no_grad()
    def get_ref_logprobs(self, input_ids, attention_mask, labels,
                         pixel_values=None, image_grid_thw=None,
                         image_flags=None, **kwargs):
        self.ref_model.eval()
        logps, resp_mask, _ = self.get_logprobs(
            self.ref_model, input_ids, attention_mask, labels,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            image_flags=image_flags, **kwargs)
        return logps, resp_mask

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        计算DTP-DPO总损失
        L = L_treat + gamma1 * L_prevent + gamma2 * L_stable

        内存优化：ref model前向逐步执行并及时释放中间结果
        """
        source_method = getattr(self.args, 'source_method', 'dtp_dpo')
        chosen_pv = batch['chosen_pixel_values']
        chosen_grid = batch.get('chosen_image_grid_thw', None)
        chosen_flags = batch.get('chosen_image_flags', None)
        chosen_ids = batch['chosen_input_ids']
        chosen_attn_mask = batch['chosen_attention_mask']
        chosen_labels = batch['chosen_labels']

        rejected_pv = batch['rejected_pixel_values']
        rejected_grid = batch.get('rejected_image_grid_thw', None)
        rejected_flags = batch.get('rejected_image_flags', None)
        rejected_ids = batch['rejected_input_ids']
        rejected_attn_mask = batch['rejected_attention_mask']
        rejected_labels = batch['rejected_labels']

        instability_scores = batch['instability_scores']
        # Type-decomposed HIS components (may be absent in old data → fall back to combined)
        his_sem = batch.get('his_sem', instability_scores)
        his_vis = batch.get('his_vis', instability_scores)

        # === 1. ref model前向（no_grad，先算完释放显存给policy model） ===
        ref_chosen_logps, chosen_resp_mask = self.get_ref_logprobs(
            chosen_ids, chosen_attn_mask, chosen_labels,
            pixel_values=chosen_pv, image_grid_thw=chosen_grid,
            image_flags=chosen_flags)
        ref_rejected_logps, rejected_resp_mask = self.get_ref_logprobs(
            rejected_ids, rejected_attn_mask, rejected_labels,
            pixel_values=rejected_pv, image_grid_thw=rejected_grid,
            image_flags=rejected_flags)

        needs_masked_forward = source_method in {"dtp_dpo", "opa_source"}
        if needs_masked_forward:
            # ref masked forward (for L_prevent / OPA-source CoPO)
            masked_images = mask_pixel_values(chosen_pv, self.mask_ratio,
                                              is_internvl=self.is_internvl)
            ref_masked_logps, _ = self.get_ref_logprobs(
                chosen_ids, chosen_attn_mask, chosen_labels,
                pixel_values=masked_images, image_grid_thw=chosen_grid,
                image_flags=chosen_flags)
        else:
            masked_images = None
            ref_masked_logps = None

        # detach ref logps确保不占计算图
        ref_chosen_logps = ref_chosen_logps.detach()
        ref_rejected_logps = ref_rejected_logps.detach()
        if ref_masked_logps is not None:
            ref_masked_logps = ref_masked_logps.detach()
        chosen_resp_mask = chosen_resp_mask.detach()
        rejected_resp_mask = rejected_resp_mask.detach()
        torch.cuda.empty_cache()

        # === 2. policy model前向（需要梯度） ===
        policy_chosen_logps, _, _ = self.get_logprobs(
            self.model, chosen_ids, chosen_attn_mask, chosen_labels,
            pixel_values=chosen_pv, image_grid_thw=chosen_grid,
            image_flags=chosen_flags)
        policy_rejected_logps, _, _ = self.get_logprobs(
            self.model, rejected_ids, rejected_attn_mask, rejected_labels,
            pixel_values=rejected_pv, image_grid_thw=rejected_grid,
            image_flags=rejected_flags)
        if needs_masked_forward:
            policy_masked_logps, _, _ = self.get_logprobs(
                self.model, chosen_ids, chosen_attn_mask, chosen_labels,
                pixel_values=masked_images, image_grid_thw=chosen_grid,
                image_flags=chosen_flags)
        else:
            policy_masked_logps = None

        if source_method in {"standard_dpo", "hsa_source", "opa_source"}:
            return self._compute_source_adapted_loss(
                batch=batch,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                ref_masked_logps=ref_masked_logps,
                chosen_resp_mask=chosen_resp_mask,
                rejected_resp_mask=rejected_resp_mask,
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                policy_masked_logps=policy_masked_logps,
                source_method=source_method,
            )

        # === 3. Align HIS components to chosen response length ===
        chosen_resp_len = chosen_resp_mask.shape[1]

        def _align(scores, length):
            if scores.shape[1] >= length:
                return scores[:, :length]
            return F.pad(scores, (0, length - scores.shape[1]), value=1.0)

        weights     = _align(instability_scores, chosen_resp_len)
        weights_sem = _align(his_sem, chosen_resp_len)
        weights_vis = _align(his_vis, chosen_resp_len)

        # Temperature-scaled weights for L_treat (semantic instability drives DPO)
        treat_weights = _temperature_weights(weights_sem)

        # === 4. L_treat: semantic-instability-weighted DPO ===
        # rejected also gets HIS weighting (improvement B)
        rej_his_sem = batch.get('rejected_his_sem', None)
        rej_treat_weights = _temperature_weights(_align(rej_his_sem, rejected_resp_mask.shape[1])) \
            if rej_his_sem is not None else None

        (l_target,
         chosen_rewards,
         rejected_rewards,
         target_logits,
         target_effective_len) = stability_weighted_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            chosen_mask=chosen_resp_mask,
            rejected_mask=rejected_resp_mask,
            instability_weights=treat_weights,
            rejected_instability_weights=rej_treat_weights,
            beta=self.beta)

        # === 5. L_prevent: hard-gated visual fidelity (only when visual neglect diagnosed) ===
        vis_gate = torch.sigmoid(
            (weights_vis.mean(dim=-1) - self.vis_threshold) / self.aux_gate_temp
        )
        if vis_gate.sum() > 0:
            l_visual_raw = visual_contrastive_loss(
                policy_chosen_logps, policy_masked_logps,
                ref_chosen_logps, ref_masked_logps,
                resp_mask=chosen_resp_mask, beta=self.beta)
            l_visual = torch.clamp(l_visual_raw, max=self.aux_loss_cap) * vis_gate
        else:
            l_visual = torch.zeros(weights.shape[0], device=weights.device)

        # === 6. L_stable: hard-gated reward-margin constraint (only when high instability) ===
        stab_gate = torch.sigmoid(
            (weights.mean(dim=-1) - self.stab_threshold) / self.aux_gate_temp
        )
        l_anchor_raw = (
            -F.logsigmoid(chosen_rewards - self.anchor_value)
            - F.logsigmoid(-rejected_rewards + self.anchor_value)
        )
        l_anchor = torch.clamp(l_anchor_raw, max=self.aux_loss_cap) * stab_gate

        # === 7. 总损失 ===
        loss = (l_target.mean()
                + self.gamma1 * l_visual.mean()
                + self.gamma2 * l_anchor.mean())

        return {
            'loss': loss,
            'l_target': l_target.mean().item(),
            'l_visual': l_visual.mean().item(),
            'l_anchor': l_anchor.mean().item(),
            # Raw sum margin is useful for compatibility with previous logs, but
            # target_logit is the actual DPO margin used by L_treat.
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'target_logit': target_logits.mean().item(),
            'chosen_reward_avg': (chosen_rewards / chosen_resp_mask.sum(dim=-1).clamp(min=1.0)).mean().item(),
            'rejected_reward_avg': (rejected_rewards / rejected_resp_mask.sum(dim=-1).clamp(min=1.0)).mean().item(),
            'chosen_len': chosen_resp_mask.sum(dim=-1).mean().item(),
            'rejected_len': rejected_resp_mask.sum(dim=-1).mean().item(),
            'target_effective_len': target_effective_len.mean().item(),
            'vis_gate': vis_gate.mean().item(),
            'stab_gate': stab_gate.mean().item(),
        }

    def _compute_source_adapted_loss(
        self,
        batch: Dict,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        ref_masked_logps: torch.Tensor,
        chosen_resp_mask: torch.Tensor,
        rejected_resp_mask: torch.Tensor,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        policy_masked_logps: torch.Tensor,
        source_method: str,
    ) -> Dict[str, torch.Tensor]:
        """Source-adapted baselines under the same Qwen/DTP data pipeline.

        hsa_source: response-level DPO reweighted by object-label hallucination
        severity, matching HSA-DPO's severity-supervision role without using
        DTP's token-level HIS during optimization.

        opa_source: standard DPO plus ungated OPA-style CoPO and AncPO terms.
        """
        (l_dpo,
         chosen_rewards,
         rejected_rewards,
         target_logits,
         target_effective_len) = standard_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            chosen_mask=chosen_resp_mask,
            rejected_mask=rejected_resp_mask,
            beta=self.beta)

        if source_method == "hsa_source":
            severity = batch.get("hsa_severity")
            if severity is None:
                severity = torch.ones_like(l_dpo)
            severity = severity.to(l_dpo.device).view_as(l_dpo).clamp(min=0.0, max=1.0)
            weight = 1.0 + getattr(self.args, "hsa_alpha", 1.0) * severity
            l_target = l_dpo * weight
            l_visual = torch.zeros_like(l_dpo)
            l_anchor = torch.zeros_like(l_dpo)
            loss = l_target.mean()
        elif source_method == "opa_source":
            l_target = l_dpo
            l_visual = visual_contrastive_loss(
                policy_chosen_logps,
                policy_masked_logps,
                ref_chosen_logps,
                ref_masked_logps,
                resp_mask=chosen_resp_mask,
                beta=self.beta)
            l_anchor = (
                -F.logsigmoid(chosen_rewards - self.anchor_value)
                -F.logsigmoid(-rejected_rewards + self.anchor_value)
            )
            loss = (
                l_target.mean()
                + getattr(self.args, "opa_copo_coef", self.gamma1) * l_visual.mean()
                + getattr(self.args, "opa_anchor_coef", self.gamma2) * l_anchor.mean()
            )
        else:
            l_target = l_dpo
            l_visual = torch.zeros_like(l_dpo)
            l_anchor = torch.zeros_like(l_dpo)
            loss = l_target.mean()

        return {
            'loss': loss,
            'l_target': l_target.mean().item(),
            'l_visual': l_visual.mean().item(),
            'l_anchor': l_anchor.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'target_logit': target_logits.mean().item(),
            'chosen_reward_avg': (chosen_rewards / chosen_resp_mask.sum(dim=-1).clamp(min=1.0)).mean().item(),
            'rejected_reward_avg': (rejected_rewards / rejected_resp_mask.sum(dim=-1).clamp(min=1.0)).mean().item(),
            'chosen_len': chosen_resp_mask.sum(dim=-1).mean().item(),
            'rejected_len': rejected_resp_mask.sum(dim=-1).mean().item(),
            'target_effective_len': target_effective_len.mean().item(),
            'vis_gate': 0.0,
            'stab_gate': 0.0,
        }
