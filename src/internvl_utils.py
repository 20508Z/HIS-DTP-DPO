"""
InternVL2.5-2B image processing and input construction utilities.
Adapted from the official InternVL2.5-2B README.
"""

import sys
import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def _get_conv_template(model_path):
    """Import get_conv_template from InternVL model directory."""
    model_dir = os.path.abspath(model_path)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from conversation import get_conv_template
    return get_conv_template


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """Split image into tiles based on aspect ratio. max_num=6 to limit VRAM."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_for_internvl(image, input_size=448, max_num=6):
    """Process a PIL Image into pixel_values tensor for InternVL2.5-2B.

    Args:
        image: PIL Image (already opened)
        input_size: tile size (448 for InternVL2.5)
        max_num: max tiles per image (6 to limit VRAM for training)
    Returns:
        pixel_values: [num_tiles, 3, 448, 448]
        num_patches: int, number of tiles
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values, pixel_values.shape[0]


def build_internvl_input_ids(tokenizer, query, response=None,
                             num_patches=1, num_image_token=256,
                             template_name='internvl2_5',
                             system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
                             model_path=None, max_length=2048):
    """Build input_ids for InternVL2.5-2B with proper image token placement.

    Returns:
        dict with input_ids, attention_mask, labels (if response given), prompt_len
    """
    get_conv_template = _get_conv_template(model_path)

    # Build image token string: <img><IMG_CONTEXT>*256*num_patches</img>
    image_tokens = (IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * num_image_token * num_patches
                    + IMG_END_TOKEN)

    question = image_tokens + '\n' + query

    # Prompt-only for measuring prompt length
    template = get_conv_template(template_name)
    template.system_message = system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    prompt_text = template.get_prompt()

    prompt_inputs = tokenizer(prompt_text, return_tensors='pt',
                              padding=False, truncation=True,
                              max_length=max_length)
    prompt_len = prompt_inputs['input_ids'].shape[1]

    if response is None:
        return {
            'input_ids': prompt_inputs['input_ids'],
            'attention_mask': prompt_inputs['attention_mask'],
            'prompt_len': prompt_len,
        }

    # Full input (prompt + response + sep)
    template_full = get_conv_template(template_name)
    template_full.system_message = system_message
    template_full.append_message(template_full.roles[0], question)
    template_full.append_message(template_full.roles[1], response)
    full_text = template_full.get_prompt()

    full_inputs = tokenizer(full_text, return_tensors='pt',
                            padding=False, truncation=True,
                            max_length=max_length)

    labels = full_inputs['input_ids'].clone()
    labels[0, :prompt_len] = -100

    return {
        'input_ids': full_inputs['input_ids'],
        'attention_mask': full_inputs['attention_mask'],
        'labels': labels,
        'prompt_len': prompt_len,
    }


def build_image_flags(input_ids, img_context_token_id):
    """Build image_flags tensor for InternVL forward pass.

    Args:
        input_ids: [batch, seq_len]
        img_context_token_id: token id for <IMG_CONTEXT>
    Returns:
        image_flags: [num_image_patches, 1] — one flag per ViT tile
    """
    # Count total IMG_CONTEXT tokens to determine number of ViT patches needed
    # Each tile contributes num_image_token (256) context tokens
    # image_flags shape should be [num_tiles_in_batch, 1], all 1s for real images
    # This is passed to model.forward() which does vit_embeds[image_flags == 1]
    pass  # handled inline in forward
