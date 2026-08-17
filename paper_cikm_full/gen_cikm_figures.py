"""Prompt record for the conceptual figures used during paper preparation.

The local image-generation wrapper is intentionally not redistributed. These
prompts are documentation only and are unrelated to training and evaluation.
"""

PROMPT_WORKFLOW = """Create a polished academic paper figure for a CIKM paper about multimodal information credibility.

Canvas: wide horizontal figure, white background, clean vector/infographic style, no decorative gradients, no tiny text. Use only a few large English labels that are easy to read.

Show a left-to-right workflow with three stages:
1. "Generate" stage: an image-question input enters a vision-language model; the model emits a fluent answer with one risky object mention highlighted in red.
2. "Diagnose" stage: two internal evidence streams are shown: layer-wise semantic convergence (a blue stable curve and a red unstable curve) and visual grounding coherence (blue focused attention versus red drifting attention over image patches).
3. "Act" stage: high-risk claims are routed to verification, and HIS-guided preference optimization feeds back to the model.

Visual metaphors:
- Use a small realistic image thumbnail of a person on a motorcycle in a rural scene, because the paper's COCO examples include motorcycle/person grounding.
- Show risky generated objects as red underline marks, but do not include long sentences.
- Use blue for stable/supported evidence, red for unstable/unsupported evidence, and green for verified/mitigated outputs.
- Include one compact formula block with only "HIS = semantic instability + visual drift" in large readable type.

Do not invent numeric results. Do not include benchmark names or tables inside the image. Keep all text large, sparse, and print-readable."""


PROMPT_SIGNALS = """Create a single-column academic figure explaining internal signatures behind hallucination in a large vision-language model.

Canvas: portrait/square, white background, clean scientific style, suitable for LaTeX paper inclusion. Use only minimal, large, readable labels.

Top half: semantic convergence.
- Draw transformer layers as a horizontal stack.
- Plot two entropy trajectories: blue line decreases smoothly and is labeled "supported"; red dashed line remains high/oscillatory and is labeled "unsupported".
- Add a small bracket labeled "layer window".

Bottom half: visual grounding coherence.
- Show two image-patch grids.
- Left grid has a blue focused attention blob on the true object region and a stable centroid arrow.
- Right grid has red scattered attention blobs with centroid arrows moving between tokens.
- Label the contrast "focused" versus "drifting".

Avoid fake numerical values. Avoid dense paragraphs. Make it visually crisp and publication-ready."""

