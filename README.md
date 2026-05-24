# Anonymous Artifact for HIS-Guard

This anonymous artifact contains source code for the submitted paper on internal stability signals for selective verification and hallucination mitigation in vision-language models.

The package intentionally excludes run logs, checkpoints, local model copies, datasets, generated results, paper build products, and any machine-specific paths. It is intended for code review and reproduction by reviewers who prepare the required public datasets and base models.

## Contents

- `train.py`: main training entry point for HIS-guided preference optimization.
- `src/`: dataset, trainer, stability-diagnostic, and model utility code.
- `scripts/`: preference-data construction and lightweight launch examples.
- `eval/`: CHAIR, POPE, MME, MMHal, and unsupported-claim ranking evaluation scripts.
- `requirements.txt`: Python dependency list used by the implementation.

## Paper-Code Terminology

The paper uses `HIS-Guard` for the overall framework, `HIS` for the internal
instability diagnostic, and `DTP-DPO` for the training objective. Some code
symbols preserve earlier names for compatibility: `clss` corresponds to the
semantic convergence score, and `ctss` corresponds to visual grounding
coherence computed from cross-token image-attention distribution consistency.

## Required External Assets

Reviewers should provide these assets locally:

- Qwen2.5-VL or InternVL2.5 base model checkpoints.
- Visual Genome images and object annotations for preference-pair construction.
- COCO val2014 images and annotations for CHAIR and POPE evaluation.
- POPE random, popular, and adversarial question files.
- Optional MMHal-Bench and MME hallucination-subset data.

No dataset files or model weights are included in this anonymous package.

## Expected Directory Layout

The scripts assume a layout like:

```text
artifact/
├── train.py
├── src/
├── scripts/
├── eval/
├── models/
│   ├── Qwen2.5-VL-3B/
│   ├── Qwen2.5-VL-7B/
│   └── InternVL2_5-2B/
├── data/
│   ├── visual_genome/
│   ├── coco/
│   └── POPE/
├── checkpoints/
└── outputs/
```

`models/`, `data/`, `checkpoints/`, and `outputs/` are local runtime directories and are not part of the submission artifact.

## Example Workflow

Build preference data:

```bash
bash scripts/run_build_data_example.sh
```

Train HIS-Guard on Qwen2.5-VL-3B:

```bash
bash scripts/run_train_qwen3b_example.sh
```

Evaluate POPE and CHAIR:

```bash
bash scripts/run_eval_example.sh models/Qwen2.5-VL-3B checkpoints/his_guard_qwen3b/epoch_3 outputs/qwen3b
```

The examples are templates. Paths may be changed to match the reviewer's local storage.

## Reproducibility Scope

This package is intended to reproduce the method implementation and evaluation
pipeline once reviewers provide the external datasets and base checkpoints.
It does not include experiment logs, generated result files, or trained LoRA
checkpoints. The main experiments in the paper used Python 3.10, PyTorch 2.4.1,
Transformers 4.49.0, PEFT/Accelerate, and 4 RTX 4090 GPUs.

## Anonymity Notes

This artifact is anonymized for double-blind review. It excludes identity metadata, personal accounts, local absolute paths, previous run outputs, logs, checkpoints, and generated result files.
