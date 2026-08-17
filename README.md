# HIS-Guard: Internal Stability Signals for Vision-Language Models

Official implementation of **Internal Stability Signals for Selective Verification and Hallucination Mitigation in Vision-Language Models** (CIKM 2026).

HIS-Guard uses internal model signals to identify unstable response tokens and combines them with a Diagnose-Treat-Prevent (DTP) preference-optimization objective. The code supports Qwen2.5-VL and InternVL2.5 model families.

## Repository layout

- train.py: main HIS-guided DTP-DPO training entry point.
- src/: data preparation, trainer, internal-stability analysis, and model utilities.
- scripts/: preference-data construction, filtering, merging, and training/evaluation examples.
- eval/: CHAIR, POPE, MME, MMHal, and unsupported-claim evaluation entry points.
- paper_cikm_full/gen_cikm_figures.py: figure-prompt record used during paper preparation. The local image-generation wrapper is not redistributed and is not required for training or evaluation.
- requirements.txt: Python dependencies.

## Method terminology

The paper uses HIS-Guard for the framework, HIS for the internal hallucination-instability diagnostic, and DTP-DPO for the training objective. In the implementation, clss is the semantic convergence score (top-K LogitLens entropy reduction), ctss is the visual-attention consistency score (cross-token JSD), and his_sem/his_vis are the corresponding semantic/visual signals.

The DTP objective is implemented as:

- L_treat: HIS-weighted per-token DPO.
- L_prevent: visual-fidelity gating using original and masked-image response probabilities.
- L_stable: combined-instability reward-margin anchoring.

## Requirements

The experiments were developed with Python 3.10, PyTorch 2.4.1, Transformers 4.49.0, PEFT, Accelerate, and CUDA. Install the pinned dependencies with:

    pip install -r requirements.txt

## External assets

The repository does not redistribute model weights, datasets, generated outputs, checkpoints, or machine-specific paths. Prepare the following assets locally:

- Qwen2.5-VL or InternVL2.5 base checkpoints.
- Visual Genome images and object annotations for preference-pair construction.
- COCO val2014 images/annotations and POPE question files for CHAIR/POPE.
- Optional MMHal-Bench and MME hallucination-subset data.

The example scripts assume:

    artifact/
    |- train.py
    |- src/  scripts/  eval/
    |- models/
    |   |- Qwen2.5-VL-3B/
    |   |- Qwen2.5-VL-7B/
    |   - InternVL2_5-2B/
    |- data/
    |   |- visual_genome/  coco/  POPE/
    |- checkpoints/
    - outputs/

## Preference data

Run:

    bash scripts/run_build_data_example.sh

Visual Genome object labels are used to construct supported/unsupported preference pairs. Token-level localization and weighting are computed by the internal HIS implementation; no external token-level hallucination mask is passed to the optimizer. The reported build produced 4,959 pairs; the exact count can vary with filtering, available annotations, and preprocessing versions.

## Training examples

The launchers are templates; update paths and batch settings for the available hardware.

    bash scripts/run_train_qwen3b_example.sh
    bash scripts/run_train_qwen7b_example.sh
    bash scripts/run_train_internvl2b_example.sh

The defaults correspond to the main settings used in the paper: learning rate 1e-6, DPO beta=0.1, LoRA rank/alpha 64/128, visual weight gamma_visual=0.2, anchor weight gamma_anchor=0.1, and mask ratio 0.3. The Qwen examples use three epochs; the InternVL2.5-2B example uses one epoch.

## Evaluation

For the standard CHAIR/POPE workflow:

    bash scripts/run_eval_example.sh models/Qwen2.5-VL-3B checkpoints/his_guard_qwen3b/epoch_3 outputs/qwen3b

Additional entry points in eval/ cover MME, MMHal, and unsupported-claim ranking. Their dataset paths must be configured for the local environment.

## Reproducibility scope

This repository contains the method implementation and evaluation pipeline. It intentionally excludes training logs, generated result files, trained LoRA checkpoints, model weights, and datasets, so the exact numerical results in the paper are not claimed to be reproduced by a checkout alone. Results depend on the externally supplied assets, hardware, and preprocessing versions.

## License

This project is released under the Apache License 2.0. See LICENSE.

