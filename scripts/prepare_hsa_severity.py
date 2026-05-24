#!/usr/bin/env python3
"""Add object-label hallucination severity for HSA-source adaptation.

The original preference data stores image paths and rejected responses, but not
response-level hallucination severity. This script reconstructs a lightweight
severity signal from Visual Genome object labels:

    hsa_severity = hallucinated_sentence_count / sentence_count

This keeps HSA-source comparisons separate from DTP-DPO's internal HIS signal.
"""

import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--vg_objects_file", default="data/visual_genome/objects.json")
    p.add_argument("--output", required=True)
    return p.parse_args()


def normalize_name(name):
    return name.lower().strip()


def load_vg_object_map(path):
    with open(path, encoding="utf-8") as f:
        vg_data = json.load(f)
    mapping = {}
    for item in vg_data:
        names = set()
        for obj in item.get("objects", []):
            for name in obj.get("names", []):
                n = normalize_name(name)
                if n:
                    names.add(n)
                    names.update(part for part in n.split() if part)
                    if n.endswith("s") and len(n) > 3:
                        names.add(n[:-1])
        mapping[str(item["image_id"])] = names
    return mapping


def image_id_from_path(image_path):
    return Path(image_path).stem


def sentence_severity(text, gt_objects):
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sentences:
        return 1.0
    hallucinated = 0
    for sent in sentences:
        compact_words = set(re.findall(r"[a-zA-Z]+", sent.lower()))
        has_overlap = bool(compact_words & gt_objects)
        if not has_overlap and len(compact_words) > 3:
            hallucinated += 1
    return hallucinated / max(len(sentences), 1)


def main():
    args = parse_args()
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    vg_objects = load_vg_object_map(args.vg_objects_file)

    missing = 0
    severities = []
    for item in tqdm(data, desc="HSA severity"):
        img_id = image_id_from_path(item["image_path"])
        gt = vg_objects.get(img_id)
        if not gt:
            missing += 1
            severity = 1.0
        else:
            severity = sentence_severity(item.get("rejected", ""), gt)
        item["hsa_severity"] = float(severity)
        severities.append(severity)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    avg = sum(severities) / max(len(severities), 1)
    print(f"Saved {len(data)} rows to {args.output}")
    print(f"Missing VG labels: {missing}")
    print(f"Mean hsa_severity: {avg:.4f}")


if __name__ == "__main__":
    main()
