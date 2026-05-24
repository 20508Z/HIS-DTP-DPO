"""Merge sharded CHAIR generations and recompute CHAIR metrics."""

import argparse
import json
import os

from eval_chair import compute_chair


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_ann_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--generation_files", nargs="+", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    generations = []
    seen = set()
    for path in args.generation_files:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            for item in json.load(f):
                image_id = item["image_id"]
                if image_id in seen:
                    continue
                seen.add(image_id)
                generations.append(item)

    with open(args.coco_ann_file) as f:
        coco_ann = json.load(f)

    metrics = compute_chair(generations, coco_ann)
    metrics["num_images"] = len(generations)

    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Merged {len(generations)} images")
    print(f"CHAIR_s: {metrics['CHAIR_s']:.2f}")
    print(f"CHAIR_i: {metrics['CHAIR_i']:.2f}")


if __name__ == "__main__":
    main()
