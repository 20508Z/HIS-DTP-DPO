"""
Filter preference data to keep high-quality pairs for DTP-DPO training.

Filtering criteria:
1. len_ratio in [0.15, 0.75]: chosen should be meaningfully shorter than rejected
   (removes hallucinations) but not trivially short
2. chosen_len >= 20 words: chosen must have enough content
3. his_mean > 0.45: keep samples with meaningful instability (DTP-DPO's target)
4. overlap_ratio < 0.7: chosen and rejected must differ substantially
"""

import json
import numpy as np
import argparse

def char_overlap_ratio(chosen, rejected):
    min_len = min(len(chosen), len(rejected))
    shared = sum(1 for i in range(min_len) if chosen[i] == rejected[i])
    return shared / max(len(rejected), 1)

def filter_data(data, len_ratio_min=0.15, len_ratio_max=0.75,
                min_chosen_words=20, his_min=0.45, overlap_max=0.7):
    kept = []
    stats = {"total": len(data), "too_similar_len": 0, "too_short_chosen": 0,
             "low_his": 0, "high_overlap": 0, "kept": 0}

    for d in data:
        c_words = len(d['chosen'].split())
        r_words = len(d['rejected'].split())
        len_ratio = c_words / max(r_words, 1)
        his_mean = float(np.mean(d['instability_scores']))
        overlap = char_overlap_ratio(d['chosen'], d['rejected'])

        if not (len_ratio_min <= len_ratio <= len_ratio_max):
            stats["too_similar_len"] += 1
            continue
        if c_words < min_chosen_words:
            stats["too_short_chosen"] += 1
            continue
        if his_mean < his_min:
            stats["low_his"] += 1
            continue
        if overlap > overlap_max:
            stats["high_overlap"] += 1
            continue

        kept.append(d)

    stats["kept"] = len(kept)
    return kept, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/preference_data.json")
    parser.add_argument("--output", default="data/preference_data_filtered.json")
    parser.add_argument("--len_ratio_min", type=float, default=0.15)
    parser.add_argument("--len_ratio_max", type=float, default=0.75)
    parser.add_argument("--min_chosen_words", type=int, default=20)
    parser.add_argument("--his_min", type=float, default=0.45)
    parser.add_argument("--overlap_max", type=float, default=0.7)
    args = parser.parse_args()

    data = json.load(open(args.input))
    kept, stats = filter_data(
        data,
        len_ratio_min=args.len_ratio_min,
        len_ratio_max=args.len_ratio_max,
        min_chosen_words=args.min_chosen_words,
        his_min=args.his_min,
        overlap_max=args.overlap_max,
    )

    print(f"Total: {stats['total']}")
    print(f"Removed (len_ratio out of [{args.len_ratio_min},{args.len_ratio_max}]): {stats['too_similar_len']}")
    print(f"Removed (chosen < {args.min_chosen_words} words): {stats['too_short_chosen']}")
    print(f"Removed (HIS < {args.his_min}): {stats['low_his']}")
    print(f"Removed (overlap > {args.overlap_max}): {stats['high_overlap']}")
    print(f"Kept: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")

    with open(args.output, 'w') as f:
        json.dump(kept, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
