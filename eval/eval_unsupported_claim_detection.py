#!/usr/bin/env python3
"""
Intrinsic unsupported-claim detection for LVLM object mentions.

This evaluator treats generated object mentions as claims and asks whether
model-internal instability scores can rank unsupported claims above supported
ones. It is intentionally lightweight: it reuses generated responses plus
saved HIS / HIS_sem / HIS_vis arrays from preference-data JSON files, so the
metric pass does not need to reload the LVLM.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

DEFAULT_EXCLUDE_TERMS = {
    "a", "an", "the", "this", "that", "these", "those", "there", "here", "it", "its",
    "they", "their", "them", "he", "she", "his", "her", "we", "you", "i", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "1", "2", "3", "4",
    "5", "6", "7", "8", "9", "10", "is", "are", "was", "were", "be", "being", "been",
    "has", "have", "had", "do", "does", "did", "to", "of", "for", "from", "with", "without",
    "in", "on", "at", "by", "as", "and", "or", "but", "if", "then", "than", "while",
    "where", "when", "what", "which", "who", "how", "not", "no", "yes", "can", "could",
    "would", "should", "may", "might", "will", "shall", "also", "very", "more", "most",
    "some", "any", "several", "many", "few", "both", "each", "other", "another", "same",
    "left", "right", "top", "bottom", "front", "back", "side", "left side", "right side",
    "background", "foreground", "in the background", "near", "next", "under", "over", "above",
    "below", "around", "through", "across", "behind", "between", "toward", "towards",
    "visible", "shown", "seen", "appears", "appear", "sitting", "standing", "lying", "wearing",
    "holding", "looking", "made", "use", "used", "called", "including", "such", "part",
    "image", "photo", "picture", "scene", "view", "area", "element", "feature", "structure",
    "object", "item", "thing", "stuff", "style", "design", "color", "colored", "white", "black",
    "red", "blue", "green", "yellow", "orange", "brown", "gray", "grey", "purple", "pink",
    "large", "small", "big", "little", "tall", "short", "long", "wide", "modern", "urban",
    "commercial", "residential", "clean", "well", "maintained", "ha",
}


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("ses"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def image_id_from_path(path: str) -> int:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return int(stem)


def is_excluded_term(term: str, excluded: set) -> bool:
    if not term:
        return True
    if term in excluded:
        return True
    if term.isdigit():
        return True
    words = term.split()
    if all(w in excluded or w.isdigit() for w in words):
        return True
    return False


def load_aliases(alias_path: str, excluded: set) -> Dict[str, str]:
    aliases = {}
    if not alias_path or not os.path.exists(alias_path):
        return aliases
    with open(alias_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [normalize_phrase(p) for p in line.strip().split(",")]
            parts = [p for p in parts if p and not is_excluded_term(p, excluded)]
            if not parts:
                continue
            canonical = parts[0]
            for part in parts:
                aliases[part] = canonical
    return aliases


def canonicalize(name: str, aliases: Dict[str, str]) -> str:
    norm = normalize_phrase(name)
    return aliases.get(norm, norm)


def load_vg_objects(
    objects_path: str,
    aliases: Dict[str, str],
    min_object_freq: int,
    excluded: set,
) -> Tuple[Dict[int, set], set]:
    with open(objects_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_objects = {}
    freq = Counter()
    for item in data:
        names = set()
        for obj in item.get("objects", []):
            for name in obj.get("names", []):
                canon = canonicalize(name, aliases)
                if canon and not is_excluded_term(canon, excluded):
                    names.add(canon)
        image_objects[int(item["image_id"])] = names
        freq.update(names)

    vocabulary = {
        name for name, count in freq.items()
        if count >= min_object_freq and not is_excluded_term(name, excluded)
    }
    return image_objects, vocabulary


def word_tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text.lower())]


def build_word_scores(num_words: int, scores: Sequence[float]) -> List[float]:
    if num_words <= 0:
        return []
    if not scores:
        return [0.0] * num_words
    if len(scores) == num_words:
        return [float(x) for x in scores]
    mapped = []
    denom = max(1, num_words - 1)
    last = len(scores) - 1
    for i in range(num_words):
        idx = round((i / denom) * last) if last > 0 else 0
        mapped.append(float(scores[idx]))
    return mapped


def extract_mentions(
    text: str,
    vocabulary: set,
    aliases: Dict[str, str],
    max_ngram: int,
) -> List[Dict]:
    toks = word_tokens(text)
    words = [w for w, _, _ in toks]
    mentions = []
    used = set()

    for n in range(max_ngram, 0, -1):
        for i in range(0, len(words) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            phrase = " ".join(words[i:i + n])
            canon = aliases.get(normalize_phrase(phrase), normalize_phrase(phrase))
            if canon not in vocabulary:
                continue
            mentions.append({
                "canonical": canon,
                "surface": text[toks[i][1]:toks[i + n - 1][2]],
                "word_start": i,
                "word_end": i + n,
                "char_start": toks[i][1],
                "char_end": toks[i + n - 1][2],
            })
            used.update(range(i, i + n))

    mentions.sort(key=lambda x: (x["word_start"], x["word_end"]))
    return mentions


def mention_score(word_scores: Sequence[float], start: int, end: int, mode: str) -> float:
    vals = [float(v) for v in word_scores[start:end]]
    if not vals:
        return 0.0
    if mode == "mean":
        return sum(vals) / len(vals)
    if mode == "first":
        return vals[0]
    return max(vals)


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = average_ranks(scores)
    pos_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / max(1, tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def precision_recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> Tuple[float, float]:
    if not labels:
        return math.nan, math.nan
    k = min(k, len(labels))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = sum(labels[i] for i in order)
    precision = hits / k if k else math.nan
    recall = hits / sum(labels) if sum(labels) else math.nan
    return precision, recall


def safe_float(x: float) -> float:
    return None if isinstance(x, float) and math.isnan(x) else float(x)


def evaluate_field(records: List[Dict], field: str, ks: Iterable[int]) -> Dict:
    labels = [int(r["unsupported"]) for r in records]
    scores = [float(r["scores"][field]) for r in records]
    out = {
        "auroc": safe_float(auroc(labels, scores)),
        "auprc": safe_float(auprc(labels, scores)),
    }
    for k in ks:
        p, r = precision_recall_at_k(labels, scores, k)
        out[f"precision@{k}"] = safe_float(p)
        out[f"recall@{k}"] = safe_float(r)
    return out


def format_metric(value) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}"


def markdown_table(results: Dict[str, Dict], ks: Sequence[int]) -> str:
    headers = ["Method", "AUROC", "AUPRC"] + [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks]
    rows = ["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method, metrics in results.items():
        row = [
            method,
            format_metric(metrics["auroc"]),
            format_metric(metrics["auprc"]),
        ]
        row.extend(format_metric(metrics[f"precision@{k}"]) for k in ks)
        row.extend(format_metric(metrics[f"recall@{k}"]) for k in ks)
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/preference_data_filtered.json")
    parser.add_argument("--vg_objects_file", default="data/visual_genome/objects.json")
    parser.add_argument("--object_alias_file", default="data/visual_genome/object_alias.txt")
    parser.add_argument("--response_field", default="rejected")
    parser.add_argument("--score_fields", nargs="+",
                        default=["instability_scores", "his_sem", "his_vis"])
    parser.add_argument("--score_names", nargs="+", default=["HIS", "HIS_sem", "HIS_vis"])
    parser.add_argument("--min_object_freq", type=int, default=5)
    parser.add_argument("--exclude_terms_file", default=None,
                        help="Optional newline/comma separated object terms to exclude from the mention vocabulary.")
    parser.add_argument("--max_ngram", type=int, default=4)
    parser.add_argument("--score_pooling", choices=["max", "mean", "first"], default="max")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--output_file", default="outputs/unsupported_claim_detection.json")
    parser.add_argument("--mention_output_file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(base, args.data_path)
    objects_path = args.vg_objects_file if os.path.isabs(args.vg_objects_file) else os.path.join(base, args.vg_objects_file)
    alias_path = args.object_alias_file if os.path.isabs(args.object_alias_file) else os.path.join(base, args.object_alias_file)

    if len(args.score_names) != len(args.score_fields):
        raise ValueError("--score_names must have the same length as --score_fields")

    excluded = set(DEFAULT_EXCLUDE_TERMS)
    if args.exclude_terms_file:
        exclude_path = args.exclude_terms_file if os.path.isabs(args.exclude_terms_file) else os.path.join(base, args.exclude_terms_file)
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                for part in line.strip().split(","):
                    term = normalize_phrase(part)
                    if term:
                        excluded.add(term)

    aliases = load_aliases(alias_path, excluded)
    image_objects, vocabulary = load_vg_objects(objects_path, aliases, args.min_object_freq, excluded)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    records = []
    skipped = defaultdict(int)
    for item_idx, item in enumerate(data):
        text = item.get(args.response_field, "")
        if not text:
            skipped["empty_response"] += 1
            continue
        try:
            image_id = image_id_from_path(item["image_path"])
        except Exception:
            skipped["bad_image_id"] += 1
            continue
        gt_objects = image_objects.get(image_id)
        if gt_objects is None:
            skipped["missing_gt"] += 1
            continue

        mentions = extract_mentions(text, vocabulary, aliases, args.max_ngram)
        if not mentions:
            skipped["no_mentions"] += 1
            continue

        words = word_tokens(text)
        per_field_word_scores = {}
        missing_score = False
        for field, name in zip(args.score_fields, args.score_names):
            if field not in item:
                missing_score = True
                skipped[f"missing_{field}"] += 1
                break
            per_field_word_scores[name] = build_word_scores(len(words), item[field])
        if missing_score:
            continue

        for mention in mentions:
            scores = {
                name: mention_score(
                    word_scores,
                    mention["word_start"],
                    mention["word_end"],
                    args.score_pooling,
                )
                for name, word_scores in per_field_word_scores.items()
            }
            records.append({
                "item_index": item_idx,
                "image_id": image_id,
                "surface": mention["surface"],
                "canonical": mention["canonical"],
                "supported": mention["canonical"] in gt_objects,
                "unsupported": int(mention["canonical"] not in gt_objects),
                "scores": scores,
            })

    if not records:
        raise RuntimeError("No object mentions were extracted; check vocabulary and input data.")

    results = {
        name: evaluate_field(records, name, args.ks)
        for name in args.score_names
    }

    summary = {
        "data_path": data_path,
        "num_samples": len(data),
        "num_mentions": len(records),
        "num_unsupported": sum(r["unsupported"] for r in records),
        "unsupported_rate": sum(r["unsupported"] for r in records) / len(records),
        "min_object_freq": args.min_object_freq,
            "score_pooling": args.score_pooling,
            "excluded_terms": len(excluded),
        "skipped": dict(skipped),
        "metrics": results,
        "markdown_table": markdown_table(results, args.ks),
    }

    output_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(base, args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.mention_output_file:
        mention_path = args.mention_output_file if os.path.isabs(args.mention_output_file) else os.path.join(base, args.mention_output_file)
        os.makedirs(os.path.dirname(mention_path), exist_ok=True)
        with open(mention_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Mentions: {summary['num_mentions']} | unsupported: {summary['num_unsupported']} ({100 * summary['unsupported_rate']:.2f}%)")
    print(summary["markdown_table"])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
