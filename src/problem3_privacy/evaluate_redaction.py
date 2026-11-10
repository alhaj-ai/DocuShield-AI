"""Measures precision/recall per PII type: does detect_all_pii() find the
same spans we planted in build_eval_set.py?

A predicted span counts as correct if it overlaps a ground-truth span of
the same type (we don't require the exact same character boundaries,
since e.g. detecting "Dr. Jean Martin" vs "Jean Martin" should both count
as finding the name).
"""
import json
from collections import defaultdict
from pathlib import Path

from build_eval_set import load_eval_set
from pii_detectors import detect_all_pii

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def spans_overlap(a, b):
    return a["start"] < b["end"] and a["end"] > b["start"]


def score_example(gt_spans, pred_spans):
    """Returns per-type true positive / false positive / false negative counts
    for one document."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_gt = [False] * len(gt_spans)

    for pred in pred_spans:
        found_match = False
        for i, gt in enumerate(gt_spans):
            if matched_gt[i]:
                continue
            if gt["type"] == pred["type"] and spans_overlap(gt, pred):
                matched_gt[i] = True
                found_match = True
                counts[pred["type"]]["tp"] += 1
                break
        if not found_match:
            counts[pred["type"]]["fp"] += 1

    for i, gt in enumerate(gt_spans):
        if not matched_gt[i]:
            counts[gt["type"]]["fn"] += 1

    return counts


def main():
    examples = load_eval_set()

    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for ex in examples:
        pred_spans = detect_all_pii(ex["text"])
        counts = score_example(ex["pii_spans"], pred_spans)
        for pii_type, c in counts.items():
            totals[pii_type]["tp"] += c["tp"]
            totals[pii_type]["fp"] += c["fp"]
            totals[pii_type]["fn"] += c["fn"]

    report = {}
    for pii_type, c in totals.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[pii_type] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "problem3_pii_detection_eval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'TYPE':12s} {'P':>6s} {'R':>6s} {'F1':>6s}   TP  FP  FN")
    for pii_type, m in sorted(report.items()):
        print(f"{pii_type:12s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}  "
              f"{m['tp']:3d} {m['fp']:3d} {m['fn']:3d}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
