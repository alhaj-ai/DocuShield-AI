"""Evaluate the trained LayoutLMv3 extraction pipeline on the held-out
synthetic test set: field-level accuracy, table extraction accuracy,
per-document processing time, and success/failure examples.
"""
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from layout_extract import LayoutExtractor

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic" / "problem1"
OUT_DIR = ROOT / "outputs"

SINGLE_FIELDS = ["lab_name", "lab_address", "patient_name", "dob", "patient_id",
                  "report_date", "physician_name", "physician_id"]
TABLE_COLS = ["test_name", "value", "unit", "reference_range", "flag"]


def norm(s: str) -> str:
    return " ".join(s.strip().casefold().split())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def eval_document(extractor: LayoutExtractor, entry: dict) -> dict:
    image = Image.open(DATA_DIR / entry["image"]).convert("RGB")
    gt = json.loads((DATA_DIR / entry["label"]).read_text())

    t0 = time.perf_counter()
    pred = extractor.extract(image)
    elapsed = time.perf_counter() - t0

    field_results = {}
    for key in SINGLE_FIELDS:
        gt_text = gt.get("fields", {}).get(key, {}).get("text", "")
        if not gt_text:
            continue
        pred_text = pred["fields"].get(key, {}).get("text", "")
        field_results[key] = {
            "gt": gt_text, "pred": pred_text,
            "exact_match": norm(gt_text) == norm(pred_text),
            "similarity": sim(gt_text, pred_text),
        }

    gt_rows = gt.get("table", {}).get("rows", [])
    pred_rows = pred["table"]["rows"]
    row_results = []
    for i, gt_row in enumerate(gt_rows):
        pred_row = pred_rows[i] if i < len(pred_rows) else {c: "" for c in TABLE_COLS}
        cell_matches = {c: norm(gt_row.get(c, "")) == norm(pred_row.get(c, "")) for c in TABLE_COLS}
        row_results.append({"gt": gt_row, "pred": pred_row, "cell_matches": cell_matches,
                             "full_row_match": all(cell_matches.values())})

    field_acc = sum(r["exact_match"] for r in field_results.values()) / max(1, len(field_results))
    row_acc = sum(r["full_row_match"] for r in row_results) / max(1, len(row_results))
    cell_acc = (sum(sum(r["cell_matches"].values()) for r in row_results) /
                max(1, sum(len(r["cell_matches"]) for r in row_results)))

    return {
        "doc_id": entry["doc_id"], "language": entry["language"], "augmented": entry["augmented"],
        "elapsed_seconds": elapsed,
        "field_results": field_results, "row_results": row_results,
        "field_accuracy": field_acc, "row_accuracy": row_acc, "cell_accuracy": cell_acc,
        "n_gt_rows": len(gt_rows), "n_pred_rows": len(pred_rows),
    }


def main():
    manifest = json.loads((DATA_DIR / "manifest_test.json").read_text())
    extractor = LayoutExtractor()

    results = [eval_document(extractor, entry) for entry in manifest]

    per_field = {}
    for key in SINGLE_FIELDS:
        vals = [r["field_results"][key]["exact_match"] for r in results if key in r["field_results"]]
        sims = [r["field_results"][key]["similarity"] for r in results if key in r["field_results"]]
        if vals:
            per_field[key] = {"exact_match_accuracy": sum(vals) / len(vals),
                               "avg_similarity": sum(sims) / len(sims), "n": len(vals)}

    summary = {
        "n_documents": len(results),
        "overall_field_accuracy": sum(r["field_accuracy"] for r in results) / len(results),
        "overall_row_accuracy": sum(r["row_accuracy"] for r in results) / len(results),
        "overall_cell_accuracy": sum(r["cell_accuracy"] for r in results) / len(results),
        "avg_processing_time_seconds": sum(r["elapsed_seconds"] for r in results) / len(results),
        "per_field": per_field,
        "en_field_accuracy": sum(r["field_accuracy"] for r in results if r["language"] == "en") /
                              max(1, sum(1 for r in results if r["language"] == "en")),
        "fr_field_accuracy": sum(r["field_accuracy"] for r in results if r["language"] == "fr") /
                              max(1, sum(1 for r in results if r["language"] == "fr")),
        "augmented_field_accuracy": sum(r["field_accuracy"] for r in results if r["augmented"]) /
                                     max(1, sum(1 for r in results if r["augmented"])),
        "clean_field_accuracy": sum(r["field_accuracy"] for r in results if not r["augmented"]) /
                                 max(1, sum(1 for r in results if not r["augmented"])),
    }

    ranked = sorted(results, key=lambda r: r["field_accuracy"])
    examples = {"failures": ranked[:5], "successes": ranked[-5:]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "problem1_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "problem1_eval_full.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / "problem1_eval_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
