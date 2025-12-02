"""The crucial Problem 3 question: does redacting PII break the Problem 2
models? We take Problem 1's synthetic lab reports (real documents, known to
be "healthcare"), redact them with each strategy, and check:

1. Does the Problem 2 domain classifier still say "healthcare"?
2. Does the Problem 2 NER model still find the same clinical entities
   (test names, chemicals, etc.) it found in the original, un-redacted text?

For (2) there's no hand-labeled ground truth for this text, so we use the
model's own predictions on the ORIGINAL text as a reference point, and see
how much of that is still recovered after redaction. This is a standard
practical trick when you don't have gold labels: it measures "how much did
redaction change what the model sees," not "is the model correct."
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "problem2_ner_classification"))

from pii_detectors import detect_all_pii, guess_language
from redaction import redact_text
from classify_and_extract import classify_document, extract_entities

ROOT = Path(__file__).resolve().parents[2]
PROBLEM1_CACHE = ROOT / "data" / "processed" / "problem1"
OUT_DIR = ROOT / "outputs"

MODES = ["strict", "typed", "surrogate"]


def load_problem1_documents():
    docs = []
    for split in ["test", "train"]:
        cache_file = PROBLEM1_CACHE / f"{split}_ocr_labels.json"
        if not cache_file.exists():
            continue
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        for item in cache.values():
            text = " ".join(item["words"])
            if len(text) > 40:
                docs.append({"text": text, "language": item["language"]})
    return docs


def entity_set(entities):
    # Compare entities as a simple bag of (family, normalized text) pairs,
    # so small position shifts after redaction don't matter.
    return {(e["family"], e["text"].strip().lower()) for e in entities}


def score_entity_overlap(reference_entities, predicted_entities):
    ref_set = entity_set(reference_entities)
    pred_set = entity_set(predicted_entities)
    if not ref_set:
        return 1.0  # nothing to lose
    kept = ref_set & pred_set
    return len(kept) / len(ref_set)  # fraction of original entities still found


def main():
    docs = load_problem1_documents()
    print(f"Evaluating downstream utility on {len(docs)} Problem 1 documents...")

    # classifier accuracy: all these documents are genuinely healthcare
    # documents, so "healthcare" is the correct label regardless of redaction
    classifier_correct = {"original": 0, **{m: 0 for m in MODES}}
    entity_recall = {m: [] for m in MODES}

    for doc in docs:
        text = doc["text"]
        language = doc["language"]

        original_pred = classify_document(text)
        if original_pred["label"] == "healthcare":
            classifier_correct["original"] += 1
        original_entities = extract_entities(text)

        pii_spans = detect_all_pii(text)
        for mode in MODES:
            redacted_text, _ = redact_text(text, pii_spans, mode=mode, language=language)

            pred = classify_document(redacted_text)
            if pred["label"] == "healthcare":
                classifier_correct[mode] += 1

            redacted_entities = extract_entities(redacted_text)
            entity_recall[mode].append(score_entity_overlap(original_entities, redacted_entities))

    n = len(docs)
    report = {
        "n_documents": n,
        "classifier_accuracy": {
            "original": classifier_correct["original"] / n,
            **{m: classifier_correct[m] / n for m in MODES},
        },
        "ner_entity_recall_vs_original": {
            m: sum(entity_recall[m]) / len(entity_recall[m]) for m in MODES
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "problem3_downstream_utility.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
