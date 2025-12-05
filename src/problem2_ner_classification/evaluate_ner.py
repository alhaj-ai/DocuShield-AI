"""Evaluates trained NER models: per-entity P/R/F1 by language, plus the
cross-lingual transfer comparison (EN-only model zero-shot on FR test vs.
the joint EN+FR model on the same FR test set).
"""
import json
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer

from prepare_data import load_split
from labels import ID2LABEL, EN_ONLY_SUPPORTED_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def predict_labels(model_dir, examples, max_length=192, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()

    all_true, all_pred = [], []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        words_batch = [e["words"] for e in batch]
        enc = tokenizer(words_batch, is_split_into_words=True, truncation=True,
                         max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        preds = logits.argmax(-1).cpu().numpy()

        for b_idx, e in enumerate(batch):
            word_ids = enc.word_ids(batch_index=b_idx)
            pred_labels = [None] * len(e["words"])
            for tok_idx, wid in enumerate(word_ids):
                if wid is not None and pred_labels[wid] is None:
                    pred_labels[wid] = ID2LABEL[int(preds[b_idx][tok_idx])]
            pred_labels = [l or "O" for l in pred_labels]
            all_true.append(e["labels"])
            all_pred.append(pred_labels)
    return all_true, all_pred


def evaluate_model(model_dir, split_name):
    examples = load_split(split_name)
    true_labels, pred_labels = predict_labels(model_dir, examples)
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return report


def main():
    joint_dir = ROOT / "models" / "problem2_xlmr_joint"
    enonly_dir = ROOT / "models" / "problem2_xlmr_en_only"

    results = {}
    print("Evaluating joint EN+FR model on English test set...")
    results["joint_on_en"] = evaluate_model(joint_dir, "test_en")
    print("Evaluating joint EN+FR model on French test set...")
    results["joint_on_fr"] = evaluate_model(joint_dir, "test_fr")
    print("Evaluating EN-only model on French test set (zero-shot cross-lingual transfer)...")
    results["en_only_zero_shot_on_fr"] = evaluate_model(enonly_dir, "test_fr")
    print("Evaluating EN-only model on English test set (in-language sanity check)...")
    results["en_only_on_en"] = evaluate_model(enonly_dir, "test_en")

    # Cross-lingual transfer summary for the shared DISEASE class
    transfer_summary = {
        "disease_f1_en_only_zero_shot_fr": results["en_only_zero_shot_on_fr"].get("DISEASE", {}).get("f1-score", 0.0),
        "disease_f1_joint_fr": results["joint_on_fr"].get("DISEASE", {}).get("f1-score", 0.0),
        "overall_f1_en_only_zero_shot_fr": results["en_only_zero_shot_on_fr"]["micro avg"]["f1-score"],
        "overall_f1_joint_fr": results["joint_on_fr"]["micro avg"]["f1-score"],
        "note": ("EN-only training data has zero examples of the FR-only families "
                 f"({sorted(set(['ANAT','PROC','LIVB','CHEM','PHYS','PHEN','GEOG','DEVI','OBJC']) - EN_ONLY_SUPPORTED_FAMILIES)}), "
                 "so its zero-shot recall on those classes is structurally ~0 regardless of cross-lingual "
                 "representation quality -- only the shared DISEASE class is a fair transfer comparison."),
    }
    results["cross_lingual_transfer_summary"] = transfer_summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "problem2_ner_eval.json").write_text(
        json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    print(json.dumps(transfer_summary, indent=2))
    print(f"Full report saved to {OUT_DIR / 'problem2_ner_eval.json'}")


if __name__ == "__main__":
    main()
