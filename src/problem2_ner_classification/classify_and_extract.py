"""Combines the Task B domain classifier and the Task A NER model into one
simple function: given a document's text, return its predicted domain plus
the entities found in it, as JSON. Demonstrates the output format the spec
asks for, on a handful of the held-out test documents.
"""
import json
from pathlib import Path

import torch
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification,
                           AutoTokenizer)

from prepare_domain_data import load_domain_split
from build_domain_classifier import ID2LABEL as DOMAIN_ID2LABEL
from labels import ID2LABEL as NER_ID2LABEL

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = ROOT / "models" / "problem2_domain_classifier"
NER_DIR = ROOT / "models" / "problem2_xlmr_joint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load everything once, at import time, so we don't reload models per call.
_clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
_clf_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR).to(DEVICE).eval()

_ner_tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
_ner_model = AutoModelForTokenClassification.from_pretrained(NER_DIR).to(DEVICE).eval()


def classify_document(text, max_length=256):
    inputs = _clf_tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _clf_model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(probs.argmax())
    return {"label": DOMAIN_ID2LABEL[pred_id], "confidence": float(probs[pred_id])}


def extract_entities(text, max_length=256):
    # NER works on a list of pre-split words, so split on whitespace first.
    words = text.split()
    if not words:
        return []

    inputs = _ner_tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _ner_model(**inputs).logits[0]
    preds = logits.argmax(-1).cpu().numpy()
    word_ids = inputs.word_ids(batch_index=0)

    # Take the first subtoken's prediction for each word.
    word_labels = [None] * len(words)
    for token_idx, wid in enumerate(word_ids):
        if wid is not None and word_labels[wid] is None:
            word_labels[wid] = NER_ID2LABEL[int(preds[token_idx])]
    word_labels = [l or "O" for l in word_labels]

    # Group consecutive B-/I- words of the same type into one entity.
    entities = []
    current_words, current_family = [], None
    for word, label in zip(words, word_labels):
        if label == "O":
            if current_words:
                entities.append({"family": current_family, "text": " ".join(current_words)})
            current_words, current_family = [], None
            continue
        prefix, family = label.split("-", 1)
        if prefix == "B" or family != current_family:
            if current_words:
                entities.append({"family": current_family, "text": " ".join(current_words)})
            current_words, current_family = [word], family
        else:
            current_words.append(word)
    if current_words:
        entities.append({"family": current_family, "text": " ".join(current_words)})
    return entities


def analyze_document(text):
    return {
        "classification": classify_document(text),
        "entities": extract_entities(text),
    }


def main():
    test_docs = load_domain_split("domain_test")
    results = []
    for doc in test_docs[:8]:
        result = analyze_document(doc["text"])
        result["true_label"] = doc["label"]
        result["text_preview"] = doc["text"][:150]
        results.append(result)

    out_path = ROOT / "outputs" / "problem2_sample_classification_output.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} example outputs to {out_path}")


if __name__ == "__main__":
    main()
