"""Task B: fine-tunes XLM-R-base for cross-domain document classification
(Healthcare / Legal / Finance / Multi-domain), with a TF-IDF + Logistic
Regression baseline for comparison.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           Trainer, TrainingArguments)
from datasets import Dataset

from prepare_domain_data import load_domain_split

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "xlm-roberta-base"
OUT_DIR = ROOT / "models" / "problem2_domain_classifier"
LABELS = ["healthcare", "legal", "finance", "multi_domain"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def run_tfidf_baseline(train, test):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vec.fit_transform([e["text"] for e in train])
    X_test = vec.transform([e["text"] for e in test])
    y_train = [e["label"] for e in train]
    y_test = [e["label"] for e in test]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=LABELS).tolist(),
        "labels": LABELS,
        "classification_report": classification_report(y_test, preds, labels=LABELS, zero_division=0, output_dict=True),
    }


def tokenize_fn(examples, tokenizer, max_length=256):
    enc = tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    enc["labels"] = [LABEL2ID[l] for l in examples["label"]]
    return enc


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
    }


def run_xlmr_classifier(train, test, epochs=8):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Turn the raw {"text": ..., "label": ...} examples into tokenized tensors.
    # We drop the original columns after tokenizing so the old string "label"
    # column doesn't clash with the new integer "labels" column we create.
    raw_train = Dataset.from_list(train)
    raw_test = Dataset.from_list(test)
    train_ds = raw_train.map(lambda ex: tokenize_fn(ex, tokenizer), batched=True,
                              remove_columns=raw_train.column_names)
    test_ds = raw_test.map(lambda ex: tokenize_fn(ex, tokenizer), batched=True,
                            remove_columns=raw_test.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                       compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=1)
    y_true = [ID2LABEL[i] for i in preds_out.label_ids]
    y_pred = [ID2LABEL[i] for i in preds]

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True),
        "train_time_seconds": train_time,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    return result


def main():
    train = load_domain_split("domain_train")
    test = load_domain_split("domain_test")
    print(f"train={len(train)} test={len(test)}")

    print("Running TF-IDF + Logistic Regression baseline...")
    tfidf_result = run_tfidf_baseline(train, test)
    print("TF-IDF baseline accuracy:", tfidf_result["accuracy"])

    print("Fine-tuning XLM-R sequence classifier...")
    xlmr_result = run_xlmr_classifier(train, test)
    print("XLM-R classifier accuracy:", xlmr_result["accuracy"])

    out = {"tfidf_logreg_baseline": tfidf_result, "xlmr_classifier": xlmr_result}
    out_path = ROOT / "outputs" / "problem2_domain_classifier_eval.json"
    out_path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
