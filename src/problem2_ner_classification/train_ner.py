"""Fine-tunes XLM-RoBERTa-base for token classification on the unified
EN (NCBI Disease) + FR (Quaero French Med) corpus.

Two training conditions are used elsewhere for the cross-lingual transfer
analysis (see evaluate_ner.py): this script trains whichever split(s) you
point it at via --train-split.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import evaluate as hf_evaluate
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                           DataCollatorForTokenClassification, Trainer, TrainingArguments)

from prepare_data import load_split
from labels import LABEL_LIST, ID2LABEL, LABEL2ID

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "xlm-roberta-base"
seqeval = hf_evaluate.load("seqeval")


def tokenize_and_align(examples, tokenizer, max_length=192):
    tokenized = tokenizer(examples["words"], is_split_into_words=True, truncation=True,
                           max_length=max_length, padding="max_length")
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev_wid = [], None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_wid:
                label_ids.append(LABEL2ID[labels[wid]])
            else:
                label_ids.append(-100)  # only label the first subtoken of each word
            prev_wid = wid
        all_labels.append(label_ids)
        prev_wid = None
    tokenized["labels"] = all_labels
    return tokenized


def build_hf_dataset(split_name, tokenizer):
    from datasets import Dataset
    examples = load_split(split_name)
    ds = Dataset.from_list(examples)
    return ds.map(lambda ex: tokenize_and_align(ex, tokenizer), batched=True,
                   remove_columns=ds.column_names)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[ID2LABEL[p_] for p_, l_ in zip(pred, lab) if l_ != -100]
                         for pred, lab in zip(predictions, labels)]
    true_labels = [[ID2LABEL[l_] for p_, l_ in zip(pred, lab) if l_ != -100]
                    for pred, lab in zip(predictions, labels)]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    out = {"precision": results["overall_precision"], "recall": results["overall_recall"],
           "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="train_joint", help="train_joint | train_en | train_fr")
    ap.add_argument("--eval-split", default="test_fr", help="split to watch during training")
    ap.add_argument("--out-name", default="xlmr_joint")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    out_dir = ROOT / "models" / f"problem2_{args.out_name}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_hf_dataset(args.train_split, tokenizer)
    eval_ds = build_hf_dataset(args.eval_split, tokenizer)
    print(f"train={len(train_ds)} ({args.train_split})  eval={len(eval_ds)} ({args.eval_split})")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
                       data_collator=collator, compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    metrics = trainer.evaluate()
    metrics["train_time_seconds"] = train_time
    print(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
