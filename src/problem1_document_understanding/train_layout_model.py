"""Fine-tunes LayoutLMv3-base for token classification on the synthetic
medical lab report dataset."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import evaluate as hf_evaluate
from transformers import LayoutLMv3ForTokenClassification, Trainer, TrainingArguments

from dataset import LayoutDataset, get_processor, MODEL_NAME
from labels import LABEL_LIST, ID2LABEL, LABEL2ID

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "problem1_layoutlmv3"

seqeval = hf_evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [ID2LABEL[p_] for (p_, l_) in zip(pred, lab) if l_ != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l_] for (p_, l_) in zip(pred, lab) if l_ != -100]
        for pred, lab in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    cli_args = ap.parse_args()

    processor = get_processor()
    print("Building train dataset (OCR + label alignment)...")
    train_ds = LayoutDataset("train", processor, limit=cli_args.train_limit)
    print("Building test dataset...")
    test_ds = LayoutDataset("test", processor, limit=cli_args.test_limit)
    print(f"train={len(train_ds)} test={len(test_ds)}")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=cli_args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    metrics = trainer.evaluate()
    metrics["train_time_seconds"] = train_time
    print(metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    (OUT_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved model to {OUT_DIR}")


if __name__ == "__main__":
    main()
