"""Makes a few simple bar charts from our real results, for the results
report. Nothing fancy -- just matplotlib bar charts saved as PNG files.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bar_chart(labels, values, title, ylabel, filename, ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"saved {filename}")


def chart_problem1():
    data = json.loads((OUT_DIR / "problem1_eval_summary.json").read_text())
    bar_chart(
        ["English", "French", "Clean scan", "Noisy scan"],
        [data["en_field_accuracy"], data["fr_field_accuracy"],
         data["clean_field_accuracy"], data["augmented_field_accuracy"]],
        "Problem 1: field extraction accuracy", "accuracy",
        "problem1_field_accuracy.png",
    )


def chart_problem2_ner():
    data = json.loads((OUT_DIR / "problem2_ner_eval.json").read_text())
    summary = data["cross_lingual_transfer_summary"]
    bar_chart(
        ["EN-only\n(zero-shot FR)", "Joint EN+FR\n(on FR)"],
        [summary["disease_f1_en_only_zero_shot_fr"], summary["disease_f1_joint_fr"]],
        "Problem 2: cross-lingual transfer (DISEASE class F1)", "F1 score",
        "problem2_cross_lingual_transfer.png",
    )


def chart_problem2_classifier():
    data = json.loads((OUT_DIR / "problem2_domain_classifier_eval.json").read_text())
    bar_chart(
        ["TF-IDF +\nLogistic Regression", "XLM-R\n(fine-tuned)"],
        [data["tfidf_logreg_baseline"]["accuracy"], data["xlmr_classifier"]["accuracy"]],
        "Problem 2: domain classifier accuracy", "accuracy",
        "problem2_classifier_accuracy.png",
    )


def chart_problem3_pii():
    data = json.loads((OUT_DIR / "problem3_pii_detection_eval.json").read_text())
    labels = sorted(data.keys())
    f1s = [data[l]["f1"] for l in labels]
    bar_chart(labels, f1s, "Problem 3: PII detection F1 by type", "F1 score",
               "problem3_pii_detection_f1.png")


def chart_problem3_utility():
    data = json.loads((OUT_DIR / "problem3_downstream_utility.json").read_text())
    recall = data["ner_entity_recall_vs_original"]
    bar_chart(
        list(recall.keys()), list(recall.values()),
        "Problem 3: NER entity recall after redaction\n(vs. entities found in original text)",
        "fraction of entities retained",
        "problem3_downstream_utility.png",
    )


if __name__ == "__main__":
    chart_problem1()
    chart_problem2_ner()
    chart_problem2_classifier()
    chart_problem3_pii()
    chart_problem3_utility()
