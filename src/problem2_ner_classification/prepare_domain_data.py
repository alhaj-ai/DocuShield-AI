"""Builds the cross-domain classification dataset for Task B:
Healthcare / Legal / Finance / Multi-domain.

- Healthcare: OCR'd text from Problem 1's synthetic lab reports (real
  document-length text, reusing that pipeline's output) topped up with
  NCBI/Quaero sentences if needed.
- Legal: mattmdjaga/text-anonymization-benchmark-train (European Court of
  Human Rights case documents -- the MAPA-style dataset named in the spec).
- Finance: gretelai/synthetic_pii_finance_multilingual (English + French).
- Multi-domain: synthetic documents built by concatenating a Healthcare
  snippet with a Legal or Finance snippet, per the spec's own example of
  a "medical-legal document."
"""
import json
import random
from pathlib import Path

import pandas as pd

from prepare_data import load_ncbi, load_quaero

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "problem2"
PROBLEM1_DIR = ROOT / "data" / "processed" / "problem1"

N_PER_CLASS = 50


def load_healthcare_docs(n: int, rng: random.Random) -> list[str]:
    docs = []
    for split in ["train", "test"]:
        cache_file = PROBLEM1_DIR / f"{split}_ocr_labels.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            for item in cache.values():
                text = " ".join(item["words"])
                if len(text) > 40:
                    docs.append(text)
    if len(docs) < n:
        # top up with grouped NCBI/Quaero sentences (5 sentences per doc)
        extra_sents = [" ".join(e["words"]) for e in load_ncbi("dev")] + \
                      [" ".join(e["words"]) for e in load_quaero("test")]
        rng.shuffle(extra_sents)
        for i in range(0, len(extra_sents) - 5, 5):
            docs.append(" ".join(extra_sents[i:i + 5]))
            if len(docs) >= n * 2:
                break
    rng.shuffle(docs)
    return docs[:n]


def load_legal_docs(n: int, rng: random.Random) -> list[str]:
    df = pd.read_parquet("hf://datasets/mattmdjaga/text-anonymization-benchmark-train/data/train-00000-of-00001.parquet")
    texts = df["text"].dropna().tolist()
    rng.shuffle(texts)
    # truncate very long court documents to a representative excerpt
    return [t[:3000] for t in texts[:n]]


def load_finance_docs(n: int, rng: random.Random) -> list[str]:
    dfs = []
    for f in ["English_train-00000-of-00001.parquet", "France_train-00000-of-00001.parquet"]:
        dfs.append(pd.read_parquet(f"hf://datasets/gretelai/synthetic_pii_finance_multilingual/data/{f}"))
    df = pd.concat(dfs, ignore_index=True)
    texts = df["generated_text"].dropna().tolist()
    rng.shuffle(texts)
    return texts[:n]


def build_multidomain_docs(healthcare, legal, finance, n, rng):
    docs = []
    for i in range(n):
        hc = rng.choice(healthcare)
        other_pool, other_label = rng.choice([(legal, "legal"), (finance, "finance")])
        other = rng.choice(other_pool)
        combined = hc[:800] + "\n\n---\n\n" + other[:800]
        docs.append({"text": combined, "combined_with": other_label})
    return docs


def build_dataset(n_per_class=N_PER_CLASS, seed=42):
    rng = random.Random(seed)
    healthcare = load_healthcare_docs(n_per_class, rng)
    legal = load_legal_docs(n_per_class, rng)
    finance = load_finance_docs(n_per_class, rng)
    multidomain = build_multidomain_docs(healthcare, legal, finance, n_per_class, rng)

    examples = (
        [{"text": t, "label": "healthcare"} for t in healthcare] +
        [{"text": t, "label": "legal"} for t in legal] +
        [{"text": t, "label": "finance"} for t in finance] +
        [{"text": d["text"], "label": "multi_domain"} for d in multidomain]
    )
    rng.shuffle(examples)

    n_test = max(1, int(0.2 * len(examples)))
    test_examples = examples[:n_test]
    train_examples = examples[n_test:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, exs in [("domain_train", train_examples), ("domain_test", test_examples)]:
        (OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in exs), encoding="utf-8"
        )
        print(f"{name}: {len(exs)} docs")

    return train_examples, test_examples


def load_domain_split(name: str) -> list[dict]:
    path = OUT_DIR / f"{name}.jsonl"
    if not path.exists():
        build_dataset()
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    build_dataset()
