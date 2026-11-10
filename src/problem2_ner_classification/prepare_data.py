"""Loads NCBI Disease (EN) and Quaero French Med (FR), converts both to a
unified word-level BIO scheme (see labels.py), and writes cached JSONL
splits to data/processed/problem2/.

Note on data sources: the canonical `ncbi/ncbi_disease` and
`mnaguib/QuaeroFrenchMed` repos ship with Hugging Face "loading scripts",
which the current `datasets` library version no longer executes. We use
`clinicalnlplab/NCBI_disease_{train,dev,test}` instead (row counts verified
to match the official corpus: 5424/923/940 sentences -- same data,
re-hosted in a span-tagged text format) and load Quaero's raw parquet
files directly from its repo, bypassing the broken script.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from labels import QUAERO_ID2NAME, QUAERO_TO_FAMILY

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "problem2"

SPAN_RE = re.compile(r'<span class="disease">(.*?)</span>')


def _parse_ncbi_sentence(processed_text: str) -> tuple[list[str], list[str]]:
    """'... the <span class="disease">adenomatous polyposis coli tumour</span> suppressor .'
    -> (words, bio_labels) with family DISEASE."""
    words, labels = [], []
    pos = 0
    for m in SPAN_RE.finditer(processed_text):
        before = processed_text[pos:m.start()]
        for w in before.split():
            words.append(w); labels.append("O")
        span_words = m.group(1).split()
        for i, w in enumerate(span_words):
            words.append(w)
            labels.append("B-DISEASE" if i == 0 else "I-DISEASE")
        pos = m.end()
    for w in processed_text[pos:].split():
        words.append(w); labels.append("O")
    return words, labels


def load_ncbi(split: str) -> list[dict]:
    """split in {train, dev, test}"""
    ds = load_dataset(f"clinicalnlplab/NCBI_disease_{split}")["train"]
    examples = []
    for row in ds:
        words, bio = _parse_ncbi_sentence(row["processed"])
        if words:
            examples.append({"words": words, "labels": bio, "language": "en"})
    return examples


def load_quaero(split: str) -> list[dict]:
    """split in {train, test}"""
    url = f"hf://datasets/mnaguib/QuaeroFrenchMed/data/{split}.parquet"
    ds = load_dataset("parquet", data_files={split: url})[split]
    examples = []
    for row in ds:
        words = row["words"]
        tag_ids = row["ner_tags"]
        bio = []
        prev_family = None
        for tid in tag_ids:
            name = QUAERO_ID2NAME[tid]
            if name == "O":
                bio.append("O")
                prev_family = None
                continue
            family = QUAERO_TO_FAMILY[name]
            bio.append(f"{'I' if family == prev_family else 'B'}-{family}")
            prev_family = family
        if words:
            examples.append({"words": words, "labels": bio, "language": "fr"})
    return examples


def build_splits(n_train_en=500, n_train_fr=500, seed=42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    ncbi_train_full = load_ncbi("train")
    ncbi_test = load_ncbi("test")
    quaero_train_full = load_quaero("train")
    quaero_test = load_quaero("test")

    train_en = rng.sample(ncbi_train_full, min(n_train_en, len(ncbi_train_full)))
    train_fr = rng.sample(quaero_train_full, min(n_train_fr, len(quaero_train_full)))

    splits = {
        "train_en": train_en, "train_fr": train_fr,
        "train_joint": train_en + train_fr,
        "test_en": ncbi_test, "test_fr": quaero_test,
    }
    for name, examples in splits.items():
        (OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in examples), encoding="utf-8"
        )
        print(f"{name}: {len(examples)} sentences")
    return splits


def load_split(name: str) -> list[dict]:
    path = OUT_DIR / f"{name}.jsonl"
    if not path.exists():
        build_splits()
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    build_splits()
