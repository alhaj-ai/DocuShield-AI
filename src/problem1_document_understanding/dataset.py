"""Builds the LayoutLMv3 token-classification dataset from synthetic docs:
OCR each image once (cached to disk), auto-label words via labels.py, then
encode with LayoutLMv3Processor on the fly in __getitem__.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import LayoutLMv3Processor

from ocr_utils import run_ocr
from labels import build_word_labels, LABEL2ID

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic" / "problem1"
CACHE_DIR = ROOT / "data" / "processed" / "problem1"

MODEL_NAME = "microsoft/layoutlmv3-base"


def normalize_box(box, width, height):
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


def get_processor():
    return LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)


def _cache_path(split):
    return CACHE_DIR / f"{split}_ocr_labels.json"


def build_ocr_cache(split, limit=None):
    """Runs OCR once per doc and caches words+labels to disk (idempotent, resumable)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(split)
    manifest = json.loads((DATA_DIR / f"manifest_{split}.json").read_text())
    if limit:
        manifest = manifest[:limit]

    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())

    changed = False
    for entry in manifest:
        doc_id = entry["doc_id"]
        if doc_id in cache:
            continue
        img = Image.open(DATA_DIR / entry["image"]).convert("RGB")
        gt = json.loads((DATA_DIR / entry["label"]).read_text())
        words = run_ocr(img)
        word_labels = build_word_labels(words, gt)
        cache[doc_id] = {
            "words": [w["text"] for w in words],
            "boxes": [w["bbox"] for w in words],
            "labels": word_labels,
            "width": gt["width"], "height": gt["height"],
            "image": entry["image"], "language": entry["language"],
        }
        changed = True

    if changed:
        cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    return {doc_id: cache[doc_id] for doc_id in [e["doc_id"] for e in manifest]}


class LayoutDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, max_length=512, limit=None):
        self.data = build_ocr_cache(split, limit=limit)
        self.doc_ids = list(self.data.keys())
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        item = self.data[doc_id]
        image = Image.open(DATA_DIR / item["image"]).convert("RGB")
        words = item["words"]
        boxes = [normalize_box(b, item["width"], item["height"]) for b in item["boxes"]]
        label_ids = [LABEL2ID[l] for l in item["labels"]]

        if len(words) == 0:
            words, boxes, label_ids = [""], [[0, 0, 0, 0]], [LABEL2ID["O"]]

        encoding = self.processor(
            images=image, text=words, boxes=boxes, word_labels=label_ids,
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
