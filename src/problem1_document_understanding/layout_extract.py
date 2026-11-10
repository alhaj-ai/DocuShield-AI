"""Inference pipeline: document image -> OCR -> LayoutLMv3 token classification
-> layout blocks (header/table/text/image) + key-value fields + parsed table.
"""
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from ocr_utils import run_ocr
from dataset import normalize_box
from labels import ID2LABEL, FAMILY_TO_BLOCK, SINGLE_FIELD_TO_FAMILY, TABLE_CELL_TO_FAMILY

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "problem1_layoutlmv3"

FAMILY_TO_SINGLE_FIELD = {v: k for k, v in SINGLE_FIELD_TO_FAMILY.items()}
FAMILY_TO_TABLE_COL = {v: k for k, v in TABLE_CELL_TO_FAMILY.items()}
TABLE_FAMILIES = set(TABLE_CELL_TO_FAMILY.values())
ROW_Y_THRESHOLD = 14  # px, for clustering table entities into rows


class LayoutExtractor:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir)).to(self.device)
        self.model.eval()

    def predict_word_labels(self, image: Image.Image, words: list[dict]) -> list[str]:
        if not words:
            return []
        w, h = image.size
        texts = [wd["text"] for wd in words]
        boxes = [normalize_box(wd["bbox"], w, h) for wd in words]

        encoding = self.processor(
            images=image, text=texts, boxes=boxes,
            truncation=True, padding="max_length", max_length=512, return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits[0]
        preds = logits.argmax(-1).cpu().numpy()

        word_label = [None] * len(words)
        for token_idx, wid in enumerate(word_ids):
            if wid is None or word_label[wid] is not None:
                continue
            word_label[wid] = ID2LABEL[int(preds[token_idx])]
        return [l or "O" for l in word_label]

    @staticmethod
    def _union_bbox(bboxes):
        xs0 = [b[0] for b in bboxes]; ys0 = [b[1] for b in bboxes]
        xs1 = [b[2] for b in bboxes]; ys1 = [b[3] for b in bboxes]
        return [min(xs0), min(ys0), max(xs1), max(ys1)]

    def decode_entities(self, words: list[dict], word_labels: list[str]) -> list[dict]:
        """BIO decoding: consecutive B-/I- of the same family become one entity span."""
        entities = []
        current = None
        for word, label in zip(words, word_labels):
            if label == "O":
                current = None
                continue
            prefix, family = label.split("-", 1)
            if prefix == "B" or current is None or current["family"] != family:
                current = {"family": family, "words": [word["text"]], "bboxes": [word["bbox"]]}
                entities.append(current)
            else:
                current["words"].append(word["text"])
                current["bboxes"].append(word["bbox"])
        for e in entities:
            e["text"] = " ".join(e["words"])
            e["bbox"] = self._union_bbox(e["bboxes"])
        return entities

    @staticmethod
    def detect_image_regions(image: Image.Image, words: list[dict], min_area=1500, min_side=30):
        """Heuristic: rectangular non-white regions not covered by any OCR word
        (used to find embedded logos/images since the token classifier only
        sees text tokens)."""
        arr = np.array(image.convert("L"))
        h, w = arr.shape
        mask = np.ones_like(arr, dtype=np.uint8) * 255
        for wd in words:
            x0, y0, x1, y1 = [int(v) for v in wd["bbox"]]
            mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 255
            mask[y0:y1, x0:x1] = 255
        non_white = (arr < 245).astype(np.uint8) * 255
        for wd in words:
            x0, y0, x1, y1 = [int(v) for v in wd["bbox"]]
            non_white[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = 0

        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw * ch >= min_area and cw >= min_side and ch >= min_side:
                regions.append({"type": "image", "bbox": [x, y, x + cw, y + ch]})
        return regions

    def extract(self, image: Image.Image) -> dict:
        words = run_ocr(image)
        word_labels = self.predict_word_labels(image, words)
        entities = self.decode_entities(words, word_labels)

        fields = {}
        for e in entities:
            if e["family"] in FAMILY_TO_SINGLE_FIELD:
                key = FAMILY_TO_SINGLE_FIELD[e["family"]]
                if key not in fields:  # keep first occurrence
                    fields[key] = {"text": e["text"], "bbox": e["bbox"]}

        table_entities = [e for e in entities if e["family"] in TABLE_FAMILIES]
        table_entities.sort(key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2)
        rows = []
        for e in table_entities:
            ycenter = (e["bbox"][1] + e["bbox"][3]) / 2
            placed = False
            for row in rows:
                if abs(row["_y"] - ycenter) <= ROW_Y_THRESHOLD:
                    col = FAMILY_TO_TABLE_COL[e["family"]]
                    if col not in row["cells"]:
                        row["cells"][col] = e["text"]
                    placed = True
                    break
            if not placed:
                col = FAMILY_TO_TABLE_COL[e["family"]]
                rows.append({"_y": ycenter, "cells": {col: e["text"]}})

        columns = ["test_name", "value", "unit", "reference_range", "flag"]
        table_rows = [{c: row["cells"].get(c, "") for c in columns} for row in rows]

        blocks = []
        block_words = {"header": [], "patient_info": [], "table": [], "footer": []}
        for e in entities:
            block_type = FAMILY_TO_BLOCK.get(e["family"])
            if block_type:
                block_words[block_type].append(e["bbox"])
        for block_type, bboxes in block_words.items():
            if bboxes:
                out_type = "table" if block_type == "table" else "text"
                blocks.append({"type": out_type, "subtype": block_type, "bbox": self._union_bbox(bboxes)})
        blocks.extend(self.detect_image_regions(image, words))

        return {
            "fields": fields,
            "table": {"columns": columns, "rows": table_rows},
            "blocks": blocks,
            "entities": [{"family": e["family"], "text": e["text"], "bbox": e["bbox"]} for e in entities],
        }
