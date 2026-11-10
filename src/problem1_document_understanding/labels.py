"""Entity label scheme for the LayoutLMv3 token classifier and the logic
that turns per-document ground truth (from generate_synthetic_docs.py) into
per-OCR-word BIO labels for training.
"""
from typing import Optional

FIELD_FAMILIES = [
    "LABNAME", "LABADDR", "PATIENT_NAME", "DOB", "PATIENT_ID", "REPORT_DATE",
    "PHYSICIAN_NAME", "PHYSICIAN_ID", "TEST_NAME", "VALUE", "UNIT", "REF_RANGE", "FLAG",
]

LABEL_LIST = ["O"] + [f"{p}-{fam}" for fam in FIELD_FAMILIES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Maps a ground-truth "fields" dict key -> entity family
SINGLE_FIELD_TO_FAMILY = {
    "lab_name": "LABNAME", "lab_address": "LABADDR",
    "patient_name": "PATIENT_NAME", "dob": "DOB", "patient_id": "PATIENT_ID",
    "report_date": "REPORT_DATE", "physician_name": "PHYSICIAN_NAME", "physician_id": "PHYSICIAN_ID",
}
# Maps a table row cell key -> entity family
TABLE_CELL_TO_FAMILY = {
    "test_name": "TEST_NAME", "value": "VALUE", "unit": "UNIT",
    "reference_range": "REF_RANGE", "flag": "FLAG",
}

# Coarse block grouping used to derive layout segments from predicted entities
FAMILY_TO_BLOCK = {
    "LABNAME": "header", "LABADDR": "header",
    "PATIENT_NAME": "patient_info", "DOB": "patient_info",
    "PATIENT_ID": "patient_info", "REPORT_DATE": "patient_info",
    "PHYSICIAN_NAME": "footer", "PHYSICIAN_ID": "footer",
    "TEST_NAME": "table", "VALUE": "table", "UNIT": "table",
    "REF_RANGE": "table", "FLAG": "table",
}


def _center_inside(word_bbox, span_bbox, margin=4):
    cx = (word_bbox[0] + word_bbox[2]) / 2
    cy = (word_bbox[1] + word_bbox[3]) / 2
    x0, y0, x1, y1 = span_bbox
    return (x0 - margin) <= cx <= (x1 + margin) and (y0 - margin) <= cy <= (y1 + margin)


def _flatten_gt_spans(ground_truth: dict):
    """Returns list of (family, bbox) spans, ordered so that multi-word spans
    can be resolved deterministically (reading order top-to-bottom, left-to-right)."""
    spans = []
    for key, span in ground_truth.get("fields", {}).items():
        family = SINGLE_FIELD_TO_FAMILY.get(key)
        if family and span.get("text"):
            spans.append((family, span["bbox"]))
    for row in ground_truth.get("table", {}).get("_row_spans", []):
        for key, span in row.items():
            family = TABLE_CELL_TO_FAMILY.get(key)
            if family and span.get("text"):
                spans.append((family, span["bbox"]))
    spans.sort(key=lambda s: (s[1][1], s[1][0]))
    return spans


def build_word_labels(ocr_words: list[dict], ground_truth: dict) -> list[str]:
    """ocr_words: list of {"text":str, "bbox":[x0,y0,x1,y1]} in reading order.
    Returns a parallel list of BIO label strings."""
    spans = _flatten_gt_spans(ground_truth)
    labels = ["O"] * len(ocr_words)
    active_family: Optional[str] = None
    active_span_bbox = None

    for i, word in enumerate(ocr_words):
        wbbox = word["bbox"]
        matched_family = None
        for family, span_bbox in spans:
            if _center_inside(wbbox, span_bbox):
                matched_family = family
                matched_bbox = span_bbox
                break
        if matched_family is None:
            labels[i] = "O"
            active_family, active_span_bbox = None, None
            continue

        is_continuation = (matched_family == active_family and matched_bbox == active_span_bbox)
        labels[i] = f"{'I' if is_continuation else 'B'}-{matched_family}"
        active_family, active_span_bbox = matched_family, matched_bbox

    return labels
