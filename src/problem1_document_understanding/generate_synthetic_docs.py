"""Generate synthetic medical lab report images + ground-truth annotations.

Each document is drawn from scratch with PIL so exact field text and pixel
bounding boxes are known (used later to auto-label OCR tokens for LayoutLMv3
training, and as ground truth for extraction evaluation).
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "synthetic" / "problem1"

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_REGULAR = FONT_DIR / "arial.ttf"
FONT_BOLD = FONT_DIR / "arialbd.ttf"

CANVAS_W, CANVAS_H = 1000, 1294
MARGIN = 60

LAB_NAMES = {
    "en": ["City Health Labs", "Northgate Diagnostics", "Unity Medical Laboratory",
           "Prime Clinical Labs", "Riverside Pathology Center"],
    "fr": ["Laboratoire Central de Biologie", "Institut Medical Saint-Louis",
           "Clinique du Nord - Laboratoire", "Laboratoire BioSante",
           "Centre d'Analyses Medicales Lille"],
}

LABELS = {
    "en": {
        "title": "LABORATORY REPORT",
        "patient_name": "Patient Name", "dob": "Date of Birth",
        "patient_id": "Patient ID", "report_date": "Report Date",
        "table_title": "Laboratory Results",
        "col_test": "Test Name", "col_value": "Value", "col_unit": "Unit",
        "col_ref": "Reference Range", "col_flag": "Flag",
        "physician": "Physician", "physician_id": "Medical ID",
    },
    "fr": {
        "title": "RAPPORT DE LABORATOIRE",
        "patient_name": "Nom du Patient", "dob": "Date de Naissance",
        "patient_id": "ID Patient", "report_date": "Date du Rapport",
        "table_title": "Resultats de Laboratoire",
        "col_test": "Analyse", "col_value": "Valeur", "col_unit": "Unite",
        "col_ref": "Plage de Reference", "col_flag": "Indicateur",
        "physician": "Medecin", "physician_id": "ID Medical",
    },
}

# (test_name_en, test_name_fr, unit, ref_low, ref_high, decimals)
TEST_PANEL = [
    ("Hemoglobin", "Hemoglobine", "g/dL", 12.0, 16.0, 1),
    ("White Blood Cell Count", "Leucocytes", "x10^9/L", 4.0, 11.0, 1),
    ("Glucose", "Glucose", "mg/dL", 70, 100, 0),
    ("Cholesterol Total", "Cholesterol Total", "mg/dL", 125, 200, 0),
    ("Creatinine", "Creatinine", "mg/dL", 0.6, 1.3, 2),
    ("Sodium", "Sodium", "mmol/L", 135, 145, 0),
    ("Potassium", "Potassium", "mmol/L", 3.5, 5.1, 1),
    ("Platelet Count", "Plaquettes", "x10^9/L", 150, 400, 0),
    ("Urea", "Uree", "mg/dL", 7, 20, 0),
    ("Calcium", "Calcium", "mg/dL", 8.5, 10.5, 1),
]


def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


@dataclass
class DocResult:
    doc_id: str
    language: str
    image: Image.Image
    ground_truth: dict = field(default_factory=dict)


def _draw_text(draw: ImageDraw.ImageDraw, xy, text, font, fill="black"):
    draw.text(xy, text, font=font, fill=fill)
    bbox = draw.textbbox(xy, text, font=font)
    return {"text": text, "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]]}


def generate_document(doc_id: str, language: str, rng: random.Random) -> DocResult:
    fake = Faker("fr_FR" if language == "fr" else "en_US")
    fake.seed_instance(rng.randint(0, 2**31))
    L = LABELS[language]

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)
    gt = {"doc_id": doc_id, "language": language, "width": CANVAS_W, "height": CANVAS_H,
          "fields": {}, "table": {"columns": ["test_name", "value", "unit", "reference_range", "flag"], "rows": []},
          "blocks": []}

    # --- Header: logo (image block) + lab name/address (header block) ---
    logo_bbox = [MARGIN, 40, MARGIN + 80, 40 + 80]
    draw.rectangle(logo_bbox, outline="black", width=2)
    draw.line([logo_bbox[0] + 40, logo_bbox[1] + 15, logo_bbox[0] + 40, logo_bbox[3] - 15], fill="black", width=6)
    draw.line([logo_bbox[0] + 15, logo_bbox[1] + 40, logo_bbox[2] - 15, logo_bbox[1] + 40], fill="black", width=6)
    gt["blocks"].append({"type": "image", "bbox": logo_bbox})

    lab_name = rng.choice(LAB_NAMES[language])
    header_x = MARGIN + 100
    f_title = _font(True, 22)
    f_body = _font(False, 15)
    f_small = _font(False, 13)

    name_span = _draw_text(draw, (header_x, 40), lab_name, f_title)
    addr_text = fake.address().replace("\n", ", ")
    addr_span = _draw_text(draw, (header_x, 70), addr_text, f_body)
    phone_span = _draw_text(draw, (header_x, 95), fake.phone_number(), f_body)
    gt["fields"]["lab_name"] = name_span
    gt["fields"]["lab_address"] = addr_span
    header_block_bbox = [MARGIN, 40, CANVAS_W - MARGIN, 130]
    gt["blocks"].append({"type": "header", "bbox": header_block_bbox})

    draw.line([MARGIN, 145, CANVAS_W - MARGIN, 145], fill="black", width=2)
    title_span = _draw_text(draw, (MARGIN, 155), L["title"], _font(True, 18))

    # --- Patient info block (2x2 grid) ---
    py = 195
    col1_label_x, col1_val_x = MARGIN, MARGIN + 150
    col2_label_x, col2_val_x = 520, 650

    patient_name = fake.name()
    dob = fake.date_of_birth(minimum_age=1, maximum_age=95).strftime("%d/%m/%Y")
    patient_id = f"PT-{rng.randint(100000, 999999)}"
    report_date = fake.date_between(start_date="-30d", end_date="today").strftime("%d/%m/%Y")

    _draw_text(draw, (col1_label_x, py), L["patient_name"] + ":", f_small)
    gt["fields"]["patient_name"] = _draw_text(draw, (col1_val_x, py), patient_name, f_body)
    _draw_text(draw, (col2_label_x, py), L["dob"] + ":", f_small)
    gt["fields"]["dob"] = _draw_text(draw, (col2_val_x, py), dob, f_body)

    py2 = py + 35
    _draw_text(draw, (col1_label_x, py2), L["patient_id"] + ":", f_small)
    gt["fields"]["patient_id"] = _draw_text(draw, (col1_val_x, py2), patient_id, f_body)
    _draw_text(draw, (col2_label_x, py2), L["report_date"] + ":", f_small)
    gt["fields"]["report_date"] = _draw_text(draw, (col2_val_x, py2), report_date, f_body)

    patient_block_bbox = [MARGIN, 190, CANVAS_W - MARGIN, py2 + 25]
    gt["blocks"].append({"type": "text", "subtype": "patient_info", "bbox": patient_block_bbox})

    draw.line([MARGIN, py2 + 40, CANVAS_W - MARGIN, py2 + 40], fill="black", width=2)

    # --- Table ---
    table_title_y = py2 + 55
    _draw_text(draw, (MARGIN, table_title_y), L["table_title"], _font(True, 16))

    col_x = {"test_name": MARGIN, "value": 400, "unit": 500, "reference_range": 600, "flag": 850}
    header_y = table_title_y + 30
    for key in ["col_test", "col_value", "col_unit", "col_ref", "col_flag"]:
        field_key = key.replace("col_", "")
        field_key = {"test": "test_name", "value": "value", "unit": "unit", "ref": "reference_range", "flag": "flag"}[field_key]
        _draw_text(draw, (col_x[field_key], header_y), L[key], _font(True, 13))
    draw.line([MARGIN, header_y + 22, CANVAS_W - MARGIN, header_y + 22], fill="black", width=1)

    n_rows = rng.randint(5, 8)
    tests = rng.sample(TEST_PANEL, n_rows)
    row_y = header_y + 32
    row_h = 28
    for name_en, name_fr, unit, lo, hi, dec in tests:
        name = name_fr if language == "fr" else name_en
        # ~25% chance of an abnormal (flagged) value
        if rng.random() < 0.25:
            if rng.random() < 0.5:
                val = round(rng.uniform(lo * 0.5, lo * 0.95), dec)
                flag = "L"
            else:
                val = round(rng.uniform(hi * 1.05, hi * 1.5), dec)
                flag = "H"
        else:
            val = round(rng.uniform(lo, hi), dec)
            flag = ""
        val_str = f"{val:.{dec}f}" if dec > 0 else str(int(val))
        ref_str = f"{lo:g}-{hi:g}"

        row = {}
        row["test_name"] = _draw_text(draw, (col_x["test_name"], row_y), name, f_body)
        row["value"] = _draw_text(draw, (col_x["value"], row_y), val_str, f_body)
        row["unit"] = _draw_text(draw, (col_x["unit"], row_y), unit, f_body)
        row["reference_range"] = _draw_text(draw, (col_x["reference_range"], row_y), ref_str, f_body)
        if flag:
            row["flag"] = _draw_text(draw, (col_x["flag"], row_y), flag, _font(True, 14), fill="darkred" if flag == "H" else "darkblue")
        else:
            row["flag"] = {"text": "", "bbox": [col_x["flag"], row_y, col_x["flag"], row_y]}
        gt["table"]["rows"].append({k: v["text"] for k, v in row.items()})
        gt["table"]["_row_spans"] = gt["table"].get("_row_spans", [])
        gt["table"]["_row_spans"].append(row)
        row_y += row_h

    table_block_bbox = [MARGIN, table_title_y, CANVAS_W - MARGIN, row_y]
    gt["blocks"].append({"type": "table", "bbox": table_block_bbox})
    draw.line([MARGIN, row_y, CANVAS_W - MARGIN, row_y], fill="black", width=2)

    # --- Footer: physician info ---
    footer_y = row_y + 30
    physician_name = "Dr. " + fake.name()
    physician_id = f"MD-{rng.randint(10000, 99999)}" if language == "en" else f"FR{rng.randint(100000000, 999999999)}"

    _draw_text(draw, (MARGIN, footer_y), L["physician"] + ":", f_small)
    gt["fields"]["physician_name"] = _draw_text(draw, (MARGIN + 100, footer_y), physician_name, f_body)
    _draw_text(draw, (520, footer_y), L["physician_id"] + ":", f_small)
    gt["fields"]["physician_id"] = _draw_text(draw, (650, footer_y), physician_id, f_body)

    footer_block_bbox = [MARGIN, footer_y - 5, CANVAS_W - MARGIN, footer_y + 30]
    gt["blocks"].append({"type": "text", "subtype": "footer", "bbox": footer_block_bbox})

    return DocResult(doc_id=doc_id, language=language, image=img, ground_truth=gt)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Light scan-like noise: small rotation + blur + jpeg-ish artifacts."""
    from PIL import ImageFilter
    import io

    angle = rng.uniform(-3, 3)
    img2 = img.rotate(angle, expand=False, fillcolor="white", resample=Image.BICUBIC)
    if rng.random() < 0.5:
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    buf = io.BytesIO()
    img2.save(buf, format="JPEG", quality=rng.randint(55, 80))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--augment-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        img_dir = OUT_DIR / "images" / split
        lbl_dir = OUT_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        for i in range(n):
            language = "en" if i % 2 == 0 else "fr"
            doc_id = f"{split}_{language}_{i:04d}"
            result = generate_document(doc_id, language, rng)

            is_augmented = rng.random() < args.augment_frac
            img_out = augment(result.image, rng) if is_augmented else result.image
            result.ground_truth["augmented"] = is_augmented

            img_path = img_dir / f"{doc_id}.png"
            lbl_path = lbl_dir / f"{doc_id}.json"
            img_out.save(img_path)
            lbl_path.write_text(json.dumps(result.ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({"doc_id": doc_id, "language": language, "image": str(img_path.relative_to(OUT_DIR)),
                              "label": str(lbl_path.relative_to(OUT_DIR)), "augmented": is_augmented})

        (OUT_DIR / f"manifest_{split}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{split}: generated {n} documents ({sum(m['augmented'] for m in manifest)} augmented) -> {img_dir}")


if __name__ == "__main__":
    main()
