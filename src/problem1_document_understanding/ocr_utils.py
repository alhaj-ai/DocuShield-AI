"""Tesseract OCR wrapper: image -> word-level text + pixel bounding boxes."""
from PIL import Image
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_LANG = "eng+fra"


def run_ocr(image: Image.Image, lang: str = OCR_LANG) -> list[dict]:
    """Returns word-level OCR results in reading order:
    [{"text": str, "bbox": [x0, y0, x1, y1], "conf": float}, ...]
    """
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "conf": conf,
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i],
        })
    return words
