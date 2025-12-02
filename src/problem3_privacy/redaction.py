"""Three ways to redact the PII spans found by pii_detectors.py:

- strict:    replace every PII span with the same generic marker [REDACTED] or [محجوب]
- typed:     replace with a marker that names the PII type, e.g. [NAME] or [رقم_وطني]
- surrogate: replace with a fake but realistic-looking value, e.g. a made-up
             name instead of the real one, so the document stays readable
"""
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from faker import Faker
    _faker_en = Faker("en_US")
    _faker_fr = Faker("fr_FR")
    try:
        _faker_ar = Faker("ar_AA")
    except Exception:
        _faker_ar = _faker_en
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False
    _faker_en = None
    _faker_fr = None
    _faker_ar = None

ARABIC_TYPED_LABELS = {
    "NAME": "اسم",
    "EMAIL": "بريد_إلكتروني",
    "PHONE": "هاتف",
    "ADDRESS": "عنوان",
    "DATE": "تاريخ",
    "SSN": "رقم_تأميني",
    "YEMENI_NATIONAL_ID": "رقم_وطني_يمني",
    "SAUDI_NATIONAL_ID": "هوية_وطنية",
    "MEDICAL_ID": "رقم_طبي",
}


def _make_surrogate(pii_type: str, language: str = "ar") -> str:
    rnd = random.randint(10000, 99999)
    if pii_type == "NAME":
        return "محمد علي الحاج" if language == "ar" else "John Doe"
    if pii_type == "EMAIL":
        return f"patient_{rnd}@clinic.ye" if language == "ar" else f"user_{rnd}@example.com"
    if pii_type == "PHONE":
        return f"+967 77{random.randint(1000000, 9999999)}" if language == "ar" else f"+1-555-01{rnd%100:02d}"
    if pii_type == "YEMENI_NATIONAL_ID":
        return f"0101{random.randint(1000000, 9999999)}"
    if pii_type == "SAUDI_NATIONAL_ID":
        return f"10{rnd}{rnd%1000:03d}"
    if pii_type == "MEDICAL_ID":
        return f"MRN-YEM-{rnd}"
    if pii_type == "DATE":
        return "2026-04-12"
    if pii_type == "SSN":
        return f"999-{rnd%100:02d}-{rnd%10000:04d}"
    return "[UNKNOWN]"


def redact_text(
    text: str,
    entities: List[Dict[str, Any]],
    mode: str = "typed",
    language: str = "ar",
    surrogate_map: Optional[Dict[Tuple[str, str], str]] = None,
    arabic_labels: bool = False
) -> Tuple[str, Dict[Tuple[str, str], str]]:
    if mode not in ("strict", "typed", "surrogate"):
        raise ValueError(f"unknown redaction mode: {mode}")

    if surrogate_map is None:
        surrogate_map = {}

    # Replace spans from the end backwards so char offsets remain valid
    entities_sorted = sorted(entities, key=lambda e: e["start"], reverse=True)

    redacted = text
    for ent in entities_sorted:
        original = ent["text"]

        if mode == "strict":
            replacement = "[محجوب]" if (language == "ar" or arabic_labels) else "[REDACTED]"
        elif mode == "typed":
            if arabic_labels and ent["type"] in ARABIC_TYPED_LABELS:
                replacement = f"[{ARABIC_TYPED_LABELS[ent['type']]}]"
            else:
                replacement = f"[{ent['type']}]"
        else:  # surrogate
            key = (ent["type"], original.strip().lower())
            if key not in surrogate_map:
                surrogate_map[key] = _make_surrogate(ent["type"], language=language)
            replacement = surrogate_map[key]

        redacted = redacted[:ent["start"]] + replacement + redacted[ent["end"]:]

    return redacted, surrogate_map
