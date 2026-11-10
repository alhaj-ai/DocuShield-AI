"""Unit tests for Arabic PII detection and redaction."""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem3_privacy.arabic_pii import (
    detect_saudi_national_ids,
    detect_arabic_phones,
    detect_arabic_medical_ids,
    detect_arabic_names,
    detect_arabic_pii,
)
from src.problem3_privacy.pii_detectors import detect_all_pii, guess_language
from src.problem3_privacy.redaction import redact_text


def test_saudi_national_id():
    text = "رقم الهوية الوطنية للمريض هو 1098765432 والمقيم رقمه 2012345678"
    matches = detect_saudi_national_ids(text)
    assert len(matches) == 2
    assert matches[0]["text"] == "1098765432"
    assert matches[1]["text"] == "2012345678"


def test_arabic_phones():
    text = "يرجى الاتصال على 0501234567 أو عبر الرقم الدولي +966551234567 أو فرع اليمن 771234567"
    matches = detect_arabic_phones(text)
    assert len(matches) >= 2
    found_texts = [m["text"] for m in matches]
    assert any("0501234567" in t for t in found_texts)


def test_arabic_medical_id():
    text = "تقرير طبي - رقم الملف: MED-99482 للطوارئ"
    matches = detect_arabic_medical_ids(text)
    assert len(matches) == 1
    assert matches[0]["text"] == "MED-99482"


def test_arabic_names():
    text = "تم فحص المريض محمد عبدالله الحاج بواسطة دكتور أحمد علي الغامدي في العيادة"
    matches = detect_arabic_names(text)
    assert len(matches) >= 1
    found_names = [m["text"] for m in matches]
    assert any("محمد عبدالله الحاج" in n for n in found_names)


def test_detect_all_pii_arabic():
    text = (
        "تقرير طبي صادر بتاريخ 2024-05-18. "
        "المريض محمد علي، رقم الهوية 1029384756، "
        "رقم الملف: MRN-10029، هاتف: 0558765432، "
        "البريد: mohammed@example.com"
    )
    assert guess_language(text) == "ar"
    pii = detect_all_pii(text)
    types = [p["type"] for p in pii]
    assert "SAUDI_NATIONAL_ID" in types
    assert "EMAIL" in types
    assert "PHONE" in types
    assert "MEDICAL_ID" in types


def test_redaction_modes_arabic():
    text = "المريض خالد عمر هوية رقم 1029384756 وهاتف 0551122334"
    pii = detect_all_pii(text)

    # Strict mode
    strict_redacted, _ = redact_text(text, pii, mode="strict", language="ar")
    assert "1029384756" not in strict_redacted
    assert "[محجوب]" in strict_redacted

    # Typed mode with Arabic labels
    typed_redacted, _ = redact_text(text, pii, mode="typed", language="ar", arabic_labels=True)
    assert "1029384756" not in typed_redacted
    assert "[هوية_وطنية]" in typed_redacted

    print("\n[SUCCESS] All Arabic PII tests passed successfully!")


if __name__ == "__main__":
    test_saudi_national_id()
    test_arabic_phones()
    test_arabic_medical_id()
    test_arabic_names()
    test_detect_all_pii_arabic()
    test_redaction_modes_arabic()
