"""Unit tests for Yemeni & Arabic PII detection and redaction."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem3_privacy.arabic_pii import (
    detect_yemeni_national_ids,
    detect_yemeni_and_arabic_phones,
    detect_arabic_medical_ids,
    detect_arabic_names,
    detect_arabic_pii,
)
from src.problem3_privacy.pii_detectors import detect_all_pii, guess_language
from src.problem3_privacy.redaction import redact_text


def test_yemeni_national_id():
    text = "الرقم الوطني اليمني للمريض المسجل في صنعاء: 01010198452 والبطاقة 12045678901"
    matches = detect_yemeni_national_ids(text)
    assert len(matches) >= 2
    found = [m["text"] for m in matches]
    assert "01010198452" in found


def test_yemeni_phones():
    text = "للتواصل: يمن موبايل 771234567 أو سبأفون 712345678 أو يو 731234567 أو دولي +967779876543"
    matches = detect_yemeni_and_arabic_phones(text)
    assert len(matches) >= 3
    found_texts = [m["text"] for m in matches]
    assert any("771234567" in t for t in found_texts)


def test_arabic_medical_id():
    text = "مستشفى الثورة - صنعاء - رقم الملف: YEM-MED-99482"
    matches = detect_arabic_medical_ids(text)
    assert len(matches) == 1
    assert matches[0]["text"] == "YEM-MED-99482"


def test_arabic_names():
    text = "تم فحص المريض محمد ناصر العميسي بواسطة استشاري دكتور عبدالكريم الحاج في قسم الطوارئ"
    matches = detect_arabic_names(text)
    assert len(matches) >= 1
    found_names = [m["text"] for m in matches]
    assert any("محمد ناصر العميسي" in n for n in found_names)


def test_detect_all_pii_yemeni():
    text = (
        "تقرير طبي - هيئة مستشفى الكويت (صنعاء) بتاريخ 2026-05-14. "
        "المريض محمد علي الحاج، الرقم الوطني 01010198452، "
        "رقم الملف: YEM-MRN-10029، هاتف: 771234567، "
        "البريد: mohammed@hospital.ye"
    )
    assert guess_language(text) == "ar"
    pii = detect_all_pii(text)
    types = [p["type"] for p in pii]
    assert "YEMENI_NATIONAL_ID" in types
    assert "EMAIL" in types
    assert "PHONE" in types
    assert "MEDICAL_ID" in types


def test_redaction_modes_yemeni():
    text = "المريض عصام الصنعاني الرقم الوطني 01010198452 هاتف 771234567"
    pii = detect_all_pii(text)

    # Strict mode
    strict_redacted, _ = redact_text(text, pii, mode="strict", language="ar")
    assert "01010198452" not in strict_redacted
    assert "[محجوب]" in strict_redacted

    # Typed mode with Arabic labels
    typed_redacted, _ = redact_text(text, pii, mode="typed", language="ar", arabic_labels=True)
    assert "01010198452" not in typed_redacted
    assert "[رقم_وطني_يمني]" in typed_redacted or "[هوية_وطنية]" in typed_redacted

    print("\n[SUCCESS] All Yemeni & Arabic PII tests passed successfully!")


if __name__ == "__main__":
    test_yemeni_national_id()
    test_yemeni_phones()
    test_arabic_medical_id()
    test_arabic_names()
    test_detect_all_pii_yemeni()
    test_redaction_modes_yemeni()
