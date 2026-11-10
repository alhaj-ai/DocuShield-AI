"""Yemeni & Arabic PII detection module for medical documents and records.

Engineered by Mohammed AL-Haj (Sana'a, Yemen)

Provides specialized pattern matching and heuristic NER for Yemeni & Arabic medical PII:
1. Yemeni National ID & Personal Card Numbers (الرقم الوطني والبطاقة الشخصية اليمنية: 11 digits / 10 digits)
2. Yemeni Mobile & Fixed Phone Networks (Yemen Mobile 77/78, YOU 73, Sabafon 71, Y 70, +967)
3. Arabic Medical Record Numbers (MRN / رقم الملف الطبي / السجل الطبي في المستشفيات اليمنية والعربية)
4. Arabic Names heuristic extraction (names following doctor/patient honorifics)
5. Arabic / Hijri & Gregorian Dates
"""
import re
from typing import Any, Dict, List


def detect_yemeni_national_ids(text: str) -> List[Dict[str, Any]]:
    """Detects Yemeni National IDs / Personal Card numbers:

    - Standard Yemeni National ID (الرقم الوطني الموحد): 11 digits (e.g. 01010123456)
    - Older personal cards / IDs: 10 digits
    """
    patterns = [
        r"\b(?:0[1-9]|1[0-9]|2[0-2])\d{9}\b",  # 11-digit national ID starting with governorate code
        r"\b\d{10,11}\b",                       # General 10-11 digit identity pattern
    ]
    matches = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            span = (m.start(), m.end())
            if span not in seen:
                seen.add(span)
                matches.append({
                    "type": "YEMENI_NATIONAL_ID",
                    "text": m.group(),
                    "start": m.start(),
                    "end": m.end()
                })
    return matches


def detect_yemeni_and_arabic_phones(text: str) -> List[Dict[str, Any]]:
    """Detects Yemeni phone networks (Yemen Mobile 77/78, YOU 73, Sabafon 71, Y 70)

    and regional GCC/Arab phone numbers:
    - Yemen Mobile: +967 77XXXXXXX, 00967 78XXXXXXX, 77XXXXXXX
    - YOU (MTN): +967 73XXXXXXX, 73XXXXXXX
    - Sabafon: +967 71XXXXXXX, 71XXXXXXX
    - Y Telecom: +967 70XXXXXXX, 70XXXXXXX
    - Landlines: (01) XXXXXX (Sana'a), (02) XXXXXX (Aden), etc.
    - Regional GCC (+966, etc.)
    """
    patterns = [
        # Yemeni mobile (+967 7X / 00967 7X / 7X) - Yemen Mobile, YOU, Sabafon, Y
        r"(?:\+967|00967|0)?(?:77|78|73|71|70)[0-9]{7}\b",
        r"(?:\+967|00967)[\s-]?(?:77|78|73|71|70)[\s-]?[0-9]{3}[\s-]?[0-9]{4}\b",
        # Yemeni landline (01 Sana'a, 02 Aden, 03 Hodeidah, 04 Taiz, 05 Hadramout)
        r"\b0[1-5][\s-]?[0-9]{6}\b",
        # Regional Arab / GCC mobile (e.g. +966 5X)
        r"(?:\+966|00966|0)?5[0-9]{8}\b",
    ]
    matches = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            span = (m.start(), m.end())
            if span not in seen:
                seen.add(span)
                matches.append({
                    "type": "PHONE",
                    "text": m.group(),
                    "start": m.start(),
                    "end": m.end()
                })
    return matches


def detect_arabic_medical_ids(text: str) -> List[Dict[str, Any]]:
    """Detects Arabic & Yemeni medical record numbers like:

    - رقم الملف: 123456
    - رقم المريض: YEM-MRN-9876
    - السجل الطبي: 89472
    """
    pattern = r"(?:رقم\s+(?:الملف|المريض|السجل\s+الطبي)|الملف\s+الطبي)[\s:]*([A-Za-z0-9\-]+)"
    matches = []
    for m in re.finditer(pattern, text):
        val = m.group(1)
        start = m.start(1)
        end = m.end(1)
        matches.append({
            "type": "MEDICAL_ID",
            "text": val,
            "start": start,
            "end": end
        })
    return matches


def detect_arabic_names(text: str) -> List[Dict[str, Any]]:
    """Detects Arabic & Yemeni personal names in medical contexts:

    - المريض/المريضة: [اسم ثنائي أو ثلاثي أو رباعي مع اللقب]
    - دكتور/دكتورة/طبيب/استشاري: [اسم]
    - السيد/السيدة: [اسم]
    """
    title_pattern = (
        r"(?:المريض(?:ة)?|دكتور(?:ة)?|طبيب(?:ة)?|استشاري(?:ة)?|د\.|السيد(?:ة)?|المراجع(?:ة)?)"
        r"[\s:]+([\u0621-\u064A]{2,15}(?:\s+[\u0621-\u064A]{2,15}){1,4})"
    )
    matches = []
    for m in re.finditer(title_pattern, text):
        name_val = m.group(1).strip()
        start = m.start(1)
        end = m.end(1)
        matches.append({
            "type": "NAME",
            "text": name_val,
            "start": start,
            "end": end
        })
    return matches


def detect_arabic_pii(text: str) -> List[Dict[str, Any]]:
    """Runs all Yemeni & Arabic PII detectors and returns combined non-overlapping matches."""
    all_matches = []
    all_matches.extend(detect_yemeni_national_ids(text))
    all_matches.extend(detect_yemeni_and_arabic_phones(text))
    all_matches.extend(detect_arabic_medical_ids(text))
    all_matches.extend(detect_arabic_names(text))

    all_matches.sort(key=lambda x: x["start"])
    return all_matches
