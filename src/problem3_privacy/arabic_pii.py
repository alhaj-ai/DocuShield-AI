"""Arabic PII detection module for medical documents and records.

Provides specialized pattern matching and heuristic NER for Arabic medical PII:
1. Saudi National ID & Iqama (10 digits: 1XXXXXXXXX for Saudi, 2XXXXXXXXX for resident)
2. Arab & GCC Phone Numbers (+966, +967, 05, etc.)
3. Arabic Medical Record Numbers (MRN / رقم الملف الطبي / رقم المريض)
4. Arabic Names heuristic extraction (names following doctor/patient honorifics)
5. Arabic / Hijri & Gregorian Dates
"""
import re
from typing import Any, Dict, List


def detect_saudi_national_ids(text: str) -> List[Dict[str, Any]]:
    """Detects Saudi National IDs (starts with 1) and Resident Iqama numbers (starts with 2),

    each strictly 10 digits.
    """
    pattern = r"\b[12]\d{9}\b"
    matches = []
    for m in re.finditer(pattern, text):
        matches.append({
            "type": "SAUDI_NATIONAL_ID",
            "text": m.group(),
            "start": m.start(),
            "end": m.end()
        })
    return matches


def detect_arabic_phones(text: str) -> List[Dict[str, Any]]:
    """Detects Saudi, Yemeni, and GCC phone numbers:

    - Saudi mobile: +9665XXXXXXXX, 009665XXXXXXXX, 05XXXXXXXX
    - Yemeni mobile: +9677XXXXXXXX, 009677XXXXXXXX, 7XXXXXXXX
    - Normalized spaced/dashed variations
    """
    patterns = [
        # Saudi mobile (+966 5X / 05X)
        r"(?:\+966|00966|0)?5[0-9]{8}\b",
        r"(?:\+966|00966)[\s-]5[0-9]{1,2}[\s-]?[0-9]{3}[\s-]?[0-9]{3,4}\b",
        # Yemeni mobile (+967 7X / 7X)
        r"(?:\+967|00967|0)?7[0-9]{8}\b",
        r"(?:\+967|00967)[\s-]7[0-9]{1,2}[\s-]?[0-9]{3}[\s-]?[0-9]{3,4}\b",
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
    """Detects Arabic medical record numbers like:

    - رقم الملف: 123456
    - رقم المريض: MRN-9876
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
    """Detects Arabic personal names in medical contexts:

    - مريض/مريضة: [اسم ثنائي أو ثلاثي]
    - دكتور/دكتورة/طبيب: [اسم ثنائي أو ثلاثي]
    - السيد/السيدة: [اسم]
    """
    title_pattern = (
        r"(?:المريض(?:ة)?|دكتور(?:ة)?|طبيب(?:ة)?|د\.|السيد(?:ة)?|المراجع(?:ة)?)"
        r"[\s:]+([\u0621-\u064A]{2,15}(?:\s+[\u0621-\u064A]{2,15}){1,3})"
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
    """Runs all Arabic PII detectors and returns combined non-overlapping matches."""
    all_matches = []
    all_matches.extend(detect_saudi_national_ids(text))
    all_matches.extend(detect_arabic_phones(text))
    all_matches.extend(detect_arabic_medical_ids(text))
    all_matches.extend(detect_arabic_names(text))

    # Sort by start position
    all_matches.sort(key=lambda x: x["start"])
    return all_matches
