"""Simple PII detection for English, French, and Arabic medical documents.

Provides hybrid detection strategies:
1. Regex patterns for structured PII: emails, phones, SSNs, Saudi National IDs, medical IDs, dates.
2. Lightweight spaCy models for English/French names, and heuristic extraction for Arabic names.
"""
import re
from typing import Any, Dict, List

from .arabic_pii import detect_arabic_pii

# Lazy load spaCy models to prevent import crashes if models are missing
_nlp_en = None
_nlp_fr = None


def _get_nlp(lang: str):
    global _nlp_en, _nlp_fr
    try:
        import spacy
        if lang == "fr":
            if _nlp_fr is None:
                try:
                    _nlp_fr = spacy.load("fr_core_news_sm")
                except Exception:
                    _nlp_fr = spacy.blank("fr")
            return _nlp_fr
        else:
            if _nlp_en is None:
                try:
                    _nlp_en = spacy.load("en_core_web_sm")
                except Exception:
                    _nlp_en = spacy.blank("en")
            return _nlp_en
    except ImportError:
        return None


def detect_emails(text: str) -> List[Dict[str, Any]]:
    pattern = r"[\w.+-]+@[\w-]+\.[\w.-]+"
    matches = []
    for m in re.finditer(pattern, text):
        matches.append({"type": "EMAIL", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


def detect_phones(text: str) -> List[Dict[str, Any]]:
    # French mobile/landline numbers, e.g. "+33 6 12 34 56 78" or "06 12 34 56 78"
    fr_pattern = r"(?:\+33\s?\(?0?\)?|0)[1-9](?:[\s.-]?\d{2}){4}"
    # US/generic numbers, e.g. "(555) 123-4567" or "555-123-4567"
    us_pattern = r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"

    matches = []
    for pattern in (fr_pattern, us_pattern):
        for m in re.finditer(pattern, text):
            matches.append({"type": "PHONE", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


def detect_ssns(text: str) -> List[Dict[str, Any]]:
    # French social security number: 15 digits, usually written in groups,
    # e.g. "2 78 03 75 116 025 43"
    fr_pattern = r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"
    # US social security number, e.g. "123-45-6789"
    us_pattern = r"\b\d{3}-\d{2}-\d{4}\b"

    matches = []
    for pattern in (fr_pattern, us_pattern):
        for m in re.finditer(pattern, text):
            matches.append({"type": "SSN", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


def detect_medical_ids(text: str) -> List[Dict[str, Any]]:
    # Matches things like "FR789456123", "MD-87483", "PT-988598"
    patterns = [
        r"\bFR\d{9}\b",
        r"\bMD-\d{4,6}\b",
        r"\bPT-\d{4,6}\b",
    ]
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            matches.append({"type": "MEDICAL_ID", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


def detect_dates(text: str) -> List[Dict[str, Any]]:
    # Numeric dates like "15/03/1978", "2024-05-18", or "15-03-2024"
    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
    ]
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            matches.append({"type": "DATE", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


def detect_addresses(text: str) -> List[Dict[str, Any]]:
    fr_pattern = (r"\d{1,4}[,]?\s+(?:Rue|Avenue|Boulevard|Chemin|All[ée]e|Impasse|Place|Quai)"
                  r"\s+[^,\n]+,\s*\d{5}\s+[A-Za-zÀ-ÿ\- ]+")
    us_pattern = r"\d{1,6}[A-Za-z0-9 .,'\-]{5,60},\s*[A-Za-z .'\-]+,\s*[A-Z]{2}\s*\d{5}"

    matches = []
    for m in re.finditer(fr_pattern, text, flags=re.IGNORECASE):
        matches.append({"type": "ADDRESS", "text": m.group(), "start": m.start(), "end": m.end()})
    for m in re.finditer(us_pattern, text):
        matches.append({"type": "ADDRESS", "text": m.group(), "start": m.start(), "end": m.end()})
    return matches


_FRENCH_WORDS = {"le", "la", "les", "de", "des", "du", "et", "est", "une", "un",
                 "au", "aux", "pour", "avec", "patient", "medecin", "medical"}
_ENGLISH_WORDS = {"the", "and", "is", "of", "a", "an", "for", "with",
                  "patient", "physician", "medical"}


def guess_language(text: str) -> str:
    """Guesses language: 'ar' for Arabic, 'fr' for French, 'en' for English."""
    # Check for Arabic characters
    if len(re.findall(r"[\u0600-\u06FF]", text)) >= 3:
        return "ar"
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", text.lower())
    fr_count = sum(1 for w in words if w in _FRENCH_WORDS)
    en_count = sum(1 for w in words if w in _ENGLISH_WORDS)
    return "fr" if fr_count > en_count else "en"


def detect_names(text: str) -> List[Dict[str, Any]]:
    lang = guess_language(text)
    if lang == "ar":
        # Arabic names are handled via arabic_pii heuristic detectors
        return []
    nlp = _get_nlp(lang)
    if nlp is None:
        return []
    matches = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("PER", "PERSON"):
            matches.append({"type": "NAME", "text": ent.text, "start": ent.start_char, "end": ent.end_char})
    return matches


def _remove_overlaps(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches = sorted(matches, key=lambda m: (m["end"] - m["start"]), reverse=True)
    kept = []
    for m in matches:
        overlaps = False
        for k in kept:
            if m["start"] < k["end"] and m["end"] > k["start"]:
                overlaps = True
                break
        if not overlaps:
            kept.append(m)
    kept.sort(key=lambda m: m["start"])
    return kept


def detect_all_pii(text: str) -> List[Dict[str, Any]]:
    """Runs every detector (English, French, and Arabic) and returns merged,

    non-overlapping PII spans.
    """
    all_matches = []
    # Western / Multilingual detectors
    all_matches += detect_names(text)
    all_matches += detect_addresses(text)
    all_matches += detect_medical_ids(text)
    all_matches += detect_ssns(text)
    all_matches += detect_phones(text)
    all_matches += detect_emails(text)
    all_matches += detect_dates(text)
    # Arabic specialized detectors
    all_matches += detect_arabic_pii(text)
    return _remove_overlaps(all_matches)
