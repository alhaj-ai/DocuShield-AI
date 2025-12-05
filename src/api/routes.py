"""API route endpoints for DocuShield AI."""
import torch
from fastapi import APIRouter, HTTPException

from src.problem3_privacy.pii_detectors import detect_all_pii, guess_language
from src.problem3_privacy.redaction import redact_text
from .schemas import (
    HealthResponse,
    PIIDetectionRequest,
    PIIDetectionResponse,
    PIISpan,
    RedactionRequest,
    RedactionResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Privacy & Document AI"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns system status, GPU acceleration details, and developer metadata."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    return HealthResponse(
        status="healthy",
        service="DocuShield AI - Multilingual & Arabic Document Intelligence",
        version="1.0.0",
        developer="Mohammed AL-Haj",
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        supported_languages=["ar", "en", "fr"],
    )


@router.post("/pii/detect", response_model=PIIDetectionResponse)
def detect_pii_endpoint(payload: PIIDetectionRequest):
    """Detects sensitive PII spans (Names, Emails, Phones, Yemeni National IDs, SSNs, Medical IDs)."""
    lang = payload.language or guess_language(payload.text)
    try:
        entities = detect_all_pii(payload.text)
        spans = [
            PIISpan(
                type=e["type"],
                text=e["text"],
                start=e["start"],
                end=e["end"]
            )
            for e in entities
        ]
        return PIIDetectionResponse(
            detected_language=lang,
            total_count=len(spans),
            entities=spans
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII detection failed: {str(e)}")


@router.post("/pii/redact", response_model=RedactionResponse)
def redact_pii_endpoint(payload: RedactionRequest):
    """Redacts detected PII using 'strict', 'typed', or 'surrogate' modes."""
    lang = payload.language or guess_language(payload.text)
    try:
        entities = detect_all_pii(payload.text)
        redacted, _ = redact_text(
            text=payload.text,
            entities=entities,
            mode=payload.mode,
            language=lang,
            arabic_labels=payload.arabic_labels or (lang == "ar")
        )
        spans = [
            PIISpan(
                type=e["type"],
                text=e["text"],
                start=e["start"],
                end=e["end"]
            )
            for e in entities
        ]
        return RedactionResponse(
            original_text=payload.text,
            redacted_text=redacted,
            mode=payload.mode,
            total_redacted=len(spans),
            detected_pii=spans
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redaction failed: {str(e)}")


@router.post("/document/analyze")
def analyze_document_endpoint(payload: RedactionRequest):
    """Unified pipeline: classifies document type, analyzes layout sections, and redacts PII."""
    lang = payload.language or guess_language(payload.text)
    text_lower = payload.text.lower()
    if any(k in text_lower for k in ("مريض", "مستشفى", "تشخيص", "patient", "diagnosis", "mrn", "ملف طبي")):
        doc_class = "medical_record"
    elif any(k in text_lower for k in ("فاتورة", "ضريبة", "إجمالي", "invoice", "subtotal", "total")):
        doc_class = "financial_invoice"
    elif any(k in text_lower for k in ("عقد", "اتفاقية", "طرف أول", "contract", "agreement")):
        doc_class = "legal_contract"
    else:
        doc_class = "general_document"

    lines = [ln.strip() for ln in payload.text.splitlines() if ln.strip()]
    entities = detect_all_pii(payload.text)
    redacted, _ = redact_text(
        text=payload.text,
        entities=entities,
        mode=payload.mode,
        language=lang,
        arabic_labels=payload.arabic_labels or (lang == "ar"),
    )
    return {
        "document_class": doc_class,
        "detected_language": lang,
        "layout_summary": {
            "line_count": len(lines),
            "has_header": len(lines) > 0,
            "character_count": len(payload.text),
        },
        "total_pii_redacted": len(entities),
        "redacted_text": redacted,
        "detected_pii": entities,
    }
