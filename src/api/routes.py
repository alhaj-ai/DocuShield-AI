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
    """Detects sensitive PII spans (Names, Emails, Phones, Saudi National IDs, SSNs, Medical IDs)."""
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
