"""Pydantic schemas for DocuShield AI API."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PIISpan(BaseModel):
    type: str = Field(..., description="Type of PII (e.g. SAUDI_NATIONAL_ID, PHONE, EMAIL, NAME)")
    text: str = Field(..., description="The raw PII text extracted")
    start: int = Field(..., description="Character start index")
    end: int = Field(..., description="Character end index")


class PIIDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to scan for sensitive PII")
    language: Optional[str] = Field(None, description="Optional language override ('ar', 'en', 'fr')")


class PIIDetectionResponse(BaseModel):
    detected_language: str
    total_count: int
    entities: List[PIISpan]


class RedactionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to redact")
    mode: Literal["strict", "typed", "surrogate"] = Field("typed", description="Redaction strategy")
    language: Optional[str] = Field(None, description="Language override ('ar', 'en', 'fr')")
    arabic_labels: bool = Field(False, description="Use Arabic label markers (e.g. [هوية_وطنية], [محجوب])")


class RedactionResponse(BaseModel):
    original_text: str
    redacted_text: str
    mode: str
    total_redacted: int
    detected_pii: List[PIISpan]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    developer: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    supported_languages: List[str]
