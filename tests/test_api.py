"""Unit tests for DocuShield AI FastAPI endpoints."""
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["developer"] == "Mohammed AL-Haj"
    assert "ar" in data["supported_languages"]


def test_pii_detection_endpoint():
    payload = {
        "text": "المريض أحمد سالم، رقم الهوية 1098765432، هاتف: 0551234567، بريد: ahmed@clinic.sa"
    }
    response = client.post("/api/v1/pii/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["detected_language"] == "ar"
    assert data["total_count"] >= 3
    types = [e["type"] for e in data["entities"]]
    assert "SAUDI_NATIONAL_ID" in types


def test_pii_redaction_endpoint():
    payload = {
        "text": "Report for John Doe, Email: john.doe@hospital.com, SSN: 123-45-6789",
        "mode": "typed"
    }
    response = client.post("/api/v1/pii/redact", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "john.doe@hospital.com" not in data["redacted_text"]
    assert "[EMAIL]" in data["redacted_text"]
    assert "[SSN]" in data["redacted_text"]


if __name__ == "__main__":
    test_health_endpoint()
    test_pii_detection_endpoint()
    test_pii_redaction_endpoint()
    print("\n[SUCCESS] All FastAPI endpoint tests passed!")
