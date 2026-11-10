"""DocuShield AI - FastAPI Application Entrypoint.

Developed & Extended by Mohammed AL-Haj
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routes import router

app = FastAPI(
    title="DocuShield AI — Multilingual & Arabic Document Intelligence",
    description=(
        "Production-grade privacy-preserving Document AI API. "
        "Supports Arabic, English, and French document analysis, "
        "Saudi National ID & GCC PII anonymization, and Layout-aware processing. "
        "Engineered by Mohammed AL-Haj."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
