"""FastAPI entrypoint. Mounts routers from app/api/. Serves frontend/index.html.

/health reports whether the model artifacts loaded and whether an LLM
credential is present. It reports the PRESENCE of a credential as a boolean and
never any part of its value.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import batch, disputes, score
from app.services import llm_service
from app.services.risk_service import model_info

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="ChargeShield",
    description="Cost-sensitive fraud risk scoring + bounded chargeback defence. "
                "The LLM proposes; a deterministic policy engine decides.",
    version="0.1.0",
)

app.include_router(score.router)
app.include_router(disputes.router)
app.include_router(batch.router)

FRONTEND = Path("frontend/index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": model_info(),
        # Presence only. Never any part of the value.
        "llm_credential_present": llm_service.is_enabled(),
        "llm_fallback": "deterministic template (app/services/template_response.py)",
    }


@app.get("/", include_in_schema=False)
def index():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"service": "ChargeShield", "docs": "/docs"}
