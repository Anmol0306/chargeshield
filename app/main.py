"""FastAPI entrypoint. Mounts routers from app/api/. Serves frontend/index.html.

/health reports whether the model artifacts loaded and whether an LLM
credential is present. It reports the PRESENCE of a credential as a boolean and
never any part of its value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import json

from fastapi import FastAPI, HTTPException
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


@app.get("/metrics")
def metrics() -> dict:
    """Everything the dashboard needs, in one call.

    Reads the committed evaluation artifacts rather than recomputing. The page
    is a view of what `make all` produced -- if a number on screen disagrees
    with the repo, the repo is wrong, not the page.
    """
    out: dict = {}
    for key, path in (
        ("evaluation", "evaluation/metrics.json"),
        ("batch", "evaluation/batch_results.json"),
        ("calibration", "evaluation/calibration_metrics.json"),
        ("bands", "artifacts/policy_bands.json"),
    ):
        p = Path(path)
        out[key] = json.loads(p.read_text()) if p.exists() else None
    if all(v is None for v in out.values()):
        raise HTTPException(status_code=503, detail="run `make all` first")
    return out


@app.get("/", include_in_schema=False)
def index():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"service": "ChargeShield", "docs": "/docs"}
