"""POST /batch/run -> policy comparison over the anchored dispute queue."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.policy.thresholds import load_policy_config
from app.services.audit_service import AuditLog
from app.services.batch_runner import DISPUTES_PATH, load_requirements, run_batch

router = APIRouter(tags=["batch"])

RESULTS_PATH = Path("evaluation/batch_results.json")


@router.post("/batch/run")
def run(limit: int | None = None, write_audit: bool = False) -> dict:
    """Re-run the gate over the dispute queue.

    `write_audit` defaults to False so that hitting this endpoint during a demo
    does not append 5,013 records to the audit log every time.
    """
    if not DISPUTES_PATH.exists():
        raise HTTPException(status_code=503,
                            detail=f"{DISPUTES_PATH} not found — run `make link`")
    disputes = json.loads(DISPUTES_PATH.read_text())["disputes"]
    if limit is not None:
        disputes = disputes[:limit]
    return run_batch(disputes, load_policy_config(), load_requirements(),
                     AuditLog() if write_audit else None)


@router.get("/batch/results")
def results() -> dict:
    """Last committed batch result, without recomputing."""
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="run POST /batch/run first")
    return json.loads(RESULTS_PATH.read_text())
