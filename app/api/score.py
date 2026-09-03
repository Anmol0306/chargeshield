"""POST /score -> {p_fraud, risk band for this amount}"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import ScoreRequest, ScoreResponse
from app.policy.thresholds import load_policy_config
from app.services.risk_service import ModelUnavailable, score

router = APIRouter(tags=["score"])


@router.post("/score", response_model=ScoreResponse)
def score_transaction(request: ScoreRequest) -> ScoreResponse:
    try:
        result = score(request.features)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    band = note = threshold = None
    if request.amount_inr is not None:
        cfg = load_policy_config()
        threshold = cfg.indifference_threshold(request.amount_inr)
        band = cfg.review_band(request.amount_inr)
        note = (
            f"For an INR {request.amount_inr:,.0f} dispute, contesting is "
            f"cost-minimising below p={threshold:.3f}. The bands are "
            f"amount-dependent; this is not a global threshold."
        )

    return ScoreResponse(**result, indifference_threshold=threshold,
                         review_band=band, band_note=note)
