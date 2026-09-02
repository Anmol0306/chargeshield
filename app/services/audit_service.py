"""One JSON record per decision. Append-only.

WHY THIS IS NOT OPTIONAL
  "The policy engine overrode the LLM" is only a claim you can defend if you
  can produce the record showing which rule fired, on what inputs, under which
  cost assumptions. A gate without an audit trail is a gate you are asking
  people to take on trust, which is the thing this project exists not to do.

WHAT EACH RECORD MUST CONTAIN
  Enough to RECONSTRUCT the decision without the code that made it: the inputs,
  the rule that fired, the bands in force at the time, and a fingerprint of the
  cost assumptions. Thresholds move when config/costs.yaml changes, so a record
  that only stored the action would silently become unexplainable.

JSONL, not SQLite, for now
  Append-only, greppable, no schema migration, and trivially diffable in a
  demo. db_models.py's SQLite tables remain the Tier-2 path if the batch runner
  needs querying. See DECISIONS.md.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("evidence/audit_log.jsonl")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def build_record(
    *,
    dispute_id: str,
    decision: Any,
    config: Any,
    proposal: dict | None = None,
    proposal_source: str = "none",
) -> dict:
    """Assemble one audit record. Pure -- returns a dict, writes nothing."""
    d = _jsonable(decision)
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dispute_id": dispute_id,
        "action": d["action"],
        "rule": d["rule"],
        "rationale": d["rationale"],
        "proposal_honoured": d["proposal_honoured"],
        "inputs": {
            "p_fraud": d["p_fraud"],
            "amount_inr": d["amount_inr"],
        },
        "bands_in_force": {
            "indifference_threshold": d["indifference_threshold"],
            "review_band": d["review_band"],
            "amount_cap_inr": config.auto_action_amount_cap_inr,
        },
        "evidence": {
            "missing_required": d["missing_required_evidence"],
            "fabricated": d["fabricated_evidence"],
        },
        # Fingerprint: thresholds move when these move, so a record without
        # them becomes unexplainable the moment config/costs.yaml is edited.
        "cost_assumptions": {
            "scenario": config.scenario,
            "representment_cost_inr": config.representment_cost_inr,
            "assumed_win_rate_if_legitimate": config.assumed_win_rate_if_legitimate,
            "human_review_cost_inr": config.human_review_cost_inr,
            "assumed_dispute_fraud_rate": config.assumed_dispute_fraud_rate,
            "source": config.source,
        },
        "proposal": {
            "source": proposal_source,
            "proposed_action": d.get("proposed_action"),
            "body": proposal,
        },
    }


class AuditLog:
    """Append-only JSONL writer.

    Opened in append mode per write rather than held open, so a crash mid-batch
    leaves every record before it intact and readable.
    """

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def append_many(self, records: list[dict]) -> None:
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def find(self, dispute_id: str) -> dict | None:
        """Last record for a dispute. GET /audit/{id} is built on this."""
        match = [r for r in self.read_all() if r["dispute_id"] == dispute_id]
        return match[-1] if match else None

    def reset(self) -> None:
        """Truncate. For batch reruns and tests -- never called by the API."""
        if self.path.exists():
            self.path.unlink()
