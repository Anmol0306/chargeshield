"""
Build docs/index.html — a static, hostable copy of the interface.

WHY THIS EXISTS
  frontend/index.html needs a running API. A reviewer with a link and no
  checkout cannot run one. This produces a single self-contained page that can
  be served by GitHub Pages from the docs/ folder.

WHAT IS AND IS NOT PRECOMPUTED
  The five scenario decisions are produced HERE, at build time, by the real
  app.policy.action_policy.decide() -- the same function the API calls. They
  are not hand-written, and they are not a JavaScript reimplementation of the
  gate. A second implementation of the decision rules is exactly the drift this
  project argues against elsewhere (see DECISIONS: p*(A) has one
  implementation), so there is none here either.

  The page says so on its face. A viewer should never have to guess whether a
  decision was computed by the engine or typed by its author.

  Metrics come from the committed evaluation artifacts, which is what the live
  /metrics endpoint serves too -- so those are identical either way.

Run: python -m scripts.build_static
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import date

from app.policy.action_policy import decide
from app.policy.thresholds import load_policy_config
from app.services.batch_runner import load_requirements

SRC = pathlib.Path("frontend/index.html")
OUT = pathlib.Path("docs/index.html")

COMPLETE = {"shipping_proof": ["doc_ship_001"], "billing_proof": ["doc_bill_001"]}

# Must stay in step with SCENARIOS in frontend/index.html. A test asserts every
# scenario id in the page has a decision here, so a mismatch fails the build
# rather than showing an error to a viewer.
SCENARIOS = [
    dict(dispute_id="rec_clean", reason_code="NON_RECEIPT", amount_inr=6070.0,
         p_fraud=0.05, evidence=COMPLETE,
         cited=["shipping_proof", "billing_proof"], proposed="CONTEST"),
    dict(dispute_id="rec_fabricated", reason_code="NON_RECEIPT", amount_inr=6070.0,
         p_fraud=0.05, evidence=COMPLETE,
         cited=["shipping_proof", "access_activity_log"], proposed="CONTEST"),
    dict(dispute_id="rec_missing", reason_code="NON_RECEIPT", amount_inr=6070.0,
         p_fraud=0.05, evidence={"billing_proof": ["doc_bill_001"]},
         cited=["billing_proof"], proposed="CONTEST"),
    dict(dispute_id="rec_highp", reason_code="FRAUD", amount_inr=6070.0,
         p_fraud=0.97,
         evidence={"customer_communication": ["doc_c"], "billing_proof": ["doc_b"]},
         cited=["customer_communication", "billing_proof"], proposed="CONTEST"),
    dict(dispute_id="rec_cap", reason_code="NON_RECEIPT", amount_inr=90000.0,
         p_fraud=0.05, evidence=COMPLETE,
         cited=["shipping_proof", "billing_proof"], proposed="CONTEST"),
]

BANNER = """<div class="banner">
<b>Static preview.</b> The four sections below are the real interface. Figures in
&sect;1 and &sect;3 are read from the committed evaluation artifacts, exactly as the
live <code>/metrics</code> endpoint serves them. The determinations in &sect;2 were
produced by the actual policy engine — <code>app/policy/action_policy.py</code> —
at build time, not written by hand and not reimplemented in JavaScript.<br><br>
For the gate running live, one HTTP request per click:
<code>git clone</code> &rarr; <code>make all</code> &rarr; <code>make api</code>.
Built {built} from commit <code>{commit}</code>.
</div>"""


def decision_payload() -> dict:
    cfg, reqs = load_policy_config(), load_requirements()
    out = {}
    for s in SCENARIOS:
        d = decide(config=cfg, p_fraud=s["p_fraud"], amount_inr=s["amount_inr"],
                   reason_code=s["reason_code"], evidence=s["evidence"],
                   requirements=reqs, proposed_action=s["proposed"],
                   cited_evidence=s["cited"])
        out[s["dispute_id"]] = {
            "dispute_id": s["dispute_id"],
            "action": d.action,
            "rule": d.rule,
            "rationale": d.rationale,
            "proposal_honoured": d.proposal_honoured,
            "proposal_source": "constructed",
            "proposed_action": d.proposed_action,
            "missing_required_evidence": list(d.missing_required_evidence),
            "fabricated_evidence": list(d.fabricated_evidence),
            "indifference_threshold": d.indifference_threshold,
            "review_band": list(d.review_band),
            "evaluated": [{"rule": e.rule, "outcome": e.outcome, "detail": e.detail}
                          for e in d.evaluated],
        }
    return out


def metrics_payload() -> dict:
    out = {}
    for key, path in (("evaluation", "evaluation/metrics.json"),
                      ("batch", "evaluation/batch_results.json"),
                      ("calibration", "evaluation/calibration_metrics.json"),
                      ("bands", "artifacts/policy_bands.json")):
        p = pathlib.Path(path)
        out[key] = json.loads(p.read_text()) if p.exists() else None
    return out


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    payload = {"decisions": decision_payload(), "metrics": metrics_payload()}

    html = SRC.read_text()
    inject = ("<script>window.__STATIC__ = "
              + json.dumps(payload, separators=(",", ":"))
              + ";</script>\n")
    html = html.replace("<script>\nconst $ = id =>", inject + "<script>\nconst $ = id =>")
    html = html.replace('<div id="static-banner"></div>',
                        BANNER.format(built=date.today().isoformat(), commit=commit))
    html = html.replace("<title>ChargeShield — decision record</title>",
                        "<title>ChargeShield — decision record</title>\n"
                        '<meta name="description" content="Cost-sensitive fraud '
                        'scoring and bounded chargeback defence. The model '
                        'proposes; a deterministic policy engine decides.">')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    pathlib.Path("docs/.nojekyll").write_text("")   # serve _-prefixed files as-is

    actions = {v["action"] for v in payload["decisions"].values()}
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  scenarios precomputed by the real engine: {len(payload['decisions'])}")
    for k, v in payload["decisions"].items():
        print(f"    {k:<16} {v['action']:<13} {v['rule']}")
    assert len(actions) >= 3, "scenarios should not all reach the same action"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
