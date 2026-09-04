"""
LLM proposal generation. The ONLY module that knows which provider is in use.

CONTRACT
    propose(...) -> (DisputeProposal, source)   source in {"llm", "template"}

  It always returns a valid DisputeProposal. There is no failure mode that
  propagates to the caller: no key, no network, a timeout, malformed JSON, or
  schema violations twice in a row all end at the deterministic template in
  app/services/template_response.py. The pipeline cannot be broken by the model
  provider.

SECRET HANDLING
  The key is read from the environment (LLM_API_KEY, falling back to
  OPENAI_API_KEY) and handed to the client constructor. It is never logged,
  never printed, never placed in an exception message, never written to the
  audit log, and never returned. `_scrub()` is defence in depth: every string
  this module logs passes through it, so even a provider exception that
  somehow echoed a credential cannot reach a log line. This module never reads
  a .env file -- the environment is the interface.

WHAT IS SENT TO THE PROVIDER
  The reason code, the dispute amount, the evidence FIELD NAMES on file, and
  the requirement list for that reason code. Deliberately NOT sent:

    - document ids       opaque, and the model has no use for them
    - p_fraud            the model's job is evidence assessment and drafting.
                         The fraud probability is the deterministic system's
                         input. Keeping them apart means the LLM cannot
                         launder a score into a decision, and keeps the roles
                         clean: LLM does language, model does probability,
                         policy does deciding.
    - anything from .env or the raw dataset

VALIDATION IS SHAPE, NOT CONTENT
  Responses are validated against DisputeProposal. Fabricated evidence
  citations are NOT stripped here, on purpose: if this module quietly repaired
  them, app/policy/action_policy.py's fabrication rule would never fire and
  there would be no audit record showing the model tried. Content trust is the
  gate's job. This module's job is to guarantee a well-formed object.

RETRY POLICY
  One retry, then the template. A provider returning garbage twice is a
  provider to stop calling, not one to keep paying.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from pydantic import ValidationError

from app.models.schemas import DisputeProposal
from app.services.template_response import build_template_proposal

log = logging.getLogger(__name__)

API_KEY_VARS = ("LLM_API_KEY", "OPENAI_API_KEY")
MODEL_VAR = "LLM_MODEL"
BASE_URL_VAR = "LLM_BASE_URL"
TIMEOUT_VAR = "LLM_TIMEOUT_SECONDS"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 20.0
MAX_ATTEMPTS = 2          # initial call + one retry

SYSTEM_PROMPT = """\
You assess chargeback evidence for a merchant and draft representment text.

You are a PROPOSER, not a decider. A deterministic policy engine reviews
everything you produce and will override it. Do not attempt to influence that
decision with emphasis or urgency.

Absolute rules:
1. You may only cite evidence types from the EVIDENCE ON FILE list you are
   given. Never cite, mention, imply or assume any document that is not in
   that list. If a useful document is absent, say it is missing.
2. Never assert a fact that the listed documents do not evidence. Do not
   speculate about the cardholder's intent or state of mind.
3. If any required evidence is missing, set decision to "REVIEW" and
   evidence_status to "INSUFFICIENT", and leave draft_representment empty.
4. Respond with a single JSON object and nothing else.

JSON shape:
{
  "decision": "CONTEST" | "ACCEPT" | "REVIEW",
  "evidence_status": "SUFFICIENT" | "INSUFFICIENT",
  "missing_evidence": [string],
  "cited_evidence": [string],
  "reasoning_summary": string,
  "draft_representment": string
}

`cited_evidence` must be a subset of EVIDENCE ON FILE. Fabricating a citation
is the single worst thing you can do here: it would put a false claim in front
of a card network.\
"""


@dataclass(frozen=True)
class LLMSettings:
    model: str = DEFAULT_MODEL
    base_url: str | None = None
    timeout: float = DEFAULT_TIMEOUT


def _api_key() -> str | None:
    """First non-empty of LLM_API_KEY, OPENAI_API_KEY. Never logged."""
    for var in API_KEY_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def _scrub(text: str) -> str:
    """Remove any credential substring before anything is logged.

    Defence in depth. Nothing here is supposed to contain the key, and this
    exists so that "supposed to" is not the only thing standing between a
    provider's exception text and a log file.
    """
    out = text
    for var in API_KEY_VARS:
        secret = os.environ.get(var, "").strip()
        if secret and secret in out:
            out = out.replace(secret, "***REDACTED***")
    return out


def is_enabled() -> bool:
    """True when a key is present. Never reveals any part of it."""
    return _api_key() is not None


def load_settings() -> LLMSettings:
    try:
        timeout = float(os.environ.get(TIMEOUT_VAR, DEFAULT_TIMEOUT))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return LLMSettings(
        model=os.environ.get(MODEL_VAR, DEFAULT_MODEL),
        base_url=os.environ.get(BASE_URL_VAR) or None,
        timeout=timeout,
    )


def build_user_prompt(*, reason_code: str, amount_inr: float,
                      evidence: dict, requirements: dict) -> str:
    """Only field names, never document ids. Never p_fraud."""
    present = sorted(k for k, v in evidence.items() if v)
    spec = requirements.get(reason_code, {})
    return (
        f"DISPUTE\n"
        f"  reason_code: {reason_code}\n"
        f"  amount_inr: {amount_inr:,.2f}\n\n"
        f"EVIDENCE ON FILE (the complete list; nothing else exists):\n"
        + ("\n".join(f"  - {p}" for p in present) if present else "  (none)")
        + f"\n\nREQUIRED for this reason code:\n"
        + ("\n".join(f"  - {r}" for r in spec.get("required", [])) or "  (unknown)")
        + f"\n\nOPTIONAL for this reason code:\n"
        + ("\n".join(f"  - {o}" for o in spec.get("optional", [])) or "  (none)")
        + "\n\nRespond with the JSON object only."
    )


def _looks_like_unsupported_parameter(exc: Exception) -> bool:
    """Did the provider reject a REQUEST PARAMETER rather than the request?

    JSON mode is not universally supported. OpenAI accepts it, Groq accepts it
    on most models, some models on some gateways do not. A 400 naming
    response_format means "drop that parameter", not "give up" -- and since the
    system prompt already demands a bare JSON object, dropping it usually still
    yields parseable output.

    Inspecting the message here is safe: nothing is logged from it, and the
    decision it drives is a retry shape, not a credential path.
    """
    if getattr(exc, "status_code", None) not in (400, 404, 422):
        return False
    blob = f"{getattr(exc, 'code', '')} {exc}".lower()
    return "response_format" in blob or "json_object" in blob or "json mode" in blob


def _call_provider(prompt: str, settings: LLMSettings, api_key: str,
                   json_mode: bool = True) -> str:
    """Single provider call. Imported lazily so the template path works even if
    the provider SDK is not installed.

    `json_mode` is separable because it is the one parameter that varies across
    OpenAI-compatible providers. Everything else in this call is portable, which
    is what lets LLM_BASE_URL + LLM_MODEL swap providers with no code change.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=settings.base_url,
                    timeout=settings.timeout)
    kwargs = {
        "model": settings.model,
        "temperature": 0,       # a compliance artifact should not be creative
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _invoke(call_provider, prompt, settings, api_key, json_mode: bool) -> str:
    """Call the provider, passing json_mode only if the callable accepts it.

    Tests inject a 3-argument stub; the real client takes a fourth. Inspecting
    the signature keeps every existing test working without each one having to
    know about a parameter it does not care about.
    """
    import inspect

    try:
        params = inspect.signature(call_provider).parameters
        if len(params) >= 4:
            return call_provider(prompt, settings, api_key, json_mode)
    except (TypeError, ValueError):
        pass
    return call_provider(prompt, settings, api_key)


def propose(
    *,
    reason_code: str,
    amount_inr: float,
    evidence: dict,
    requirements: dict,
    dispute_id: str | None = None,
    settings: LLMSettings | None = None,
    call_provider=_call_provider,
) -> tuple[DisputeProposal, str]:
    """Return (proposal, source). Never raises.

    `call_provider` is injectable so tests can exercise malformed responses,
    timeouts and schema violations without a network or a key.
    """
    def template() -> tuple[DisputeProposal, str]:
        return build_template_proposal(
            reason_code=reason_code, evidence=evidence,
            requirements=requirements, amount_inr=amount_inr,
            dispute_id=dispute_id,
        ), "template"

    api_key = _api_key()
    if api_key is None:
        log.info("no LLM credential in environment; using deterministic template")
        return template()

    settings = settings or load_settings()
    prompt = build_user_prompt(reason_code=reason_code, amount_inr=amount_inr,
                               evidence=evidence, requirements=requirements)

    json_mode = True
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = _invoke(call_provider, prompt, settings, api_key, json_mode)
            return DisputeProposal.model_validate(json.loads(raw)), "llm"
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("LLM response rejected (attempt %d/%d): %s",
                        attempt, MAX_ATTEMPTS, _scrub(str(exc))[:400])
        except Exception as exc:                      # provider/network/timeout
            # Log the TYPE, not the message: provider exceptions can carry
            # request context, and the type is what actually distinguishes a
            # timeout from an auth failure.
            log.warning("LLM call failed (attempt %d/%d): %s",
                        attempt, MAX_ATTEMPTS, type(exc).__name__)
            if json_mode and _looks_like_unsupported_parameter(exc):
                log.info("provider rejected JSON mode; retrying without it")
                json_mode = False

    log.info("LLM unusable after %d attempts; using deterministic template",
             MAX_ATTEMPTS)
    return template()
