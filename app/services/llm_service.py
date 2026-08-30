"""
Provider-agnostic LLM wrapper. ONE public function:

    propose_response(case: DisputeCase) -> DisputeProposal

Nothing above this file knows which provider answered.

FALLBACK CHAIN — build the template path FIRST, before any API call:
    structured output -> pydantic validation
      -> fail: retry once with the validation error appended
      -> fail: deterministic template (app/services/template_response.py)
      -> always: policy engine validates before anything is filed

SYSTEM PROMPT MUST INCLUDE:
  "Never state that evidence exists unless it is present in the supplied
   evidence set. Never invent customer, delivery, authentication, refund,
   or communication records."
"""
