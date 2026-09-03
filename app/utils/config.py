"""Load config/*.yaml. No magic numbers anywhere else in the codebase.

Secrets are NOT loaded here and no module in this project reads .env.
Credentials come from the process environment only -- see
app/services/llm_service.py. Keeping the read in one place means the
answer to "what could leak a key?" is one file, not a grep.
"""
