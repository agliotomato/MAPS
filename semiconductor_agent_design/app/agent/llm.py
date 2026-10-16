"""
LLM client.

Default provider is OpenAI, with optional Anthropic support.
"""
from __future__ import annotations

import json
import logging
import os
import re

_log = logging.getLogger(__name__)

_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "openai").strip().lower()
_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")
_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))
_CALL_BUDGET = int(os.getenv("AGENT_LLM_CALL_BUDGET", "50"))
_CALL_COUNT = 0

_client = None


def remaining_call_budget() -> int | None:
    """Return remaining LLM call budget, or None when budget is disabled."""
    if _CALL_BUDGET <= 0:
        return None
    return max(0, _CALL_BUDGET - _CALL_COUNT)


def _consume_call_budget() -> None:
    global _CALL_COUNT
    if _CALL_BUDGET <= 0:
        return
    if _CALL_COUNT >= _CALL_BUDGET:
        raise RuntimeError(
            f"LLM call budget exhausted: {_CALL_COUNT}/{_CALL_BUDGET}. "
            "Increase AGENT_LLM_CALL_BUDGET to allow more calls."
        )
    _CALL_COUNT += 1


def get_client():
    global _client
    if _client is None:
        if _PROVIDER == "anthropic":
            import anthropic

            _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif _PROVIDER == "openai":
            from openai import OpenAI

            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError(f"Unsupported AGENT_LLM_PROVIDER: {_PROVIDER}")
    return _client


def call_llm(system: str, user: str) -> str:
    """Return a free-form response."""
    _consume_call_budget()
    client = get_client()

    if _PROVIDER == "anthropic":
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def call_llm_json(system: str, user: str) -> dict:
    """
    Return JSON output from the model.
    """
    raw = call_llm(system, user)

    # Strip code fences if the model returns them.
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        _log.warning("JSON parse failed | raw=%s", raw[:300])
        return {}
