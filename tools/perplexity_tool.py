#!/usr/bin/env python3
"""
Perplexity Tool — research-grounded question answering via Perplexity API.

Uses the OpenAI-compatible chat completions endpoint at api.perplexity.ai.
Perplexity models natively search the web and return citation-backed answers,
making this ideal for research, fact-checking, and current-events queries.

Available models:
- sonar           — fast, lightweight, default
- sonar-pro       — premium search-grounded research
- sonar-reasoning — deep reasoning with chain-of-thought
- sonar-deep-research — exhaustive multi-step research (expensive)

Requires: PERPLEXITY_API_KEY env var (from https://docs.perplexity.ai)
"""

import json
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# Which models support the "search" domain filter and web search options
_CITATION_MODELS = {"sonar", "sonar-pro", "sonar-reasoning", "sonar-deep-research"}

# Retry configuration
_MAX_RETRIES = 2
_RETRY_BACKOFF = 2.0  # seconds, multiplied by attempt number
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _get_api_key() -> str:
    """Return the Perplexity API key, checking env vars and a sidecar file."""
    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if key and key != "***":
        return key
    # Fallback: sidecar file in ~/.hermes/
    sidecar = os.path.join(
        os.path.expanduser("~/.hermes"), "perplexity_key.txt",
    )
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as fh:
                val = fh.read().strip()
            if val and val != "***":
                return val
        except OSError:
            pass
    return ""


def perplexity_ask(
    query: str,
    model: str = "sonar",
    temperature: float = 0.2,
) -> str:
    """Send a research question to Perplexity and return the grounded answer.

    Args:
        query: The question or research topic to investigate.
        model: Perplexity model to use. One of: sonar, sonar-pro,
               sonar-reasoning, sonar-deep-research. Default: sonar.
        temperature: Sampling temperature (0.0–2.0). Default 0.2.

    Returns:
        JSON string with ``answer`` and ``citations`` (URLs when available).
    """
    api_key = _get_api_key()
    if not api_key:
        return json.dumps({
            "error": "PERPLEXITY_API_KEY not set",
            "hint": "Set PERPLEXITY_API_KEY env var or write key to ~/.hermes/perplexity_key.txt",
        })

    if model not in _CITATION_MODELS:
        model = "sonar"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant. Answer the question "
                "thoroughly with citations to sources. Be concise but complete. "
                "When appropriate, structure the answer with clear sections."
            ),
        },
        {"role": "user", "content": query},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": max(0.0, min(2.0, temperature)),
        "max_tokens": 4096,
    }

    # Enable web search and citations for citation-capable models
    if model in {"sonar-pro", "sonar-reasoning", "sonar-deep-research"}:
        payload["search_domain_filter"] = None  # no domain restriction
        payload["return_images"] = False
        payload["return_related_questions"] = False
        payload["search_recency_filter"] = None

    last_error = None
    data = None
    for attempt in range(1 + _MAX_RETRIES):  # 1 initial + N retries
        try:
            response = httpx.post(
                f"{PERPLEXITY_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            if response.status_code == 200:
                data = response.json()
                break  # success
            # On rate-limit or server error, retry; on auth errors, fail fast
            if response.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Perplexity %d on attempt %d, retrying in %.1fs",
                    response.status_code, attempt + 1, delay,
                )
                last_error = (response.status_code, response.text[:500])
                time.sleep(delay)
                continue
            # Non-retryable — fail immediately
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Perplexity API error: %s %s", exc.response.status_code, exc.response.text[:500])
            return json.dumps({
                "error": f"Perplexity API returned {exc.response.status_code}",
                "detail": exc.response.text[:500],
                "fallback": "Use web_search with the same query instead.",
            })
        except httpx.RequestError as exc:
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF * (attempt + 1)
                logger.warning("Perplexity network error (attempt %d): %s — retrying in %.1fs", attempt + 1, exc, delay)
                last_error = exc
                time.sleep(delay)
                continue
            logger.error("Perplexity network error after %d retries: %s", _MAX_RETRIES, exc)
            return json.dumps({
                "error": f"Network error contacting Perplexity after {_MAX_RETRIES} retries: {exc}",
                "fallback": "Use web_search with the same query instead.",
            })

    if data is None:
        # All retries exhausted without a successful response
        status = last_error[0] if isinstance(last_error, tuple) else "network"
        detail = last_error[1] if isinstance(last_error, tuple) else str(last_error)
        return json.dumps({
            "error": f"Perplexity unavailable after {_MAX_RETRIES} retries (last: {status})",
            "detail": detail[:500],
            "fallback": "Use web_search with the same query instead.",
        })

    # Extract the assistant message
    choices = data.get("choices", [])
    if not choices:
        return json.dumps({"error": "No response from Perplexity", "raw": data})

    message = choices[0].get("message", {})
    answer = message.get("content", "")

    # Collect citations from the response
    citations = data.get("citations", [])

    # Perplexity sometimes returns citations in the message context
    if not citations:
        citations = message.get("citations", [])

    result = {
        "answer": answer,
        "model": data.get("model", model),
        "citations": citations if citations else None,
    }

    # Include usage for budgeting
    if "usage" in data:
        result["usage"] = data["usage"]

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

PERPLEXITY_ASK_SCHEMA = {
    "name": "perplexity_ask",
    "description": (
        "Research a question using Perplexity AI, which searches the web and "
        "returns a citation-backed answer. Use this for fact-checking, deep "
        "research on current events or specialized topics, academic questions, "
        "or any query that benefits from grounded, web-searched responses with "
        "citations to real sources. "
        "Models: sonar (fast, default), sonar-pro (premium research), "
        "sonar-reasoning (deep reasoning), sonar-deep-research (exhaustive)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research question or topic to investigate.",
            },
            "model": {
                "type": "string",
                "enum": ["sonar", "sonar-pro", "sonar-reasoning", "sonar-deep-research"],
                "description": (
                    "Perplexity model to use. sonar=fast/general, sonar-pro=premium "
                    "search-grounded, sonar-reasoning=deep chain-of-thought, "
                    "sonar-deep-research=exhaustive multi-step research."
                ),
                "default": "sonar",
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature 0.0–2.0. Lower = more deterministic. Default 0.2.",
                "minimum": 0.0,
                "maximum": 2.0,
                "default": 0.2,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry


def check_perplexity_api_key() -> bool:
    """Return True when the Perplexity API key is configured."""
    return bool(_get_api_key())


registry.register(
    name="perplexity_ask",
    toolset="research",
    schema=PERPLEXITY_ASK_SCHEMA,
    handler=lambda args, **kw: perplexity_ask(
        query=args.get("query", ""),
        model=args.get("model", "sonar"),
        temperature=args.get("temperature", 0.2),
    ),
    check_fn=check_perplexity_api_key,
    requires_env=["PERPLEXITY_API_KEY"],
    emoji="🔎",
    max_result_size_chars=100_000,
)
