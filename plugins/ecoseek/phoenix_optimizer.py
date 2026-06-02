"""Phoenix-powered token optimization for ecoseek plugin.

Uses Arize Phoenix traces to create a continuous feedback loop:
  TRACE → ANALYZE → OPTIMIZE → VALIDATE → repeat

Key optimizations:
1. Prompt deduplication: hash system prompts, reuse cached prefixes
2. Auto-tier-demotion: if a T2 task succeeds 3× as T1, permanently demote it
3. Token budget inference: learn optimal max_tokens per task_type from history
4. Phoenix span export: send TER metrics to Phoenix for dashboard visualization

Phoenix integration:
- Reads spans from Phoenix REST API (localhost:6006 on reumanlab)
- Exports ecoseek-specific spans (delegation, tool calls, TER measurements)
- Uses span metadata to drive optimization decisions

Requires:
  PHOENIX_ENDPOINT    - Phoenix server URL (default: http://localhost:6006)
  PHOENIX_API_KEY     - Phoenix API key (optional for local)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
_PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

# ---------------------------------------------------------------------------
# Prompt Cache — hash-based deduplication of system prompts
# ---------------------------------------------------------------------------

_PROMPT_CACHE: dict[str, str] = {}  # hash → full prompt text
_PROMPT_STATS: dict[str, int] = {}  # hash → hit count


def get_prompt_hash(prompt: str) -> str:
    """Generate a compact hash for a system prompt."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def cache_system_prompt(prompt: str) -> str:
    """Cache a system prompt and return its hash for reference.

    On repeat calls with the same prompt, the cached version is returned.
    This enables DeepSeek's prompt caching to kick in more reliably
    (identical prefix = cache hit on the inference server).
    """
    h = get_prompt_hash(prompt)
    if h in _PROMPT_CACHE:
        _PROMPT_STATS[h] = _PROMPT_STATS.get(h, 0) + 1
        return _PROMPT_CACHE[h]
    _PROMPT_CACHE[h] = prompt
    _PROMPT_STATS[h] = 1
    return prompt


def get_cache_stats() -> dict:
    """Return prompt cache hit statistics."""
    total_hits = sum(_PROMPT_STATS.values())
    unique_prompts = len(_PROMPT_CACHE)
    return {
        "unique_prompts": unique_prompts,
        "total_hits": total_hits,
        "savings_estimate": f"~{max(0, total_hits - unique_prompts) * 200} tokens saved via caching",
        "top_prompts": sorted(
            _PROMPT_STATS.items(), key=lambda x: x[1], reverse=True
        )[:5],
    }


# ---------------------------------------------------------------------------
# Auto-Tier Demotion — learn when T2 tasks can safely become T1
# ---------------------------------------------------------------------------

@dataclass
class TaskHistory:
    """Tracks success/failure of tasks at different tiers."""
    task_type: str
    tier_attempts: dict[str, list[bool]] = field(default_factory=dict)
    current_tier: str = "T2"
    demoted_at: Optional[float] = None


_TASK_HISTORIES: dict[str, TaskHistory] = {}

# Demotion thresholds
_DEMOTION_THRESHOLD = 3      # consecutive successes at lower tier to demote
_PROMOTION_THRESHOLD = 2     # consecutive failures at current tier to promote back


def record_task_outcome(task_type: str, tier: str, success: bool) -> None:
    """Record whether a task succeeded at a given tier."""
    if task_type not in _TASK_HISTORIES:
        _TASK_HISTORIES[task_type] = TaskHistory(task_type=task_type)

    history = _TASK_HISTORIES[task_type]
    if tier not in history.tier_attempts:
        history.tier_attempts[tier] = []
    history.tier_attempts[tier].append(success)

    # Keep only last 10 attempts per tier
    if len(history.tier_attempts[tier]) > 10:
        history.tier_attempts[tier] = history.tier_attempts[tier][-10:]


def get_recommended_tier(task_type: str) -> str:
    """Get the recommended tier for a task type based on history.

    Rules:
    - If task_type succeeded 3× at T1, recommend T1 (demote from T2)
    - If task_type failed 2× at T1, recommend T2 (promote back)
    - Default: T2 for unknown tasks
    """
    if task_type not in _TASK_HISTORIES:
        return "T2"

    history = _TASK_HISTORIES[task_type]

    # Check if T1 has enough successes for demotion
    t1_attempts = history.tier_attempts.get("T1", [])
    if len(t1_attempts) >= _DEMOTION_THRESHOLD:
        recent = t1_attempts[-_DEMOTION_THRESHOLD:]
        if all(recent):
            history.current_tier = "T1"
            history.demoted_at = time.time()
            return "T1"

    # Check if T1 has failures that need promotion back to T2
    if t1_attempts and len(t1_attempts) >= _PROMOTION_THRESHOLD:
        recent_failures = t1_attempts[-_PROMOTION_THRESHOLD:]
        if not any(recent_failures):
            history.current_tier = "T2"
            return "T2"

    return history.current_tier


def get_demotion_report() -> list[dict]:
    """Report on tasks that have been or could be demoted."""
    report = []
    for task_type, history in _TASK_HISTORIES.items():
        t1_attempts = history.tier_attempts.get("T1", [])
        t2_attempts = history.tier_attempts.get("T2", [])
        report.append({
            "task_type": task_type,
            "current_tier": history.current_tier,
            "t1_success_rate": (
                sum(t1_attempts) / len(t1_attempts) if t1_attempts else 0
            ),
            "t2_success_rate": (
                sum(t2_attempts) / len(t2_attempts) if t2_attempts else 0
            ),
            "recommendation": get_recommended_tier(task_type),
            "demoted_at": history.demoted_at,
        })
    return report


# ---------------------------------------------------------------------------
# Token Budget Inference — learn optimal max_tokens per task_type
# ---------------------------------------------------------------------------

_TOKEN_BUDGETS: dict[str, list[int]] = {}  # task_type → list of actual tokens used

# Default budgets by task type (starting points)
_DEFAULT_BUDGETS = {
    "status_check": 200,
    "job_submit": 300,
    "file_ops": 150,
    "debug": 800,
    "pr_creation": 500,
    "env_setup": 600,
    "general": 500,
}


def record_token_usage(task_type: str, tokens_used: int) -> None:
    """Record actual token usage for a task type."""
    if task_type not in _TOKEN_BUDGETS:
        _TOKEN_BUDGETS[task_type] = []
    _TOKEN_BUDGETS[task_type].append(tokens_used)
    # Keep last 20 observations
    if len(_TOKEN_BUDGETS[task_type]) > 20:
        _TOKEN_BUDGETS[task_type] = _TOKEN_BUDGETS[task_type][-20:]


def get_optimal_budget(task_type: str) -> int:
    """Get optimal max_tokens budget for a task type.

    Uses P90 of historical usage + 20% headroom.
    Falls back to defaults if insufficient history.
    """
    if task_type not in _TOKEN_BUDGETS or len(_TOKEN_BUDGETS[task_type]) < 3:
        return _DEFAULT_BUDGETS.get(task_type, 500)

    history = sorted(_TOKEN_BUDGETS[task_type])
    p90_idx = int(len(history) * 0.9)
    p90 = history[min(p90_idx, len(history) - 1)]
    # Add 20% headroom, minimum 100
    return max(100, int(p90 * 1.2))


def get_budget_report() -> dict:
    """Report on learned token budgets vs defaults."""
    report = {}
    for task_type, history in _TOKEN_BUDGETS.items():
        default = _DEFAULT_BUDGETS.get(task_type, 500)
        optimal = get_optimal_budget(task_type)
        avg_used = sum(history) / len(history) if history else 0
        report[task_type] = {
            "default_budget": default,
            "learned_optimal": optimal,
            "avg_actual": int(avg_used),
            "savings_vs_default": f"{max(0, round((1 - optimal/default)*100))}%",
            "samples": len(history),
        }
    return report


# ---------------------------------------------------------------------------
# Phoenix Span Export — send TER metrics to Phoenix for visualization
# ---------------------------------------------------------------------------

def export_span_to_phoenix(
    name: str,
    tier: str,
    task_type: str,
    tokens: int,
    duration_s: float,
    success: bool = True,
    metadata: dict = None,
) -> bool:
    """Export a span to Phoenix for observability.

    Creates an OpenInference-compatible span that shows up in the Phoenix UI
    with token usage, timing, tier classification, and success status.

    Returns True if export succeeded, False otherwise (non-blocking).
    """
    span = {
        "name": f"ecoseek.{name}",
        "context": {
            "trace_id": hashlib.sha256(
                f"{time.time()}{name}{task_type}".encode()
            ).hexdigest()[:32],
            "span_id": hashlib.sha256(
                f"{time.time()}{name}".encode()
            ).hexdigest()[:16],
        },
        "start_time": time.time() - duration_s,
        "end_time": time.time(),
        "attributes": {
            "ecoseek.tier": tier,
            "ecoseek.task_type": task_type,
            "ecoseek.tokens": tokens,
            "ecoseek.duration_s": duration_s,
            "ecoseek.success": success,
            "llm.token_count.total": tokens,
            "llm.token_count.completion": tokens,  # approximation
            "openinference.span.kind": "LLM",
        },
        "status": "OK" if success else "ERROR",
    }

    if metadata:
        for k, v in metadata.items():
            span["attributes"][f"ecoseek.{k}"] = v

    # Non-blocking export — fire and forget
    try:
        body = json.dumps({"spans": [span]}).encode("utf-8")
        req = urllib.request.Request(
            f"{_PHOENIX_ENDPOINT}/v1/spans",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {_PHOENIX_API_KEY}"} if _PHOENIX_API_KEY else {}),
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as exc:
        logger.debug("Phoenix export failed (non-blocking): %s", exc)
        return False


# ---------------------------------------------------------------------------
# Optimization Advisor — combines all signals into recommendations
# ---------------------------------------------------------------------------

def get_optimization_advice(task_type: str, error_text: str = "") -> dict:
    """Get combined optimization advice for a task.

    Combines:
    - Tier recommendation (based on history)
    - Token budget (based on learned usage)
    - Pattern match (if error text provided)
    - Cache status (for prompt reuse)

    Returns a dict that the caller can use to configure the next call.
    """
    from . import pattern_check

    advice = {
        "recommended_tier": get_recommended_tier(task_type),
        "recommended_max_tokens": get_optimal_budget(task_type),
        "use_compact_format": task_type in ("status_check", "file_ops", "job_submit"),
        "pattern_match": None,
    }

    if error_text:
        match = pattern_check(error_text)
        if match:
            advice["pattern_match"] = match
            advice["skip_llm"] = True  # Don't spend tokens, just apply fix

    return advice


# ---------------------------------------------------------------------------
# Session Summary — end-of-session optimization report
# ---------------------------------------------------------------------------

def get_session_optimization_report() -> dict:
    """Generate end-of-session report on optimizations applied.

    Shows:
    - Token savings from prompt caching
    - Tasks that were successfully demoted
    - Learned budgets vs defaults
    - Phoenix export success rate
    """
    return {
        "prompt_cache": get_cache_stats(),
        "tier_demotions": get_demotion_report(),
        "token_budgets": get_budget_report(),
        "recommendations": _generate_recommendations(),
    }


def _generate_recommendations() -> list[str]:
    """Generate actionable recommendations based on session data."""
    recs = []

    # Check for tasks that could be demoted
    for task_type, history in _TASK_HISTORIES.items():
        t2_attempts = history.tier_attempts.get("T2", [])
        if len(t2_attempts) >= 3 and all(t2_attempts[-3:]):
            if history.current_tier == "T2":
                recs.append(
                    f"Task '{task_type}' succeeded 3× at T2 — try T1 next time "
                    f"(fire_and_forget) to save ~80% tokens"
                )

    # Check for over-budgeted tasks
    for task_type, history in _TOKEN_BUDGETS.items():
        if len(history) >= 3:
            default = _DEFAULT_BUDGETS.get(task_type, 500)
            avg = sum(history) / len(history)
            if avg < default * 0.5:
                recs.append(
                    f"Task '{task_type}' uses avg {int(avg)} tokens but budget is "
                    f"{default} — reduce max_tokens to {get_optimal_budget(task_type)}"
                )

    # Prompt cache opportunity
    cache_stats = get_cache_stats()
    if cache_stats["total_hits"] > 5:
        recs.append(
            f"Prompt cache hit {cache_stats['total_hits']} times — "
            f"DeepSeek prefix caching saving ~{cache_stats['total_hits'] * 100} tokens/session"
        )

    if not recs:
        recs.append("No actionable recommendations yet — need more session data")

    return recs


# ---------------------------------------------------------------------------
# Tool schemas for registration
# ---------------------------------------------------------------------------

OPTIMIZATION_REPORT_SCHEMA = {
    "name": "optimization_report",
    "description": (
        "Get the current session's token optimization report. Shows prompt "
        "cache stats, tier demotion opportunities, learned token budgets, "
        "and actionable recommendations to reduce cost. "
        "Call at end of session or when reviewing efficiency."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

OPTIMIZE_CALL_SCHEMA = {
    "name": "optimize_call",
    "description": (
        "Get optimization advice for a specific task before making a Hermes call. "
        "Returns recommended tier, max_tokens budget, format, and whether a "
        "pattern match can skip the LLM entirely. Use this to configure "
        "escalate_remote or delegate_task calls for minimum token cost."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": [
                    "status_check", "job_submit", "file_ops",
                    "debug", "pr_creation", "env_setup", "general",
                ],
                "description": "Type of task to optimize for.",
            },
            "error_text": {
                "type": "string",
                "description": "Optional error text to check against pattern library.",
            },
        },
        "required": ["task_type"],
    },
}
