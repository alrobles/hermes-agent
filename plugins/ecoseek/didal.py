"""DiDAL — Dialectical Dual-Agent Loop protocol for EcoSeek.

Implements the structured dialogue protocol between Alpha (Emily local) and
Beta (Hermes remote) for iterative task refinement with mutual critique.

Protocol message types:
  plan             — Alpha proposes a plan of action
  code             — Alpha sends code for execution
  execution_result — Beta returns execution output
  critique         — Either agent identifies issues or suggests improvements
  final            — Both agents agree the task is complete

The loop: plan → delegate → execute → critique → refine → [loop or final]
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary: direct connection to hermes.ecoseek.org
_REMOTE_URL = os.environ.get("HERMES_REMOTE_URL", "https://hermes.ecoseek.org").rstrip("/")
_API_KEY = os.environ.get("HERMES_ECOSEEK_API_KEY", "")
_MODEL = os.environ.get("HERMES_REMOTE_MODEL", "hermes")
_TIMEOUT = int(os.environ.get("HERMES_REMOTE_TIMEOUT", "300"))

# Legacy fallback (broker)
_BROKER_URL = os.environ.get("ECOSEEK_BROKER_URL", "").rstrip("/")
_BROKER_KEY = os.environ.get("ECOSEEK_BROKER_KEY", "")


def _get_remote_endpoint() -> tuple[str, str]:
    """Return (url, auth_header) for the best available remote Hermes."""
    if _REMOTE_URL and _API_KEY:
        return _REMOTE_URL, f"Bearer {_API_KEY}"
    if _BROKER_URL and _BROKER_KEY:
        return _BROKER_URL, f"Bearer {_BROKER_KEY}"
    return "", ""
_MAX_TURNS = int(os.environ.get("DIDAL_MAX_TURNS", "20"))
_STUCK_THRESHOLD = int(os.environ.get("DIDAL_STUCK_THRESHOLD", "3"))

# Valid message types in the DiDAL protocol
MESSAGE_TYPES = ("plan", "code", "execution_result", "critique", "final")


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def make_message(
    sender: str,
    msg_type: str,
    content: str,
    task_id: str,
    turn: int,
    trace_id: str = "",
) -> dict:
    """Create a DiDAL protocol message."""
    if msg_type not in MESSAGE_TYPES:
        raise ValueError(f"Invalid message type: {msg_type!r}. Must be one of {MESSAGE_TYPES}")
    return {
        "from": sender,
        "type": msg_type,
        "content": content,
        "metadata": {
            "turn": turn,
            "task_id": task_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_id": trace_id,
        },
    }


def detect_stuck_loop(history: list[dict]) -> bool:
    """Return True if the last N errors are substantially the same."""
    errors = [
        m for m in history
        if m.get("type") in ("execution_result", "critique")
        and "error" in m.get("content", "").lower()
    ]
    if len(errors) < _STUCK_THRESHOLD:
        return False
    recent = errors[-_STUCK_THRESHOLD:]
    first = recent[0].get("content", "")[:200]
    return all(r.get("content", "")[:200] == first for r in recent)


# ---------------------------------------------------------------------------
# Remote Beta communication
# ---------------------------------------------------------------------------

def _send_to_beta(system_prompt: str, messages: list[dict]) -> dict:
    """Send a request to Beta (Hermes remote) via hermes.ecoseek.org."""
    base_url, auth = _get_remote_endpoint()
    if not base_url:
        raise RuntimeError("No remote endpoint configured")

    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = "assistant" if m.get("from") == "beta" else "user"
        api_messages.append({"role": role, "content": m["content"]})

    body = json.dumps({
        "model": _MODEL,
        "messages": api_messages,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if auth:
        headers["Authorization"] = auth

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Beta returned no choices")
    return {
        "content": choices[0].get("message", {}).get("content", ""),
        "model": data.get("model", _MODEL),
        "usage": data.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# Dialectical exchange tool
# ---------------------------------------------------------------------------

BETA_SYSTEM_PROMPT = """\
You are Beta, the execution specialist in the EcoSeek DiDAL (Dialectical Dual-Agent Loop) system.

Your role:
1. Execute plans and code from Alpha in your sandbox or on the HPC cluster
2. Critically review Alpha's proposals — point out errors, missing dependencies, edge cases
3. Report results honestly — never claim success if something failed
4. Suggest improvements when you spot better approaches
5. When you agree the task is complete, say FINAL: followed by the summary

You have access to: shell, file editing, ku-hpc (Slurm → KU HPC cluster with A100/MI210 GPUs),
ecoagent tools (GBIF, SDM, phylogenetic analysis), web search, and GitHub CLI.

Respond with a JSON object:
{"type": "execution_result|critique|final", "content": "your response"}
"""


def dialectical_exchange(
    task: str,
    plan: str = "",
    max_turns: int = 0,
    task_id: Optional[str] = None,
) -> str:
    """Initiate a dialectical exchange between Alpha (local) and Beta (remote).

    Alpha proposes a plan; Beta executes, critiques, and refines. The loop
    continues until consensus (both agree on FINAL) or max_turns is reached.

    Parameters
    ----------
    task : str
        The user's original task description.
    plan : str, optional
        Alpha's initial plan. If empty, Beta receives only the task.
    max_turns : int, optional
        Override max dialogue turns (default: DIDAL_MAX_TURNS env, or 20).
    task_id : str, optional
        Hermes task ID for tracing.

    Returns
    -------
    str
        JSON with the dialogue history and final result.
    """
    if not ((_REMOTE_URL and _API_KEY) or (_BROKER_URL and _BROKER_KEY)):
        return json.dumps({
            "success": False,
            "error": "hermes_not_configured",
            "message": (
                "DiDAL requires HERMES_ECOSEEK_API_KEY (or legacy ECOSEEK_BROKER_URL "
                "+ ECOSEEK_BROKER_KEY). Set in ~/.hermes/.env to enable."
            ),
        })

    effective_max = max_turns if max_turns > 0 else _MAX_TURNS
    dialogue_id = task_id or str(uuid.uuid4())[:8]
    history: list[dict] = []
    turn = 0

    # Alpha's opening message
    alpha_content = f"Task: {task}"
    if plan:
        alpha_content += f"\n\nPlan:\n{plan}"
    opening = make_message("alpha", "plan", alpha_content, dialogue_id, turn)
    history.append(opening)
    turn += 1

    logger.info("didal[%s] started: %s", dialogue_id, task[:100])

    final_result = None

    while turn < effective_max:
        # Send to Beta
        try:
            beta_resp = _send_to_beta(BETA_SYSTEM_PROMPT, history)
        except Exception as exc:
            logger.warning("didal[%s] beta error at turn %d: %s", dialogue_id, turn, exc)
            error_msg = make_message(
                "beta", "execution_result",
                f"Error communicating with Beta: {exc}",
                dialogue_id, turn,
            )
            history.append(error_msg)
            turn += 1

            if detect_stuck_loop(history):
                logger.warning("didal[%s] stuck loop detected, stopping", dialogue_id)
                break
            continue

        beta_content = beta_resp["content"]

        # Try to parse structured response from Beta
        beta_type = "execution_result"
        if beta_content.strip().upper().startswith("FINAL:"):
            beta_type = "final"
            final_result = beta_content[6:].strip()
        else:
            try:
                parsed = json.loads(beta_content)
                if isinstance(parsed, dict) and "type" in parsed:
                    beta_type = parsed["type"] if parsed["type"] in MESSAGE_TYPES else "execution_result"
                    beta_content = parsed.get("content", beta_content)
                    if beta_type == "final":
                        final_result = beta_content
            except (json.JSONDecodeError, KeyError):
                pass

        beta_msg = make_message("beta", beta_type, beta_content, dialogue_id, turn)
        beta_msg["metadata"]["model"] = beta_resp.get("model", "")
        beta_msg["metadata"]["usage"] = beta_resp.get("usage", {})
        history.append(beta_msg)
        turn += 1

        logger.info(
            "didal[%s] turn %d: beta %s (%d chars)",
            dialogue_id, turn, beta_type, len(beta_content),
        )

        # Check termination conditions
        if beta_type == "final":
            break

        if detect_stuck_loop(history):
            logger.warning("didal[%s] stuck loop detected at turn %d", dialogue_id, turn)
            break

        # If Beta critiqued, we return control to Alpha (the calling agent)
        # so it can refine and call dialectical_exchange again with an updated plan
        if beta_type == "critique":
            break

    total_tokens = sum(
        m.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
        for m in history
    )

    return json.dumps({
        "success": final_result is not None,
        "dialogue_id": dialogue_id,
        "turns": turn,
        "final_result": final_result,
        "last_beta_response": history[-1]["content"] if history else "",
        "history": history,
        "total_tokens": total_tokens,
        "stuck_loop": detect_stuck_loop(history),
    })
