"""ecoseek — Hermes plugin for EcoSeek ecological intelligence.

Provides the ``escalate_remote`` tool that lets a local Emily agent delegate
heavy-computation tasks to the remote Hermes instance on reumanlab.  The remote
Hermes has access to DeepSeek v4 Pro, KU HPC cluster (A100/MI210), and advanced
ecological tools.

Activation is handled by the Hermes plugin system — enable with:
  hermes plugins enable ecoseek

Optional env vars (set in ~/.hermes/.env):
  ECOSEEK_BROKER_URL  - Broker endpoint (default: https://broker.ecoseek.org)
  ECOSEEK_BROKER_KEY  - Broker session key for authenticated requests
  ECOSEEK_MODEL       - Model name on remote Hermes (default: openclaw/main)
  ECOSEEK_TIMEOUT     - Request timeout in seconds (default: 300)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BROKER_URL = os.environ.get("ECOSEEK_BROKER_URL", "https://broker.ecoseek.org").rstrip("/")
_BROKER_KEY = os.environ.get("ECOSEEK_BROKER_KEY", "")
_MODEL = os.environ.get("ECOSEEK_MODEL", "openclaw/main")
_TIMEOUT = int(os.environ.get("ECOSEEK_TIMEOUT", "300"))


def _is_configured() -> bool:
    """Return True when the remote broker is reachable (URL + key present)."""
    return bool(_BROKER_URL and _BROKER_KEY)


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def escalate_remote(
    task: str,
    context: str = "",
    urgency: str = "normal",
    task_id: Optional[str] = None,
) -> str:
    """Send a task to the remote Hermes agent on reumanlab.

    Use this when the current task requires:
      - Heavy computation (HPC cluster, GPUs)
      - Access to reumanlab resources (ku-hpc, GitHub, deployed services)
      - Capabilities beyond the local LLM (advanced reasoning, large context)
      - Running ecological pipelines that need large datasets or spatial data

    Parameters
    ----------
    task : str
        Clear description of what the remote agent should do.
    context : str, optional
        Background information or system instructions for the remote agent.
    urgency : str, optional
        One of "normal", "high", "background". Affects timeout behavior.
    task_id : str, optional
        Internal Hermes task ID (injected by the tool framework).

    Returns
    -------
    str
        JSON string with the remote agent's response.
    """
    if not _is_configured():
        return json.dumps({
            "success": False,
            "error": "ecoseek_not_configured",
            "message": (
                "Remote escalation is not configured. Set ECOSEEK_BROKER_URL "
                "and ECOSEEK_BROKER_KEY in ~/.hermes/.env to enable."
            ),
        })

    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": task})

    timeout = _TIMEOUT
    if urgency == "high":
        timeout = min(_TIMEOUT, 120)
    elif urgency == "background":
        timeout = max(_TIMEOUT, 600)

    body = json.dumps({
        "model": _MODEL,
        "messages": messages,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _BROKER_KEY:
        headers["Authorization"] = f"Bearer {_BROKER_KEY}"

    req = urllib.request.Request(
        f"{_BROKER_URL}/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices", [])
        if not choices:
            return json.dumps({
                "success": False,
                "error": "empty_response",
                "message": "Remote Hermes returned no choices.",
            })

        content = choices[0].get("message", {}).get("content", "")
        model_used = data.get("model", _MODEL)
        usage = data.get("usage", {})

        logger.info(
            "ecoseek escalate_remote: model=%s tokens=%s",
            model_used,
            usage.get("total_tokens", "?"),
        )

        return json.dumps({
            "success": True,
            "remote_response": content,
            "model": model_used,
            "usage": usage,
        })

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.warning(
            "ecoseek escalate_remote HTTP %s: %s",
            exc.code,
            error_body[:200],
        )
        return json.dumps({
            "success": False,
            "error": f"http_{exc.code}",
            "message": f"Remote Hermes returned HTTP {exc.code}.",
            "detail": error_body[:500],
        })

    except urllib.error.URLError as exc:
        logger.warning("ecoseek escalate_remote URL error: %s", exc.reason)
        return json.dumps({
            "success": False,
            "error": "connection_error",
            "message": f"Cannot reach remote Hermes: {exc.reason}",
        })

    except Exception as exc:
        logger.exception("ecoseek escalate_remote unexpected error")
        return json.dumps({
            "success": False,
            "error": "unexpected_error",
            "message": str(exc)[:300],
        })


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

try:
    from tools.registry import registry

    registry.register(
        name="escalate_remote",
        toolset="ecoseek",
        schema={
            "name": "escalate_remote",
            "description": (
                "Escalate a task to the remote Hermes agent on reumanlab. "
                "The remote agent has access to DeepSeek v4 Pro, the KU HPC "
                "cluster (A100/MI210 GPUs via Slurm), GitHub CLI, and advanced "
                "ecological tools. Use this when the task requires heavy "
                "computation, HPC resources, large datasets, advanced reasoning "
                "beyond your local model, or access to reumanlab infrastructure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Clear description of what the remote agent should accomplish. "
                            "Be specific about inputs, expected outputs, and any constraints."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional background information or system-level instructions "
                            "for the remote agent. Use this to provide ecological context, "
                            "specify data sources, or set methodology preferences."
                        ),
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["normal", "high", "background"],
                        "description": (
                            "Task urgency. 'high' for quick lookups (shorter timeout), "
                            "'background' for long-running HPC jobs (longer timeout), "
                            "'normal' for standard tasks."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
        handler=lambda args, **kw: escalate_remote(
            task=args.get("task", ""),
            context=args.get("context", ""),
            urgency=args.get("urgency", "normal"),
            task_id=kw.get("task_id"),
        ),
        check_fn=_is_configured,
        requires_env=[],
    )
except ImportError:
    pass
