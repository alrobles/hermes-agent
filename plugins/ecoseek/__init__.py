"""ecoseek — Hermes plugin for EcoSeek DiDAL Phase 2.

Provides tools for the dual-agent architecture:

  **Alpha tools** (used by Emily local to talk to Beta):
  ``escalate_remote`` — simple one-shot delegation to Hermes remote
  ``dialectical_exchange`` — DiDAL structured debate with Beta (plan → execute → critique → refine)

  **Beta tools** (used by Hermes remote for execution):
  ``eco_analyze`` — structured interface to EcoAgent MCP server (GBIF, SDM, diversity)
  ``ku_hpc`` — KU HPC cluster operations via Slurm (submit, status, cancel, output)

Communication goes directly to hermes.ecoseek.org (preferred) with fallback
to the legacy broker at broker.ecoseek.org.

Activation is handled by the Hermes plugin system — enable with:
  hermes plugins enable ecoseek

Env vars (set in ~/.hermes/.env):
  HERMES_REMOTE_URL       - Remote Hermes endpoint (default: https://hermes.ecoseek.org)
  HERMES_ECOSEEK_API_KEY  - API key for hermes.ecoseek.org
  HERMES_REMOTE_MODEL     - Model name on remote (default: hermes)
  HERMES_REMOTE_TIMEOUT   - Request timeout in seconds (default: 300)
  DIDAL_MAX_TURNS         - Max dialogue turns (default: 20)
  DIDAL_STUCK_THRESHOLD   - Repeated errors before stopping (default: 3)
  ECOAGENT_URL            - EcoAgent tool server URL (default: http://localhost:8200)
  ECOAGENT_TIMEOUT        - EcoAgent request timeout (default: 120)
  KU_HPC_TIMEOUT          - Slurm command timeout (default: 30)

  Legacy (fallback):
  ECOSEEK_BROKER_URL      - Broker endpoint
  ECOSEEK_BROKER_KEY      - Broker session key
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

# Primary: direct connection to hermes.ecoseek.org
_REMOTE_URL = os.environ.get("HERMES_REMOTE_URL", "https://hermes.ecoseek.org").rstrip("/")
_API_KEY = os.environ.get("HERMES_ECOSEEK_API_KEY", "")
_MODEL = os.environ.get("HERMES_REMOTE_MODEL", "hermes")
_TIMEOUT = int(os.environ.get("HERMES_REMOTE_TIMEOUT", "300"))

# Legacy fallback (broker)
_BROKER_URL = os.environ.get("ECOSEEK_BROKER_URL", "").rstrip("/")
_BROKER_KEY = os.environ.get("ECOSEEK_BROKER_KEY", "")


def _is_configured() -> bool:
    """Return True when we can reach a remote Hermes (direct or via broker)."""
    return bool((_REMOTE_URL and _API_KEY) or (_BROKER_URL and _BROKER_KEY))


def _get_remote_endpoint() -> tuple[str, str]:
    """Return (url, auth_header) for the best available remote Hermes."""
    if _REMOTE_URL and _API_KEY:
        return _REMOTE_URL, f"Bearer {_API_KEY}"
    if _BROKER_URL and _BROKER_KEY:
        return _BROKER_URL, f"Bearer {_BROKER_KEY}"
    return "", ""


# ---------------------------------------------------------------------------
# Tool implementation — escalate_remote
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
            "error": "hermes_not_configured",
            "message": (
                "Remote escalation is not configured. Set HERMES_ECOSEEK_API_KEY "
                "in ~/.hermes/.env to enable (or legacy ECOSEEK_BROKER_URL + ECOSEEK_BROKER_KEY)."
            ),
        })

    base_url, auth = _get_remote_endpoint()

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
    if auth:
        headers["Authorization"] = auth

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
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
# Tool schemas (shared between register() and legacy registration)
# ---------------------------------------------------------------------------

ESCALATE_REMOTE_SCHEMA = {
    "name": "escalate_remote",
    "description": (
        "Escalate a task to the remote Hermes agent on reumanlab. "
        "The remote agent has access to DeepSeek v4 Pro, the KU HPC "
        "cluster (A100/MI210 GPUs via Slurm), GitHub CLI, and advanced "
        "ecological tools. Use this for simple one-shot delegation when "
        "you just need the remote agent to do something and return."
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
}

DIALECTICAL_EXCHANGE_SCHEMA = {
    "name": "dialectical_exchange",
    "description": (
        "Start a DiDAL (Dialectical Dual-Agent Loop) exchange with Beta "
        "(remote Hermes on reumanlab). Unlike escalate_remote, this tool "
        "enables structured debate: you propose a plan, Beta executes and "
        "critiques, you refine, and the loop continues until consensus. "
        "Use this for complex multi-step tasks that benefit from iterative "
        "refinement — SDM pipelines, HPC workflows, code review, or any "
        "task where execution + critique improves the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The user's original task description. Be specific about "
                    "what needs to be accomplished."
                ),
            },
            "plan": {
                "type": "string",
                "description": (
                    "Your proposed plan of action. Include steps, code, data "
                    "sources, and expected outputs. Beta will execute this and "
                    "provide critique. If empty, Beta receives only the task."
                ),
            },
            "max_turns": {
                "type": "integer",
                "description": (
                    "Maximum dialogue turns before stopping. Default: 20. "
                    "Lower for simple tasks, higher for complex workflows."
                ),
            },
        },
        "required": ["task"],
    },
}


# ---------------------------------------------------------------------------
# register(ctx) — Plugin system entry point
# ---------------------------------------------------------------------------
# Called by the Hermes plugin loader for both bundled and user plugins.
# Uses ctx.register_tool() which delegates to tools.registry.register().
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register all EcoSeek tools. Called by the Hermes plugin loader."""
    # Import sibling modules. The plugin loader sets __name__ to
    # "hermes_plugins.ecoseek" and __package__ to the same value,
    # so relative imports resolve against that namespace.
    from importlib import import_module
    didal_mod = import_module(".didal", package=__name__)
    eco_mod = import_module(".eco_analyze", package=__name__)
    hpc_mod = import_module(".ku_hpc", package=__name__)

    _didal_exchange = didal_mod.dialectical_exchange
    _eco_analyze = eco_mod.eco_analyze
    _check_ecoagent = eco_mod.check_ecoagent_available
    _ku_hpc = hpc_mod.ku_hpc
    _check_slurm = hpc_mod.check_slurm_available

    # -- Alpha tools (local Emily → remote Beta) ---------------------------

    ctx.register_tool(
        name="escalate_remote",
        toolset="ecoseek",
        schema=ESCALATE_REMOTE_SCHEMA,
        handler=lambda args, **kw: escalate_remote(
            task=args.get("task", ""),
            context=args.get("context", ""),
            urgency=args.get("urgency", "normal"),
            task_id=kw.get("task_id"),
        ),
        check_fn=_is_configured,
    )

    ctx.register_tool(
        name="dialectical_exchange",
        toolset="ecoseek",
        schema=DIALECTICAL_EXCHANGE_SCHEMA,
        handler=lambda args, **kw: _didal_exchange(
            task=args.get("task", ""),
            plan=args.get("plan", ""),
            max_turns=args.get("max_turns", 0),
            task_id=kw.get("task_id"),
        ),
        check_fn=_is_configured,
    )

    # -- Beta tools (available on reumanlab) --------------------------------

    ctx.register_tool(
        name="eco_analyze",
        toolset="ecoseek",
        schema=eco_mod.ECO_ANALYZE_SCHEMA,
        handler=lambda args, **kw: _eco_analyze(
            action=args.get("action", ""),
            params=args.get("params"),
            task_id=kw.get("task_id"),
        ),
        check_fn=_check_ecoagent,
    )

    ctx.register_tool(
        name="ku_hpc",
        toolset="ecoseek",
        schema=hpc_mod.KU_HPC_SCHEMA,
        handler=lambda args, **kw: _ku_hpc(
            action=args.get("action", ""),
            script=args.get("script", ""),
            job_id=args.get("job_id", ""),
            partition=args.get("partition", ""),
            command=args.get("command", ""),
            extra_args=args.get("extra_args", ""),
            task_id=kw.get("task_id"),
        ),
        check_fn=_check_slurm,
    )

    logger.info("ecoseek plugin registered: 4 tools (escalate_remote, dialectical_exchange, eco_analyze, ku_hpc)")
