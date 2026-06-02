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
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TER Instrumentation — auto-log every call for continuous improvement
# ---------------------------------------------------------------------------

_CALL_LOG: list[dict] = []  # In-memory session log, accessible via get_session_metrics()


def _log_call(tier: str, task_type: str, tokens: int, duration_s: float,
              quality: float = 1.0, pattern: str = None) -> None:
    """Record a tool call for TER measurement. Zero overhead."""
    _CALL_LOG.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": tier,
        "task_type": task_type,
        "tokens": tokens,
        "duration_s": round(duration_s, 1),
        "quality": quality,
        "pattern": pattern,
    })


def _classify_task(task_text: str) -> str:
    """Infer task type from instruction text for TER categorization."""
    t = task_text.lower()
    if any(k in t for k in ("squeue", "sacct", "status", "monitor", "check")):
        return "status_check"
    if any(k in t for k in ("sbatch", "submit", "launch", "run job")):
        return "job_submit"
    if any(k in t for k in ("git push", "git commit", "gh pr", "pull request")):
        return "pr_creation"
    if any(k in t for k in ("fix", "patch", "debug", "error")):
        return "debug"
    if any(k in t for k in ("build", "container", "apptainer", "singularity")):
        return "env_setup"
    if any(k in t for k in ("ls", "cat", "find", "du", "df", "wc")):
        return "file_ops"
    return "general"


def get_session_metrics() -> dict:
    """Return current session's TER metrics. Call at end of session."""
    if not _CALL_LOG:
        return {"calls": 0, "total_tokens": 0, "avg_duration_s": 0}
    total_tokens = sum(c["tokens"] for c in _CALL_LOG)
    avg_dur = sum(c["duration_s"] for c in _CALL_LOG) / len(_CALL_LOG)
    tiers = {}
    for c in _CALL_LOG:
        tiers[c["tier"]] = tiers.get(c["tier"], 0) + 1
    return {
        "calls": len(_CALL_LOG),
        "total_tokens": total_tokens,
        "avg_duration_s": round(avg_dur, 1),
        "tier_distribution": tiers,
        "log": _CALL_LOG,
    }

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
    response_format: str = "compact",
    max_tokens: int = 0,
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
    response_format : str, optional
        Response format: "compact" (JSON-only, no prose — default),
        "structured" (key:value pairs), "full" (natural language).
    max_tokens : int, optional
        Maximum response tokens. 0 = no limit. Use 200-500 for status checks.
    task_id : str, optional
        Internal Hermes task ID (injected by the tool framework).

    Returns
    -------
    str
        JSON string with the remote agent's response.
    """
    _t0 = time.time()

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

    # Build system prompt based on response_format
    system_parts = []
    if response_format == "compact":
        system_parts.append(
            "RESPONSE RULES: Reply ONLY with a JSON object. No prose, no markdown, "
            "no explanation. Schema: {\"result\": ..., \"error\": null} or "
            "{\"result\": null, \"error\": \"msg\"}. Keep values minimal."
        )
    elif response_format == "structured":
        system_parts.append(
            "RESPONSE RULES: Reply with key:value pairs, one per line. "
            "No prose, no markdown. Example: status:RUNNING\njobs:5\nerrors:0"
        )
    if context:
        system_parts.append(context)
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    messages.append({"role": "user", "content": task})

    timeout = _TIMEOUT
    if urgency == "high":
        timeout = min(_TIMEOUT, 120)
    elif urgency == "background":
        timeout = max(_TIMEOUT, 600)

    request_body: dict = {
        "model": _MODEL,
        "messages": messages,
    }
    if max_tokens > 0:
        request_body["max_tokens"] = max_tokens

    body = json.dumps(request_body).encode("utf-8")

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

        # --- Observation Purification (SupervisorAgent-inspired) ---
        # Strip noise from response before returning to caller.
        # Saves ~30% prompt tokens on subsequent calls by reducing context size.
        try:
            from .observation_purifier import purify_hermes_response, purify_output
            _task_type_hint = _classify_task(task)
            if response_format == "compact":
                # For compact mode, purify the raw output embedded in response
                content = purify_output(content, _task_type_hint)
            else:
                content = purify_hermes_response(content, _task_type_hint)
        except Exception:
            pass  # Non-blocking — purification is best-effort

        logger.info(
            "ecoseek escalate_remote: model=%s tokens=%s",
            model_used,
            usage.get("total_tokens", "?"),
        )

        # TER instrumentation: auto-log for continuous improvement
        _task_type = _classify_task(task)
        _tier = "T1" if response_format == "compact" and max_tokens > 0 else "T2"
        _tokens = usage.get("total_tokens", 0)
        _duration = time.time() - _t0
        _log_call(
            tier=_tier,
            task_type=_task_type,
            tokens=_tokens,
            duration_s=_duration,
        )

        # Phoenix optimizer: record outcome + export span
        try:
            from .phoenix_optimizer import (
                record_task_outcome, record_token_usage, export_span_to_phoenix
            )
            record_task_outcome(_task_type, _tier, success=True)
            record_token_usage(_task_type, usage.get("completion_tokens", 0))
            export_span_to_phoenix(
                name="escalate_remote",
                tier=_tier,
                task_type=_task_type,
                tokens=_tokens,
                duration_s=_duration,
                metadata={"model": model_used, "format": response_format},
            )
        except Exception:
            pass  # Non-blocking — optimizer is best-effort

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
# Pattern Library — known errors with automatic fixes (T0 delegation)
# ---------------------------------------------------------------------------

_PATTERN_LIBRARY = [
    {
        "error": r"libRlapack\.so|libRblas\.so",
        "fix_cmd": "export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "description": "R LAPACK/BLAS library not found in container",
        "confidence": 0.99,
    },
    {
        "error": r"oom-kill|Out of memory|Cannot allocate memory",
        "fix_cmd": "sed -i 's/--mem=.*/--mem=64G/' {script}",
        "description": "Job killed by OOM — increase memory allocation",
        "confidence": 0.95,
    },
    {
        "error": r"convergence|ll_range|max_pdist",
        "fix_cmd": "sed -i 's/num_starts.*=.*/num_starts = 1500/' {script}",
        "description": "Optimization did not converge — increase starting points",
        "confidence": 0.80,
    },
    {
        "error": r"ModuleNotFoundError|there is no package",
        "fix_cmd": "apptainer exec {sif} Rscript -e 'install.packages(\"{pkg}\")'",
        "description": "Missing R/Python package inside container",
        "confidence": 0.70,
    },
    {
        "error": r"DUE TO TIME LIMIT|TIMEOUT|exceeded.*time",
        "fix_cmd": "sed -i 's/--time=.*/--time=24:00:00/' {script}",
        "description": "Job exceeded walltime — increase time limit",
        "confidence": 0.90,
    },
    {
        "error": r"Permission denied|EACCES",
        "fix_cmd": "chmod +x {script}",
        "description": "Permission denied on script/file",
        "confidence": 0.85,
    },
]


def pattern_check(error_text: str) -> Optional[dict]:
    """Check if an error matches a known pattern. Returns fix info or None.

    This is the LLM-free adaptive filter — avoids spending Devin tokens
    on problems that have known solutions.

    Returns:
        dict with {pattern, fix_cmd, confidence, description} if match found
        None if no pattern matches (escalate to Devin)
    """
    import re
    for pattern in _PATTERN_LIBRARY:
        if re.search(pattern["error"], error_text, re.IGNORECASE):
            return {
                "matched": True,
                "pattern": pattern["error"],
                "fix_cmd": pattern["fix_cmd"],
                "confidence": pattern["confidence"],
                "description": pattern["description"],
            }
    return None


# ---------------------------------------------------------------------------
# fire_and_forget — T1 delegation (Devin initiates, Hermes executes + reports)
# ---------------------------------------------------------------------------

def fire_and_forget(
    task: str,
    report_to: str = "github",
    monitor: bool = True,
    auto_fix: bool = True,
    task_id: Optional[str] = None,
) -> str:
    """Submit task to Hermes and return immediately. Zero follow-up tokens.

    Hermes handles execution, monitoring, error recovery, and reporting.
    Devin only sees results via git pull (0 additional tokens).

    Args:
        task: What to do (natural language or direct command)
        report_to: Where to report ("github" = push to repo, "none" = silent)
        monitor: If True, Hermes monitors until completion
        auto_fix: If True, Hermes applies known patterns on failure
        task_id: Optional tracking ID

    Returns:
        JSON with task acknowledgment and tracking info
    """
    _t0 = time.time()

    # Build the autonomous instruction
    instructions = [task]
    if monitor:
        instructions.append(
            "Monitor this task until completion. "
            "If it fails, check error against known patterns and auto-fix if confident."
        )
    if auto_fix:
        instructions.append(
            "Known fix patterns: libRlapack→LD_LIBRARY_PATH, OOM→increase mem, "
            "convergence→increase starts, timeout→increase walltime."
        )
    if report_to == "github":
        instructions.append(
            "When complete, push a status summary to the repo via: "
            "cd ~/work/xsdm_1000_sp && git pull && "
            "echo 'STATUS: completed at $(date)' >> docs/status/auto_report.md && "
            "git add -A && git commit -m 'auto: task completed' && git push"
        )

    full_task = " ".join(instructions)

    # Use escalate_remote with minimal response expectation
    result = escalate_remote(
        task=full_task,
        context="AUTONOMOUS MODE: Execute fully, report via GitHub, "
                "only escalate back if truly stuck after 3 fix attempts.",
        urgency="background",
        response_format="compact",
        max_tokens=100,  # We only need acknowledgment
        task_id=task_id,
    )

    # Override tier to T1 (fire-and-forget)
    if _CALL_LOG:
        _CALL_LOG[-1]["tier"] = "T1"
        _CALL_LOG[-1]["task_type"] = "fire_and_forget"

    return result


FIRE_AND_FORGET_SCHEMA = {
    "name": "fire_and_forget",
    "description": (
        "Submit a task to Hermes and return immediately. Hermes handles "
        "execution, monitoring, auto-fix on known errors, and reports "
        "results via GitHub push. Devin spends ~100 tokens total. "
        "Use for HPC jobs, builds, monitoring tasks — anything that "
        "doesn't need real-time Devin reasoning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What Hermes should do autonomously.",
            },
            "report_to": {
                "type": "string",
                "enum": ["github", "none"],
                "description": "Where to report completion. Default: github.",
            },
            "monitor": {
                "type": "boolean",
                "description": "Monitor until completion. Default: true.",
            },
            "auto_fix": {
                "type": "boolean",
                "description": "Auto-apply known fixes on failure. Default: true.",
            },
        },
        "required": ["task"],
    },
}

PATTERN_CHECK_SCHEMA = {
    "name": "pattern_check",
    "description": (
        "Check if an error matches a known fix pattern. Returns the fix "
        "command if matched (no LLM cost), or None if truly novel. "
        "Use this BEFORE escalating errors to save tokens — if pattern "
        "matches, just apply the fix directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "error_text": {
                "type": "string",
                "description": "The error message or log output to check.",
            },
        },
        "required": ["error_text"],
    },
}


# ---------------------------------------------------------------------------
# Tool schemas (shared between register() and legacy registration)
# ---------------------------------------------------------------------------

ESCALATE_REMOTE_SCHEMA = {
    "name": "escalate_remote",
    "description": (
        "Escalate a task to the remote Hermes agent on reumanlab. "
        "The remote agent has access to DeepSeek v4 Pro, the KU HPC "
        "cluster (A100/MI210 GPUs via Slurm), GitHub CLI, and advanced "
        "ecological tools. Use for one-shot delegation. "
        "TIP: Use response_format='compact' + max_tokens=300 for status "
        "checks to save 5-10x tokens vs full prose responses."
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
            "response_format": {
                "type": "string",
                "enum": ["compact", "structured", "full"],
                "description": (
                    "Response format. 'compact' (default): JSON-only, no prose — "
                    "best for programmatic use, saves ~70% tokens. 'structured': "
                    "key:value pairs. 'full': natural language (legacy behavior)."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "description": (
                    "Max response tokens. 0 = unlimited. Use 200-500 for status "
                    "checks, 1000+ for complex tasks. Saves cost when you only "
                    "need a yes/no or a short result."
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
            response_format=args.get("response_format", "compact"),
            max_tokens=args.get("max_tokens", 0),
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

    # -- T1 delegation tools (Devin → fire-and-forget → Hermes autonomous) --

    ctx.register_tool(
        name="fire_and_forget",
        toolset="ecoseek",
        schema=FIRE_AND_FORGET_SCHEMA,
        handler=lambda args, **kw: fire_and_forget(
            task=args.get("task", ""),
            report_to=args.get("report_to", "github"),
            monitor=args.get("monitor", True),
            auto_fix=args.get("auto_fix", True),
            task_id=kw.get("task_id"),
        ),
        check_fn=_is_configured,
    )

    ctx.register_tool(
        name="pattern_check",
        toolset="ecoseek",
        schema=PATTERN_CHECK_SCHEMA,
        handler=lambda args, **kw: json.dumps(
            pattern_check(args.get("error_text", "")) or {"matched": False}
        ),
        check_fn=lambda: True,  # No external dep needed
    )

    # -- Subagent delegation (VoltAgent-inspired fan-out) -------------------

    delegate_mod = import_module(".subagent_delegate", package=__name__)

    ctx.register_tool(
        name="delegate_task",
        toolset="ecoseek",
        schema=delegate_mod.DELEGATE_TASK_SCHEMA,
        handler=lambda args, **kw: delegate_mod.delegate_task(
            task=args.get("task", ""),
            target_agents=args.get("target_agents", []),
            context=args.get("context"),
            parallel=args.get("parallel", True),
        ),
        check_fn=_is_configured,
    )

    ctx.register_tool(
        name="list_subagents",
        toolset="ecoseek",
        schema=delegate_mod.LIST_SUBAGENTS_SCHEMA,
        handler=lambda args, **kw: delegate_mod.list_subagents(),
        check_fn=lambda: True,
    )

    # -- Deterministic HPC workflows (zero-LLM-token operations) -------------

    workflow_mod = import_module(".deterministic_workflows", package=__name__)

    ctx.register_tool(
        name="hpc_workflow",
        toolset="ecoseek",
        schema=workflow_mod.HPC_WORKFLOW_SCHEMA,
        handler=lambda args, **kw: workflow_mod.hpc_workflow(
            workflow=args.get("workflow", "status"),
            sbatch_script=args.get("sbatch_script", ""),
            job_id=args.get("job_id", ""),
            results_dir=args.get("results_dir", ""),
            work_dir=args.get("work_dir", ""),
            pattern=args.get("pattern", "*.rds"),
            push_to_github=args.get("push_to_github", False),
            max_checks=args.get("max_checks", 5),
        ),
        check_fn=_is_configured,
    )

    # -- Phoenix optimizer (trace-driven continuous improvement) -------------

    phoenix_mod = import_module(".phoenix_optimizer", package=__name__)

    ctx.register_tool(
        name="optimization_report",
        toolset="ecoseek",
        schema=phoenix_mod.OPTIMIZATION_REPORT_SCHEMA,
        handler=lambda args, **kw: json.dumps(
            phoenix_mod.get_session_optimization_report(), indent=2
        ),
        check_fn=lambda: True,
    )

    ctx.register_tool(
        name="optimize_call",
        toolset="ecoseek",
        schema=phoenix_mod.OPTIMIZE_CALL_SCHEMA,
        handler=lambda args, **kw: json.dumps(
            phoenix_mod.get_optimization_advice(
                task_type=args.get("task_type", "general"),
                error_text=args.get("error_text", ""),
            )
        ),
        check_fn=lambda: True,
    )

    logger.info(
        "ecoseek plugin registered: 11 tools "
        "(escalate_remote, dialectical_exchange, eco_analyze, ku_hpc, "
        "fire_and_forget, pattern_check, delegate_task, list_subagents, "
        "hpc_workflow, optimization_report, optimize_call)"
    )
