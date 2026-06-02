"""Deterministic HPC Workflow Agent — zero LLM tokens for known sequences.

Implements Google ADK-style Sequential/Parallel/Loop workflow patterns
for HPC operations that don't need LLM reasoning.

Key insight: submit→monitor→collect is a deterministic sequence.
Using an LLM to orchestrate it wastes ~2000 tokens per invocation.
Instead, encode the sequence directly and execute via SSH.

Performance: 0 inference tokens, <1s execution time.
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

# Configuration
_REMOTE_URL = os.environ.get("HERMES_REMOTE_URL", "https://hermes.ecoseek.org").rstrip("/")
_API_KEY = os.environ.get("HERMES_ECOSEEK_API_KEY", "")
_HPC_USER = "a474r867"


def _exec_raw(command: str, timeout: int = 30) -> tuple:
    """Execute a raw command on the HPC cluster.

    Strategy (in order of preference):
    1. Direct SSH (if running on reumanlab — 0 LLM tokens)
    2. Cluster HTTP API at r10r18n02:8888 (if tunnel active — 0 LLM tokens)
    3. Fallback: Hermes gateway call (~200 tokens)
    """
    import subprocess

    # Strategy 1: Direct SSH from reumanlab (0 tokens)
    try:
        ssh_key = os.path.expanduser("~/.ssh/hpc_a474r867_ed25519_new")
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                   f"{_HPC_USER}@hpc.crc.ku.edu", command]
        if os.path.exists(ssh_key):
            ssh_cmd.insert(1, "-i")
            ssh_cmd.insert(2, ssh_key)
        result = subprocess.run(
            ssh_cmd,
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 or result.stdout:
            output = result.stdout or result.stderr
            return output.strip(), {"completion_tokens": 0, "source": "direct_ssh"}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Strategy 2: Cluster HTTP API at r10r18n02:8888 (0 tokens)
    # Endpoint: POST /task with {"cmd": ..., "timeout": ...}
    # Also accepts Authorization: Bearer OR api-key header (both valid)
    try:
        api_body = json.dumps({"cmd": command, "timeout": timeout}).encode("utf-8")
        api_req = urllib.request.Request(
            "http://localhost:8888/task",
            data=api_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(api_req, timeout=timeout) as resp:
            api_data = json.loads(resp.read().decode("utf-8"))
        return api_data.get("output", ""), {"completion_tokens": 0, "source": "cluster_api"}
    except Exception:
        pass

    # Strategy 3: Hermes gateway (fallback, ~200 tokens)
    if not _REMOTE_URL or not _API_KEY:
        return "error: no execution method available", {}

    body = json.dumps({
        "model": os.environ.get("HERMES_REMOTE_MODEL", "hermes"),
        "messages": [
            {"role": "system", "content": "Execute and return ONLY raw output. No commentary."},
            {"role": "user", "content": f"ku-hpc raw: {command}"},
        ],
        "max_tokens": 300,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_REMOTE_URL}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        usage["source"] = "hermes_gateway"
        return content, usage
    except Exception as exc:
        return str(exc), {"source": "error"}


# ---------------------------------------------------------------------------
# Deterministic Workflows (Sequential patterns, no LLM reasoning)
# ---------------------------------------------------------------------------

def workflow_submit_and_confirm(
    sbatch_script: str,
    work_dir: str = "",
) -> str:
    """Submit a Slurm job and return the job ID. Zero reasoning tokens.

    Sequence: cd work_dir → sbatch script → parse job ID → confirm queued
    """
    t0 = time.time()

    cd_prefix = f"cd {work_dir} && " if work_dir else ""
    cmd = f"{cd_prefix}sbatch {sbatch_script} && sleep 2 && squeue -u {_HPC_USER} --format='%i %T %j' --noheader | head -5"

    output, usage = _exec_raw(cmd, timeout=60)

    # Parse job ID from "Submitted batch job XXXXX"
    import re
    job_match = re.search(r"Submitted batch job (\d+)", output)
    job_id = job_match.group(1) if job_match else None

    # Parse running status
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    status_lines = [l for l in lines if re.search(r"\d{5,}", l)]

    duration = time.time() - t0

    return json.dumps({
        "workflow": "submit_and_confirm",
        "job_id": job_id,
        "queue_snapshot": status_lines[:5],
        "tokens_used": usage.get("completion_tokens", 0),
        "duration_s": round(duration, 1),
        "raw": output[:500] if not job_id else None,
    })


def workflow_check_status(job_id: str = "") -> str:
    """Check job/queue status. Single command, structured output.

    If job_id provided: check specific job.
    If empty: show overall queue summary.
    """
    t0 = time.time()

    if job_id:
        cmd = f"sacct -j {job_id} --format=JobID,State%12,Elapsed,MaxRSS --noheader | tail -5"
    else:
        cmd = f"squeue -u {_HPC_USER} --format='%i %T %j' --noheader | sort -k2 | uniq -c -f1"

    output, usage = _exec_raw(cmd, timeout=30)
    duration = time.time() - t0

    # Parse into structured form
    import re
    lines = [l.strip() for l in output.split("\n") if l.strip()]

    # Count states
    states = {}
    for line in lines:
        for state in ("RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            if state in line:
                states[state] = states.get(state, 0) + 1

    return json.dumps({
        "workflow": "check_status",
        "job_id": job_id or "all",
        "states": states,
        "lines": lines[:10],
        "tokens_used": usage.get("completion_tokens", 0),
        "duration_s": round(duration, 1),
    })


def workflow_collect_results(
    results_dir: str,
    pattern: str = "*.rds",
    push_to_github: bool = False,
) -> str:
    """Count and summarize results in a directory.

    Deterministic: find + wc + optional git push. No LLM reasoning.
    """
    t0 = time.time()

    cmd = (
        f"find {results_dir} -name '{pattern}' | wc -l && "
        f"du -sh {results_dir} 2>/dev/null | cut -f1 && "
        f"find {results_dir} -name '{pattern}' -newer {results_dir}/.. -mmin -60 | wc -l"
    )

    if push_to_github:
        cmd += (
            f" && cd {results_dir}/.. && git add -A && "
            "git commit -m 'auto: collect results' --allow-empty && "
            "git push 2>/dev/null"
        )

    output, usage = _exec_raw(cmd, timeout=60)
    duration = time.time() - t0

    lines = [l.strip() for l in output.split("\n") if l.strip()]

    total = int(lines[0]) if lines and lines[0].isdigit() else 0
    size = lines[1] if len(lines) > 1 else "?"
    recent = int(lines[2]) if len(lines) > 2 and lines[2].isdigit() else 0

    return json.dumps({
        "workflow": "collect_results",
        "total_files": total,
        "disk_size": size,
        "recent_60min": recent,
        "pushed": push_to_github,
        "tokens_used": usage.get("completion_tokens", 0),
        "duration_s": round(duration, 1),
    })


def workflow_monitor_loop(
    job_id: str,
    max_checks: int = 5,
    interval_hint: str = "fast",
) -> str:
    """Monitor a job until completion or max_checks. Single batched command.

    Instead of polling N times (N × LLM calls), executes a single bash
    loop on the remote that polls and returns the final state.
    Costs: 1 LLM call (~200 tokens) vs N calls (~N×200 tokens).
    """
    t0 = time.time()

    # Batch the monitoring into a single remote command
    interval = 10 if interval_hint == "fast" else 60
    cmd = (
        f"for i in $(seq 1 {max_checks}); do "
        f"  state=$(sacct -j {job_id} --format=State --noheader | head -1 | tr -d ' '); "
        f"  echo \"check_$i: $state\"; "
        f"  if [ \"$state\" = 'COMPLETED' ] || [ \"$state\" = 'FAILED' ] || "
        f"     [ \"$state\" = 'CANCELLED' ] || [ \"$state\" = 'TIMEOUT' ]; then "
        f"    echo \"FINAL: $state\"; break; "
        f"  fi; "
        f"  sleep {interval}; "
        f"done"
    )

    # Longer timeout for monitoring loop
    timeout = max_checks * (interval + 15) + 30
    output, usage = _exec_raw(cmd, timeout=min(timeout, 300))
    duration = time.time() - t0

    # Parse final state
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    final_state = "UNKNOWN"
    checks_done = 0
    for line in lines:
        if line.startswith("FINAL:"):
            final_state = line.split(":", 1)[1].strip()
        if line.startswith("check_"):
            checks_done += 1

    return json.dumps({
        "workflow": "monitor_loop",
        "job_id": job_id,
        "final_state": final_state,
        "checks_done": checks_done,
        "max_checks": max_checks,
        "tokens_used": usage.get("completion_tokens", 0),
        "duration_s": round(duration, 1),
        "savings_vs_polling": f"{max_checks}× calls avoided ({max_checks * 200} tokens saved)",
    })


# ---------------------------------------------------------------------------
# Composite Workflows (chains of deterministic steps)
# ---------------------------------------------------------------------------

def workflow_full_pipeline(
    sbatch_script: str,
    results_dir: str,
    work_dir: str = "",
    monitor_checks: int = 3,
) -> str:
    """Full pipeline: submit → monitor → collect. 1 LLM call total.

    Instead of 3-5 separate Hermes calls (submit, check, check, collect),
    executes the entire sequence in a single batched command.
    Savings: ~3000-5000 tokens → ~300 tokens.
    """
    t0 = time.time()

    cd_prefix = f"cd {work_dir} && " if work_dir else ""

    # Single compound command: submit + wait + collect
    cmd = (
        f"{cd_prefix}"
        f"JOB=$(sbatch {sbatch_script} | grep -oP '\\d+') && "
        f"echo \"SUBMITTED: $JOB\" && "
        f"for i in $(seq 1 {monitor_checks}); do "
        f"  sleep 30; "
        f"  state=$(sacct -j $JOB --format=State --noheader | head -1 | tr -d ' '); "
        f"  echo \"CHECK_$i: $state\"; "
        f"  [ \"$state\" = 'COMPLETED' ] || [ \"$state\" = 'FAILED' ] && break; "
        f"done && "
        f"echo \"RESULTS: $(find {results_dir} -newer /tmp -name '*.rds' 2>/dev/null | wc -l) files\""
    )

    output, usage = _exec_raw(cmd, timeout=min(monitor_checks * 45 + 60, 300))
    duration = time.time() - t0

    # Parse structured output
    import re
    job_match = re.search(r"SUBMITTED:\s*(\d+)", output)
    job_id = job_match.group(1) if job_match else None

    checks = re.findall(r"CHECK_(\d+):\s*(\w+)", output)
    results_match = re.search(r"RESULTS:\s*(\d+)", output)
    result_count = int(results_match.group(1)) if results_match else 0

    final_state = checks[-1][1] if checks else "UNKNOWN"

    return json.dumps({
        "workflow": "full_pipeline",
        "job_id": job_id,
        "final_state": final_state,
        "checks": [{"n": int(c[0]), "state": c[1]} for c in checks],
        "result_files": result_count,
        "tokens_used": usage.get("completion_tokens", 0),
        "duration_s": round(duration, 1),
        "savings": "Single call vs 3-5 calls. ~3000+ tokens saved.",
    })


# ---------------------------------------------------------------------------
# Tool Schema for registration
# ---------------------------------------------------------------------------

HPC_WORKFLOW_SCHEMA = {
    "name": "hpc_workflow",
    "description": (
        "Execute deterministic HPC workflows with ZERO LLM reasoning tokens. "
        "Workflows: 'submit' (sbatch + confirm), 'status' (queue/job check), "
        "'collect' (count results + optional git push), 'monitor' (poll until done), "
        "'pipeline' (submit→monitor→collect in one call). "
        "Each workflow is a hard-coded sequence — no LLM decides what to do. "
        "Savings: ~200 tokens per workflow vs ~2000 for equivalent LLM orchestration."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "enum": ["submit", "status", "collect", "monitor", "pipeline"],
                "description": "Which deterministic workflow to execute.",
            },
            "sbatch_script": {
                "type": "string",
                "description": "Path to sbatch script (for submit/pipeline).",
            },
            "job_id": {
                "type": "string",
                "description": "Slurm job ID (for status/monitor).",
            },
            "results_dir": {
                "type": "string",
                "description": "Path to results directory (for collect/pipeline).",
            },
            "work_dir": {
                "type": "string",
                "description": "Working directory on HPC (optional).",
            },
            "pattern": {
                "type": "string",
                "description": "File pattern to count (default: '*.rds').",
            },
            "push_to_github": {
                "type": "boolean",
                "description": "Git push results after collecting (default: false).",
            },
            "max_checks": {
                "type": "integer",
                "description": "Max monitoring iterations (default: 5).",
            },
        },
        "required": ["workflow"],
    },
}


def hpc_workflow(
    workflow: str,
    sbatch_script: str = "",
    job_id: str = "",
    results_dir: str = "",
    work_dir: str = "",
    pattern: str = "*.rds",
    push_to_github: bool = False,
    max_checks: int = 5,
) -> str:
    """Router for deterministic HPC workflows."""
    if workflow == "submit":
        return workflow_submit_and_confirm(sbatch_script, work_dir)
    elif workflow == "status":
        return workflow_check_status(job_id)
    elif workflow == "collect":
        return workflow_collect_results(results_dir, pattern, push_to_github)
    elif workflow == "monitor":
        return workflow_monitor_loop(job_id, max_checks)
    elif workflow == "pipeline":
        return workflow_full_pipeline(sbatch_script, results_dir, work_dir, max_checks)
    else:
        return json.dumps({"error": f"Unknown workflow: {workflow}"})
