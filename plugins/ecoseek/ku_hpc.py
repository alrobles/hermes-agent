"""ku_hpc — KU HPC cluster tool for Beta executor.

Provides a structured interface to the KU HPC cluster. On reumanlab the
cluster is accessed through the ``ku-hpc`` SSH wrapper located at
``/home/reumanlab/local/bin/ku-hpc``.  The wrapper accepts subcommands:

    ku-hpc shell                 — interactive SSH session
    ku-hpc squeue-json           — job queue in pipe-delimited format
    ku-hpc submit-template NAME  — submit an sbatch template by name
    ku-hpc raw SSH_CMD...        — run an arbitrary command on the cluster

If the wrapper is not found, the tool falls back to direct Slurm commands
(for environments where sbatch/squeue are on PATH).

GPU Resources on KU HPC:
  - A100 (80GB) — bf16/fp16/tf32
  - MI210 (64GB) — bf16/fp16 (ROCm)
  - V100 (16GB) — fp16 only
  - Quadro RTX 8000 (48GB) — fp16 only

Common paths on HPC:
  - Scratch: /home/a474r867/scratch/
  - Repos: /home/a474r867/work/Github/
  - Templates: /home/a474r867/scratch/agenticplug-templates/
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_SLURM_TIMEOUT = int(os.environ.get("KU_HPC_TIMEOUT", "60"))
_KU_HPC_BIN = os.environ.get(
    "KU_HPC_BIN", "/home/reumanlab/local/bin/ku-hpc"
)

SUPPORTED_ACTIONS = (
    "submit",
    "status",
    "cancel",
    "output",
    "info",
    "account_usage",
    "raw",
)


def _has_wrapper() -> bool:
    """Return True when the ku-hpc SSH wrapper is available."""
    return os.path.isfile(_KU_HPC_BIN) and os.access(_KU_HPC_BIN, os.X_OK)


def _has_slurm() -> bool:
    """Return True when Slurm commands are available locally."""
    return shutil.which("sbatch") is not None


def check_slurm_available() -> bool:
    """Return True when we can talk to the HPC cluster."""
    return _has_wrapper() or _has_slurm()


def _run_cmd(cmd: list[str], timeout: int | None = None) -> dict:
    """Run a shell command and return structured output."""
    effective_timeout = timeout or _SLURM_TIMEOUT
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {effective_timeout}s",
            "returncode": -1,
        }
    except FileNotFoundError as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
        }


def _run_via_wrapper(slurm_cmd: str, timeout: int | None = None) -> dict:
    """Run a Slurm command through the ku-hpc SSH wrapper."""
    return _run_cmd([_KU_HPC_BIN, "raw", slurm_cmd], timeout=timeout)


def _parse_squeue_json(raw: str) -> list[dict]:
    """Parse pipe-delimited squeue output from ku-hpc squeue-json.

    Format: JobID|User|State|Time|Nodes|NodeList|JobName
    Lines starting with dashes or containing 'Access to electronic' are
    the SSH banner — skip them.
    """
    jobs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        if not parts[0] or not parts[0][0].isdigit():
            continue
        jobs.append({
            "job_id": parts[0],
            "user": parts[1] if len(parts) > 1 else "",
            "state": parts[2] if len(parts) > 2 else "",
            "time": parts[3] if len(parts) > 3 else "",
            "nodes": parts[4] if len(parts) > 4 else "",
            "nodelist": parts[5] if len(parts) > 5 else "",
            "name": parts[6] if len(parts) > 6 else "",
        })
    return jobs


def ku_hpc(
    action: str,
    script: str = "",
    job_id: str = "",
    partition: str = "",
    command: str = "",
    extra_args: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Execute an HPC cluster action via the ku-hpc wrapper or direct Slurm.

    Parameters
    ----------
    action : str
        Action to execute: submit, status, cancel, output, info,
        account_usage, raw.
    script : str
        Template name for submit (used with ku-hpc submit-template).
    job_id : str
        Slurm job ID (for status/cancel/output of a specific job).
    partition : str
        Target partition (optional, for direct Slurm submit).
    command : str
        Raw command to execute on HPC (for raw action).
    extra_args : str
        Additional arguments as a string.
    task_id : str, optional
        Hermes task ID for tracing.
    """
    if action not in SUPPORTED_ACTIONS:
        return json.dumps({
            "success": False,
            "error": "invalid_action",
            "message": f"Unknown action: {action!r}. Supported: {', '.join(SUPPORTED_ACTIONS)}",
        })

    use_wrapper = _has_wrapper()

    # --- submit ---
    if action == "submit":
        if not script:
            return json.dumps({
                "success": False,
                "error": "missing_script",
                "message": "The 'script' parameter is required for submit.",
            })

        if use_wrapper:
            result = _run_cmd(
                [_KU_HPC_BIN, "submit-template", script], timeout=60
            )
        else:
            cmd = ["sbatch"]
            if partition:
                cmd.extend(["--partition", partition])
            if extra_args:
                cmd.extend(extra_args.split())
            cmd.append(script)
            result = _run_cmd(cmd, timeout=30)

        if result["returncode"] == 0:
            submitted_id = ""
            for word in result["stdout"].split():
                if word.isdigit():
                    submitted_id = word
                    break
            logger.info("ku_hpc submit: job_id=%s script=%s", submitted_id, script)
            return json.dumps({
                "success": True,
                "action": "submit",
                "job_id": submitted_id,
                "output": result["stdout"],
                "method": "wrapper" if use_wrapper else "direct",
            })
        return json.dumps({
            "success": False,
            "error": "submit_failed",
            "action": "submit",
            "output": result["stdout"],
            "stderr": result["stderr"],
        })

    # --- status ---
    elif action == "status":
        if use_wrapper:
            result = _run_cmd([_KU_HPC_BIN, "squeue-json"], timeout=30)
            all_jobs = _parse_squeue_json(result["stdout"])
            if job_id:
                all_jobs = [j for j in all_jobs if j["job_id"] == job_id]
            else:
                user = os.environ.get("USER", "a474r867")
                all_jobs = [j for j in all_jobs if j.get("user") == user]
        else:
            cmd = ["squeue", "--format=%i|%u|%T|%M|%D|%R|%j", "--noheader"]
            if job_id:
                cmd.extend(["--job", job_id])
            else:
                cmd.extend(["--user", os.environ.get("USER", "a474r867")])
            result = _run_cmd(cmd)
            all_jobs = _parse_squeue_json(result["stdout"])

        logger.info("ku_hpc status: %d jobs", len(all_jobs))
        return json.dumps({
            "success": True,
            "action": "status",
            "jobs": all_jobs,
            "total": len(all_jobs),
            "method": "wrapper" if use_wrapper else "direct",
        })

    # --- cancel ---
    elif action == "cancel":
        if not job_id:
            return json.dumps({
                "success": False,
                "error": "missing_job_id",
                "message": "The 'job_id' parameter is required for cancel.",
            })
        if use_wrapper:
            result = _run_via_wrapper(f"scancel {job_id}")
        else:
            result = _run_cmd(["scancel", job_id])
        logger.info("ku_hpc cancel: job_id=%s rc=%d", job_id, result["returncode"])
        return json.dumps({
            "success": result["returncode"] == 0,
            "action": "cancel",
            "job_id": job_id,
            "output": result["stdout"],
            "stderr": result["stderr"],
        })

    # --- output ---
    elif action == "output":
        if not job_id:
            return json.dumps({
                "success": False,
                "error": "missing_job_id",
                "message": "The 'job_id' parameter is required for output.",
            })
        sacct_cmd = (
            f"sacct -j {job_id} "
            f"--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize "
            f"--noheader --parsable2"
        )
        if use_wrapper:
            result = _run_via_wrapper(sacct_cmd, timeout=30)
        else:
            result = _run_cmd(sacct_cmd.split())

        jobs = []
        for line in result["stdout"].splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                jobs.append({
                    "job_id": parts[0],
                    "name": parts[1] if len(parts) > 1 else "",
                    "state": parts[2] if len(parts) > 2 else "",
                    "exit_code": parts[3] if len(parts) > 3 else "",
                    "elapsed": parts[4] if len(parts) > 4 else "",
                    "max_rss": parts[5] if len(parts) > 5 else "",
                    "max_vmsize": parts[6] if len(parts) > 6 else "",
                })

        logger.info("ku_hpc output: job_id=%s entries=%d", job_id, len(jobs))
        return json.dumps({
            "success": True,
            "action": "output",
            "job_id": job_id,
            "accounting": jobs,
            "raw": result["stdout"],
        })

    # --- info ---
    elif action == "info":
        sinfo_cmd = "sinfo --format=%P|%a|%l|%D|%T|%C|%G --noheader"
        if use_wrapper:
            result = _run_via_wrapper(sinfo_cmd, timeout=30)
        else:
            result = _run_cmd(sinfo_cmd.split())

        partitions = []
        for line in result["stdout"].splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                partitions.append({
                    "partition": parts[0],
                    "avail": parts[1] if len(parts) > 1 else "",
                    "timelimit": parts[2] if len(parts) > 2 else "",
                    "nodes": parts[3] if len(parts) > 3 else "",
                    "state": parts[4] if len(parts) > 4 else "",
                    "cpus": parts[5] if len(parts) > 5 else "",
                    "gres": parts[6] if len(parts) > 6 else "",
                })

        return json.dumps({
            "success": True,
            "action": "info",
            "partitions": partitions,
            "raw": result["stdout"],
        })

    # --- account_usage ---
    elif action == "account_usage":
        user = os.environ.get("USER", "a474r867")
        sacct_cmd = (
            f"sacct --user {user} --starttime now-7days "
            f"--format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode "
            f"--noheader --parsable2"
        )
        if use_wrapper:
            result = _run_via_wrapper(sacct_cmd, timeout=30)
        else:
            result = _run_cmd(sacct_cmd.split())
        return json.dumps({
            "success": True,
            "action": "account_usage",
            "raw": result["stdout"],
        })

    # --- raw ---
    elif action == "raw":
        if not command:
            return json.dumps({
                "success": False,
                "error": "missing_command",
                "message": "The 'command' parameter is required for raw action.",
            })
        if use_wrapper:
            result = _run_via_wrapper(command, timeout=_SLURM_TIMEOUT)
        else:
            result = _run_cmd(["bash", "-c", command], timeout=_SLURM_TIMEOUT)

        return json.dumps({
            "success": result["returncode"] == 0,
            "action": "raw",
            "output": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
        })

    return json.dumps({
        "success": False,
        "error": "unhandled_action",
        "message": f"Action {action!r} is recognized but not implemented.",
    })


KU_HPC_SCHEMA = {
    "name": "ku_hpc",
    "description": (
        "Interact with the KU HPC cluster. On reumanlab this uses the ku-hpc "
        "SSH wrapper; elsewhere it falls back to direct Slurm commands. "
        "Submit sbatch templates, check job status, cancel jobs, view "
        "accounting, and query partition info. Available GPUs: A100 (80GB), "
        "MI210 (64GB), V100 (16GB), Quadro RTX 8000 (48GB)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(SUPPORTED_ACTIONS),
                "description": (
                    "HPC action to execute. "
                    "submit: submit a template (ku-hpc submit-template NAME). "
                    "status: list running/pending jobs (yours by default). "
                    "cancel: cancel a job by ID. "
                    "output: get job accounting (state, elapsed, memory). "
                    "info: list partitions, nodes, GPUs. "
                    "account_usage: recent job history (last 7 days). "
                    "raw: run an arbitrary command on the cluster via SSH."
                ),
            },
            "script": {
                "type": "string",
                "description": (
                    "Template name for submit action (e.g. 'launch_ollama_light.slurm'). "
                    "Templates are in /home/a474r867/scratch/agenticplug-templates/."
                ),
            },
            "job_id": {
                "type": "string",
                "description": "Slurm job ID (for status/cancel/output of a specific job).",
            },
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run on the HPC cluster (for raw action). "
                    "Examples: 'sinfo -p gpu', 'ls ~/scratch/', 'nvidia-smi'."
                ),
            },
            "partition": {
                "type": "string",
                "description": "Target partition (optional, for direct Slurm submit).",
            },
            "extra_args": {
                "type": "string",
                "description": (
                    "Additional sbatch arguments as a string "
                    "(e.g. '--gres=gpu:1 --time=01:00:00 --mem=32G')."
                ),
            },
        },
        "required": ["action"],
    },
}
