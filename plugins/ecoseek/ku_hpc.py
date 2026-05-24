"""ku_hpc — KU HPC cluster tool for Beta executor.

Provides a structured interface to the KU HPC cluster via Slurm. Wraps
sbatch, squeue, scancel, sacct, and sinfo into a single tool with
structured parameters and results.

Available on reumanlab which has direct SSH/Slurm access to the cluster.
The check function verifies sbatch is on PATH.

GPU Resources:
  - A100 (80GB) — bf16/fp16/tf32
  - MI210 (64GB) — bf16/fp16 (ROCm)
  - V100 (16GB) — fp16 only
  - Quadro RTX 8000 (48GB) — fp16 only

Common paths on HPC:
  - Scratch: /home/a474r867/scratch/
  - Repos: /home/a474r867/work/Github/
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_SLURM_TIMEOUT = int(os.environ.get("KU_HPC_TIMEOUT", "30"))

SUPPORTED_ACTIONS = ("submit", "status", "cancel", "output", "info", "account_usage")


def check_slurm_available() -> bool:
    """Return True when Slurm commands are available."""
    return shutil.which("sbatch") is not None


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


def ku_hpc(
    action: str,
    script: str = "",
    job_id: str = "",
    partition: str = "",
    extra_args: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Execute an HPC cluster action via Slurm.

    Parameters
    ----------
    action : str
        Slurm action: submit, status, cancel, output, info, account_usage.
    script : str
        Path to the Slurm job script (for submit action).
    job_id : str
        Slurm job ID (for status, cancel, output actions).
    partition : str
        Target partition (optional, overrides script default).
    extra_args : str
        Additional sbatch arguments (e.g. '--gres=gpu:1 --time=01:00:00').
    task_id : str, optional
        Hermes task ID for tracing.
    """
    if action not in SUPPORTED_ACTIONS:
        return json.dumps({
            "success": False,
            "error": "invalid_action",
            "message": f"Unknown action: {action!r}. Supported: {', '.join(SUPPORTED_ACTIONS)}",
        })

    if action == "submit":
        if not script:
            return json.dumps({
                "success": False,
                "error": "missing_script",
                "message": "The 'script' parameter is required for submit action.",
            })
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
            })
        return json.dumps({
            "success": False,
            "error": "submit_failed",
            "action": "submit",
            "output": result["stdout"],
            "stderr": result["stderr"],
        })

    elif action == "status":
        cmd = ["squeue", "--format=%i %j %T %M %l %P %R", "--noheader"]
        if job_id:
            cmd.extend(["--job", job_id])
        else:
            cmd.extend(["--user", os.environ.get("USER", "a474r867")])
        result = _run_cmd(cmd)

        jobs = []
        for line in result["stdout"].splitlines():
            parts = line.split(None, 6)
            if len(parts) >= 4:
                jobs.append({
                    "job_id": parts[0],
                    "name": parts[1] if len(parts) > 1 else "",
                    "state": parts[2] if len(parts) > 2 else "",
                    "time": parts[3] if len(parts) > 3 else "",
                    "time_limit": parts[4] if len(parts) > 4 else "",
                    "partition": parts[5] if len(parts) > 5 else "",
                    "nodelist": parts[6] if len(parts) > 6 else "",
                })

        logger.info("ku_hpc status: %d jobs", len(jobs))
        return json.dumps({
            "success": True,
            "action": "status",
            "jobs": jobs,
            "raw": result["stdout"],
        })

    elif action == "cancel":
        if not job_id:
            return json.dumps({
                "success": False,
                "error": "missing_job_id",
                "message": "The 'job_id' parameter is required for cancel action.",
            })
        result = _run_cmd(["scancel", job_id])
        logger.info("ku_hpc cancel: job_id=%s rc=%d", job_id, result["returncode"])
        return json.dumps({
            "success": result["returncode"] == 0,
            "action": "cancel",
            "job_id": job_id,
            "output": result["stdout"],
            "stderr": result["stderr"],
        })

    elif action == "output":
        if not job_id:
            return json.dumps({
                "success": False,
                "error": "missing_job_id",
                "message": "The 'job_id' parameter is required for output action.",
            })
        result = _run_cmd([
            "sacct", "-j", job_id,
            "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize",
            "--noheader", "--parsable2",
        ])

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

    elif action == "info":
        result = _run_cmd([
            "sinfo",
            "--format=%P %a %l %D %T %C %G",
            "--noheader",
        ])
        partitions = []
        for line in result["stdout"].splitlines():
            parts = line.split(None, 6)
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

    elif action == "account_usage":
        result = _run_cmd([
            "sacct",
            "--user", os.environ.get("USER", "a474r867"),
            "--starttime", "now-7days",
            "--format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode",
            "--noheader", "--parsable2",
        ])
        return json.dumps({
            "success": True,
            "action": "account_usage",
            "raw": result["stdout"],
        })

    return json.dumps({
        "success": False,
        "error": "unhandled_action",
        "message": f"Action {action!r} is recognized but not implemented.",
    })


KU_HPC_SCHEMA = {
    "name": "ku_hpc",
    "description": (
        "Interact with the KU HPC cluster via Slurm. Submit batch jobs, "
        "check job status, cancel jobs, view job output/accounting, and "
        "query partition info. Available GPUs: A100 (80GB), MI210 (64GB), "
        "V100 (16GB), Quadro RTX 8000 (48GB). Use this instead of raw "
        "sbatch/squeue commands for structured results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(SUPPORTED_ACTIONS),
                "description": (
                    "Slurm action to execute. "
                    "submit: submit a batch job script. "
                    "status: check running/pending jobs. "
                    "cancel: cancel a job. "
                    "output: get job accounting (state, elapsed, memory). "
                    "info: list partitions, nodes, GPUs. "
                    "account_usage: recent job history (last 7 days)."
                ),
            },
            "script": {
                "type": "string",
                "description": "Path to the Slurm job script (required for submit).",
            },
            "job_id": {
                "type": "string",
                "description": "Slurm job ID (required for status/cancel/output of a specific job).",
            },
            "partition": {
                "type": "string",
                "description": "Target partition (optional, overrides script's #SBATCH --partition).",
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
