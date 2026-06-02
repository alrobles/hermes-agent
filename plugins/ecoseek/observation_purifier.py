"""Observation Purification — SupervisorAgent-inspired token reduction.

Strips irrelevant content from tool outputs before the next LLM step.
Based on "Stop Wasting Your Tokens" (ICLR 2026): ~30% prompt token savings.

The purifier is LLM-free — it uses regex/heuristic rules to identify and
remove noise (ANSI codes, progress bars, repeated headers, empty lines,
verbose stack traces) while preserving actionable content.
"""
from __future__ import annotations

import re
from typing import Optional


# Max output chars to keep after purification (prevents context blowup)
_MAX_OUTPUT_CHARS = 2000

# Patterns that are pure noise in HPC/CLI outputs
_NOISE_PATTERNS = [
    # ANSI escape codes
    re.compile(r"\x1b\[[0-9;]*[a-zA-Z]"),
    # Progress bars (e.g., "####...   45%")
    re.compile(r"[#=\-]{10,}\s*\d+%"),
    # Repeated blank lines (collapse to single)
    re.compile(r"\n{3,}"),
    # Slurm header/footer decoration lines
    re.compile(r"^[-=]{20,}$", re.MULTILINE),
    # wget/curl progress lines
    re.compile(r"^\s*\d+K\s+[\.\s]+\d+%.*$", re.MULTILINE),
    # pip install verbose output (already installed)
    re.compile(r"^Requirement already satisfied:.*$", re.MULTILINE),
    # R package compilation noise (gcc/g++ lines)
    re.compile(r"^g?cc\s.*-[co]\s.*$", re.MULTILINE),
    re.compile(r"^g\+\+\s.*-[co]\s.*$", re.MULTILINE),
    # R build verbose lines (** byte-compile, installing to, etc.)
    re.compile(r"^\*\*\s.*$", re.MULTILINE),
    re.compile(r"^installing to /.*$", re.MULTILINE),
    # git clone/fetch progress
    re.compile(r"^(Cloning|Receiving|Resolving|Counting|Compressing).*\d+%.*$", re.MULTILINE),
    # Remote/origin lines in git output
    re.compile(r"^remote:\s*(Enumerating|Counting|Compressing|Total).*$", re.MULTILINE),
]

# Patterns to always keep (high signal)
_KEEP_PATTERNS = [
    re.compile(r"(?i)(error|fail|exception|traceback|killed|denied|timeout)"),
    re.compile(r"(?i)(success|completed|running|pending|submitted)"),
    re.compile(r"(?i)(job\s*id|jobid|slurm)"),
    re.compile(r"\d+\s+(RUNNING|PENDING|COMPLETED|FAILED|CANCELLED)"),
]

# Stack trace compression: keep first + last frame + error message
_TRACEBACK_START = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_TRACEBACK_FRAME = re.compile(r'^\s+File ".*", line \d+', re.MULTILINE)


def purify_output(raw: str, task_type: str = "general") -> str:
    """Strip noise from tool output, keeping only actionable content.

    Args:
        raw: Raw tool output (e.g., from SSH command, squeue, sacct)
        task_type: Context hint for what matters (status_check, job_submit, etc.)

    Returns:
        Cleaned output, max _MAX_OUTPUT_CHARS, with noise removed.
    """
    if not raw:
        return ""

    text = raw

    # Phase 1: Remove noise patterns
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # Phase 2: Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    # Phase 3: Compress stack traces (keep first frame + error only)
    text = _compress_tracebacks(text)

    # Phase 4: Task-specific extraction
    if task_type == "status_check":
        text = _extract_status_lines(text)
    elif task_type == "job_submit":
        text = _extract_job_ids(text)

    # Phase 5: Truncate if still too long
    if len(text) > _MAX_OUTPUT_CHARS:
        # Keep first and last portions
        half = _MAX_OUTPUT_CHARS // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]

    return text


def _compress_tracebacks(text: str) -> str:
    """Compress Python tracebacks to error message + first frame only."""
    lines = text.split("\n")
    result = []
    in_traceback = False
    tb_frames = []
    tb_start_idx = 0

    for i, line in enumerate(lines):
        if _TRACEBACK_START.match(line):
            in_traceback = True
            tb_start_idx = i
            tb_frames = []
            continue
        if in_traceback:
            if _TRACEBACK_FRAME.match(line):
                tb_frames.append(line)
                # Skip the following line (code line)
                continue
            if line.startswith("    ") and tb_frames:
                continue  # Code context line
            # End of traceback — this line is the error message
            in_traceback = False
            if tb_frames:
                result.append(f"Traceback: {tb_frames[0].strip()}")
                if len(tb_frames) > 1:
                    result.append(f"  ...{len(tb_frames)-1} more frames...")
            result.append(line)  # The actual error message
            continue
        result.append(line)

    return "\n".join(result)


def _extract_status_lines(text: str) -> str:
    """For status_check: keep only lines with job state info."""
    lines = text.split("\n")
    relevant = []
    for line in lines:
        if any(p.search(line) for p in _KEEP_PATTERNS):
            relevant.append(line)
        elif re.search(r"\d{5,}", line):  # Lines with job IDs
            relevant.append(line)
    if relevant:
        return "\n".join(relevant[:20])  # Max 20 status lines
    return text[:_MAX_OUTPUT_CHARS]  # Fallback: return truncated original


def _extract_job_ids(text: str) -> str:
    """For job_submit: extract job ID and confirmation."""
    # Look for "Submitted batch job XXXXX" pattern
    match = re.search(r"Submitted batch job (\d+)", text)
    if match:
        job_id = match.group(1)
        return f"job_id:{job_id}\nstatus:submitted"
    # Look for generic job ID patterns
    ids = re.findall(r"(?:job|JOB)\s*(?:ID|id)?[:\s]*(\d{5,})", text)
    if ids:
        return f"job_ids:{','.join(ids)}\nstatus:submitted"
    return text[:500]


def purify_hermes_response(response: str, task_type: str = "general") -> str:
    """Strip noise from Hermes LLM response content.

    Different from purify_output — this handles the LLM's prose response,
    not raw command output. Removes:
    - Unnecessary preambles ("Sure, I'll help you with that...")
    - Repeated context echoing
    - Verbose explanations when only data is needed
    """
    if not response:
        return ""

    # If response is JSON, keep as-is (already compact)
    stripped = response.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped[:_MAX_OUTPUT_CHARS]

    # Remove common LLM preambles
    preambles = [
        r"^(?:Sure|Of course|Certainly|I'll|Let me|Here's|Here is)[^\n]*(?:\.|:|!)\s*\n*",
        r"^(?:Based on|According to|Looking at)[^\n]*(?:\.|:|!)\s*\n*",
        r"^(?:The (?:output|result|response|command) shows)[^\n]*[.:]\s*\n*",
    ]
    text = stripped
    for p in preambles:
        text = re.sub(p, "", text, count=1)

    # Remove trailing summaries/offers
    outros = [
        r"\n\n(?:Let me know|Would you like|Do you want|Is there anything).*$",
        r"\n\n(?:Summary|In summary|To summarize):.*$",
    ]
    for p in outros:
        text = re.sub(p, "", text, flags=re.DOTALL)

    return text.strip()[:_MAX_OUTPUT_CHARS]
