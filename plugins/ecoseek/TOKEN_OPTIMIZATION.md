# EcoSeek Token Optimization Guide

**Goal:** Performance = Tokens / (Watt + $) — maximize useful output per cost unit.

## Changes Summary

### 1. Compact Response Mode (`response_format`)

**Before:** Every `escalate_remote` call returned ~500-1500 tokens of prose.
**After:** `response_format="compact"` forces JSON-only responses (~100-300 tokens).

```python
# Old: ~1200 tokens response
escalate_remote(task="Check job 22284217 status")
# → "I checked the job and it appears to be running well. There are 600 tasks
#    currently active with 200 completed. The job was submitted at..."

# New: ~80 tokens response
escalate_remote(task="Check job 22284217 status", response_format="compact", max_tokens=200)
# → {"result": {"state": "RUNNING", "tasks": 600, "completed": 200}, "error": null}
```

**Savings:** ~70-85% per call for status checks.

### 2. Batch Commands (`action="batch"`)

**Before:** 3 separate SSH round-trips for git pull + sbatch + squeue.
**After:** One `batch` call chains all commands in a single SSH session.

```python
# Old: 3 calls × ~500ms network + ~300 tokens each = 1.5s + 900 tokens
ku_hpc(action="raw", command="cd /work && git pull")
ku_hpc(action="raw", command="sbatch job.sh")
ku_hpc(action="status")

# New: 1 call × ~500ms + ~200 tokens
ku_hpc(action="batch", command="cd /work && git pull && sbatch job.sh && squeue -u a474r867")
```

**Savings:** ~66% fewer round-trips, ~60% fewer tokens.

### 3. DiDAL Sliding Window

**Before:** Turn N sends ALL N previous messages = O(n²) token growth.
At turn 10 with 500 tokens/msg: 10 × 500 = 5,000 tokens per call.

**After:** Only last 4 messages sent in full. Earlier turns compressed to ~150 chars each.
At turn 10: summary(6 × 150) + full(4 × 500) = 2,900 tokens per call.

**Savings:** ~42% at turn 10, grows with dialogue length. At turn 20: ~65%.

### 4. Max Tokens Budget

Force the LLM to stay within a token budget per response:
- Status checks: `max_tokens=200`
- Simple operations: `max_tokens=500`  
- Complex tasks: `max_tokens=1500`
- Unlimited (legacy): `max_tokens=0`

## Recommended Usage Patterns

### Status Check (cheapest)
```python
escalate_remote(
    task="squeue -j 22284217 --format=%T --noheader | sort | uniq -c",
    response_format="compact",
    max_tokens=200,
    urgency="high"
)
```

### Submit + Verify (batched)
```python
ku_hpc(
    action="batch",
    command="cd /home/a474r867/work/xsdm_1000_sp && git pull && "
            "sbatch templates/xsdm_v2_2var.sbatch && "
            "sleep 5 && squeue -u a474r867 --format=%i,%T,%j | head -5"
)
```

### Fire-and-Forget (zero follow-up tokens)
```python
escalate_remote(
    task="Submit job, monitor until done, push results to alrobles/test repo. "
         "Do NOT report back — just push results.",
    response_format="compact",
    urgency="background"
)
# Then: git pull alrobles/test to get results (free, 0 Hermes tokens)
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DIDAL_CONTEXT_WINDOW` | 4 | Messages to send in full (rest summarized) |
| `DIDAL_MAX_TURNS` | 20 | Max dialogue turns |
| `HERMES_REMOTE_TIMEOUT` | 300 | Request timeout (seconds) |

## Estimated Total Savings

| Pattern | Before (tokens/op) | After (tokens/op) | Reduction |
|---------|--------------------|--------------------|-----------|
| Status check | ~2,000 | ~400 | 80% |
| Submit + verify | ~4,500 | ~1,200 | 73% |
| DiDAL 10 turns | ~50,000 | ~29,000 | 42% |
| DiDAL 20 turns | ~200,000 | ~70,000 | 65% |
| Batch 3 commands | ~3,000 | ~800 | 73% |

**Net effect on Devin token consumption:** ~60-75% reduction for HPC workflows.
