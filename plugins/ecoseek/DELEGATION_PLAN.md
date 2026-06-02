# Hermes Delegation Plan: Devin → Hermes Token Reduction

**Goal:** Reduce Devin token consumption by 60-80% by delegating more work to Hermes and its subagents.

**Principle:** Performance = Tokens / (Watt + $)
- Maximize useful output per token consumed
- Hermes costs $0 (self-hosted DeepSeek on reumanlab)
- Devin costs $200/month — every token saved extends capacity

---

## Current State (Measured)

| Action | Devin tokens | Hermes tokens | Who decides |
|--------|-------------|---------------|-------------|
| Status check (squeue) | 200-900 | 0 (raw cmd) | Devin |
| Submit job (sbatch) | 200-400 | 0 (raw cmd) | Devin |
| Monitor job completion | 500-2000 (polling) | 0 | Devin |
| Parse error + decide fix | 1500-3000 | 0 | Devin |
| Apply fix + resubmit | 800-1500 | 0 | Devin |
| Create PR | 2000-4000 | 0 | Devin |
| Review results | 1000-2000 | 0 | Devin |

**Total per HPC cycle:** ~6,000-12,000 Devin tokens

---

## Target State (Hermes-Empowered)

| Action | Devin tokens | Hermes tokens | Who decides |
|--------|-------------|---------------|-------------|
| Status check | 0 | 100 | Hermes cron |
| Submit job | 100 (fire-and-forget) | 50 | Devin delegates |
| Monitor job completion | 0 | 200 (cron) | Hermes autonomous |
| Parse error + decide fix | 0 | 500 | Hermes pattern-match |
| Apply known fix + resubmit | 0 | 300 | Hermes autonomous |
| Apply unknown fix | 500 (Devin reviews) | 300 | Escalate to Devin |
| Create PR | 1500 | 0 | Devin (T3) |
| Report results to GitHub | 0 | 200 | Hermes cron |

**Total per HPC cycle:** ~2,100 Devin tokens (vs 9,000 = **77% reduction**)

---

## Architecture: 4-Tier Delegation

### Tier 0: Zero-Devin (Fully Autonomous Hermes)

Tasks Hermes handles alone, no Devin involvement:

1. **Cron watchdog** — monitors running jobs every 5 min
   - `squeue -u a474r867` → parse → if COMPLETED/FAILED → trigger next action
   - On COMPLETED: copy results, run validation, push summary to GitHub
   - On FAILED: check error against pattern library → if match, auto-fix + resubmit

2. **Pattern-match fixer** — known errors get auto-fixed
   - `libRlapack.so not found` → rebuild container with symlinks
   - `OOM killed` → re-queue with `--mem=64G`
   - `convergence failure` → re-run with `num_starts=1500`
   - Each pattern: shell script + validation + report

3. **GitHub reporter** — push status updates without Devin
   - Job completed → write `docs/status/YYYY-MM-DD.md` → git push
   - Results ready → create summary CSV → commit to repo
   - Hermes memory stores the pattern; cron executes

4. **Health checks** — cluster status, disk usage, queue depth
   - Daily: `df -h scratch`, `squeue | wc -l`, model count
   - Alert only if something breaks threshold

### Tier 1: Fire-and-Forget (Devin initiates, Hermes executes + reports)

Devin spends ~100 tokens to dispatch, then moves on:

```python
# Devin says ONE thing:
escalate_remote(
    "Submit xsdm_v2 phase 2var for species 1-1749, "
    "monitor until done, push results summary to GitHub. "
    "If libRlapack fails, rebuild container and retry.",
    response_format="compact",
    max_tokens=100
)
# Devin cost: ~200 tokens total
# Hermes does: submit + monitor + fix + report (hours of work, $0)
```

### Tier 2: Guided Delegation (Devin plans, Hermes executes, Devin reviews)

For novel tasks where Hermes needs a plan but can execute autonomously:

```python
# Devin sends plan (500 tokens):
escalate_remote(
    "Plan: 1) Extract env19 for 775 species 2) Run phase 2var "
    "3) Collect results 4) Push to GitHub. "
    "Scripts at templates/xsdm_v2_*.sbatch. "
    "Report completion via git push to docs/status/.",
    response_format="compact"
)
# Later: Devin does `git pull` to see results (0 Hermes tokens)
```

### Tier 3: Devin-Required (Novel reasoning, code creation, PRs)

Only these remain on Devin:
- Writing new R/Python code
- Designing new algorithms
- Creating PRs with documentation
- Architectural decisions
- Interpreting scientific results

---

## Implementation Plan

### Phase 1: Hermes Cron Profiles (Week 1)

Create 3 Hermes profiles using the Kanban system:

```yaml
# ~/.hermes/profiles/hpc-monitor.yaml
name: hpc-monitor
role: "HPC job monitor — check status, report to GitHub"
tools: [ku-hpc, terminal, git]
cron: "*/5 * * * *"  # every 5 minutes
skills: [kanban-worker]

# ~/.hermes/profiles/pattern-fixer.yaml  
name: pattern-fixer
role: "Auto-fix known HPC errors from pattern library"
tools: [ku-hpc, terminal, git]
trigger: "on hpc-monitor finding FAILED job"
skills: [kanban-worker]

# ~/.hermes/profiles/results-reporter.yaml
name: results-reporter
role: "Collect, summarize, and push HPC results to GitHub"
tools: [ku-hpc, terminal, git]
trigger: "on hpc-monitor finding COMPLETED job"
skills: [kanban-worker]
```

**Key insight from Hermes Kanban docs:** Workers are full OS processes with their own identity, durable state in SQLite, and human-in-the-loop via comments. Perfect for autonomous HPC monitoring.

### Phase 2: Pattern Library Expansion (Week 1-2)

Expand from 6 → 20 patterns. Each pattern is a script on the cluster:

```bash
# ~/bin/patterns/fix_librlapack.sh
#!/bin/bash
# Pattern: libRlapack.so not found
# Trigger: grep "libRlapack" in slurm error output
# Fix: add LD_LIBRARY_PATH to sbatch header
sed -i '/^#SBATCH/a export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu' "$1"
echo "FIXED: Added LD_LIBRARY_PATH for libRlapack"
```

Pattern registry in Hermes memory:
```json
{
  "patterns": [
    {"error": "libRlapack.so", "fix": "~/bin/patterns/fix_librlapack.sh", "confidence": 0.99},
    {"error": "out of memory", "fix": "~/bin/patterns/increase_mem.sh", "confidence": 0.95},
    {"error": "convergence.*ll_range", "fix": "~/bin/patterns/increase_starts.sh", "confidence": 0.8},
    {"error": "ModuleNotFoundError", "fix": "~/bin/patterns/install_module.sh", "confidence": 0.7}
  ]
}
```

### Phase 3: Adaptive Supervisor (Week 2-3)

Inspired by SupervisorAgent (ICLR 2026, 29.68% token reduction):

**LLM-free filter:** Before escalating to Devin, check:
1. Is this a known pattern? → auto-fix (T0)
2. Is this a simple variant of a known pattern? → Hermes reasons (T1)
3. Is this truly novel? → escalate to Devin (T3)

```python
def should_escalate_to_devin(error: str, context: dict) -> bool:
    """LLM-free filter — only escalate truly novel problems."""
    # Check pattern library first (zero LLM cost)
    for pattern in PATTERN_LIBRARY:
        if re.search(pattern["error"], error):
            if pattern["confidence"] >= 0.8:
                return False  # Auto-fix, don't bother Devin
    
    # Check if similar to any pattern (fuzzy match)
    similarities = [fuzz.ratio(error, p["error"]) for p in PATTERN_LIBRARY]
    if max(similarities) > 70:
        return False  # Hermes can reason about this
    
    # Truly novel — escalate
    return True
```

### Phase 4: Kanban Swarm for Complex Tasks (Week 3-4)

For multi-step HPC workflows, use Hermes' built-in swarm topology:

```
Devin creates ONE kanban task: "Run expanded xsdm pipeline for 775 species"
    │
    ├── hpc-monitor: watches job queue, reports completion
    ├── pattern-fixer: auto-resolves known failures  
    ├── results-reporter: collects + summarizes when done
    └── verifier: checks quality metrics before pushing

Devin only sees the final summary pushed to GitHub.
```

This is the **fan-out / fan-in** pattern from production literature — 75% wall-clock reduction, all executed on Hermes ($0).

### Phase 5: Minicluster Distribution (Week 4+)

When 3 machines are available:

| Machine | Hermes Profile | Autonomous Actions |
|---------|---------------|-------------------|
| M1 (gateway) | orchestrator | Route tasks, cron scheduling |
| M2 (inference) | deep-reasoner | Fix novel errors, write scripts |
| M3 (monitor) | hpc-monitor + reporter | Continuous job monitoring |

The kanban board is shared (SQLite or PostgreSQL) — any machine can read/write tasks.

---

## Concrete Improvements to ecoseek Plugin

### 1. Add `autonomous_mode` parameter

```python
def escalate_remote(task, ..., autonomous=False):
    """
    If autonomous=True, Hermes is authorized to:
    - Execute the task without waiting for Devin review
    - Auto-fix known errors
    - Push results to GitHub
    - Only escalate back if truly stuck
    """
```

### 2. Add `fire_and_forget` function

```python
def fire_and_forget(task: str, report_to: str = "github") -> str:
    """Submit task and return immediately. Hermes handles everything.
    
    Args:
        task: What to do (natural language or direct command)
        report_to: Where to report completion ("github", "slack", "none")
    
    Returns:
        Task ID for later reference (if needed)
    """
```

### 3. Add `pattern_check` before escalation

```python
def pattern_check(error: str) -> Optional[str]:
    """Check if error matches a known pattern. Returns fix command or None.
    
    If match found, Hermes applies fix autonomously.
    If no match, error is escalated to Devin.
    """
```

### 4. Add `batch_delegate` for parallel work

```python
def batch_delegate(tasks: list[str], parallel: int = 4) -> str:
    """Submit multiple tasks to Hermes in one call.
    
    Uses kanban fan-out: creates N parallel cards,
    returns when all complete (or reports partial failures).
    """
```

### 5. Add cron-based auto-reporting

```python
def setup_auto_report(job_id: str, repo: str, interval_min: int = 5):
    """Tell Hermes to monitor a job and push status to GitHub.
    
    Devin cost: ~100 tokens (one-time setup)
    Hermes cost: ~50 tokens per check ($0)
    Previous cost: ~500 tokens per manual check from Devin
    """
```

---

## Projected Token Savings

| Scenario | Before (Devin tokens/month) | After (Devin tokens/month) | Savings |
|----------|----------------------------|---------------------------|---------|
| HPC pipeline (50 jobs) | 450,000 | 105,000 | **77%** |
| Status checks (200/month) | 180,000 | 0 | **100%** |
| Error fixing (30/month) | 90,000 | 15,000 | **83%** |
| PR creation (10/month) | 30,000 | 25,000 | **17%** |
| Total | **750,000** | **145,000** | **81%** |

---

## Research Sources

1. **SupervisorAgent** (ICLR 2026) — LLM-free adaptive filter reduces tokens 29.68% without quality loss
2. **Uno-Orchestra** — Unified orchestration policy: selective decomposition + dispatch, 10× cost reduction
3. **Beam.ai Production Patterns** — Orchestrator-worker (40-60% cost cut), fan-out/fan-in (75% time cut)
4. **Hermes Kanban** — Durable multi-agent board, crash-recovery, human-in-the-loop, named profiles
5. **VoltAgent subagents** — Typed delegation with streaming, schema validation, memory persistence
6. **OpenClaw skills** — 1200+ community skills for coding, monitoring, automation

---

## Next Steps (Priority Order)

1. **[NOW]** Implement `fire_and_forget()` in ecoseek plugin
2. **[NOW]** Create 3 helper scripts on cluster (`~/bin/patterns/`)
3. **[WEEK 1]** Set up Hermes cron for job monitoring (already have the infrastructure)
4. **[WEEK 1]** Expand pattern library from 6 → 15 patterns
5. **[WEEK 2]** Add LLM-free filter (regex/fuzzy match before LLM reasoning)
6. **[WEEK 2]** Test kanban swarm with a real HPC workflow
7. **[WEEK 3]** Deploy autonomous mode + measure actual TER improvement
8. **[WEEK 4]** Minicluster distribution (when hardware arrives)
