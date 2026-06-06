"""DiDAL Hypothesis Protocol — Phase 4 of EcoSeek.

Extends the DiDAL (Dialectical Dual-Agent Loop) from binary debate into
a dialectical hypothesis refinement protocol rooted in Hegelian dialectics:

  THESIS → ANTITHESIS → SYNTHESIS

Alpha (Emily, the ecologist) proposes hypotheses (thesis).
Beta (the remote executor/critic) challenges them with evidence (antithesis).
The dialectical clash produces refined, stronger hypotheses (synthesis).

This is NOT a sports tournament. This is dialectical science — hypotheses
are strengthened through structured critique, not eliminated by ranking.
Elo ratings are an internal metric; the output is a synthesized research
program, not a leaderboard.

Phases:
  1. THESIS — Alpha generates N hypotheses from literature + ecological data
  2. DIALECTIC — Each hypothesis faces Beta's critique; clash produces refinement
  3. SYNTHESIS — The refined hypotheses are woven into a coherent research proposal

Model tiering (DiDAL philosophy: Alpha thinks, Beta executes):
  - Alpha (thesis generation + synthesis): DeepSeek (deep ecological reasoning)
  - Beta (dialectical critique): EcoCoder local via ollama GPU (fast, free iteration)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dialectical Strength (internal metric — not exposed as "ranking")
# ---------------------------------------------------------------------------

DEFAULT_STRENGTH = 1200  # Initial dialectical strength (internal Elo)
K_FACTOR = 32            # How much a single dialectical clash shifts strength


def expected_outcome(strength_a: float, strength_b: float) -> float:
    """Probability that thesis A withstands antithesis B in dialectical clash."""
    return 1.0 / (1.0 + math.pow(10, (strength_b - strength_a) / 400.0))


def update_dialectical_strength(
    prevailing: float,
    challenged: float,
    k: int = K_FACTOR,
) -> tuple[float, float]:
    """Return (new_prevailing, new_challenged) after a dialectical round.

    The prevailing hypothesis gains strength; the challenged one learns
    from the encounter and may come back stronger in the next round.
    """
    expected = expected_outcome(prevailing, challenged)
    delta = k * (1.0 - expected)
    return (
        round(prevailing + delta, 1),
        round(challenged - delta, 1),
    )


# ---------------------------------------------------------------------------
# Hypothesis (a thesis in the DiDAL protocol)
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A testable ecological thesis within the DiDAL dialectical protocol.

    Each hypothesis passes through: thesis → antithesis (Beta critique) →
    synthesis (refined version). The dialectical_strength tracks how well
    it has withstood critique across rounds.
    """

    id: str
    statement: str
    rationale: str
    predictions: list[str] = field(default_factory=list)
    testability: float = 0.5     # 0..1
    novelty: float = 0.5          # 0..1
    confidence: float = 0.5       # 0..1
    dialectical_strength: float = DEFAULT_STRENGTH  # Internal metric
    refinement_round: int = 1     # 1 = original thesis, 2+ = synthesized
    parent_id: Optional[str] = None  # id of the thesis this was synthesized from
    critique_history: list[dict] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "predictions": self.predictions,
            "testability": self.testability,
            "novelty": self.novelty,
            "confidence": self.confidence,
            "dialectical_strength": self.dialectical_strength,
            "refinement_round": self.refinement_round,
            "parent_id": self.parent_id,
            "evidence_sources": self.evidence_sources,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            id=d["id"],
            statement=d["statement"],
            rationale=d.get("rationale", ""),
            predictions=d.get("predictions", []),
            testability=d.get("testability", 0.5),
            novelty=d.get("novelty", 0.5),
            confidence=d.get("confidence", 0.5),
            dialectical_strength=d.get("dialectical_strength", DEFAULT_STRENGTH),
            refinement_round=d.get("refinement_round", 1),
            parent_id=d.get("parent_id"),
            evidence_sources=d.get("evidence_sources", []),
        )


# ---------------------------------------------------------------------------
# Phase 1: THESIS — Alpha generates hypotheses
# ---------------------------------------------------------------------------

THESIS_SYSTEM_PROMPT = """\
You are Alpha, the senior ecologist in the EcoSeek DiDAL (Dialectical Dual-Agent
Loop) system. Your role is to propose testable ecological theses (hypotheses)
that will be challenged by Beta, your dialectical counterpart.

For each thesis, provide:
1. A clear, falsifiable statement (the thesis)
2. Ecological rationale — why this might be true, grounded in theory
3. Specific, measurable predictions that follow from the thesis
4. Testability score (0-1): how easily can Beta verify this with available data?
5. Novelty score (0-1): does this challenge existing paradigms?
6. Initial confidence (0-1): your prior belief before dialectical testing

DiDAL principles:
- Each thesis must be FALSIFIABLE — Beta needs something concrete to critique
- Generate DIVERSE theses covering different mechanisms, scales, and taxa
- Favor ECOLOGICAL SPECIFICITY over vagueness
- A thesis that survives Beta's critique is stronger than one that was never tested

Output format: JSON array of thesis objects.
```json
[
  {
    "statement": "...",
    "rationale": "...",
    "predictions": ["...", "..."],
    "testability": 0.8,
    "novelty": 0.6,
    "confidence": 0.5
  }
]
```
"""


def _parse_theses_json(raw: str, n_expected: int = 5) -> list[dict]:
    """Extract thesis array from LLM response (may have markdown wrapping)."""
    # Direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed[:n_expected]
        if isinstance(parsed, dict) and "hypotheses" in parsed:
            return parsed["hypotheses"][:n_expected]
    except json.JSONDecodeError:
        pass

    # Markdown code block
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return parsed[:n_expected]
            if isinstance(parsed, dict) and "hypotheses" in parsed:
                return parsed["hypotheses"][:n_expected]
        except json.JSONDecodeError:
            pass

    # Bare array
    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed[:n_expected]
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse theses from LLM response")
    return []


def _validate_thesis(h: dict) -> bool:
    """Check required fields exist and are sensible."""
    required = ["statement", "rationale", "predictions"]
    for field in required:
        if field not in h or not h[field]:
            return False
    if not isinstance(h.get("predictions"), list):
        return False
    if len(h["predictions"]) == 0:
        return False
    for key in ("testability", "novelty", "confidence"):
        if key in h:
            h[key] = max(0.0, min(1.0, float(h[key])))
    return True


# ---------------------------------------------------------------------------
# Phase 2: DIALECTIC — Beta challenges each thesis
# ---------------------------------------------------------------------------

DIALECTIC_SYSTEM_PROMPT = """\
You are Beta, the dialectical counterpart in the EcoSeek DiDAL system.
Your role is to CHALLENGE Alpha's theses through structured critique.

For each thesis you face, your job is NOT to destroy it, but to TEST it:
1. Identify weaknesses, hidden assumptions, and logical gaps
2. Demand evidence — what data would confirm or refute this?
3. Propose counter-examples — where would this thesis fail?
4. Suggest refinements — how could the thesis be strengthened?

DiDAL principles:
- Critique must be CONSTRUCTIVE — the goal is stronger science, not winning
- Every challenge must come with a PATH FORWARD — "this is weak because X; try Y"
- Ground arguments in ecological reality — cite mechanisms, not opinions
- The thesis that survives your best critique IS the stronger thesis

Debate format — respond in this structure:
1. CHALLENGE: Identify the weakest point in the thesis
2. DEMAND: What evidence is missing? What data would test this?
3. COUNTER: Where would this thesis fail in nature?
4. REFINE: How could the thesis be improved?

Keep responses under 200 words. Be rigorous but constructive.
"""

# ---------------------------------------------------------------------------
# Phase 2b: Dialectical Judgment (Beta's assessment after the clash)
# ---------------------------------------------------------------------------

DIALECTICAL_JUDGMENT_PROMPT = """\
You are Beta, evaluating the outcome of a dialectical clash between two
ecological theses in the DiDAL protocol.

After both sides have been heard, assess which thesis withstood critique
better based on:

1. Evidence resilience (0-10): Did the thesis hold up under scrutiny?
2. Logical coherence (0-10): Is the reasoning sound after challenge?
3. Novelty preservation (0-10): Did novelty survive critique?
4. Testability (0-10): Can this actually be tested with available data?
5. Ecological grounding (0-10): Is it rooted in ecological theory?

The thesis that better withstood dialectical testing prevails — not
because it "won", but because it proved more robust under fire.

Output JSON:
```json
{
  "prevailing": "A" or "B",
  "assessment": {
    "A": {"resilience": 8, "coherence": 7, "novelty": 6, "testability": 9, "grounding": 8},
    "B": {"resilience": 6, "coherence": 8, "novelty": 9, "testability": 5, "grounding": 7}
  },
  "rationale": "Why the prevailing thesis proved more dialectically robust"
}
```
"""


# ---------------------------------------------------------------------------
# Phase 2c: SYNTHESIS — Refine the prevailing thesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """\
You are Alpha, synthesizing the outcome of a dialectical clash in the DiDAL
protocol. A thesis has withstood Beta's critique — now it must be refined.

The Hegelian dialectic: THESIS + ANTITHESIS → SYNTHESIS.

Given the prevailing thesis and the critiques it survived:
1. INCORPORATE valid challenges — fix the weaknesses Beta exposed
2. STRENGTHEN predictions — make them more specific and falsifiable
3. DEEPEN the rationale — add ecological mechanisms revealed by the debate
4. UPDATE confidence — the thesis is now stronger for having been tested
5. DOCUMENT changes — what was learned from the dialectical process

This is NOT "evolution" — it is DIALECTICAL SYNTHESIS. The thesis emerges
stronger precisely because it was challenged.

Output JSON:
```json
{
  "statement": "refined thesis statement",
  "rationale": "deepened rationale incorporating dialectical insights",
  "predictions": ["more specific prediction 1", "more specific prediction 2"],
  "testability": 0.9,
  "novelty": 0.7,
  "confidence": 0.6,
  "dialectical_insight": "What the critique revealed and how the thesis improved"
}
```
"""


# ---------------------------------------------------------------------------
# Phase 3: SYNTHESIS — Weave refined theses into a research program
# ---------------------------------------------------------------------------

RESEARCH_PROGRAM_PROMPT = """\
You are Alpha, the senior ecologist in EcoSeek DiDAL, synthesizing the results
of a dialectical hypothesis refinement session into a coherent research program.

The theses before you have survived Beta's dialectical testing. They are not
"ranked" — they are REFINED. Each has been strengthened through critique.

Your task: weave them into a RESEARCH PROGRAM — not a competition result.

Sections:
1. Research Question: The original ecological inquiry
2. Dialectical Process: How the theses were tested and refined (method)
3. Refined Theses: Each thesis with its dialectical journey (original →
   critique → synthesis), current strength, and ecological significance
4. Integrated Framework: How the theses connect — do they form a coherent
   picture? Where do they complement or contradict each other?
5. Proposed Investigation: How to empirically test the synthesized theses
   (methods, data sources: GBIF, SDM, field, experimental)
6. Expected Contributions: What this research program would contribute to
   ecological knowledge
7. Open Questions: What the dialectical process revealed as still unknown

Output as clean markdown suitable for a research proposal or grant application.
Remember: this is a DIALECTICAL SYNTHESIS, not a tournament leaderboard.
"""


# ---------------------------------------------------------------------------
# LLM Call Helpers (model-tiered per DiDAL roles)
# ---------------------------------------------------------------------------

# Alpha calls: DeepSeek (deep ecological reasoning for thesis + synthesis)
# Beta calls: EcoCoder local (fast critique, no cost per round)


def _alpha_call(system_prompt: str, user_message: str,
                max_tokens: int = 2000) -> str:
    """Alpha (thesis generation, synthesis) uses DeepSeek for depth."""
    import urllib.error
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set — Alpha needs DeepSeek")

    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Alpha (DeepSeek) returned no choices")
    return choices[0]["message"]["content"]


def _beta_call(system_prompt: str, user_message: str,
               model: str = "ecocoder:7b",
               max_tokens: int = 500) -> str:
    """Beta (dialectical critique) uses local EcoCoder GPU for fast iteration."""
    import urllib.error
    import urllib.request

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://172.27.112.1:11434")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Beta (Ollama) returned no choices")
    return choices[0]["message"]["content"]


# ---------------------------------------------------------------------------
# Core DiDAL Protocol Functions
# ---------------------------------------------------------------------------


def generate_theses(
    question: str,
    n: int = 5,
    domain: str = "ecology",
    context: str = "",
) -> list[Hypothesis]:
    """Phase 1: THESIS — Alpha generates N testable ecological theses.

    Alpha proposes; Beta will challenge them in Phase 2.
    """
    user_msg = f"Research question: {question}\nDomain: {domain}\nGenerate {n} theses."
    if context:
        user_msg += f"\n\nAdditional context:\n{context}"

    raw = _alpha_call(THESIS_SYSTEM_PROMPT, user_msg, max_tokens=2000)
    parsed = _parse_theses_json(raw, n_expected=n)

    theses = []
    for i, h in enumerate(parsed):
        if not _validate_thesis(h):
            logger.warning("Thesis %d failed validation, skipping", i)
            continue
        theses.append(Hypothesis(
            id=f"th_{uuid.uuid4().hex[:8]}",
            statement=h["statement"],
            rationale=h["rationale"],
            predictions=h["predictions"],
            testability=h.get("testability", 0.5),
            novelty=h.get("novelty", 0.5),
            confidence=h.get("confidence", 0.5),
            refinement_round=1,
        ))

    logger.info("Alpha generated %d/%d valid theses", len(theses), n)
    return theses


def dialectical_clash(
    thesis_a: Hypothesis,
    thesis_b: Hypothesis,
    question: str,
    max_exchanges: int = 4,
    model: str = "ecocoder:7b",
) -> dict:
    """Phase 2: DIALECTIC — Beta challenges both theses in structured debate.

    This is NOT a competition. It is a dialectical process where both
    theses are tested by Beta's critique. The one that withstands
    scrutiny better demonstrates greater dialectical strength.

    Returns dict with transcript, exchanges, tokens_used.
    """
    # Beta's challenge to thesis A
    def build_beta_prompt(thesis: Hypothesis, opponent: Hypothesis) -> str:
        return f"""\
You are Beta, challenging the following ecological thesis in the DiDAL protocol.

RESEARCH QUESTION: {question}

THESIS UNDER SCRUTINY: {thesis.statement}
RATIONALE: {thesis.rationale}
PREDICTIONS: {json.dumps(thesis.predictions)}

OPPOSING THESIS (for context): {opponent.statement}

Remember: your role is CONSTRUCTIVE CRITIQUE. Challenge rigorously but
always suggest how the thesis could be strengthened.
"""

    transcript_lines = []
    total_tokens = 0

    # Round 1: Beta challenges thesis A
    prompt_a = build_beta_prompt(thesis_a, thesis_b)
    msg_a = f"Challenge this thesis: {thesis_a.statement}"
    try:
        crit_a = _beta_call(DIALECTIC_SYSTEM_PROMPT, msg_a, model=model, max_tokens=300)
    except Exception as e:
        logger.warning("Beta critique of thesis A failed: %s", e)
        crit_a = f"[Dialectical error: {e}]"
    transcript_lines.append(f"[Exchange 1] Beta challenges Thesis A:\n{crit_a}")

    # Round 1: Beta challenges thesis B
    prompt_b = build_beta_prompt(thesis_b, thesis_a)
    msg_b = f"Challenge this thesis: {thesis_b.statement}"
    try:
        crit_b = _beta_call(DIALECTIC_SYSTEM_PROMPT, msg_b, model=model, max_tokens=300)
    except Exception as e:
        logger.warning("Beta critique of thesis B failed: %s", e)
        crit_b = f"[Dialectical error: {e}]"
    transcript_lines.append(f"[Exchange 1] Beta challenges Thesis B:\n{crit_b}")

    # Rounds 2..max_exchanges: alternating deeper critique
    for ex in range(2, max_exchanges + 1):
        # Beta deepens critique of A
        msg = f"Deepen your critique. Previous challenge: {crit_a[:300]}\nResponse so far: {crit_b[:300]}"
        try:
            crit_a = _beta_call(DIALECTIC_SYSTEM_PROMPT, msg, model=model, max_tokens=300)
        except Exception as e:
            crit_a = f"[Dialectical error: {e}]"
        transcript_lines.append(f"[Exchange {ex}] Beta deepens on Thesis A:\n{crit_a}")

        # Beta deepens critique of B
        msg = f"Deepen your critique. Previous challenge: {crit_b[:300]}\nResponse so far: {crit_a[:300]}"
        try:
            crit_b = _beta_call(DIALECTIC_SYSTEM_PROMPT, msg, model=model, max_tokens=300)
        except Exception as e:
            crit_b = f"[Dialectical error: {e}]"
        transcript_lines.append(f"[Exchange {ex}] Beta deepens on Thesis B:\n{crit_b}")

    transcript = "\n\n".join(transcript_lines)
    return {
        "transcript": transcript,
        "exchanges": max_exchanges,
        "total_tokens": total_tokens,
    }


def judge_dialectic(transcript: str, thesis_a: Hypothesis,
                    thesis_b: Hypothesis) -> dict:
    """Phase 2b: DIALECTICAL JUDGMENT — Alpha assesses which thesis
    withstood Beta's critique better.

    Returns dict with prevailing thesis, assessment scores, rationale.
    """
    user_msg = (
        f"Dialectical transcript:\n{transcript}\n\n"
        f"Thesis A: {thesis_a.statement}\n"
        f"Thesis B: {thesis_b.statement}\n\n"
        f"Assess which thesis better withstood dialectical testing."
    )

    raw = _alpha_call(DIALECTICAL_JUDGMENT_PROMPT, user_msg, max_tokens=500)
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            result = {"prevailing": "A", "rationale": "Could not parse judgment"}
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
            except json.JSONDecodeError:
                result = {"prevailing": "A", "rationale": "Could not parse judgment"}
        else:
            logger.warning("Could not parse dialectical judgment, defaulting to A")
            result = {"prevailing": "A", "rationale": "Could not parse judgment"}

    return result


def synthesize_thesis(prevailing: Hypothesis,
                      critique_transcript: str) -> Hypothesis:
    """Phase 2c: SYNTHESIS — Alpha refines the prevailing thesis using
    insights from Beta's critique.

    Hegelian: THESIS + ANTITHESIS → SYNTHESIS.
    """
    user_msg = (
        f"Prevailing thesis: {prevailing.statement}\n"
        f"Original rationale: {prevailing.rationale}\n"
        f"Original predictions: {json.dumps(prevailing.predictions)}\n\n"
        f"Beta's dialectical critique:\n{critique_transcript}\n\n"
        f"Synthesize a refined thesis. Return JSON."
    )

    raw = _alpha_call(SYNTHESIS_SYSTEM_PROMPT, user_msg, max_tokens=800)
    try:
        synthesized = json.loads(raw)
        if isinstance(synthesized, list):
            logger.warning("Synthesis returned list instead of dict, keeping original")
            return prevailing
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                synthesized = json.loads(match.group(1))
                if isinstance(synthesized, list):
                    return prevailing
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Could not parse synthesis, keeping original")
                return prevailing
        else:
            logger.warning("Could not parse synthesis, keeping original")
            return prevailing

    return Hypothesis(
        id=f"th_{uuid.uuid4().hex[:8]}",
        statement=synthesized.get("statement", prevailing.statement),
        rationale=synthesized.get("rationale", prevailing.rationale),
        predictions=synthesized.get("predictions", prevailing.predictions),
        testability=float(synthesized.get("testability", prevailing.testability)),
        novelty=float(synthesized.get("novelty", prevailing.novelty)),
        confidence=float(synthesized.get("confidence", prevailing.confidence)),
        dialectical_strength=prevailing.dialectical_strength,
        refinement_round=prevailing.refinement_round + 1,
        parent_id=prevailing.id,
    )


def weave_research_program(
    question: str,
    theses: list[Hypothesis],
    dialectical_log: list[dict],
) -> str:
    """Phase 3: SYNTHESIS — Alpha weaves all refined theses into a
    coherent ecological research program.

    This is the final output: not a ranking, but a research proposal
    born from dialectical refinement.
    """
    # Order by dialectical strength (internal metric, not exposed as "ranking")
    ordered = sorted(theses, key=lambda h: h.dialectical_strength, reverse=True)
    theses_text = "\n\n".join(
        f"### Thesis {i+1}: {h.statement}\n"
        f"* Dialectical strength: {h.dialectical_strength:.0f} "
        f"(refined through {h.refinement_round} round(s))*\n\n"
        f"Rationale: {h.rationale}\n\n"
        f"Predictions: {', '.join(h.predictions)}\n\n"
        f"Testability: {h.testability:.0%} | Novelty: {h.novelty:.0%} | "
        f"Confidence: {h.confidence:.0%}"
        for i, h in enumerate(ordered)
    )

    process_summary = "\n".join(
        f"Round {log.get('round', '?')}: "
        f"Thesis {log.get('prevailing_id', '?')} withstood critique "
        f"against thesis {log.get('challenged_id', '?')}"
        for log in dialectical_log[-10:]
    )

    user_msg = (
        f"Research question: {question}\n\n"
        f"Refined theses (ordered by dialectical strength):\n{theses_text}\n\n"
        f"Dialectical process summary:\n{process_summary}\n\n"
        f"Weave these into a coherent ecological research program."
    )

    return _alpha_call(RESEARCH_PROGRAM_PROMPT, user_msg, max_tokens=3000)


# ---------------------------------------------------------------------------
# DiDAL Hypothesis Protocol — Main Entry Point
# ---------------------------------------------------------------------------


def dialectical_hypothesis_refinement(
    question: str,
    n_theses: int = 5,
    n_rounds: int = 3,
    domain: str = "ecology",
    context: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Run the full DiDAL Hypothesis Protocol.

    THESIS → DIALECTIC → SYNTHESIS

    1. Alpha generates N ecological theses (thesis)
    2. Beta challenges each through structured critique (antithesis)
    3. Alpha synthesizes refined theses + research program (synthesis)

    This is dialectical science, not a tournament. Theses are strengthened
    through critique, not eliminated by ranking.

    Returns JSON with refined theses, dialectical log, and research program.
    """
    t0 = time.time()
    trace_id = task_id or str(uuid.uuid4())[:8]
    dialectical_log: list[dict] = []

    # ==================================================================
    # Phase 1: THESIS — Alpha generates
    # ==================================================================
    logger.info("didal[%s] Phase 1: Alpha generating %d theses", trace_id, n_theses)
    theses = generate_theses(question, n=n_theses, domain=domain, context=context)

    if len(theses) < 3:
        return json.dumps({
            "success": False,
            "error": "insufficient_theses",
            "message": (
                f"Alpha generated only {len(theses)} valid theses "
                f"(need ≥3 for dialectical testing)."
            ),
            "trace_id": trace_id,
        })

    dialectical_log.append({
        "phase": "thesis",
        "n_requested": n_theses,
        "n_generated": len(theses),
        "theses": [t.to_dict() for t in theses],
    })

    # ==================================================================
    # Phase 2: DIALECTIC — Beta challenges, Alpha synthesizes
    # ==================================================================
    logger.info("didal[%s] Phase 2: %d rounds of dialectical testing", trace_id, n_rounds)

    for round_num in range(1, n_rounds + 1):
        # Pair theses by dialectical strength for focused clash
        sorted_theses = sorted(theses, key=lambda h: h.dialectical_strength, reverse=True)

        for i in range(0, len(sorted_theses) - 1, 2):
            th_a = sorted_theses[i]
            th_b = sorted_theses[i + 1]

            logger.info(
                "didal[%s] Round %d: %s (%.0f) ↔ %s (%.0f)",
                trace_id, round_num, th_a.id, th_a.dialectical_strength,
                th_b.id, th_b.dialectical_strength,
            )

            # Beta challenges both theses
            clash = dialectical_clash(th_a, th_b, question)
            transcript = clash["transcript"]

            # Alpha judges which withstood critique better
            judgment = judge_dialectic(transcript, th_a, th_b)
            prevailing_id = th_a.id if judgment.get("prevailing") == "A" else th_b.id
            challenged_id = th_b.id if judgment.get("prevailing") == "A" else th_a.id

            # Update dialectical strength
            prevailing = th_a if prevailing_id == th_a.id else th_b
            challenged = th_b if prevailing_id == th_a.id else th_b
            new_prev, new_chall = update_dialectical_strength(
                prevailing.dialectical_strength, challenged.dialectical_strength
            )
            prevailing.dialectical_strength = new_prev
            challenged.dialectical_strength = new_chall

            # Alpha synthesizes the prevailing thesis
            synthesized = synthesize_thesis(prevailing, transcript)
            synthesized.dialectical_strength = new_prev

            # Replace the original with the synthesized version
            for j, th in enumerate(theses):
                if th.id == prevailing.id:
                    theses[j] = synthesized
                    break

            clash_log = {
                "phase": "dialectic",
                "round": round_num,
                "thesis_a": th_a.to_dict(),
                "thesis_b": th_b.to_dict(),
                "prevailing_id": prevailing_id,
                "challenged_id": challenged_id,
                "strength_after": {"prevailing": new_prev, "challenged": new_chall},
                "judgment": judgment,
                "synthesized": synthesized.to_dict() if synthesized.id != prevailing.id else None,
            }
            dialectical_log.append(clash_log)

            logger.info(
                "didal[%s] Round %d: %s prevails over %s (strength: %.0f > %.0f)",
                trace_id, round_num, prevailing_id, challenged_id, new_prev, new_chall,
            )

    # ==================================================================
    # Phase 3: SYNTHESIS — Alpha weaves into research program
    # ==================================================================
    logger.info("didal[%s] Phase 3: Alpha weaving research program", trace_id)
    research_program = weave_research_program(question, theses, dialectical_log)

    # Order by dialectical strength for the output
    ordered = sorted(theses, key=lambda h: h.dialectical_strength, reverse=True)

    duration_s = round(time.time() - t0, 1)
    result = {
        "success": True,
        "trace_id": trace_id,
        "research_question": question,
        "domain": domain,
        "dialectical_rounds": n_rounds,
        "protocol": "DiDAL Hypothesis Protocol — THESIS → ANTITHESIS → SYNTHESIS",
        "dialectical_log": dialectical_log,
        "refined_theses": [t.to_dict() for t in ordered],
        "research_program": research_program,
        "duration_s": duration_s,
    }

    logger.info(
        "didal[%s] Protocol complete: %d theses refined in %.1fs",
        trace_id, len(ordered), duration_s,
    )

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool Schema for Hermes registration
# ---------------------------------------------------------------------------

DIALECTICAL_HYPOTHESIS_SCHEMA = {
    "name": "dialectical_hypothesis",
    "description": (
        "Run the DiDAL Hypothesis Protocol: THESIS → ANTITHESIS → SYNTHESIS. "
        "Alpha (the ecologist) generates ecological hypotheses. Beta (the "
        "dialectical counterpart) challenges them through structured critique. "
        "Alpha synthesizes the refined theses into a coherent research program. "
        "This is dialectical science — hypotheses are strengthened through "
        "critique, not eliminated by ranking. "
        "Part of the EcoSeek DiDAL (Dialectical Dual-Agent Loop) protocol."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The ecological research question to investigate dialectically.",
            },
            "n_theses": {
                "type": "integer",
                "description": "Number of initial theses Alpha should generate (3-10, default 5).",
            },
            "n_rounds": {
                "type": "integer",
                "description": "Rounds of dialectical testing (1-5, default 3).",
            },
            "domain": {
                "type": "string",
                "description": "Scientific domain (default: ecology).",
            },
            "context": {
                "type": "string",
                "description": "Additional context, data sources, or constraints for Alpha.",
            },
        },
        "required": ["question"],
    },
}
