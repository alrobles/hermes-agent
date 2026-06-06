"""Hypothesis Engine — Phase 4 of EcoSeek DiDAL.

Extends the dual-agent DiDAL protocol into a multi-agent hypothesis tournament
inspired by Google DeepMind's Co-Scientist (Nature 2026).

Architecture:
  Phase 1: GENERATE — GenerationAgent proposes N hypotheses from literature + data
  Phase 2: TOURNAMENT — Elo-ranked debate matches between hypotheses
  Phase 3: META-REVIEW — MetaReviewAgent synthesizes ranked proposal

Model tiering:
  - Generation + MetaReview: deepseek (high quality, paid)
  - Debate matches: EcoCoder local via ollama (GPU, free)
  - Ranking: lightweight LLM call or heuristic
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
# Elo Rating System
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1200
ELO_K_FACTOR = 32


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given their Elo ratings."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_elo(
    winner_rating: float,
    loser_rating: float,
    k: int = ELO_K_FACTOR,
) -> tuple[float, float]:
    """Return (new_winner_rating, new_loser_rating) after a match."""
    expected_win = expected_score(winner_rating, loser_rating)
    delta = k * (1.0 - expected_win)
    return (
        round(winner_rating + delta, 1),
        round(loser_rating - delta, 1),
    )


# ---------------------------------------------------------------------------
# Hypothesis Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A testable ecological hypothesis with tournament state."""

    id: str
    statement: str
    rationale: str
    predictions: list[str] = field(default_factory=list)
    testability: float = 0.5  # 0..1 how easily testable
    novelty: float = 0.5       # 0..1 how novel
    confidence: float = 0.5     # 0..1 initial confidence
    elo: float = DEFAULT_ELO
    generation: int = 1          # generation number (1 = original, 2+ = evolved)
    parent_id: Optional[str] = None  # id of parent hypothesis (if evolved)
    round_history: list[dict] = field(default_factory=list)
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
            "elo": self.elo,
            "generation": self.generation,
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
            elo=d.get("elo", DEFAULT_ELO),
            generation=d.get("generation", 1),
            parent_id=d.get("parent_id"),
            evidence_sources=d.get("evidence_sources", []),
        )


# ---------------------------------------------------------------------------
# Generation Agent
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are a senior ecologist specialized in hypothesis generation.
Your task is to generate diverse, testable, and novel ecological hypotheses
for a given research question.

For each hypothesis, provide:
1. A clear, falsifiable statement
2. Ecological rationale (why this might be true)
3. Specific, measurable predictions
4. Testability score (0-1)
5. Novelty score (0-1)
6. Initial confidence score (0-1)

Rules:
- Generate HYPOTHESES that are DIVERSE — cover different mechanisms, scales, taxa
- Each hypothesis MUST be falsifiable with available data (GBIF, SDM, climate)
- Prefer specific over vague (e.g., "Andean hummingbirds above 3000m" not "high-altitude birds")
- Novelty > 0.7 means the hypothesis challenges existing paradigms

Output format: JSON array of hypothesis objects.
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


def _parse_hypotheses_json(raw: str, n_expected: int = 5) -> list[dict]:
    """Extract hypothesis array from LLM response (may have markdown wrapping)."""
    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed[:n_expected]
        if isinstance(parsed, dict) and "hypotheses" in parsed:
            return parsed["hypotheses"][:n_expected]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
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

    # Try finding array brackets
    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed[:n_expected]
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse hypotheses from LLM response")
    return []


def _validate_hypothesis(h: dict) -> bool:
    """Check required fields exist and are sensible."""
    required = ["statement", "rationale", "predictions"]
    for field in required:
        if field not in h or not h[field]:
            return False
    if not isinstance(h.get("predictions"), list):
        return False
    if len(h["predictions"]) == 0:
        return False
    # Clamp scores to [0, 1]
    for key in ("testability", "novelty", "confidence"):
        if key in h:
            h[key] = max(0.0, min(1.0, float(h[key])))
    return True


# ---------------------------------------------------------------------------
# Ranking Agent
# ---------------------------------------------------------------------------

RANKING_SYSTEM_PROMPT = """\
You are a scientific peer reviewer evaluating a debate between two ecological
hypotheses. After reading the debate transcript, determine which hypothesis
is stronger based on:

1. Evidence quality (0-10): How well-supported are the claims?
2. Logical coherence (0-10): Is the reasoning sound?
3. Novelty (0-10): How original is the hypothesis?
4. Testability (0-10): Can it be empirically tested with available data?
5. Ecological validity (0-10): Is it grounded in ecological theory?

Output JSON:
```json
{
  "winner": "A" or "B",
  "scores": {
    "A": {"evidence": 8, "coherence": 7, "novelty": 6, "testability": 9, "validity": 8},
    "B": {"evidence": 6, "coherence": 8, "novelty": 9, "testability": 5, "validity": 7}
  },
  "rationale": "Brief explanation of the decision"
}
```
"""


# ---------------------------------------------------------------------------
# Evolution Agent
# ---------------------------------------------------------------------------

EVOLUTION_SYSTEM_PROMPT = """\
You are an ecological theorist specializing in hypothesis refinement.
Given a winning hypothesis and critiques from the debate, evolve it:

1. Incorporate valid critiques — fix weaknesses identified by the opponent
2. Strengthen predictions — make them more specific and measurable
3. Expand rationale — add ecological mechanisms or references
4. Update confidence — based on how well it survived the debate

Output JSON:
```json
{
  "statement": "refined statement",
  "rationale": "expanded rationale incorporating debate insights",
  "predictions": ["more specific prediction 1", "more specific prediction 2"],
  "testability": 0.9,
  "novelty": 0.7,
  "confidence": 0.6,
  "changes_summary": "What was changed and why"
}
```
"""


# ---------------------------------------------------------------------------
# Meta-Review Agent
# ---------------------------------------------------------------------------

META_REVIEW_SYSTEM_PROMPT = """\
You are a senior research program officer synthesizing a tournament of ecological
hypotheses into a research proposal. Given the ranked hypotheses and debate
outcomes, produce a structured proposal:

Sections:
1. Title: Concise, informative
2. Abstract: 150-250 words summarizing the research program
3. Introduction: Context and significance of the research question
4. Hypotheses (ranked): Each with Elo score, rationale, and how it emerged from debate
5. Proposed Methods: How to test the top hypotheses (GBIF, SDM, field, experimental)
6. Expected Outcomes: What we will learn
7. References: Key papers and data sources cited

Output as clean markdown suitable for a grant proposal.
"""


# ---------------------------------------------------------------------------
# LLM Call Helpers (model-tiered)
# ---------------------------------------------------------------------------

def _call_deepseek(system_prompt: str, user_message: str,
                   max_tokens: int = 2000) -> str:
    """Call DeepSeek API for high-quality generation/meta-review."""
    import urllib.error
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

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
        raise RuntimeError("DeepSeek returned no choices")
    return choices[0]["message"]["content"]


def _call_ollama(system_prompt: str, user_message: str,
                 model: str = "ecocoder:7b",
                 max_tokens: int = 500) -> str:
    """Call local Ollama (EcoCoder GPU) for debate matches."""
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
        raise RuntimeError("Ollama returned no choices")
    return choices[0]["message"]["content"]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def generate_hypotheses(
    question: str,
    n: int = 5,
    domain: str = "ecology",
    context: str = "",
) -> list[Hypothesis]:
    """Generate N diverse, testable ecological hypotheses for a question.

    Uses DeepSeek for high-quality generation.
    """
    user_msg = f"Research question: {question}\nDomain: {domain}\nGenerate {n} hypotheses."
    if context:
        user_msg += f"\n\nAdditional context:\n{context}"

    raw = _call_deepseek(GENERATION_SYSTEM_PROMPT, user_msg, max_tokens=2000)
    parsed = _parse_hypotheses_json(raw, n_expected=n)

    hypotheses = []
    for i, h in enumerate(parsed):
        if not _validate_hypothesis(h):
            logger.warning("Hypothesis %d failed validation, skipping", i)
            continue
        hypotheses.append(Hypothesis(
            id=f"h_{uuid.uuid4().hex[:8]}",
            statement=h["statement"],
            rationale=h["rationale"],
            predictions=h["predictions"],
            testability=h.get("testability", 0.5),
            novelty=h.get("novelty", 0.5),
            confidence=h.get("confidence", 0.5),
            generation=1,
        ))

    logger.info("Generated %d/%d valid hypotheses", len(hypotheses), n)
    return hypotheses


def rank_debate(transcript: str, hypothesis_a: Hypothesis,
                hypothesis_b: Hypothesis) -> dict:
    """Judge a debate match between two hypotheses.

    Uses DeepSeek for quality evaluation.
    """
    user_msg = (
        f"Debate transcript:\n{transcript}\n\n"
        f"Hypothesis A: {hypothesis_a.statement}\n"
        f"Hypothesis B: {hypothesis_b.statement}\n\n"
        f"Evaluate and declare the winner."
    )

    raw = _call_deepseek(RANKING_SYSTEM_PROMPT, user_msg, max_tokens=500)
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            # Got a list instead of dict — default to A
            result = {"winner": "A", "rationale": "Could not parse ranking (got list)"}
    except json.JSONDecodeError:
        # Fallback: parse from text
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
            except json.JSONDecodeError:
                result = {"winner": "A", "rationale": "Could not parse ranking"}
        else:
            logger.warning("Could not parse ranking result, defaulting to A")
            result = {"winner": "A", "rationale": "Could not parse ranking"}

    return result


def evolve_hypothesis(winner: Hypothesis, loser_critiques: str) -> Hypothesis:
    """Refine a winning hypothesis using critiques from the debate.

    Uses DeepSeek for quality refinement.
    """
    user_msg = (
        f"Original hypothesis: {winner.statement}\n"
        f"Rationale: {winner.rationale}\n"
        f"Predictions: {json.dumps(winner.predictions)}\n\n"
        f"Critiques from debate:\n{loser_critiques}\n\n"
        f"Evolve this hypothesis. Return JSON."
    )

    raw = _call_deepseek(EVOLUTION_SYSTEM_PROMPT, user_msg, max_tokens=800)
    try:
        evolved = json.loads(raw)
        if isinstance(evolved, list):
            logger.warning("Evolution returned list instead of dict, using original")
            return winner
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                evolved = json.loads(match.group(1))
                if isinstance(evolved, list):
                    return winner
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Could not parse evolution, returning original")
                return winner
        else:
            logger.warning("Could not parse evolution, returning original")
            return winner

    return Hypothesis(
        id=f"h_{uuid.uuid4().hex[:8]}",
        statement=evolved.get("statement", winner.statement),
        rationale=evolved.get("rationale", winner.rationale),
        predictions=evolved.get("predictions", winner.predictions),
        testability=float(evolved.get("testability", winner.testability)),
        novelty=float(evolved.get("novelty", winner.novelty)),
        confidence=float(evolved.get("confidence", winner.confidence)),
        elo=winner.elo,
        generation=winner.generation + 1,
        parent_id=winner.id,
    )


def debate_match(
    hypothesis_a: Hypothesis,
    hypothesis_b: Hypothesis,
    question: str,
    max_turns: int = 4,
    model: str = "ecocoder:7b",
) -> dict:
    """Run a 1v1 debate between two hypotheses using local EcoCoder.

    Returns dict with transcript, turn_count, tokens_used.
    """
    debate_prompt = f"""\
You are defending the following ecological hypothesis in a scientific debate.

RESEARCH QUESTION: {question}

YOUR HYPOTHESIS: {hypothesis_a.statement}
RATIONALE: {hypothesis_a.rationale}
PREDICTIONS: {json.dumps(hypothesis_a.predictions)}

YOUR OPPONENT'S HYPOTHESIS: {hypothesis_b.statement}

Debate format — respond in exactly this structure each turn:
1. CLAIM: State your argument concisely
2. EVIDENCE: Cite ecological mechanisms, data, or literature
3. REBUTTAL: Address your opponent's last point (if any)

Keep responses under 200 words. Be specific and evidence-based.
Do NOT concede unless the evidence clearly favors your opponent.
"""

    transcript_lines = []
    total_tokens = 0
    turn = 0

    # Round 1: A opens
    a_msg = f"Opening statement for hypothesis A. Defend: {hypothesis_a.statement}"
    try:
        a_resp = _call_ollama(debate_prompt, a_msg, model=model, max_tokens=300)
    except Exception as e:
        logger.warning("Debate A opening failed: %s", e)
        a_resp = f"[Error: {e}]"
    transcript_lines.append(f"[Turn {turn}] A (opening): {a_resp}")
    turn += 1

    # Round 1: B responds
    b_prompt = debate_prompt.replace(
        hypothesis_a.statement, hypothesis_b.statement
    ).replace(
        hypothesis_a.rationale, hypothesis_b.rationale
    ).replace(
        json.dumps(hypothesis_a.predictions), json.dumps(hypothesis_b.predictions)
    ).replace(
        hypothesis_b.statement, hypothesis_a.statement
    )
    b_msg = f"Respond to opponent's opening: {a_resp[:300]}"
    try:
        b_resp = _call_ollama(b_prompt, b_msg, model=model, max_tokens=300)
    except Exception as e:
        logger.warning("Debate B response failed: %s", e)
        b_resp = f"[Error: {e}]"
    transcript_lines.append(f"[Turn {turn}] B (response): {b_resp}")
    turn += 1

    # Rounds 2..max_turns: alternating
    for t in range(turn, max_turns * 2):
        if t % 2 == 0:
            # A's turn
            msg = f"Turn {t//2 + 1}. Respond to opponent: {b_resp[:300]}"
            try:
                a_resp = _call_ollama(debate_prompt, msg, model=model, max_tokens=300)
            except Exception as e:
                a_resp = f"[Error: {e}]"
            transcript_lines.append(f"[Turn {t}] A: {a_resp}")
        else:
            # B's turn
            msg = f"Turn {t//2 + 1}. Respond to opponent: {a_resp[:300]}"
            try:
                b_resp = _call_ollama(b_prompt, msg, model=model, max_tokens=300)
            except Exception as e:
                b_resp = f"[Error: {e}]"
            transcript_lines.append(f"[Turn {t}] B: {b_resp}")

    transcript = "\n\n".join(transcript_lines)
    return {
        "transcript": transcript,
        "turns": max_turns * 2,
        "total_tokens": total_tokens,
    }


def synthesize_proposal(
    question: str,
    hypotheses: list[Hypothesis],
    tournament_log: list[dict],
) -> str:
    """Generate a research proposal from ranked hypotheses.

    Uses DeepSeek for high-quality synthesis.
    """
    ranked = sorted(hypotheses, key=lambda h: h.elo, reverse=True)
    hypotheses_text = "\n\n".join(
        f"### {i+1}. {h.statement} (Elo: {h.elo}, Generation: {h.generation})\n"
        f"Rationale: {h.rationale}\n"
        f"Predictions: {', '.join(h.predictions)}\n"
        f"Testability: {h.testability}, Novelty: {h.novelty}"
        for i, h in enumerate(ranked)
    )

    summary_text = "\n".join(
        f"Round {log.get('round', '?')}: {log.get('winner_id', '?')} "
        f"defeated {log.get('loser_id', '?')}"
        for log in tournament_log[-10:]  # Last 10 matches
    )

    user_msg = (
        f"Research question: {question}\n\n"
        f"Ranked hypotheses:\n{hypotheses_text}\n\n"
        f"Tournament summary:\n{summary_text}\n\n"
        f"Synthesize into a research proposal."
    )

    return _call_deepseek(META_REVIEW_SYSTEM_PROMPT, user_msg, max_tokens=3000)


def hypothesis_tournament(
    question: str,
    n_hypotheses: int = 5,
    n_rounds: int = 3,
    domain: str = "ecology",
    context: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Orchestrate a full hypothesis tournament.

    Phase 1: Generate N hypotheses
    Phase 2: Elo tournament (Swiss-system pairing, n_rounds)
    Phase 3: Meta-review synthesis

    Returns JSON with ranked hypotheses, tournament log, and proposal.
    """
    t0 = time.time()
    trace_id = task_id or str(uuid.uuid4())[:8]
    tournament_log: list[dict] = []

    # ------------------------------------------------------------------
    # Phase 1: GENERATE
    # ------------------------------------------------------------------
    logger.info("tournament[%s] Phase 1: generating %d hypotheses", trace_id, n_hypotheses)
    hypotheses = generate_hypotheses(
        question, n=n_hypotheses, domain=domain, context=context
    )

    if len(hypotheses) < 3:
        return json.dumps({
            "success": False,
            "error": "not_enough_hypotheses",
            "message": f"Only {len(hypotheses)} valid hypotheses generated (need ≥3).",
            "trace_id": trace_id,
        })

    tournament_log.append({
        "phase": "generate",
        "n_requested": n_hypotheses,
        "n_generated": len(hypotheses),
        "hypotheses": [h.to_dict() for h in hypotheses],
    })

    # ------------------------------------------------------------------
    # Phase 2: TOURNAMENT
    # ------------------------------------------------------------------
    logger.info("tournament[%s] Phase 2: %d rounds of debate", trace_id, n_rounds)

    for round_num in range(1, n_rounds + 1):
        # Sort by Elo for Swiss-style pairing (adjacent pairs)
        sorted_h = sorted(hypotheses, key=lambda h: h.elo, reverse=True)
        round_matches = []

        for i in range(0, len(sorted_h) - 1, 2):
            h_a = sorted_h[i]
            h_b = sorted_h[i + 1]

            logger.info(
                "tournament[%s] Round %d: %s (Elo %.0f) vs %s (Elo %.0f)",
                trace_id, round_num, h_a.id, h_a.elo, h_b.id, h_b.elo,
            )

            # Run debate
            match_result = debate_match(h_a, h_b, question)
            transcript = match_result["transcript"]

            # Rank the debate
            ranking = rank_debate(transcript, h_a, h_b)
            winner_id = h_a.id if ranking.get("winner") == "A" else h_b.id
            loser_id = h_b.id if ranking.get("winner") == "A" else h_a.id

            # Update Elo
            winner = h_a if winner_id == h_a.id else h_b
            loser = h_b if winner_id == h_a.id else h_b
            new_w, new_l = update_elo(winner.elo, loser.elo)
            winner.elo = new_w
            loser.elo = new_l

            # Evolve the winner
            evolved = evolve_hypothesis(winner, transcript)
            evolved.elo = new_w  # Carry forward the new Elo
            # Replace the winner with evolved version
            for j, h in enumerate(hypotheses):
                if h.id == winner.id:
                    hypotheses[j] = evolved
                    break

            match_log = {
                "phase": "tournament",
                "round": round_num,
                "hypothesis_a": h_a.to_dict(),
                "hypothesis_b": h_b.to_dict(),
                "winner_id": winner_id,
                "loser_id": loser_id,
                "elo_after": {"winner": new_w, "loser": new_l},
                "ranking": ranking,
                "evolved": evolved.to_dict() if evolved.id != winner.id else None,
            }
            round_matches.append(match_log)
            tournament_log.append(match_log)

            logger.info(
                "tournament[%s] Round %d match: %s beats %s (Elo: %.0f > %.0f)",
                trace_id, round_num, winner_id, loser_id, new_w, new_l,
            )

    # ------------------------------------------------------------------
    # Phase 3: META-REVIEW
    # ------------------------------------------------------------------
    logger.info("tournament[%s] Phase 3: synthesizing proposal", trace_id)
    proposal = synthesize_proposal(question, hypotheses, tournament_log)

    # Final rankings
    ranked = sorted(hypotheses, key=lambda h: h.elo, reverse=True)

    duration_s = round(time.time() - t0, 1)
    result = {
        "success": True,
        "trace_id": trace_id,
        "research_question": question,
        "domain": domain,
        "n_rounds": n_rounds,
        "tournament_log": tournament_log,
        "ranked_hypotheses": [h.to_dict() for h in ranked],
        "proposal": proposal,
        "duration_s": duration_s,
    }

    logger.info(
        "tournament[%s] complete: %d hypotheses ranked in %.1fs",
        trace_id, len(ranked), duration_s,
    )

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool Schema for Hermes registration
# ---------------------------------------------------------------------------

HYPOTHESIS_TOURNAMENT_SCHEMA = {
    "name": "hypothesis_tournament",
    "description": (
        "Run a multi-agent hypothesis tournament for ecological research. "
        "Generates N hypotheses, debates them in Elo-ranked matches, "
        "evolves winners, and synthesizes a research proposal. "
        "Inspired by Google DeepMind's Co-Scientist (Nature 2026)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The ecological research question to investigate.",
            },
            "n_hypotheses": {
                "type": "integer",
                "description": "Number of hypotheses to generate (3-10, default 5).",
            },
            "n_rounds": {
                "type": "integer",
                "description": "Number of tournament rounds (1-5, default 3).",
            },
            "domain": {
                "type": "string",
                "description": "Scientific domain (default: ecology).",
            },
            "context": {
                "type": "string",
                "description": "Additional context, data sources, or constraints.",
            },
        },
        "required": ["question"],
    },
}
