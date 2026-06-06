"""Tests for EcoSeek DiDAL Hypothesis Protocol — Phase 4.

Covers:
  - Dialectical strength math (expected_outcome, update_dialectical_strength)
  - Hypothesis dataclass (to_dict, from_dict, validation)
  - JSON parsing (_parse_theses_json)
  - Thesis validation (_validate_thesis)
  - Thesis generation (mocked Alpha/DeepSeek)
  - Dialectical clash (mocked Beta/Ollama)
  - Protocol orchestrator (mock integration)
"""

import json
import sys
import pytest

sys.path.insert(0, "plugins/ecoseek")

from hypothesis_engine import (
    expected_outcome,
    update_dialectical_strength,
    DEFAULT_STRENGTH,
    K_FACTOR,
    Hypothesis,
    _parse_theses_json,
    _validate_thesis,
)


# ===========================================================================
# Dialectical Strength Math
# ===========================================================================


class TestDialecticalStrength:
    """The math behind dialectical strength (internal metric, not ranking)."""

    def test_equal_strength(self):
        """Equal strength → 0.5 probability of prevailing."""
        assert expected_outcome(1200, 1200) == pytest.approx(0.5)

    def test_stronger_prevails_more_often(self):
        """Higher dialectical strength → higher probability of withstanding critique."""
        assert expected_outcome(1400, 1200) > 0.5

    def test_weaker_prevails_less_often(self):
        assert expected_outcome(1000, 1200) < 0.5

    def test_400_point_gap(self):
        assert expected_outcome(1600, 1200) == pytest.approx(0.909, abs=0.01)

    def test_strength_bounds(self):
        prob = expected_outcome(3000, 0)
        assert 0.0 < prob < 1.0

    def test_update_prevailing_gains(self):
        """Thesis that withstands critique gains dialectical strength."""
        p, c = update_dialectical_strength(1200, 1200)
        assert p > 1200
        assert c < 1200
        assert p + c == pytest.approx(2400)

    def test_update_upset(self):
        """Weaker thesis prevailing gains more strength (learned more)."""
        p, c = update_dialectical_strength(1000, 1400)
        delta = p - 1000
        assert delta > K_FACTOR / 2

    def test_update_favorite(self):
        """Already-strong thesis gains less from prevailing."""
        p, c = update_dialectical_strength(1400, 1000)
        delta = p - 1400
        assert delta < K_FACTOR / 2


# ===========================================================================
# Hypothesis Dataclass
# ===========================================================================


class TestHypothesis:
    def test_roundtrip(self):
        """to_dict → from_dict round-trips correctly."""
        h = Hypothesis(
            id="th_abc123",
            statement="Climate change reduces Andean hummingbird range by 30%",
            rationale="Warming temperatures shift cloud forest upward",
            predictions=["Range contraction >30% by 2050", "Altitude shift >500m"],
            testability=0.8,
            novelty=0.6,
            confidence=0.5,
            dialectical_strength=1250,
            refinement_round=2,
            parent_id="th_parent",
            evidence_sources=["GBIF:12345", "WorldClim v2"],
        )
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        assert h2.id == h.id
        assert h2.statement == h.statement
        assert h2.rationale == h.rationale
        assert h2.predictions == h.predictions
        assert h2.testability == h.testability
        assert h2.novelty == h.novelty
        assert h2.dialectical_strength == h.dialectical_strength
        assert h2.refinement_round == h.refinement_round
        assert h2.parent_id == h.parent_id

    def test_defaults(self):
        """Default values reflect an untested thesis."""
        h = Hypothesis(id="th1", statement="Test", rationale="Test rationale")
        assert h.testability == 0.5
        assert h.novelty == 0.5
        assert h.dialectical_strength == DEFAULT_STRENGTH
        assert h.refinement_round == 1
        assert h.parent_id is None
        assert h.predictions == []

    def test_from_dict_missing_fields(self):
        d = {"id": "th1", "statement": "Test"}
        h = Hypothesis.from_dict(d)
        assert h.rationale == ""
        assert h.predictions == []
        assert h.testability == 0.5


# ===========================================================================
# JSON Parsing
# ===========================================================================


class TestParseTheses:
    def test_direct_array(self):
        raw = json.dumps([
            {"statement": "T1", "rationale": "R1", "predictions": ["P1"]},
            {"statement": "T2", "rationale": "R2", "predictions": ["P2"]},
        ])
        result = _parse_theses_json(raw, n_expected=5)
        assert len(result) == 2
        assert result[0]["statement"] == "T1"

    def test_markdown_code_block(self):
        raw = '```json\n[{"statement": "T1", "rationale": "R", "predictions": ["P"]}]\n```'
        result = _parse_theses_json(raw)
        assert len(result) == 1
        assert result[0]["statement"] == "T1"

    def test_dict_wrapper(self):
        raw = json.dumps({"hypotheses": [
            {"statement": "T1", "rationale": "R", "predictions": ["P"]},
        ]})
        result = _parse_theses_json(raw)
        assert len(result) == 1

    def test_truncates_to_n(self):
        raw = json.dumps([
            {"statement": f"T{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
            for i in range(10)
        ])
        result = _parse_theses_json(raw, n_expected=3)
        assert len(result) == 3

    def test_invalid_json(self):
        result = _parse_theses_json("not json at all")
        assert result == []

    def test_bare_array_in_text(self):
        raw = 'Some text\n[{"statement": "T1", "rationale": "R", "predictions": ["P"]}]\nMore'
        result = _parse_theses_json(raw)
        assert len(result) == 1

    def test_no_code_block_marker(self):
        raw = '[{"statement": "T1", "rationale": "R", "predictions": ["P"]}]'
        result = _parse_theses_json(raw)
        assert len(result) == 1


# ===========================================================================
# Thesis Validation
# ===========================================================================


class TestValidateThesis:
    def test_valid(self):
        t = {"statement": "S", "rationale": "R", "predictions": ["P1", "P2"]}
        assert _validate_thesis(t) is True

    def test_missing_statement(self):
        t = {"rationale": "R", "predictions": ["P1"]}
        assert _validate_thesis(t) is False

    def test_empty_statement(self):
        t = {"statement": "", "rationale": "R", "predictions": ["P1"]}
        assert _validate_thesis(t) is False

    def test_missing_predictions(self):
        t = {"statement": "S", "rationale": "R"}
        assert _validate_thesis(t) is False

    def test_empty_predictions(self):
        t = {"statement": "S", "rationale": "R", "predictions": []}
        assert _validate_thesis(t) is False

    def test_predictions_not_list(self):
        t = {"statement": "S", "rationale": "R", "predictions": "not a list"}
        assert _validate_thesis(t) is False

    def test_clamps_scores(self):
        t = {
            "statement": "S", "rationale": "R", "predictions": ["P"],
            "testability": 2.5, "novelty": -0.5, "confidence": 1.5,
        }
        assert _validate_thesis(t) is True
        assert t["testability"] == 1.0
        assert t["novelty"] == 0.0
        assert t["confidence"] == 1.0


# ===========================================================================
# Integration: DiDAL Hypothesis Protocol (mocked)
# ===========================================================================


class TestDiDALHypothesisProtocol:
    """Integration tests for dialectical_hypothesis_refinement with mocked LLMs."""

    def test_too_few_theses(self, monkeypatch):
        """Returns error if Alpha generates < 3 valid theses."""
        monkeypatch.setattr(
            "hypothesis_engine._alpha_call",
            lambda *a, **kw: json.dumps([
                {"statement": "T1", "rationale": "R", "predictions": ["P"]},
            ]),
        )
        result = json.loads(
            __import__("hypothesis_engine").dialectical_hypothesis_refinement(
                "Test question?", n_theses=5, n_rounds=1
            )
        )
        assert result["success"] is False
        assert result["error"] == "insufficient_theses"

    def test_full_protocol(self, monkeypatch):
        """Full DiDAL protocol: THESIS → DIALECTIC → SYNTHESIS."""
        call_count = {"alpha": 0, "beta": 0}

        def mock_alpha(system_prompt, user_message, max_tokens=2000):
            call_count["alpha"] += 1
            if "propose testable ecological theses" in system_prompt.lower():
                # Phase 1: Alpha generates theses
                return json.dumps([
                    {
                        "statement": f"Thesis {i} about ecological dynamics",
                        "rationale": f"Rationale {i} grounded in theory",
                        "predictions": [f"Prediction {i}a", f"Prediction {i}b"],
                        "testability": 0.7 + i * 0.05,
                        "novelty": 0.5 + i * 0.1,
                        "confidence": 0.5,
                    }
                    for i in range(5)
                ])
            elif "evaluating the outcome" in system_prompt.lower():
                # Phase 2b: Alpha judges dialectical clash
                return json.dumps({
                    "prevailing": "A",
                    "assessment": {
                        "A": {"resilience": 8, "coherence": 7, "novelty": 6, "testability": 9, "grounding": 8},
                        "B": {"resilience": 6, "coherence": 5, "novelty": 7, "testability": 4, "grounding": 6},
                    },
                    "rationale": "Thesis A showed greater dialectical resilience.",
                })
            elif "synthesizing the outcome" in system_prompt.lower():
                # Phase 2c: Alpha synthesizes refined thesis
                return json.dumps({
                    "statement": "Synthesized thesis with deeper specificity",
                    "rationale": "Deepened rationale from dialectical insights",
                    "predictions": ["Refined prediction 1", "Refined prediction 2"],
                    "testability": 0.85,
                    "novelty": 0.7,
                    "confidence": 0.6,
                    "dialectical_insight": "Critique exposed assumption X, now addressed",
                })
            else:
                # Phase 3: Alpha weaves research program
                return "# Ecological Research Program\n\n## Abstract\nDialectically refined."

        def mock_beta(system_prompt, user_message, model="ecocoder:7b", max_tokens=500):
            call_count["beta"] += 1
            return (
                "CHALLENGE: The thesis assumes uniform response across taxa.\n"
                "DEMAND: Need species-level data from GBIF to verify.\n"
                "COUNTER: In tropical systems, this pattern often reverses.\n"
                "REFINE: Specify taxonomic scope and test with occurrence data."
            )

        monkeypatch.setattr("hypothesis_engine._alpha_call", mock_alpha)
        monkeypatch.setattr("hypothesis_engine._beta_call", mock_beta)

        result = json.loads(
            __import__("hypothesis_engine").dialectical_hypothesis_refinement(
                "How does climate change affect biodiversity?",
                n_theses=5,
                n_rounds=2,
            )
        )

        assert result["success"] is True
        assert len(result["refined_theses"]) == 5
        assert "research_program" in result
        assert "dialectical_log" in result
        assert result["protocol"] == "DiDAL Hypothesis Protocol — THESIS → ANTITHESIS → SYNTHESIS"

        # 1 thesis gen + (n_rounds * floor(n/2)) judgments + (n_rounds * floor(n/2)) syntheses + 1 program
        expected_alpha = 1 + (2 * 2) + (2 * 2) + 1
        assert call_count["alpha"] == expected_alpha, \
            f"Expected {expected_alpha} Alpha calls, got {call_count['alpha']}"

        # Each round: 2 matches × max_exchanges × 2 (A and B each round)
        expected_beta = 2 * 2 * 4 * 2  # 2 rounds, 2 pairs, 4 exchanges, A+B each
        assert call_count["beta"] == expected_beta, \
            f"Expected {expected_beta} Beta calls, got {call_count['beta']}"

    def test_theses_ordered_by_dialectical_strength(self, monkeypatch):
        """Refined theses are ordered by dialectical strength, not a 'ranking'."""
        monkeypatch.setattr(
            "hypothesis_engine._alpha_call",
            lambda *a, **kw: json.dumps([
                {
                    "statement": f"T{i}",
                    "rationale": f"R{i}",
                    "predictions": [f"P{i}a", f"P{i}b"],
                    "testability": 0.8,
                    "novelty": 0.5 + i * 0.1,
                    "confidence": 0.5,
                }
                for i in range(4)
            ]),
        )
        monkeypatch.setattr(
            "hypothesis_engine._beta_call",
            lambda *a, **kw: "CHALLENGE: Valid.\nDEMAND: Data.\nCOUNTER: Possible.\nREFINE: Specify.",
        )

        result = json.loads(
            __import__("hypothesis_engine").dialectical_hypothesis_refinement(
                "Test?", n_theses=4, n_rounds=1
            )
        )

        assert result["success"] is True
        strengths = [t["dialectical_strength"] for t in result["refined_theses"]]
        assert strengths == sorted(strengths, reverse=True), \
            f"Theses not ordered by dialectical strength: {strengths}"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_odd_number_of_theses(self, monkeypatch):
        """5 theses → 2 dialectical pairs, 1 thesis rests this round."""
        monkeypatch.setattr(
            "hypothesis_engine._alpha_call",
            lambda *a, **kw: json.dumps([
                {"statement": f"T{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
                for i in range(5)
            ]),
        )
        monkeypatch.setattr(
            "hypothesis_engine._beta_call",
            lambda *a, **kw: "CHALLENGE: Point.\nDEMAND: Evidence.\nCOUNTER: Alt.\nREFINE: Fix.",
        )

        result = json.loads(
            __import__("hypothesis_engine").dialectical_hypothesis_refinement(
                "Test?", n_theses=5, n_rounds=1
            )
        )
        assert result["success"] is True
        assert len(result["refined_theses"]) == 5

    def test_beta_unavailable(self, monkeypatch):
        """Protocol continues gracefully when Beta (Ollama) is down."""
        monkeypatch.setattr(
            "hypothesis_engine._alpha_call",
            lambda *a, **kw: json.dumps([
                {"statement": f"T{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
                for i in range(4)
            ]),
        )

        def beta_down(*a, **kw):
            raise ConnectionError("Beta (Ollama) not available")

        monkeypatch.setattr("hypothesis_engine._beta_call", beta_down)

        result = json.loads(
            __import__("hypothesis_engine").dialectical_hypothesis_refinement(
                "Test?", n_theses=4, n_rounds=1
            )
        )
        # Should complete — Beta errors are caught and logged
        assert result["success"] is True
