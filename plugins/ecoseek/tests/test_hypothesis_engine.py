"""Tests for EcoSeek Hypothesis Engine — Phase 4.

Covers:
  - Elo math (expected_score, update_elo)
  - Hypothesis dataclass (to_dict, from_dict, validation)
  - JSON parsing (_parse_hypotheses_json)
  - Hypothesis validation (_validate_hypothesis)
  - Generation (mocked DeepSeek)
  - Debate match (mocked Ollama)
  - Tournament orchestrator (mock integration)
"""

import json
import sys
import pytest

# Ensure the plugin package is importable
sys.path.insert(0, "plugins/ecoseek")

from hypothesis_engine import (
    expected_score,
    update_elo,
    DEFAULT_ELO,
    ELO_K_FACTOR,
    Hypothesis,
    _parse_hypotheses_json,
    _validate_hypothesis,
)


# ===========================================================================
# Elo Math
# ===========================================================================


class TestEloMath:
    def test_equal_ratings(self):
        """Equal ratings → 0.5 probability each."""
        assert expected_score(1200, 1200) == pytest.approx(0.5)

    def test_favorite_wins(self):
        """Higher rating → higher win probability."""
        assert expected_score(1400, 1200) > 0.5

    def test_underdog_loses(self):
        """Lower rating → lower win probability."""
        assert expected_score(1000, 1200) < 0.5

    def test_400_point_gap(self):
        """400 point gap → ~0.91 probability for favorite."""
        assert expected_score(1600, 1200) == pytest.approx(0.909, abs=0.01)

    def test_elo_bounds(self):
        """Elo stays within reasonable range."""
        prob = expected_score(3000, 0)
        assert 0.0 < prob < 1.0

    def test_update_elo_winner_gains(self):
        """Winner gains rating, loser loses rating."""
        w, l = update_elo(1200, 1200)
        assert w > 1200
        assert l < 1200
        assert w + l == pytest.approx(2400)

    def test_update_elo_upset(self):
        """Upset: winner had lower rating, gains more."""
        w, l = update_elo(1000, 1400)
        delta = w - 1000
        assert delta > ELO_K_FACTOR / 2  # Bigger gain for upset

    def test_update_elo_favorite(self):
        """Favorite wins, gains less."""
        w, l = update_elo(1400, 1000)
        delta = w - 1400
        assert delta < ELO_K_FACTOR / 2  # Smaller gain


# ===========================================================================
# Hypothesis Dataclass
# ===========================================================================


class TestHypothesis:
    def test_roundtrip(self):
        """to_dict → from_dict round-trips correctly."""
        h = Hypothesis(
            id="h_abc123",
            statement="Climate change reduces Andean hummingbird range by 30%",
            rationale="Warming temperatures shift cloud forest upward",
            predictions=["Range contraction >30% by 2050", "Altitude shift >500m"],
            testability=0.8,
            novelty=0.6,
            confidence=0.5,
            elo=1250,
            generation=2,
            parent_id="h_parent",
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
        assert h2.elo == h.elo
        assert h2.generation == h.generation
        assert h2.parent_id == h.parent_id

    def test_defaults(self):
        """Default values are sensible."""
        h = Hypothesis(id="h1", statement="Test", rationale="Test rationale")
        assert h.testability == 0.5
        assert h.novelty == 0.5
        assert h.elo == DEFAULT_ELO
        assert h.generation == 1
        assert h.parent_id is None
        assert h.predictions == []

    def test_from_dict_missing_fields(self):
        """from_dict handles missing optional fields."""
        d = {"id": "h1", "statement": "Test"}
        h = Hypothesis.from_dict(d)
        assert h.rationale == ""
        assert h.predictions == []
        assert h.testability == 0.5


# ===========================================================================
# JSON Parsing
# ===========================================================================


class TestParseHypotheses:
    def test_direct_array(self):
        raw = json.dumps([
            {"statement": "H1", "rationale": "R1", "predictions": ["P1"]},
            {"statement": "H2", "rationale": "R2", "predictions": ["P2"]},
        ])
        result = _parse_hypotheses_json(raw, n_expected=5)
        assert len(result) == 2
        assert result[0]["statement"] == "H1"

    def test_markdown_code_block(self):
        raw = '```json\n[{"statement": "H1", "rationale": "R", "predictions": ["P"]}]\n```'
        result = _parse_hypotheses_json(raw)
        assert len(result) == 1
        assert result[0]["statement"] == "H1"

    def test_dict_wrapper(self):
        raw = json.dumps({"hypotheses": [
            {"statement": "H1", "rationale": "R", "predictions": ["P"]},
        ]})
        result = _parse_hypotheses_json(raw)
        assert len(result) == 1

    def test_truncates_to_n(self):
        raw = json.dumps([
            {"statement": f"H{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
            for i in range(10)
        ])
        result = _parse_hypotheses_json(raw, n_expected=3)
        assert len(result) == 3

    def test_invalid_json(self):
        result = _parse_hypotheses_json("not json at all")
        assert result == []

    def test_bare_array_in_text(self):
        raw = 'Some text before\n[{"statement": "H1", "rationale": "R", "predictions": ["P"]}]\nMore text'
        result = _parse_hypotheses_json(raw)
        assert len(result) == 1

    def test_no_code_block_marker(self):
        raw = '[{"statement": "H1", "rationale": "R", "predictions": ["P"]}]'
        result = _parse_hypotheses_json(raw)
        assert len(result) == 1


# ===========================================================================
# Hypothesis Validation
# ===========================================================================


class TestValidateHypothesis:
    def test_valid(self):
        h = {"statement": "S", "rationale": "R", "predictions": ["P1", "P2"]}
        assert _validate_hypothesis(h) is True

    def test_missing_statement(self):
        h = {"rationale": "R", "predictions": ["P1"]}
        assert _validate_hypothesis(h) is False

    def test_empty_statement(self):
        h = {"statement": "", "rationale": "R", "predictions": ["P1"]}
        assert _validate_hypothesis(h) is False

    def test_missing_predictions(self):
        h = {"statement": "S", "rationale": "R"}
        assert _validate_hypothesis(h) is False

    def test_empty_predictions(self):
        h = {"statement": "S", "rationale": "R", "predictions": []}
        assert _validate_hypothesis(h) is False

    def test_predictions_not_list(self):
        h = {"statement": "S", "rationale": "R", "predictions": "not a list"}
        assert _validate_hypothesis(h) is False

    def test_clamps_scores(self):
        h = {
            "statement": "S", "rationale": "R", "predictions": ["P"],
            "testability": 2.5, "novelty": -0.5, "confidence": 1.5,
        }
        assert _validate_hypothesis(h) is True  # Still valid after clamping
        assert h["testability"] == 1.0
        assert h["novelty"] == 0.0
        assert h["confidence"] == 1.0


# ===========================================================================
# Integration: Tournament Orchestrator (mocked)
# ===========================================================================


class TestHypothesisTournament:
    """Integration tests for hypothesis_tournament with mocked LLM calls."""

    def test_tournament_too_few_hypotheses(self, monkeypatch):
        """Returns error if < 3 valid hypotheses generated."""
        monkeypatch.setattr(
            "hypothesis_engine._call_deepseek",
            lambda *a, **kw: json.dumps([
                {"statement": "H1", "rationale": "R", "predictions": ["P"]},
            ]),
        )
        result = json.loads(
            __import__("hypothesis_engine").hypothesis_tournament(
                "Test question?", n_hypotheses=5, n_rounds=1
            )
        )
        assert result["success"] is False
        assert result["error"] == "not_enough_hypotheses"

    def test_tournament_success_flow(self, monkeypatch):
        """Full tournament with mocked generation + debate + ranking."""
        call_count = {"deepseek": 0, "ollama": 0}

        def mock_deepseek(system_prompt, user_message, max_tokens=2000):
            call_count["deepseek"] += 1
            if "Generate" in system_prompt or "generate" in system_prompt.lower():
                # Phase 1: generation
                return json.dumps([
                    {
                        "statement": f"H{i} about climate and biodiversity",
                        "rationale": f"Rationale {i} based on ecological theory",
                        "predictions": [f"Prediction {i}a", f"Prediction {i}b"],
                        "testability": 0.7 + i * 0.05,
                        "novelty": 0.5 + i * 0.1,
                        "confidence": 0.5,
                    }
                    for i in range(5)
                ])
            elif "peer reviewer" in system_prompt.lower():
                # Phase 2: ranking
                return json.dumps({
                    "winner": "A",
                    "scores": {
                        "A": {"evidence": 8, "coherence": 7, "novelty": 6, "testability": 9, "validity": 8},
                        "B": {"evidence": 6, "coherence": 5, "novelty": 7, "testability": 4, "validity": 6},
                    },
                    "rationale": "A is better supported by evidence.",
                })
            elif "Evolve" in system_prompt or "refinement" in system_prompt:
                # Phase 2: evolution
                return json.dumps({
                    "statement": "Evolved statement with more specificity",
                    "rationale": "Expanded rationale with debate insights",
                    "predictions": ["Refined prediction 1", "Refined prediction 2"],
                    "testability": 0.85,
                    "novelty": 0.7,
                    "confidence": 0.6,
                    "changes_summary": "Added specificity, addressed critiques.",
                })
            else:
                # Phase 3: meta-review
                return "# Research Proposal\n\n## Abstract\nTest proposal."

        def mock_ollama(system_prompt, user_message, model="ecocoder:7b", max_tokens=500):
            call_count["ollama"] += 1
            return "CLAIM: This hypothesis is correct.\nEVIDENCE: Ecological data supports it.\nREBUTTAL: Opponent lacks evidence."

        monkeypatch.setattr("hypothesis_engine._call_deepseek", mock_deepseek)
        monkeypatch.setattr("hypothesis_engine._call_ollama", mock_ollama)

        result = json.loads(
            __import__("hypothesis_engine").hypothesis_tournament(
                "How does climate change affect biodiversity?",
                n_hypotheses=5,
                n_rounds=2,
            )
        )

        assert result["success"] is True
        assert len(result["ranked_hypotheses"]) == 5
        assert "proposal" in result
        assert "tournament_log" in result

        # 1 generation call + (n_rounds * floor(n/2)) ranking calls 
        # + (n_rounds * floor(n/2)) evolution calls + 1 meta-review call
        expected_deepseek = 1 + (2 * 2) + (2 * 2) + 1  # gen + rank + evolve + meta
        assert call_count["deepseek"] == expected_deepseek, \
            f"Expected {expected_deepseek} DeepSeek calls, got {call_count['deepseek']}"

        # Each round: 2 matches × (2 open + (max_turns*2 - 2) turns)
        # With 4 max_turns: 2 matches × 8 calls = 16 per round
        expected_ollama = 2 * 2 * (4 * 2)  # 2 rounds, 2 matches, 8 calls each
        assert call_count["ollama"] == expected_ollama, \
            f"Expected {expected_ollama} Ollama calls, got {call_count['ollama']}"

    def test_tournament_generates_ranked_hypotheses(self, monkeypatch):
        """Verify hypotheses are returned ranked by Elo."""
        monkeypatch.setattr(
            "hypothesis_engine._call_deepseek",
            lambda *a, **kw: json.dumps([
                {
                    "statement": f"H{i}",
                    "rationale": f"R{i}",
                    "predictions": [f"P{i}a", f"P{i}b"],
                    "testability": 0.8,
                    "novelty": 0.5 + i * 0.1,
                    "confidence": 0.5,
                }
                for i in range(4)  # 4 is even, ensures 2 matches/round
            ]),
        )
        monkeypatch.setattr(
            "hypothesis_engine._call_ollama",
            lambda *a, **kw: "CLAIM: Valid argument.\nEVIDENCE: Strong data.\nREBUTTAL: Weak counter.",
        )

        result = json.loads(
            __import__("hypothesis_engine").hypothesis_tournament(
                "Test?", n_hypotheses=4, n_rounds=1
            )
        )

        assert result["success"] is True
        elos = [h["elo"] for h in result["ranked_hypotheses"]]
        # Elos should be descending (ranked)
        assert elos == sorted(elos, reverse=True), f"Elos not ranked: {elos}"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_odd_number_hypotheses(self, monkeypatch):
        """5 hypotheses → 2 matches (one hypothesis gets a bye)."""
        monkeypatch.setattr(
            "hypothesis_engine._call_deepseek",
            lambda *a, **kw: json.dumps([
                {"statement": f"H{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
                for i in range(5)
            ]),
        )
        monkeypatch.setattr(
            "hypothesis_engine._call_ollama",
            lambda *a, **kw: "CLAIM: Argument.\nEVIDENCE: Data.\nREBUTTAL: Counter.",
        )

        result = json.loads(
            __import__("hypothesis_engine").hypothesis_tournament(
                "Test?", n_hypotheses=5, n_rounds=1
            )
        )
        assert result["success"] is True
        # 5 hypotheses: 2 matches, 1 hypothesis sits out
        assert len(result["ranked_hypotheses"]) == 5

    def test_debate_error_handling(self, monkeypatch):
        """Debate continues gracefully when Ollama fails."""
        monkeypatch.setattr(
            "hypothesis_engine._call_deepseek",
            lambda *a, **kw: json.dumps([
                {"statement": f"H{i}", "rationale": f"R{i}", "predictions": [f"P{i}"]}
                for i in range(4)
            ]),
        )

        def flaky_ollama(*a, **kw):
            raise ConnectionError("Ollama not available")

        monkeypatch.setattr("hypothesis_engine._call_ollama", flaky_ollama)

        result = json.loads(
            __import__("hypothesis_engine").hypothesis_tournament(
                "Test?", n_hypotheses=4, n_rounds=1
            )
        )
        # Should still complete (debate errors are caught)
        assert result["success"] is True
