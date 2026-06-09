"""Tests for confidence scoring — pure logic, no LLM calls."""

from __future__ import annotations

import pytest

from src.debate.schemas import DebateConfig, RiskAssessment, TradeDecision
from src.debate.scoring import ScoringResult, score_decision, should_execute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk() -> RiskAssessment:
    """Minimal RiskAssessment fixture."""
    return RiskAssessment(
        volatility_score=0.5,
        position_size_pct=2.0,
        max_exposure_pct=10.0,
        correlation_risk="low",
        sizing_method="atr",
    )


def _decision(confidence: int) -> TradeDecision:
    """Build a TradeDecision with the given confidence."""
    return TradeDecision(
        symbol="RELIANCE",
        exchange="NSE",
        action="BUY",
        confidence=confidence,
        signal_type="momentum",
        debate_rounds=[],
        risk_assessment=_risk(),
        reasoning="test",
        suggested_qty=10,
        suggested_price=2500.0,
    )


# Default DebateConfig thresholds:
#   confidence_skip_below  = 60
#   confidence_log_below   = 80
#   confidence_trade_above = 80


class TestScoreDecision:
    """score_decision routes to TRADE / LOG / SKIP based on thresholds."""

    def test_high_confidence_returns_trade(self):
        result = score_decision(_decision(85), DebateConfig())
        assert result.action == "TRADE"
        assert result.confidence == 85

    def test_medium_confidence_returns_log(self):
        result = score_decision(_decision(70), DebateConfig())
        assert result.action == "LOG"
        assert result.confidence == 70

    def test_low_confidence_returns_skip(self):
        result = score_decision(_decision(40), DebateConfig())
        assert result.action == "SKIP"
        assert result.confidence == 40

    # --- boundary tests ---------------------------------------------------

    def test_boundary_exactly_at_trade_above_returns_trade(self):
        """80 >= trade_above (80) -> TRADE."""
        result = score_decision(_decision(80), DebateConfig())
        assert result.action == "TRADE"

    def test_boundary_exactly_at_log_below_returns_trade(self):
        """log_below == trade_above == 80 -> first branch wins -> TRADE."""
        result = score_decision(_decision(80), DebateConfig())
        assert result.action == "TRADE"

    def test_boundary_79_returns_log(self):
        """79 < trade_above (80) but >= skip_below (60) -> LOG."""
        result = score_decision(_decision(79), DebateConfig())
        assert result.action == "LOG"

    def test_boundary_59_returns_skip(self):
        result = score_decision(_decision(59), DebateConfig())
        assert result.action == "SKIP"


class TestShouldExecute:
    """should_execute is True only when action == TRADE."""

    def test_trade_action_returns_true(self):
        sr = ScoringResult(action="TRADE", confidence=90, reason="ok")
        assert should_execute(sr) is True

    def test_log_action_returns_false(self):
        sr = ScoringResult(action="LOG", confidence=70, reason="ok")
        assert should_execute(sr) is False

    def test_skip_action_returns_false(self):
        sr = ScoringResult(action="SKIP", confidence=30, reason="ok")
        assert should_execute(sr) is False


class TestCustomConfig:
    """Custom DebateConfig thresholds override defaults."""

    def test_custom_thresholds(self):
        cfg = DebateConfig(
            confidence_skip_below=50,
            confidence_log_below=70,
            confidence_trade_above=90,
        )
        assert score_decision(_decision(95), cfg).action == "TRADE"
        assert score_decision(_decision(90), cfg).action == "TRADE"
        assert score_decision(_decision(89), cfg).action == "LOG"
        assert score_decision(_decision(70), cfg).action == "LOG"
        assert score_decision(_decision(49), cfg).action == "SKIP"


class TestScoringResultFrozen:
    """ScoringResult is immutable."""

    def test_cannot_mutate_action(self):
        sr = ScoringResult(action="TRADE", confidence=90, reason="ok")
        with pytest.raises(AttributeError):
            sr.action = "SKIP"  # type: ignore[misc]

    def test_cannot_mutate_confidence(self):
        sr = ScoringResult(action="TRADE", confidence=90, reason="ok")
        with pytest.raises(AttributeError):
            sr.confidence = 50  # type: ignore[misc]
