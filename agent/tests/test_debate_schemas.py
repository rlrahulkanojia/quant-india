"""Tests for adversarial debate pipeline dataclass schemas.

Covers:
  - Creating each dataclass with valid data
  - All are frozen (can't mutate)
  - DebateConfig defaults match spec
  - Custom DebateConfig overrides
  - Threshold ordering: skip_below < log_below <= trade_above
  - Valid action values: BUY, SELL, HOLD, SKIP
  - Valid sizing_method values: atr, kelly, fixed
  - Valid combined_signal values: bullish, bearish, neutral
"""

from __future__ import annotations

import pytest

from src.debate.schemas import (
    AnalysisReport,
    DebateConfig,
    DebateRound,
    RiskAssessment,
    TradeDecision,
)


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------

class TestAnalysisReport:
    def test_create_with_valid_data(self):
        report = AnalysisReport(
            symbol="RELIANCE",
            exchange="NSE",
            timestamp="2026-06-09T10:00:00",
            technical={"rsi": 55.0, "macd_signal": "bullish"},
            fundamental={"pe_ratio": 25.3, "debt_equity": 0.4},
            sentiment={"news_score": 0.7, "social_score": 0.6},
            combined_signal="bullish",
            signal_strength=0.75,
        )
        assert report.symbol == "RELIANCE"
        assert report.exchange == "NSE"
        assert report.timestamp == "2026-06-09T10:00:00"
        assert report.technical["rsi"] == 55.0
        assert report.fundamental["pe_ratio"] == 25.3
        assert report.sentiment["news_score"] == 0.7
        assert report.combined_signal == "bullish"
        assert report.signal_strength == 0.75

    def test_frozen(self):
        report = AnalysisReport(
            symbol="TCS",
            exchange="NSE",
            timestamp="2026-06-09T10:00:00",
            technical={},
            fundamental={},
            sentiment={},
            combined_signal="neutral",
            signal_strength=0.5,
        )
        with pytest.raises(AttributeError):
            report.symbol = "INFY"  # type: ignore[misc]

    @pytest.mark.parametrize("signal", ["bullish", "bearish", "neutral"])
    def test_valid_combined_signal_values(self, signal: str):
        report = AnalysisReport(
            symbol="HDFC",
            exchange="NSE",
            timestamp="2026-06-09T10:00:00",
            technical={},
            fundamental={},
            sentiment={},
            combined_signal=signal,
            signal_strength=0.5,
        )
        assert report.combined_signal == signal


# ---------------------------------------------------------------------------
# DebateRound
# ---------------------------------------------------------------------------

class TestDebateRound:
    def test_create_with_valid_data(self):
        dr = DebateRound(
            round_number=1,
            bull_argument="Strong earnings growth and sector tailwinds.",
            bear_argument="Overvalued relative to peers and rising rates.",
        )
        assert dr.round_number == 1
        assert "earnings" in dr.bull_argument
        assert "Overvalued" in dr.bear_argument

    def test_frozen(self):
        dr = DebateRound(
            round_number=1,
            bull_argument="bull case",
            bear_argument="bear case",
        )
        with pytest.raises(AttributeError):
            dr.round_number = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RiskAssessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_create_with_valid_data(self):
        ra = RiskAssessment(
            volatility_score=0.65,
            position_size_pct=2.5,
            max_exposure_pct=10.0,
            correlation_risk="moderate",
            sizing_method="atr",
        )
        assert ra.volatility_score == 0.65
        assert ra.position_size_pct == 2.5
        assert ra.max_exposure_pct == 10.0
        assert ra.correlation_risk == "moderate"
        assert ra.sizing_method == "atr"

    def test_frozen(self):
        ra = RiskAssessment(
            volatility_score=0.3,
            position_size_pct=1.0,
            max_exposure_pct=5.0,
            correlation_risk="low",
            sizing_method="kelly",
        )
        with pytest.raises(AttributeError):
            ra.volatility_score = 0.9  # type: ignore[misc]

    @pytest.mark.parametrize("method", ["atr", "kelly", "fixed"])
    def test_valid_sizing_method_values(self, method: str):
        ra = RiskAssessment(
            volatility_score=0.5,
            position_size_pct=2.0,
            max_exposure_pct=8.0,
            correlation_risk="low",
            sizing_method=method,
        )
        assert ra.sizing_method == method


# ---------------------------------------------------------------------------
# TradeDecision
# ---------------------------------------------------------------------------

class TestTradeDecision:
    def _make_risk(self) -> RiskAssessment:
        return RiskAssessment(
            volatility_score=0.5,
            position_size_pct=2.0,
            max_exposure_pct=10.0,
            correlation_risk="low",
            sizing_method="atr",
        )

    def _make_rounds(self) -> list[DebateRound]:
        return [
            DebateRound(round_number=1, bull_argument="bull 1", bear_argument="bear 1"),
            DebateRound(round_number=2, bull_argument="bull 2", bear_argument="bear 2"),
        ]

    def test_create_with_valid_data(self):
        td = TradeDecision(
            symbol="RELIANCE",
            exchange="NSE",
            action="BUY",
            confidence=85,
            signal_type="technical",
            debate_rounds=self._make_rounds(),
            risk_assessment=self._make_risk(),
            reasoning="Strong bullish momentum with manageable risk.",
            suggested_qty=10,
            suggested_price=2450.50,
        )
        assert td.symbol == "RELIANCE"
        assert td.exchange == "NSE"
        assert td.action == "BUY"
        assert td.confidence == 85
        assert td.signal_type == "technical"
        assert len(td.debate_rounds) == 2
        assert td.risk_assessment.sizing_method == "atr"
        assert td.reasoning.startswith("Strong")
        assert td.suggested_qty == 10
        assert td.suggested_price == 2450.50

    def test_frozen(self):
        td = TradeDecision(
            symbol="TCS",
            exchange="NSE",
            action="HOLD",
            confidence=50,
            signal_type="mixed",
            debate_rounds=[],
            risk_assessment=self._make_risk(),
            reasoning="Unclear signals.",
            suggested_qty=0,
            suggested_price=0.0,
        )
        with pytest.raises(AttributeError):
            td.action = "BUY"  # type: ignore[misc]

    @pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD", "SKIP"])
    def test_valid_action_values(self, action: str):
        td = TradeDecision(
            symbol="INFY",
            exchange="NSE",
            action=action,
            confidence=70,
            signal_type="technical",
            debate_rounds=[],
            risk_assessment=self._make_risk(),
            reasoning="Test action.",
            suggested_qty=5,
            suggested_price=1500.0,
        )
        assert td.action == action


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------

class TestDebateConfig:
    def test_defaults_match_spec(self):
        cfg = DebateConfig()
        assert cfg.rounds == 3
        assert cfg.confidence_skip_below == 60
        assert cfg.confidence_log_below == 80
        assert cfg.confidence_trade_above == 80
        # Model defaults to ANTHROPIC_MODEL env var, or claude-sonnet-4-20250514
        import os
        expected_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        assert cfg.model == expected_model

    def test_custom_overrides(self):
        cfg = DebateConfig(
            rounds=5,
            confidence_skip_below=50,
            confidence_log_below=70,
            confidence_trade_above=90,
            model="claude-opus-4-20250514",
        )
        assert cfg.rounds == 5
        assert cfg.confidence_skip_below == 50
        assert cfg.confidence_log_below == 70
        assert cfg.confidence_trade_above == 90
        assert cfg.model == "claude-opus-4-20250514"

    def test_frozen(self):
        cfg = DebateConfig()
        with pytest.raises(AttributeError):
            cfg.rounds = 10  # type: ignore[misc]

    def test_threshold_ordering_default(self):
        """skip_below < log_below <= trade_above."""
        cfg = DebateConfig()
        assert cfg.confidence_skip_below < cfg.confidence_log_below
        assert cfg.confidence_log_below <= cfg.confidence_trade_above
