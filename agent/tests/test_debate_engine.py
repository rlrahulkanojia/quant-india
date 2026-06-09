"""Tests for the debate engine orchestrator.

All LLM-backed agent functions are mocked — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.debate.engine import DebateEngine
from src.debate.schemas import (
    AnalysisReport,
    DebateConfig,
    DebateRound,
    RiskAssessment,
    TradeDecision,
)
from src.debate.scoring import ScoringResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SYMBOL = "RELIANCE"
EXCHANGE = "NSE"

MOCK_MARKET_DATA = {"close": 2500.0, "volume": 1_000_000, "sma_20": 2480.0}
MOCK_FINANCIAL_DATA = {"pe_ratio": 25.0, "roe": 0.15, "debt_to_equity": 0.4}
MOCK_SENTIMENT_DATA = {"score": 0.7, "news_count": 12, "social_volume": 300}

MOCK_TECHNICAL = {"signal": "bullish", "strength": 0.8, "indicators": {"rsi": 55}}
MOCK_FUNDAMENTAL = {"signal": "bullish", "strength": 0.7, "valuation": "fair"}
MOCK_SENTIMENT = {"signal": "bearish", "strength": 0.4, "mood": "cautious"}

MOCK_RISK = RiskAssessment(
    volatility_score=0.3,
    position_size_pct=2.0,
    max_exposure_pct=5.0,
    correlation_risk="low",
    sizing_method="atr",
)


def _make_trade_decision(confidence: int = 85) -> TradeDecision:
    """Build a TradeDecision with tuneable confidence."""
    return TradeDecision(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        action="BUY",
        confidence=confidence,
        signal_type="technical+fundamental",
        debate_rounds=[
            DebateRound(round_number=1, bull_argument="Bull r1", bear_argument="Bear r1"),
        ],
        risk_assessment=MOCK_RISK,
        reasoning="Strong technicals and fundamentals",
        suggested_qty=10,
        suggested_price=2505.0,
    )


def _make_analysis_report(
    combined_signal: str = "bullish",
    signal_strength: float = 0.63,
) -> AnalysisReport:
    """Build an AnalysisReport with sensible defaults."""
    return AnalysisReport(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        timestamp="2026-06-09T10:00:00",
        technical=MOCK_TECHNICAL,
        fundamental=MOCK_FUNDAMENTAL,
        sentiment=MOCK_SENTIMENT,
        combined_signal=combined_signal,
        signal_strength=signal_strength,
    )


# ---------------------------------------------------------------------------
# Tests — analyze()
# ---------------------------------------------------------------------------

class TestAnalyze:
    """DebateEngine.analyze() orchestrates three analysis agents."""

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_produces_full_report(
        self, mock_tech, mock_fund, mock_sent
    ):
        mock_tech.return_value = MOCK_TECHNICAL
        mock_fund.return_value = MOCK_FUNDAMENTAL
        mock_sent.return_value = MOCK_SENTIMENT

        engine = DebateEngine()
        report = engine.analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )

        assert isinstance(report, AnalysisReport)
        assert report.symbol == SYMBOL
        assert report.exchange == EXCHANGE
        assert report.technical == MOCK_TECHNICAL
        assert report.fundamental == MOCK_FUNDAMENTAL
        assert report.sentiment == MOCK_SENTIMENT
        mock_tech.assert_called_once()
        mock_fund.assert_called_once()
        mock_sent.assert_called_once()

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_majority_bullish(self, mock_tech, mock_fund, mock_sent):
        """Two bullish + one bearish → combined_signal is bullish."""
        mock_tech.return_value = {"signal": "bullish", "strength": 0.8}
        mock_fund.return_value = {"signal": "bullish", "strength": 0.7}
        mock_sent.return_value = {"signal": "bearish", "strength": 0.4}

        report = DebateEngine().analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )
        assert report.combined_signal == "bullish"

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_majority_bearish(self, mock_tech, mock_fund, mock_sent):
        """Two bearish + one bullish → combined_signal is bearish."""
        mock_tech.return_value = {"signal": "bearish", "strength": 0.6}
        mock_fund.return_value = {"signal": "bearish", "strength": 0.5}
        mock_sent.return_value = {"signal": "bullish", "strength": 0.8}

        report = DebateEngine().analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )
        assert report.combined_signal == "bearish"

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_neutral_when_mixed(self, mock_tech, mock_fund, mock_sent):
        """One of each → neutral."""
        mock_tech.return_value = {"signal": "bullish", "strength": 0.7}
        mock_fund.return_value = {"signal": "bearish", "strength": 0.6}
        mock_sent.return_value = {"signal": "neutral", "strength": 0.5}

        report = DebateEngine().analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )
        assert report.combined_signal == "neutral"

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_signal_strength_average(self, mock_tech, mock_fund, mock_sent):
        """signal_strength = average of individual strengths."""
        mock_tech.return_value = {"signal": "bullish", "strength": 0.9}
        mock_fund.return_value = {"signal": "bullish", "strength": 0.6}
        mock_sent.return_value = {"signal": "bullish", "strength": 0.3}

        report = DebateEngine().analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )
        assert report.signal_strength == pytest.approx(0.6, abs=1e-9)

    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_analyze_strength_defaults_when_missing(self, mock_tech, mock_fund, mock_sent):
        """Missing 'strength' keys → signal_strength falls back to 0.5."""
        mock_tech.return_value = {"signal": "bullish"}
        mock_fund.return_value = {"signal": "bullish"}
        mock_sent.return_value = {"signal": "bullish"}

        report = DebateEngine().analyze(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )
        assert report.signal_strength == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — debate()
# ---------------------------------------------------------------------------

class TestDebate:
    """DebateEngine.debate() runs bull/bear rounds then risk + PM decision."""

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    def test_debate_runs_3_rounds(self, mock_bull, mock_bear, mock_risk, mock_pm):
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision()

        engine = DebateEngine()  # default config → 3 rounds
        report = _make_analysis_report()
        decision = engine.debate(report)

        assert isinstance(decision, TradeDecision)
        assert mock_bull.call_count == 3
        assert mock_bear.call_count == 3
        mock_risk.assert_called_once()
        mock_pm.assert_called_once()

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    def test_debate_custom_5_rounds(self, mock_bull, mock_bear, mock_risk, mock_pm):
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision()

        config = DebateConfig(rounds=5)
        engine = DebateEngine(config=config)
        report = _make_analysis_report()
        decision = engine.debate(report)

        assert mock_bull.call_count == 5
        assert mock_bear.call_count == 5

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    def test_debate_returns_trade_decision(self, mock_bull, mock_bear, mock_risk, mock_pm):
        expected = _make_trade_decision()
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = expected

        decision = DebateEngine().debate(_make_analysis_report())

        assert decision is expected
        assert decision.symbol == SYMBOL
        assert decision.action == "BUY"

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    def test_debate_passes_model_from_config(self, mock_bull, mock_bear, mock_risk, mock_pm):
        """Each agent receives the model string from config."""
        mock_bull.return_value = "Bull"
        mock_bear.return_value = "Bear"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision()

        config = DebateConfig(model="claude-opus-4-20250514")
        DebateEngine(config=config).debate(_make_analysis_report())

        # Check model kwarg on bull calls
        for call in mock_bull.call_args_list:
            assert call.kwargs.get("model") or call[1].get("model") or "claude-opus-4-20250514" in str(call)


# ---------------------------------------------------------------------------
# Tests — evaluate()  (full pipeline)
# ---------------------------------------------------------------------------

class TestEvaluate:
    """DebateEngine.evaluate() = analyze → debate → score."""

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_evaluate_returns_tuple(
        self, mock_tech, mock_fund, mock_sent,
        mock_bull, mock_bear, mock_risk, mock_pm,
    ):
        mock_tech.return_value = MOCK_TECHNICAL
        mock_fund.return_value = MOCK_FUNDAMENTAL
        mock_sent.return_value = MOCK_SENTIMENT
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision(confidence=85)

        decision, scoring = DebateEngine().evaluate(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )

        assert isinstance(decision, TradeDecision)
        assert isinstance(scoring, ScoringResult)

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_high_confidence_scores_trade(
        self, mock_tech, mock_fund, mock_sent,
        mock_bull, mock_bear, mock_risk, mock_pm,
    ):
        """confidence >= 80 (default trade_above) → action == TRADE."""
        mock_tech.return_value = MOCK_TECHNICAL
        mock_fund.return_value = MOCK_FUNDAMENTAL
        mock_sent.return_value = MOCK_SENTIMENT
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision(confidence=90)

        _, scoring = DebateEngine().evaluate(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )

        assert scoring.action == "TRADE"

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_low_confidence_scores_skip(
        self, mock_tech, mock_fund, mock_sent,
        mock_bull, mock_bear, mock_risk, mock_pm,
    ):
        """confidence < 60 (default skip_below) → action == SKIP."""
        mock_tech.return_value = MOCK_TECHNICAL
        mock_fund.return_value = MOCK_FUNDAMENTAL
        mock_sent.return_value = MOCK_SENTIMENT
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision(confidence=40)

        _, scoring = DebateEngine().evaluate(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )

        assert scoring.action == "SKIP"

    @patch("src.debate.engine.run_portfolio_decision")
    @patch("src.debate.engine.run_risk_assessment")
    @patch("src.debate.engine.run_bear_case")
    @patch("src.debate.engine.run_bull_case")
    @patch("src.debate.engine.run_sentiment_analysis")
    @patch("src.debate.engine.run_fundamental_analysis")
    @patch("src.debate.engine.run_technical_analysis")
    def test_mid_confidence_scores_log(
        self, mock_tech, mock_fund, mock_sent,
        mock_bull, mock_bear, mock_risk, mock_pm,
    ):
        """confidence in [60, 80) → action == LOG."""
        mock_tech.return_value = MOCK_TECHNICAL
        mock_fund.return_value = MOCK_FUNDAMENTAL
        mock_sent.return_value = MOCK_SENTIMENT
        mock_bull.return_value = "Bull argument"
        mock_bear.return_value = "Bear argument"
        mock_risk.return_value = MOCK_RISK
        mock_pm.return_value = _make_trade_decision(confidence=70)

        _, scoring = DebateEngine().evaluate(
            SYMBOL, EXCHANGE, MOCK_MARKET_DATA, MOCK_FINANCIAL_DATA, MOCK_SENTIMENT_DATA,
        )

        assert scoring.action == "LOG"


# ---------------------------------------------------------------------------
# Tests — config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Verify default DebateConfig values flow through the engine."""

    def test_default_config_uses_3_rounds(self):
        engine = DebateEngine()
        assert engine._config.rounds == 3

    def test_custom_config_stored(self):
        config = DebateConfig(rounds=7, confidence_trade_above=90)
        engine = DebateEngine(config=config)
        assert engine._config.rounds == 7
        assert engine._config.confidence_trade_above == 90
