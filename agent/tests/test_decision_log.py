"""Tests for DecisionLogger — trade logging, closing, and win-rate queries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.decisions.store import DecisionStore
from src.decisions.decision_log import DecisionLogger
from src.debate.schemas import (
    AnalysisReport,
    DebateRound,
    RiskAssessment,
    TradeDecision,
)
from src.debate.scoring import ScoringResult


# ---- helpers ---------------------------------------------------------------

def _make_analysis(symbol: str = "RELIANCE") -> AnalysisReport:
    return AnalysisReport(
        symbol=symbol,
        exchange="NSE",
        timestamp="2025-06-01T10:00:00",
        technical={"rsi": 55},
        fundamental={"pe": 22.5},
        sentiment={"score": 0.7},
        combined_signal="bullish",
        signal_strength=0.75,
    )


def _make_decision(
    symbol: str = "RELIANCE",
    action: str = "BUY",
    confidence: int = 85,
    signal_type: str = "technical",
) -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
        exchange="NSE",
        action=action,
        confidence=confidence,
        signal_type=signal_type,
        debate_rounds=[
            DebateRound(round_number=1, bull_argument="Strong RSI", bear_argument="High PE"),
        ],
        risk_assessment=RiskAssessment(
            volatility_score=0.4,
            position_size_pct=2.0,
            max_exposure_pct=5.0,
            correlation_risk="low",
            sizing_method="atr",
        ),
        reasoning="RSI momentum looks strong",
        suggested_qty=10,
        suggested_price=2500.0,
    )


def _make_scoring(action: str = "TRADE", confidence: int = 85) -> ScoringResult:
    return ScoringResult(action=action, confidence=confidence, reason="above threshold")


@pytest.fixture()
def logger(tmp_path: Path) -> DecisionLogger:
    store = DecisionStore(tmp_path / "test.db")
    return DecisionLogger(store)


# ---- tests -----------------------------------------------------------------


class TestLogDecision:
    """Log a TradeDecision and retrieve it."""

    def test_log_and_retrieve(self, logger: DecisionLogger) -> None:
        decision = _make_decision()
        scoring = _make_scoring()
        analysis = _make_analysis()

        decision_id = logger.log_decision(decision, scoring, analysis, paper_order_id="paper-001")
        assert isinstance(decision_id, int)
        assert decision_id >= 1

        recent = logger.get_recent_decisions(days=1)
        assert len(recent) >= 1
        row = recent[0]
        assert row["symbol"] == "RELIANCE"
        assert row["action"] == "BUY"
        assert row["confidence"] == 85
        assert row["signal_type"] == "technical"
        assert row["paper_order_id"] == "paper-001"
        assert row["entry_price"] == 2500.0

    def test_log_without_paper_order(self, logger: DecisionLogger) -> None:
        decision_id = logger.log_decision(
            _make_decision(), _make_scoring(), _make_analysis()
        )
        recent = logger.get_recent_decisions(days=1)
        assert recent[0]["paper_order_id"] is None


class TestCloseTrade:
    """Close a trade and verify fields updated."""

    def test_close_updates_fields(self, logger: DecisionLogger) -> None:
        decision_id = logger.log_decision(
            _make_decision(), _make_scoring(), _make_analysis()
        )

        logger.close_trade(
            decision_id=decision_id,
            exit_price=2600.0,
            pnl_amount=1000.0,
            pnl_pct=4.0,
            hold_duration_minutes=120,
            exit_reason="target_hit",
        )

        row = logger._store.get_decision(decision_id)
        assert row is not None
        assert row["exit_price"] == 2600.0
        assert row["pnl_amount"] == 1000.0
        assert row["pnl_pct"] == 4.0
        assert row["hold_duration"] == 120
        assert row["exit_reason"] == "target_hit"
        assert row["closed_at"] is not None


class TestWinRate:
    """Win rate calculation with wins, losses, and edge cases."""

    def test_two_wins_one_loss(self, logger: DecisionLogger) -> None:
        for i, (pnl, pnl_pct) in enumerate([(500.0, 2.0), (300.0, 1.5), (-200.0, -1.0)]):
            did = logger.log_decision(
                _make_decision(confidence=85 + i), _make_scoring(), _make_analysis()
            )
            logger.close_trade(did, 2600.0, pnl, pnl_pct, 60, "target")

        win_rate = logger.get_win_rate(days=30)
        assert abs(win_rate - 2 / 3) < 0.01

    def test_empty_returns_zero(self, logger: DecisionLogger) -> None:
        assert logger.get_win_rate(days=30) == 0.0

    def test_signal_type_filter(self, logger: DecisionLogger) -> None:
        # 1 technical win, 1 fundamental loss
        did1 = logger.log_decision(
            _make_decision(signal_type="technical"),
            _make_scoring(),
            _make_analysis(),
        )
        logger.close_trade(did1, 2600.0, 500.0, 2.0, 60, "target")

        did2 = logger.log_decision(
            _make_decision(signal_type="fundamental"),
            _make_scoring(),
            _make_analysis(),
        )
        logger.close_trade(did2, 2400.0, -200.0, -1.0, 60, "stop_loss")

        # Overall: 50%
        assert abs(logger.get_win_rate(days=30) - 0.5) < 0.01
        # Technical only: 100%
        assert logger.get_win_rate(days=30, signal_type="technical") == 1.0
        # Fundamental only: 0%
        assert logger.get_win_rate(days=30, signal_type="fundamental") == 0.0

    def test_only_open_trades_returns_zero(self, logger: DecisionLogger) -> None:
        """Open (unclosed) trades shouldn't count for win rate."""
        logger.log_decision(_make_decision(), _make_scoring(), _make_analysis())
        assert logger.get_win_rate(days=30) == 0.0
