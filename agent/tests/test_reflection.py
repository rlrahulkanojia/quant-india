"""Tests for ReflectionService — weekly trade review with mocked Claude."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.decisions.store import DecisionStore
from src.decisions.reflection import ReflectionReport, ReflectionService


# ---- helpers ---------------------------------------------------------------

def _seed_closed_decisions(store: DecisionStore) -> None:
    """Insert 3 closed trades: 2 wins, 1 loss."""
    trades = [
        {"symbol": "RELIANCE", "action": "BUY", "confidence": 85,
         "signal_type": "technical", "entry_price": 2500.0,
         "pnl_amount": 500.0, "pnl_pct": 2.0, "hold_duration": 60,
         "exit_price": 2550.0, "exit_reason": "target_hit"},
        {"symbol": "TCS", "action": "BUY", "confidence": 78,
         "signal_type": "fundamental", "entry_price": 3500.0,
         "pnl_amount": 300.0, "pnl_pct": 1.5, "hold_duration": 90,
         "exit_price": 3550.0, "exit_reason": "target_hit"},
        {"symbol": "INFY", "action": "BUY", "confidence": 72,
         "signal_type": "technical", "entry_price": 1500.0,
         "pnl_amount": -200.0, "pnl_pct": -1.0, "hold_duration": 45,
         "exit_price": 1485.0, "exit_reason": "stop_loss"},
    ]
    for t in trades:
        did = store.save_decision({
            "symbol": t["symbol"],
            "action": t["action"],
            "confidence": t["confidence"],
            "signal_type": t["signal_type"],
            "entry_price": t["entry_price"],
        })
        store.close_decision(
            decision_id=did,
            exit_price=t["exit_price"],
            pnl_amount=t["pnl_amount"],
            pnl_pct=t["pnl_pct"],
            hold_duration=t["hold_duration"],
            exit_reason=t["exit_reason"],
        )


def _mock_claude_response() -> dict:
    """Return the JSON that our mocked Claude would produce."""
    return {
        "patterns_identified": [
            "Technical signals outperform fundamental in recent trades",
        ],
        "threshold_recommendations": [
            {
                "parameter": "confidence_skip_below",
                "current_value": 60.0,
                "recommended_value": 65.0,
                "reason": "Too many low-confidence trades slipping through",
            },
        ],
        "bias_flags": ["Slight overconfidence on losing trades"],
        "summary": "Overall positive week with 66% win rate.",
    }


@pytest.fixture()
def store(tmp_path: Path) -> DecisionStore:
    s = DecisionStore(tmp_path / "test.db")
    _seed_closed_decisions(s)
    return s


@pytest.fixture()
def mock_anthropic():
    """Patch anthropic.Anthropic so no real API call is made."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text=json.dumps(_mock_claude_response()))
    ]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("src.decisions.reflection.anthropic") as mock_mod:
        mock_mod.Anthropic.return_value = mock_client
        yield mock_mod, mock_client


# ---- tests -----------------------------------------------------------------


class TestRunWeeklyReflection:
    """Seed store with 3 closed trades; run reflection and verify."""

    def test_metrics_calculated(self, store: DecisionStore, mock_anthropic) -> None:
        _, mock_client = mock_anthropic
        svc = ReflectionService(store)

        report = svc.run_weekly_reflection()

        assert isinstance(report, ReflectionReport)
        # 2 wins / 3 total
        assert abs(report.win_rate - 2 / 3) < 0.01
        assert report.total_trades == 3
        # avg_pnl_pct: mean(2.0, 1.5, -1.0) = 0.833...
        assert abs(report.avg_pnl_pct - (2.0 + 1.5 + (-1.0)) / 3) < 0.01

    def test_claude_called_once(self, store: DecisionStore, mock_anthropic) -> None:
        _, mock_client = mock_anthropic
        svc = ReflectionService(store)

        svc.run_weekly_reflection()

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        # Should include model parameter
        assert "model" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    def test_report_saved_to_store(self, store: DecisionStore, mock_anthropic) -> None:
        svc = ReflectionService(store)
        svc.run_weekly_reflection()

        saved = store.get_latest_reflection()
        assert saved is not None
        assert saved["total_trades"] == 3
        assert abs(saved["win_rate"] - 2 / 3) < 0.01

    def test_report_has_findings_and_recommendations(
        self, store: DecisionStore, mock_anthropic
    ) -> None:
        svc = ReflectionService(store)
        report = svc.run_weekly_reflection()

        assert len(report.findings) > 0
        assert len(report.recommendations) > 0
        rec = report.recommendations[0]
        assert "parameter" in rec
        assert "recommended_value" in rec

    def test_sharpe_ratio_calculated(self, store: DecisionStore, mock_anthropic) -> None:
        svc = ReflectionService(store)
        report = svc.run_weekly_reflection()

        # pnl_pcts = [2.0, 1.5, -1.0]
        # mean = 0.833, std ≈ 1.607 -> sharpe ≈ 0.518
        assert report.sharpe_ratio > 0
        assert report.sharpe_ratio < 2.0  # sanity check

    def test_no_trades_returns_zero_metrics(self, tmp_path: Path, mock_anthropic) -> None:
        empty_store = DecisionStore(tmp_path / "empty.db")
        svc = ReflectionService(empty_store)
        report = svc.run_weekly_reflection()

        assert report.total_trades == 0
        assert report.win_rate == 0.0
        assert report.avg_pnl_pct == 0.0
        assert report.sharpe_ratio == 0.0
