"""Tests for CalibrationService — threshold tuning from reflection reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decisions.store import DecisionStore
from src.decisions.calibration import CalibrationService, CalibrationUpdate
from src.decisions.reflection import ReflectionReport
from src.debate.schemas import DebateConfig


# ---- helpers ---------------------------------------------------------------

def _make_report(recommendations: list[dict] | None = None) -> ReflectionReport:
    """Create a ReflectionReport with given recommendations."""
    if recommendations is None:
        recommendations = [
            {
                "parameter": "confidence_skip_below",
                "current_value": 60.0,
                "recommended_value": 65.0,
                "reason": "Too many low-confidence trades",
            },
        ]
    return ReflectionReport(
        period_start="2025-06-01T00:00:00",
        period_end="2025-06-08T00:00:00",
        total_trades=10,
        win_rate=0.6,
        avg_pnl_pct=1.5,
        sharpe_ratio=0.8,
        findings=["Technical signals outperform"],
        recommendations=recommendations,
    )


@pytest.fixture()
def store(tmp_path: Path) -> DecisionStore:
    return DecisionStore(tmp_path / "test.db")


@pytest.fixture()
def base_config() -> DebateConfig:
    return DebateConfig(
        rounds=3,
        confidence_skip_below=60,
        confidence_log_below=80,
        confidence_trade_above=80,
    )


@pytest.fixture()
def svc(store: DecisionStore, base_config: DebateConfig) -> CalibrationService:
    return CalibrationService(store, base_config)


# ---- tests -----------------------------------------------------------------


class TestApplyReflection:
    """Applying reflection recommendations produces CalibrationUpdates."""

    def test_single_recommendation(self, svc: CalibrationService) -> None:
        report = _make_report()
        # Need to save the report to store first to get a reflection_id
        reflection_id = svc._store.save_reflection({
            "period_start": report.period_start,
            "period_end": report.period_end,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings,
            "recommendations": report.recommendations,
        })

        updates = svc.apply_reflection(report, reflection_id=reflection_id)

        assert len(updates) == 1
        u = updates[0]
        assert isinstance(u, CalibrationUpdate)
        assert u.parameter == "confidence_skip_below"
        assert u.old_value == 60
        assert u.new_value == 65.0
        assert u.reason == "Too many low-confidence trades"
        assert u.reflection_id == reflection_id

    def test_multiple_recommendations(self, svc: CalibrationService) -> None:
        recs = [
            {"parameter": "confidence_skip_below", "current_value": 60.0,
             "recommended_value": 65.0, "reason": "Raise skip floor"},
            {"parameter": "confidence_trade_above", "current_value": 80.0,
             "recommended_value": 75.0, "reason": "Lower trade bar"},
        ]
        report = _make_report(recommendations=recs)
        rid = svc._store.save_reflection({
            "period_start": report.period_start, "period_end": report.period_end,
            "total_trades": report.total_trades, "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct, "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings, "recommendations": report.recommendations,
        })

        updates = svc.apply_reflection(report, reflection_id=rid)
        assert len(updates) == 2
        params = {u.parameter for u in updates}
        assert params == {"confidence_skip_below", "confidence_trade_above"}

    def test_unknown_parameter_skipped(self, svc: CalibrationService) -> None:
        recs = [
            {"parameter": "some_unknown_param", "current_value": 1.0,
             "recommended_value": 2.0, "reason": "Should be skipped"},
        ]
        report = _make_report(recommendations=recs)
        rid = svc._store.save_reflection({
            "period_start": report.period_start, "period_end": report.period_end,
            "total_trades": report.total_trades, "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct, "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings, "recommendations": report.recommendations,
        })

        updates = svc.apply_reflection(report, reflection_id=rid)
        assert len(updates) == 0


class TestGetCurrentConfig:
    """get_current_config reflects active calibration updates."""

    def test_reflects_active_updates(
        self, svc: CalibrationService, base_config: DebateConfig
    ) -> None:
        report = _make_report()
        rid = svc._store.save_reflection({
            "period_start": report.period_start, "period_end": report.period_end,
            "total_trades": report.total_trades, "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct, "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings, "recommendations": report.recommendations,
        })

        svc.apply_reflection(report, reflection_id=rid)

        config = svc.get_current_config()
        assert config.confidence_skip_below == 65
        # Other params should remain at base
        assert config.confidence_log_below == base_config.confidence_log_below
        assert config.confidence_trade_above == base_config.confidence_trade_above
        assert config.rounds == base_config.rounds

    def test_no_updates_returns_base(
        self, svc: CalibrationService, base_config: DebateConfig
    ) -> None:
        config = svc.get_current_config()
        assert config == base_config


class TestRevertUpdate:
    """Reverting a calibration restores config to base."""

    def test_revert_restores_config(
        self, svc: CalibrationService, base_config: DebateConfig
    ) -> None:
        report = _make_report()
        rid = svc._store.save_reflection({
            "period_start": report.period_start, "period_end": report.period_end,
            "total_trades": report.total_trades, "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct, "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings, "recommendations": report.recommendations,
        })

        updates = svc.apply_reflection(report, reflection_id=rid)
        assert len(updates) == 1

        # Config should now be adjusted
        config_before = svc.get_current_config()
        assert config_before.confidence_skip_below == 65

        # Revert
        svc.revert_update(updates[0].id)

        # Config should return to base
        config_after = svc.get_current_config()
        assert config_after.confidence_skip_below == base_config.confidence_skip_below
        assert config_after == base_config
