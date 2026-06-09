"""Tests for DecisionStore — SQLite-backed persistence for trade decisions.

Covers:
  - Schema creation: 3 tables + WAL mode
  - Trade decision CRUD: save, get, list with date filter, close
  - Reflection reports: save and get_latest
  - Calibration updates: save, get active, revert
"""

from __future__ import annotations

import json

import pytest

from src.decisions.store import DecisionStore


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    """Verify the 3-table schema and WAL journal mode."""

    def test_tables_exist(self, tmp_path):
        store = DecisionStore(str(tmp_path / "test.db"))
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in tables]
        assert "trade_decisions" in names
        assert "reflection_reports" in names
        assert "calibration_updates" in names

    def test_wal_mode_enabled(self, tmp_path):
        store = DecisionStore(str(tmp_path / "test.db"))
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# Trade decisions CRUD
# ---------------------------------------------------------------------------


class TestTradeDecisions:
    """save_decision, get_decision, list_decisions, close_decision."""

    @pytest.fixture()
    def store(self, tmp_path):
        return DecisionStore(str(tmp_path / "test.db"))

    def _make_decision(self, **overrides) -> dict:
        base = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "confidence": 75,
            "debate_log": [{"agent": "bull", "text": "strong support"}],
            "analysis_report": {"rsi": 42, "macd": "bullish"},
            "signal_type": "MOMENTUM",
            "entry_price": 2450.0,
            "paper_order_id": "PO-001",
        }
        base.update(overrides)
        return base

    def test_save_and_get_roundtrip(self, store):
        data = self._make_decision()
        row_id = store.save_decision(data)
        assert row_id >= 1

        fetched = store.get_decision(row_id)
        assert fetched is not None
        assert fetched["symbol"] == "RELIANCE"
        assert fetched["action"] == "BUY"
        assert fetched["confidence"] == 75
        assert fetched["signal_type"] == "MOMENTUM"
        assert fetched["entry_price"] == pytest.approx(2450.0)
        assert fetched["paper_order_id"] == "PO-001"

        # JSON fields should be deserialized
        assert fetched["debate_log"] == [{"agent": "bull", "text": "strong support"}]
        assert fetched["analysis_report"]["rsi"] == 42

    def test_get_missing_returns_none(self, store):
        assert store.get_decision(9999) is None

    def test_list_decisions_returns_newest_first(self, store):
        store.save_decision(self._make_decision(symbol="RELIANCE"))
        store.save_decision(self._make_decision(symbol="TCS"))
        store.save_decision(self._make_decision(symbol="INFY"))

        results = store.list_decisions()
        assert len(results) == 3
        assert results[0]["symbol"] == "INFY"  # newest first
        assert results[2]["symbol"] == "RELIANCE"  # oldest last

    def test_list_decisions_with_date_filter(self, store):
        # Insert with explicit created_at for testing
        store._conn.execute(
            """INSERT INTO trade_decisions
               (symbol, action, confidence, signal_type, entry_price, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("OLD", "BUY", 60, "TREND", 100.0, "2025-01-01 00:00:00"),
        )
        store._conn.execute(
            """INSERT INTO trade_decisions
               (symbol, action, confidence, signal_type, entry_price, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("NEW", "SELL", 80, "REVERSAL", 200.0, "2025-06-01 00:00:00"),
        )
        store._conn.commit()

        # Filter: only after 2025-03-01
        results = store.list_decisions(since_date="2025-03-01")
        assert len(results) == 1
        assert results[0]["symbol"] == "NEW"

    def test_list_decisions_respects_limit(self, store):
        for i in range(10):
            store.save_decision(self._make_decision(symbol=f"SYM{i}"))

        results = store.list_decisions(limit=3)
        assert len(results) == 3

    def test_close_decision_updates_exit_fields(self, store):
        row_id = store.save_decision(self._make_decision())

        store.close_decision(
            decision_id=row_id,
            exit_price=2550.0,
            pnl_amount=100.0,
            pnl_pct=4.08,
            hold_duration=120,
            exit_reason="target_hit",
        )

        fetched = store.get_decision(row_id)
        assert fetched is not None
        assert fetched["exit_price"] == pytest.approx(2550.0)
        assert fetched["pnl_amount"] == pytest.approx(100.0)
        assert fetched["pnl_pct"] == pytest.approx(4.08)
        assert fetched["hold_duration"] == 120
        assert fetched["exit_reason"] == "target_hit"
        assert fetched["closed_at"] is not None


# ---------------------------------------------------------------------------
# Reflection reports
# ---------------------------------------------------------------------------


class TestReflectionReports:
    """save_reflection and get_latest_reflection."""

    @pytest.fixture()
    def store(self, tmp_path):
        return DecisionStore(str(tmp_path / "test.db"))

    def test_save_and_get_latest_reflection(self, store):
        data = {
            "period_start": "2025-06-01",
            "period_end": "2025-06-07",
            "total_trades": 15,
            "win_rate": 0.6,
            "avg_pnl_pct": 1.2,
            "sharpe_ratio": 1.8,
            "findings": ["overtrading on Mondays", "good entries on gaps"],
            "recommendations": ["reduce Monday trades", "increase gap position size"],
        }
        row_id = store.save_reflection(data)
        assert row_id >= 1

        latest = store.get_latest_reflection()
        assert latest is not None
        assert latest["total_trades"] == 15
        assert latest["win_rate"] == pytest.approx(0.6)
        assert latest["avg_pnl_pct"] == pytest.approx(1.2)
        assert latest["sharpe_ratio"] == pytest.approx(1.8)

        # JSON fields deserialized
        assert latest["findings"] == ["overtrading on Mondays", "good entries on gaps"]
        assert len(latest["recommendations"]) == 2

    def test_get_latest_returns_most_recent(self, store):
        store.save_reflection({
            "period_start": "2025-05-01",
            "period_end": "2025-05-07",
            "total_trades": 10,
            "win_rate": 0.5,
            "avg_pnl_pct": 0.8,
            "sharpe_ratio": 1.2,
        })
        store.save_reflection({
            "period_start": "2025-06-01",
            "period_end": "2025-06-07",
            "total_trades": 20,
            "win_rate": 0.7,
            "avg_pnl_pct": 2.0,
            "sharpe_ratio": 2.5,
        })

        latest = store.get_latest_reflection()
        assert latest is not None
        assert latest["total_trades"] == 20
        assert latest["period_start"] == "2025-06-01"

    def test_get_latest_returns_none_when_empty(self, store):
        assert store.get_latest_reflection() is None


# ---------------------------------------------------------------------------
# Calibration updates
# ---------------------------------------------------------------------------


class TestCalibrationUpdates:
    """save_calibration_update, get_active_calibrations, revert_calibration."""

    @pytest.fixture()
    def store(self, tmp_path):
        return DecisionStore(str(tmp_path / "test.db"))

    def _create_reflection(self, store) -> int:
        """Helper: create a reflection to reference from calibration."""
        return store.save_reflection({
            "period_start": "2025-06-01",
            "period_end": "2025-06-07",
            "total_trades": 10,
            "win_rate": 0.5,
            "avg_pnl_pct": 0.8,
            "sharpe_ratio": 1.2,
        })

    def test_save_and_get_active_calibrations(self, store):
        ref_id = self._create_reflection(store)

        cal_id = store.save_calibration_update({
            "reflection_id": ref_id,
            "parameter": "confidence_threshold",
            "old_value": 60.0,
            "new_value": 70.0,
            "reason": "too many low-confidence losers",
        })
        assert cal_id >= 1

        active = store.get_active_calibrations()
        assert len(active) == 1
        assert active[0]["parameter"] == "confidence_threshold"
        assert active[0]["old_value"] == pytest.approx(60.0)
        assert active[0]["new_value"] == pytest.approx(70.0)
        assert active[0]["reason"] == "too many low-confidence losers"
        assert active[0]["reverted_at"] is None

    def test_revert_calibration_removes_from_active(self, store):
        ref_id = self._create_reflection(store)

        cal_id = store.save_calibration_update({
            "reflection_id": ref_id,
            "parameter": "stop_loss_pct",
            "old_value": 2.0,
            "new_value": 1.5,
            "reason": "tighten stops after drawdown",
        })

        # Before revert: active
        assert len(store.get_active_calibrations()) == 1

        # Revert
        store.revert_calibration(cal_id)

        # After revert: no longer active
        assert len(store.get_active_calibrations()) == 0

    def test_multiple_calibrations_only_active_returned(self, store):
        ref_id = self._create_reflection(store)

        cal1 = store.save_calibration_update({
            "reflection_id": ref_id,
            "parameter": "confidence_threshold",
            "old_value": 60.0,
            "new_value": 70.0,
            "reason": "raise threshold",
        })
        cal2 = store.save_calibration_update({
            "reflection_id": ref_id,
            "parameter": "position_size_pct",
            "old_value": 5.0,
            "new_value": 3.0,
            "reason": "reduce exposure",
        })

        # Revert only the first
        store.revert_calibration(cal1)

        active = store.get_active_calibrations()
        assert len(active) == 1
        assert active[0]["parameter"] == "position_size_pct"
