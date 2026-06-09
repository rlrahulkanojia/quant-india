"""Tests for PaperStore — SQLite-backed persistence for paper trading.

Covers:
  - Schema creation: 4 tables + WAL mode
  - Portfolio CRUD: init, update_cash, realized_delta accumulation
  - Order CRUD: save, get, list, missing-id handling
  - Position CRUD: save, get by symbol, update, delete, list all
  - Shadow fill CRUD: save, get by order_id, multiple fills, unknown order
"""

from __future__ import annotations

import json

import pytest

from src.paper.store import PaperStore


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    """Verify the 4-table schema and WAL journal mode."""

    def test_tables_exist(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in tables]
        assert "paper_portfolio" in names
        assert "paper_positions" in names
        assert "paper_orders" in names
        assert "shadow_fills" in names

    def test_wal_mode_enabled(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------


class TestPortfolioCRUD:
    """init_portfolio, get_portfolio, update_cash."""

    def test_init_creates_record(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.init_portfolio(1_000_000.0)
        p = store.get_portfolio()
        assert p is not None
        assert p["cash_balance"] == pytest.approx(1_000_000.0)
        assert p["total_realized"] == pytest.approx(0.0)

    def test_update_cash_modifies_balance(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.init_portfolio(500_000.0)
        store.update_cash(-25_000.0)
        p = store.get_portfolio()
        assert p["cash_balance"] == pytest.approx(475_000.0)

    def test_realized_delta_accumulates(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.init_portfolio(100_000.0)
        store.update_cash(0.0, realized_delta=5_000.0)
        store.update_cash(0.0, realized_delta=3_000.0)
        p = store.get_portfolio()
        assert p["total_realized"] == pytest.approx(8_000.0)
        assert p["cash_balance"] == pytest.approx(100_000.0)

    def test_get_portfolio_returns_none_when_empty(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        assert store.get_portfolio() is None


# ---------------------------------------------------------------------------
# Order CRUD
# ---------------------------------------------------------------------------


class TestOrderCRUD:
    """save_order, get_order, list_orders."""

    @staticmethod
    def _sample_order(**overrides) -> dict:
        base = {
            "id": "ord_001",
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "order_type": "MARKET",
            "qty": 10,
            "limit_price": None,
            "fill_price": 2500.0,
            "slippage": 0.50,
            "fees_total": 12.34,
            "fees_breakdown": {"stt": 6.25, "exchange_charges": 3.45, "sebi": 0.10, "gst": 0.0, "stamp_duty": 2.54},
            "status": "FILLED",
            "filled_qty": 10,
            "created_at": "2025-01-15T10:00:00",
            "filled_at": "2025-01-15T10:00:01",
        }
        base.update(overrides)
        return base

    def test_save_and_get_order(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.save_order(self._sample_order())
        order = store.get_order("ord_001")
        assert order is not None
        assert order["symbol"] == "RELIANCE"
        assert order["fill_price"] == pytest.approx(2500.0)
        assert order["fees_total"] == pytest.approx(12.34)
        # fees_breakdown should round-trip as dict
        fb = order["fees_breakdown"]
        assert isinstance(fb, dict)
        assert fb["stt"] == pytest.approx(6.25)

    def test_list_orders(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.save_order(self._sample_order(id="ord_001"))
        store.save_order(self._sample_order(id="ord_002", symbol="TCS"))
        orders = store.list_orders()
        assert len(orders) == 2

    def test_list_orders_empty(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        assert store.list_orders() == []

    def test_get_order_returns_none_for_missing(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        assert store.get_order("nonexistent") is None


# ---------------------------------------------------------------------------
# Position CRUD
# ---------------------------------------------------------------------------


class TestPositionCRUD:
    """save_position, get_position_by_symbol, update_position, delete_position."""

    @staticmethod
    def _sample_position(**overrides) -> dict:
        base = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "qty": 10,
            "avg_price": 2500.0,
            "current_price": 2520.0,
            "unrealized_pnl": 200.0,
            "side": "BUY",
            "opened_at": "2025-01-15T10:00:00",
        }
        base.update(overrides)
        return base

    def test_save_and_get_by_symbol(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        pos_id = store.save_position(self._sample_position())
        assert isinstance(pos_id, int)
        pos = store.get_position_by_symbol("RELIANCE", "NSE")
        assert pos is not None
        assert pos["qty"] == 10
        assert pos["avg_price"] == pytest.approx(2500.0)
        assert pos["side"] == "BUY"

    def test_update_position(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        pos_id = store.save_position(self._sample_position())
        store.update_position(pos_id, {"qty": 20, "avg_price": 2510.0})
        pos = store.get_position_by_symbol("RELIANCE", "NSE")
        assert pos["qty"] == 20
        assert pos["avg_price"] == pytest.approx(2510.0)

    def test_delete_position(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        pos_id = store.save_position(self._sample_position())
        store.delete_position(pos_id)
        assert store.get_position_by_symbol("RELIANCE", "NSE") is None

    def test_get_positions_returns_all(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.save_position(self._sample_position(symbol="RELIANCE"))
        store.save_position(self._sample_position(symbol="TCS"))
        store.save_position(self._sample_position(symbol="INFY"))
        positions = store.get_positions()
        assert len(positions) == 3
        symbols = {p["symbol"] for p in positions}
        assert symbols == {"RELIANCE", "TCS", "INFY"}

    def test_get_position_by_symbol_missing_returns_none(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        assert store.get_position_by_symbol("MISSING", "NSE") is None


# ---------------------------------------------------------------------------
# Shadow fill CRUD
# ---------------------------------------------------------------------------


class TestShadowFillCRUD:
    """save_shadow_fill, get_shadow_fills."""

    @staticmethod
    def _parent_order() -> dict:
        """Minimal parent order so the FK constraint is satisfied."""
        return {
            "id": "ord_001",
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "order_type": "MARKET",
            "qty": 10,
            "limit_price": None,
            "fill_price": 2500.0,
            "slippage": 0.50,
            "fees_total": 12.34,
            "fees_breakdown": {},
            "status": "FILLED",
            "filled_qty": 10,
            "created_at": "2025-01-15T10:00:00",
            "filled_at": "2025-01-15T10:00:01",
        }

    @staticmethod
    def _sample_fill(**overrides) -> dict:
        base = {
            "paper_order_id": "ord_001",
            "paper_fill_price": 2500.50,
            "market_ltp": 2500.00,
            "market_bid": 2499.50,
            "market_ask": 2500.50,
            "divergence_pct": 0.02,
            "captured_at": "2025-01-15T10:00:01",
        }
        base.update(overrides)
        return base

    def test_save_and_get_by_order_id(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.save_order(self._parent_order())
        store.save_shadow_fill(self._sample_fill())
        fills = store.get_shadow_fills("ord_001")
        assert len(fills) == 1
        assert fills[0]["paper_fill_price"] == pytest.approx(2500.50)
        assert fills[0]["divergence_pct"] == pytest.approx(0.02)

    def test_multiple_fills_per_order(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        store.save_order(self._parent_order())
        store.save_shadow_fill(self._sample_fill(paper_fill_price=2500.50))
        store.save_shadow_fill(self._sample_fill(paper_fill_price=2501.00))
        fills = store.get_shadow_fills("ord_001")
        assert len(fills) == 2

    def test_empty_for_unknown_order(self, tmp_path):
        store = PaperStore(tmp_path / "test.db")
        assert store.get_shadow_fills("unknown_order") == []
