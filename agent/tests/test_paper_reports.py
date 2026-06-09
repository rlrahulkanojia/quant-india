"""Tests for paper trading reports — daily summaries, equity curves, shadow comparison.

Covers:
  - daily_summary with filled trades on a date → correct trade_count, fees_paid
  - daily_summary with no trades on a date → zeros
  - equity_curve → list of (date, value) tuples tracking portfolio value
  - shadow_comparison → fill_count, avg_divergence, divergence_cost
"""

from __future__ import annotations

import pytest

from src.paper.store import PaperStore
from src.paper.reports import (
    DailySummary,
    ShadowComparison,
    daily_summary,
    equity_curve,
    shadow_comparison,
)


# ---------------------------------------------------------------------------
# Helpers — seed data builders
# ---------------------------------------------------------------------------


def _make_order(
    *,
    id: str = "ord_001",
    symbol: str = "RELIANCE",
    exchange: str = "NSE",
    side: str = "BUY",
    order_type: str = "MARKET",
    qty: int = 10,
    fill_price: float = 2500.0,
    fees_total: float = 12.50,
    status: str = "FILLED",
    filled_qty: int = 10,
    created_at: str = "2025-06-10T09:15:00",
    filled_at: str = "2025-06-10T09:15:01",
    **overrides,
) -> dict:
    base = {
        "id": id,
        "symbol": symbol,
        "exchange": exchange,
        "side": side,
        "order_type": order_type,
        "qty": qty,
        "limit_price": None,
        "fill_price": fill_price,
        "slippage": 0.50,
        "fees_total": fees_total,
        "fees_breakdown": {},
        "status": status,
        "filled_qty": filled_qty,
        "created_at": created_at,
        "filled_at": filled_at,
    }
    base.update(overrides)
    return base


def _make_shadow_fill(
    *,
    paper_order_id: str = "ord_001",
    paper_fill_price: float = 2500.50,
    market_ltp: float = 2500.00,
    market_bid: float = 2499.50,
    market_ask: float = 2500.50,
    divergence_pct: float = 0.02,
    qty: int = 10,
    captured_at: str = "2025-06-10T09:15:01",
    **overrides,
) -> dict:
    base = {
        "paper_order_id": paper_order_id,
        "paper_fill_price": paper_fill_price,
        "market_ltp": market_ltp,
        "market_bid": market_bid,
        "market_ask": market_ask,
        "divergence_pct": divergence_pct,
        "qty": qty,
        "captured_at": captured_at,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# DailySummary
# ---------------------------------------------------------------------------


class TestDailySummary:
    """daily_summary() — aggregate filled orders for a given date."""

    def test_trades_on_date(self, tmp_path):
        """Orders filled on the queried date → correct trade_count, fees_paid."""
        store = PaperStore(tmp_path / "test.db")
        store.save_order(_make_order(
            id="ord_001", symbol="RELIANCE", side="BUY",
            fill_price=2500.0, fees_total=12.50, filled_qty=10,
            filled_at="2025-06-10T09:15:01",
        ))
        store.save_order(_make_order(
            id="ord_002", symbol="TCS", side="BUY",
            fill_price=3400.0, fees_total=18.75, filled_qty=5,
            filled_at="2025-06-10T14:30:00",
        ))
        # Order on a different date — should be excluded
        store.save_order(_make_order(
            id="ord_003", symbol="INFY", side="SELL",
            fill_price=1500.0, fees_total=8.00, filled_qty=20,
            filled_at="2025-06-11T10:00:00",
        ))

        result = daily_summary(store, "2025-06-10")

        assert isinstance(result, DailySummary)
        assert result.date == "2025-06-10"
        assert result.trade_count == 2
        assert result.fees_paid == pytest.approx(12.50 + 18.75)

    def test_no_trades_on_date(self, tmp_path):
        """No filled orders on queried date → zeros and Nones."""
        store = PaperStore(tmp_path / "test.db")
        # Seed an order on a different date
        store.save_order(_make_order(
            id="ord_001", filled_at="2025-06-10T09:15:01",
        ))

        result = daily_summary(store, "2025-06-12")

        assert result.date == "2025-06-12"
        assert result.trade_count == 0
        assert result.fees_paid == pytest.approx(0.0)
        assert result.realized_pnl == pytest.approx(0.0)
        assert result.top_winner is None
        assert result.top_loser is None

    def test_pending_orders_excluded(self, tmp_path):
        """Orders that are PENDING (not filled) should not count."""
        store = PaperStore(tmp_path / "test.db")
        store.save_order(_make_order(
            id="ord_001", status="PENDING", filled_at=None,
            created_at="2025-06-10T09:15:00",
        ))

        result = daily_summary(store, "2025-06-10")

        assert result.trade_count == 0
        assert result.fees_paid == pytest.approx(0.0)

    def test_top_winner_and_loser(self, tmp_path):
        """When SELL orders exist, top_winner/top_loser reflect fee-based P&L."""
        store = PaperStore(tmp_path / "test.db")
        # A BUY with high fees → negative fee impact (top loser candidate)
        store.save_order(_make_order(
            id="ord_001", symbol="RELIANCE", side="BUY",
            fill_price=2500.0, fees_total=50.0, filled_qty=10,
            filled_at="2025-06-10T09:15:01",
        ))
        # A BUY with low fees → less negative (top winner candidate)
        store.save_order(_make_order(
            id="ord_002", symbol="TCS", side="BUY",
            fill_price=3400.0, fees_total=5.0, filled_qty=5,
            filled_at="2025-06-10T14:30:00",
        ))

        result = daily_summary(store, "2025-06-10")

        # top_winner = least negative fee impact, top_loser = most negative
        assert result.top_winner is not None
        assert result.top_loser is not None
        assert result.top_winner[0] == "TCS"       # lowest fee
        assert result.top_loser[0] == "RELIANCE"    # highest fee


# ---------------------------------------------------------------------------
# EquityCurve
# ---------------------------------------------------------------------------


class TestEquityCurve:
    """equity_curve() — running cash value across chronological fills."""

    def test_single_buy(self, tmp_path):
        """Starting capital minus (price * qty + fees) for one BUY."""
        store = PaperStore(tmp_path / "test.db")
        store.save_order(_make_order(
            id="ord_001", symbol="RELIANCE", side="BUY",
            fill_price=100.0, filled_qty=10, fees_total=5.0,
            filled_at="2025-06-10T09:15:01",
        ))

        result = equity_curve(store, starting_capital=10_000.0)

        # First point: starting capital
        assert result[0] == ("start", 10_000.0)
        # After BUY: 10_000 - (100*10) - 5 = 8_995
        assert result[1][0] == "2025-06-10"
        assert result[1][1] == pytest.approx(8_995.0)

    def test_buy_then_sell(self, tmp_path):
        """BUY reduces cash, SELL adds proceeds, fees always subtracted."""
        store = PaperStore(tmp_path / "test.db")
        store.save_order(_make_order(
            id="ord_001", side="BUY",
            fill_price=100.0, filled_qty=10, fees_total=5.0,
            filled_at="2025-06-10T09:15:01",
        ))
        store.save_order(_make_order(
            id="ord_002", side="SELL",
            fill_price=110.0, filled_qty=10, fees_total=6.0,
            filled_at="2025-06-11T10:00:00",
        ))

        result = equity_curve(store, starting_capital=10_000.0)

        assert len(result) == 3  # start + 2 orders
        assert result[0] == ("start", 10_000.0)
        # After BUY: 10_000 - 1_000 - 5 = 8_995
        assert result[1][1] == pytest.approx(8_995.0)
        # After SELL: 8_995 + 1_100 - 6 = 10_089
        assert result[2][1] == pytest.approx(10_089.0)

    def test_empty_store(self, tmp_path):
        """No orders → only the starting point."""
        store = PaperStore(tmp_path / "test.db")

        result = equity_curve(store, starting_capital=50_000.0)

        assert result == [("start", 50_000.0)]

    def test_unfilled_orders_excluded(self, tmp_path):
        """PENDING orders don't affect the equity curve."""
        store = PaperStore(tmp_path / "test.db")
        store.save_order(_make_order(
            id="ord_001", status="PENDING", filled_at=None,
        ))

        result = equity_curve(store, starting_capital=10_000.0)

        assert result == [("start", 10_000.0)]

    def test_chronological_order(self, tmp_path):
        """Equity curve follows fill timestamps, not insertion order."""
        store = PaperStore(tmp_path / "test.db")
        # Insert later fill first
        store.save_order(_make_order(
            id="ord_002", side="BUY",
            fill_price=200.0, filled_qty=5, fees_total=3.0,
            filled_at="2025-06-11T10:00:00",
        ))
        store.save_order(_make_order(
            id="ord_001", side="BUY",
            fill_price=100.0, filled_qty=10, fees_total=5.0,
            filled_at="2025-06-10T09:15:01",
        ))

        result = equity_curve(store, starting_capital=10_000.0)

        # Should be sorted chronologically regardless of insertion order
        assert result[1][0] == "2025-06-10"
        assert result[2][0] == "2025-06-11"


# ---------------------------------------------------------------------------
# ShadowComparison
# ---------------------------------------------------------------------------


class TestShadowComparison:
    """shadow_comparison() — aggregate shadow fills over recent days."""

    def test_basic_comparison(self, tmp_path):
        """fill_count, avg_divergence, divergence_cost computed correctly."""
        store = PaperStore(tmp_path / "test.db")

        # Need parent orders for FK
        store.save_order(_make_order(id="ord_001", filled_at="2025-06-10T09:15:01"))
        store.save_order(_make_order(id="ord_002", filled_at="2025-06-10T14:30:00"))

        # Shadow fill 1: paper paid 100.50, market was 100.00 → divergence 0.50%
        store.save_shadow_fill(_make_shadow_fill(
            paper_order_id="ord_001",
            paper_fill_price=100.50, market_ltp=100.00,
            divergence_pct=0.50, qty=10,
            captured_at="2025-06-10T09:15:01",
        ))
        # Shadow fill 2: paper paid 200.00, market was 199.00 → divergence 0.50%
        store.save_shadow_fill(_make_shadow_fill(
            paper_order_id="ord_002",
            paper_fill_price=200.00, market_ltp=199.00,
            divergence_pct=0.50, qty=5,
            captured_at="2025-06-10T14:30:00",
        ))

        result = shadow_comparison(store, days=7, as_of="2025-06-10T23:59:59")

        assert isinstance(result, ShadowComparison)
        assert result.fill_count == 2
        assert result.avg_divergence_pct == pytest.approx(0.50)
        # paper_total = 100.50*10 + 200.00*5 = 1005 + 1000 = 2005
        assert result.paper_total_value == pytest.approx(2005.0)
        # market_total = 100.00*10 + 199.00*5 = 1000 + 995 = 1995
        assert result.market_total_value == pytest.approx(1995.0)
        # divergence_cost = paper - market = 10
        assert result.divergence_cost == pytest.approx(10.0)
        assert result.period_days == 7

    def test_no_shadow_fills(self, tmp_path):
        """No shadow fills in period → zeros everywhere."""
        store = PaperStore(tmp_path / "test.db")

        result = shadow_comparison(store, days=7, as_of="2025-06-10T23:59:59")

        assert result.fill_count == 0
        assert result.avg_divergence_pct == pytest.approx(0.0)
        assert result.paper_total_value == pytest.approx(0.0)
        assert result.market_total_value == pytest.approx(0.0)
        assert result.divergence_cost == pytest.approx(0.0)

    def test_only_recent_fills_included(self, tmp_path):
        """Fills older than N days are excluded."""
        store = PaperStore(tmp_path / "test.db")

        store.save_order(_make_order(id="ord_old", filled_at="2025-05-01T09:00:00"))
        store.save_order(_make_order(id="ord_new", filled_at="2025-06-09T09:00:00"))

        # Old fill — outside the 7-day window from as_of
        store.save_shadow_fill(_make_shadow_fill(
            paper_order_id="ord_old",
            paper_fill_price=100.0, market_ltp=99.0,
            divergence_pct=1.0, qty=10,
            captured_at="2025-05-01T09:00:01",
        ))
        # Recent fill — inside the window
        store.save_shadow_fill(_make_shadow_fill(
            paper_order_id="ord_new",
            paper_fill_price=200.0, market_ltp=198.0,
            divergence_pct=1.0, qty=5,
            captured_at="2025-06-09T09:00:01",
        ))

        result = shadow_comparison(store, days=7, as_of="2025-06-10T23:59:59")

        assert result.fill_count == 1
        assert result.paper_total_value == pytest.approx(1000.0)   # 200*5
        assert result.market_total_value == pytest.approx(990.0)   # 198*5
