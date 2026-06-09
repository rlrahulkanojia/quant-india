"""Tests for paper.shadow — ShadowTracker divergence capture and reporting."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.paper import MarketSnapshot
from src.paper.shadow import DivergenceReport, ShadowFill, ShadowTracker
from src.paper.store import PaperStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> PaperStore:
    """Create a store with an order satisfying the FK constraint."""
    store = PaperStore(tmp_path / "shadow.db")
    store.init_portfolio(1_000_000.0)
    store.save_order(
        {
            "id": "ord_001",
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "side": "BUY",
            "order_type": "MARKET",
            "qty": 10,
            "limit_price": None,
            "fill_price": 2510.0,
            "slippage": 0.50,
            "fees_total": 12.0,
            "fees_breakdown": {},
            "status": "FILLED",
            "filled_qty": 10,
            "created_at": "2025-06-01T10:00:00",
            "filled_at": "2025-06-01T10:00:01",
        }
    )
    return store


def _make_market(ltp: float, bid: float, ask: float) -> MarketSnapshot:
    return MarketSnapshot(ltp=ltp, bid=bid, ask=ask, volume=100_000, avg_daily_volume=500_000)


# ---------------------------------------------------------------------------
# Capture — positive divergence
# ---------------------------------------------------------------------------


class TestCapturePositiveDivergence:
    """Paper fill price HIGHER than market LTP → positive divergence."""

    def test_returns_shadow_fill(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        market = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)

        fill = tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2510.0,
            qty=10,
            side="BUY",
            market=market,
        )

        assert isinstance(fill, ShadowFill)
        assert fill.paper_order_id == "ord_001"
        assert fill.paper_fill_price == pytest.approx(2510.0)
        assert fill.market_ltp == pytest.approx(2500.0)
        assert fill.market_bid == pytest.approx(2499.50)
        assert fill.market_ask == pytest.approx(2500.50)

    def test_divergence_pct_is_positive(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        market = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)

        fill = tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2510.0,
            qty=10,
            side="BUY",
            market=market,
        )

        # (2510 - 2500) / 2500 * 100 = 0.4%
        assert fill.divergence_pct == pytest.approx(0.4)

    def test_fill_persisted_in_store(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        market = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)

        tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2510.0,
            qty=10,
            side="BUY",
            market=market,
        )

        fills = store.get_shadow_fills("ord_001")
        assert len(fills) == 1
        assert fills[0]["qty"] == 10
        assert fills[0]["divergence_pct"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Capture — negative divergence
# ---------------------------------------------------------------------------


class TestCaptureNegativeDivergence:
    """Paper fill price LOWER than market LTP → negative divergence."""

    def test_divergence_pct_is_negative(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        market = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)

        fill = tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2490.0,
            qty=5,
            side="BUY",
            market=market,
        )

        # (2490 - 2500) / 2500 * 100 = -0.4%
        assert fill.divergence_pct == pytest.approx(-0.4)

    def test_fill_persisted_with_negative_divergence(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        market = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)

        tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2490.0,
            qty=5,
            side="BUY",
            market=market,
        )

        fills = store.get_shadow_fills("ord_001")
        assert len(fills) == 1
        assert fills[0]["divergence_pct"] == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# Divergence report — aggregation
# ---------------------------------------------------------------------------


class TestDivergenceReport:
    """get_report aggregates fill-level divergence into a summary."""

    @staticmethod
    def _seed_fills(store: PaperStore, tracker: ShadowTracker) -> None:
        """Insert two fills with known divergence values."""
        # Also need a second parent order for the second fill
        store.save_order(
            {
                "id": "ord_002",
                "symbol": "TCS",
                "exchange": "NSE",
                "side": "SELL",
                "order_type": "MARKET",
                "qty": 20,
                "limit_price": None,
                "fill_price": 3400.0,
                "slippage": 0.30,
                "fees_total": 8.0,
                "fees_breakdown": {},
                "status": "FILLED",
                "filled_qty": 20,
                "created_at": "2025-06-02T10:00:00",
                "filled_at": "2025-06-02T10:00:01",
            }
        )

        # Fill 1: +0.4% divergence, qty=10
        market1 = _make_market(ltp=2500.0, bid=2499.50, ask=2500.50)
        tracker.capture(
            paper_order_id="ord_001",
            paper_fill_price=2510.0,
            qty=10,
            side="BUY",
            market=market1,
        )

        # Fill 2: -1.0% divergence, qty=20
        market2 = _make_market(ltp=3400.0, bid=3399.0, ask=3401.0)
        tracker.capture(
            paper_order_id="ord_002",
            paper_fill_price=3366.0,
            qty=20,
            side="SELL",
            market=market2,
        )

    def test_report_fill_count(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        assert isinstance(report, DivergenceReport)
        assert report.fill_count == 2

    def test_report_avg_divergence(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        # avg of +0.4 and -1.0 = -0.3
        assert report.avg_divergence_pct == pytest.approx(-0.3)

    def test_report_max_divergence_is_absolute(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        # max(|+0.4|, |-1.0|) = 1.0
        assert report.max_divergence_pct == pytest.approx(1.0)

    def test_report_total_paper_value(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        # 2510*10 + 3366*20 = 25100 + 67320 = 92420
        assert report.total_paper_value == pytest.approx(92420.0)

    def test_report_total_market_value(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        # 2500*10 + 3400*20 = 25000 + 68000 = 93000
        assert report.total_market_value == pytest.approx(93000.0)

    def test_report_period_bounds(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)
        self._seed_fills(store, tracker)

        report = tracker.get_report(days=7)

        # period_start <= period_end
        assert report.period_start <= report.period_end
        assert report.period_start != ""
        assert report.period_end != ""


# ---------------------------------------------------------------------------
# Divergence report — empty
# ---------------------------------------------------------------------------


class TestDivergenceReportEmpty:
    """get_report returns zeroed report when no fills exist."""

    def test_empty_report_has_zero_fill_count(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)

        report = tracker.get_report(days=7)

        assert isinstance(report, DivergenceReport)
        assert report.fill_count == 0

    def test_empty_report_zero_divergence(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)

        report = tracker.get_report(days=7)

        assert report.avg_divergence_pct == pytest.approx(0.0)
        assert report.max_divergence_pct == pytest.approx(0.0)

    def test_empty_report_zero_values(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)

        report = tracker.get_report(days=7)

        assert report.total_paper_value == pytest.approx(0.0)
        assert report.total_market_value == pytest.approx(0.0)

    def test_empty_report_has_period_strings(self, tmp_path):
        store = _make_store(tmp_path)
        tracker = ShadowTracker(store)

        report = tracker.get_report(days=7)

        # Even with no fills, period bounds should be set (start of window → now)
        assert report.period_start != ""
        assert report.period_end != ""
