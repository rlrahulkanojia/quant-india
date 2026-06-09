"""Tests for paper trading fill simulator.

Covers:
  - Market BUY/SELL fills with bid-ask spread
  - Slippage is always positive (distance from LTP)
  - Zero bid/ask fallback to ltp * 0.0005
  - Limit BUY fills when ltp <= limit_price
  - Limit BUY doesn't fill when ltp > limit_price
  - Limit SELL fills when ltp >= limit_price
  - Limit SELL doesn't fill when ltp < limit_price
  - Partial fills: qty > 5% of avg_daily_volume → capped
  - Small qty fills fully
"""

from __future__ import annotations

import pytest

from src.paper import MarketSnapshot, OrderRequest
from src.paper.fill_simulator import FillResult, simulate_fill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(
    ltp: float = 100.0,
    bid: float = 99.5,
    ask: float = 100.5,
    volume: int = 10_000,
    avg_daily_volume: int = 100_000,
) -> MarketSnapshot:
    return MarketSnapshot(
        ltp=ltp, bid=bid, ask=ask, volume=volume, avg_daily_volume=avg_daily_volume,
    )


def _order(
    side: str = "BUY",
    order_type: str = "MARKET",
    qty: int = 10,
    limit_price: float | None = None,
) -> OrderRequest:
    return OrderRequest(
        symbol="RELIANCE",
        exchange="NSE",
        side=side,
        order_type=order_type,
        qty=qty,
        limit_price=limit_price,
    )


# ---------------------------------------------------------------------------
# Market order fills
# ---------------------------------------------------------------------------

class TestMarketOrderFills:
    """Market orders fill immediately at ltp ± half_spread."""

    def test_market_buy_fills_at_ltp_plus_half_spread(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5)
        order = _order(side="BUY", order_type="MARKET", qty=10)
        result = simulate_fill(order, snap)

        half_spread = (100.5 - 99.5) / 2  # 0.5
        assert result.filled is True
        assert result.fill_price == pytest.approx(100.0 + half_spread)  # 100.5
        assert result.filled_qty == 10

    def test_market_sell_fills_at_ltp_minus_half_spread(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5)
        order = _order(side="SELL", order_type="MARKET", qty=10)
        result = simulate_fill(order, snap)

        half_spread = (100.5 - 99.5) / 2  # 0.5
        assert result.filled is True
        assert result.fill_price == pytest.approx(100.0 - half_spread)  # 99.5
        assert result.filled_qty == 10


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

class TestSlippage:
    """Slippage is always positive — distance from LTP."""

    def test_market_buy_slippage_positive(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5)
        order = _order(side="BUY", order_type="MARKET", qty=10)
        result = simulate_fill(order, snap)

        assert result.slippage == pytest.approx(0.5)  # fill_price - ltp
        assert result.slippage > 0

    def test_market_sell_slippage_positive(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5)
        order = _order(side="SELL", order_type="MARKET", qty=10)
        result = simulate_fill(order, snap)

        assert result.slippage == pytest.approx(0.5)  # ltp - fill_price
        assert result.slippage > 0


# ---------------------------------------------------------------------------
# Zero bid/ask fallback
# ---------------------------------------------------------------------------

class TestZeroBidAskFallback:
    """When bid or ask is 0, half_spread falls back to ltp * 0.0005."""

    def test_zero_bid_ask_uses_fallback_spread(self):
        snap = _snap(ltp=2000.0, bid=0.0, ask=0.0)
        order = _order(side="BUY", order_type="MARKET", qty=5)
        result = simulate_fill(order, snap)

        fallback_half_spread = 2000.0 * 0.0005  # 1.0
        assert result.fill_price == pytest.approx(2000.0 + fallback_half_spread)
        assert result.slippage == pytest.approx(fallback_half_spread)

    def test_zero_bid_only_uses_fallback(self):
        snap = _snap(ltp=1000.0, bid=0.0, ask=1001.0)
        order = _order(side="SELL", order_type="MARKET", qty=5)
        result = simulate_fill(order, snap)

        fallback_half_spread = 1000.0 * 0.0005  # 0.5
        assert result.fill_price == pytest.approx(1000.0 - fallback_half_spread)


# ---------------------------------------------------------------------------
# Limit order fills
# ---------------------------------------------------------------------------

class TestLimitOrderFills:
    """Limit orders fill at limit_price when market conditions are met."""

    def test_limit_buy_fills_when_ltp_at_limit(self):
        snap = _snap(ltp=500.0)
        order = _order(side="BUY", order_type="LIMIT", qty=10, limit_price=500.0)
        result = simulate_fill(order, snap)

        assert result.filled is True
        assert result.fill_price == pytest.approx(500.0)
        assert result.slippage == pytest.approx(0.0)

    def test_limit_buy_fills_when_ltp_below_limit(self):
        snap = _snap(ltp=490.0)
        order = _order(side="BUY", order_type="LIMIT", qty=10, limit_price=500.0)
        result = simulate_fill(order, snap)

        assert result.filled is True
        assert result.fill_price == pytest.approx(500.0)

    def test_limit_buy_no_fill_when_ltp_above_limit(self):
        snap = _snap(ltp=510.0)
        order = _order(side="BUY", order_type="LIMIT", qty=10, limit_price=500.0)
        result = simulate_fill(order, snap)

        assert result.filled is False
        assert result.filled_qty == 0
        assert result.reason is not None

    def test_limit_sell_fills_when_ltp_at_limit(self):
        snap = _snap(ltp=500.0)
        order = _order(side="SELL", order_type="LIMIT", qty=10, limit_price=500.0)
        result = simulate_fill(order, snap)

        assert result.filled is True
        assert result.fill_price == pytest.approx(500.0)
        assert result.slippage == pytest.approx(0.0)

    def test_limit_sell_no_fill_when_ltp_below_limit(self):
        snap = _snap(ltp=490.0)
        order = _order(side="SELL", order_type="LIMIT", qty=10, limit_price=500.0)
        result = simulate_fill(order, snap)

        assert result.filled is False
        assert result.filled_qty == 0
        assert result.reason is not None


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

class TestPartialFills:
    """Qty > 5% of avg_daily_volume is capped; small qty fills fully."""

    def test_large_qty_capped_to_five_pct_of_adv(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5, avg_daily_volume=100_000)
        order = _order(side="BUY", order_type="MARKET", qty=10_000)
        result = simulate_fill(order, snap)

        max_fillable = int(100_000 * 0.05)  # 5000
        assert result.filled is True
        assert result.filled_qty == max_fillable
        assert result.filled_qty < order.qty

    def test_small_qty_fills_fully(self):
        snap = _snap(ltp=100.0, bid=99.5, ask=100.5, avg_daily_volume=100_000)
        order = _order(side="BUY", order_type="MARKET", qty=100)
        result = simulate_fill(order, snap)

        assert result.filled is True
        assert result.filled_qty == 100
