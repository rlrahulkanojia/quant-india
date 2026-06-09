"""Tests for paper trading shared types and Indian fee calculator.

Covers:
  - MarketSnapshot & OrderRequest dataclasses
  - FeeConfig defaults and overrides
  - calculate_fees for equity_intraday, equity_delivery, commodity
  - Edge cases: zero trade value, custom brokerage, frozen enforcement
"""

from __future__ import annotations

import pytest

from src.paper import MarketSnapshot, OrderRequest
from src.paper.fees import FeeBreakdown, FeeConfig, calculate_fees


# ---------------------------------------------------------------------------
# Shared type sanity checks
# ---------------------------------------------------------------------------

class TestSharedTypes:
    def test_market_snapshot_fields(self):
        snap = MarketSnapshot(ltp=100.0, bid=99.5, ask=100.5, volume=1000, avg_daily_volume=50000)
        assert snap.ltp == 100.0
        assert snap.bid == 99.5
        assert snap.ask == 100.5
        assert snap.volume == 1000
        assert snap.avg_daily_volume == 50000

    def test_order_request_defaults(self):
        req = OrderRequest(symbol="RELIANCE", exchange="NSE", side="BUY", order_type="MARKET", qty=10)
        assert req.limit_price is None

    def test_order_request_frozen(self):
        req = OrderRequest(symbol="RELIANCE", exchange="NSE", side="BUY", order_type="LIMIT", qty=10, limit_price=2500.0)
        with pytest.raises(AttributeError):
            req.qty = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fee calculator — equity intraday
# ---------------------------------------------------------------------------

class TestEquityIntradayFees:
    """Equity intraday: STT on SELL only, stamp on BUY only."""

    def test_buy_side(self):
        fb = calculate_fees(100_000, "BUY", "equity_intraday")
        assert fb.stt == 0.0                          # no STT on intraday BUY
        assert fb.exchange_charges == pytest.approx(3.45, abs=0.01)  # 0.00345%
        assert fb.sebi == pytest.approx(0.10, abs=0.01)              # 10/crore
        assert fb.gst == 0.0                           # no brokerage → no GST
        assert fb.stamp_duty == pytest.approx(3.0, abs=0.01)         # 0.003%
        assert fb.brokerage == 0.0

    def test_sell_side(self):
        fb = calculate_fees(100_000, "SELL", "equity_intraday")
        assert fb.stt == pytest.approx(25.0, abs=0.01)              # 0.025%
        assert fb.exchange_charges == pytest.approx(3.45, abs=0.01)
        assert fb.sebi == pytest.approx(0.10, abs=0.01)
        assert fb.stamp_duty == 0.0                    # no stamp on SELL
        assert fb.brokerage == 0.0

    def test_total_is_sum_of_components(self):
        fb = calculate_fees(100_000, "BUY", "equity_intraday")
        expected = fb.stt + fb.exchange_charges + fb.sebi + fb.gst + fb.stamp_duty + fb.brokerage
        assert fb.total == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Fee calculator — equity delivery
# ---------------------------------------------------------------------------

class TestEquityDeliveryFees:
    """Equity delivery: STT on both sides, stamp on BUY only."""

    def test_buy_side(self):
        fb = calculate_fees(100_000, "BUY", "equity_delivery")
        assert fb.stt == pytest.approx(100.0, abs=0.01)             # 0.1%
        assert fb.stamp_duty == pytest.approx(15.0, abs=0.01)       # 0.015%

    def test_sell_side(self):
        fb = calculate_fees(100_000, "SELL", "equity_delivery")
        assert fb.stt == pytest.approx(100.0, abs=0.01)             # delivery both sides
        assert fb.stamp_duty == 0.0                                  # stamp BUY only


# ---------------------------------------------------------------------------
# Fee calculator — commodity
# ---------------------------------------------------------------------------

class TestCommodityFees:
    """Commodity: CTT on SELL only (stored in stt field), stamp on BUY only."""

    def test_sell_side(self):
        fb = calculate_fees(100_000, "SELL", "commodity")
        assert fb.stt == pytest.approx(10.0, abs=0.01)              # CTT 0.01%
        assert fb.exchange_charges == pytest.approx(2.60, abs=0.01)  # 0.0026%
        assert fb.stamp_duty == 0.0

    def test_buy_side(self):
        fb = calculate_fees(100_000, "BUY", "commodity")
        assert fb.stt == 0.0                                         # no CTT on BUY
        assert fb.stamp_duty == pytest.approx(2.0, abs=0.01)        # 0.002%


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_trade_value(self):
        fb = calculate_fees(0, "BUY", "equity_intraday")
        assert fb.stt == 0.0
        assert fb.exchange_charges == 0.0
        assert fb.sebi == 0.0
        assert fb.gst == 0.0
        assert fb.stamp_duty == 0.0
        assert fb.brokerage == 0.0
        assert fb.total == 0.0

    def test_custom_brokerage_gst(self):
        fb = calculate_fees(100_000, "BUY", "equity_intraday", brokerage=20.0)
        assert fb.brokerage == 20.0
        assert fb.gst == pytest.approx(3.60, abs=0.01)  # 18% of 20

    def test_custom_fee_config(self):
        cfg = FeeConfig(stt_intraday_sell=0.001)         # override STT rate
        fb = calculate_fees(100_000, "SELL", "equity_intraday", config=cfg)
        assert fb.stt == pytest.approx(100.0, abs=0.01)  # 0.1% instead of 0.025%

    def test_fee_breakdown_is_frozen(self):
        fb = calculate_fees(100_000, "BUY", "equity_intraday")
        with pytest.raises(AttributeError):
            fb.total = 999.0  # type: ignore[misc]
