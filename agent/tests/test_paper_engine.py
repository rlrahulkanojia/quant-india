"""Tests for PaperTradingEngine — the orchestrator that composes all paper modules.

Uses tmp_path fixture for isolated SQLite databases.
Starting capital: 1_000_000 for all tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure agent/ is on sys.path
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import pytest

from src.paper import MarketSnapshot
from src.paper.engine import PaperTradingEngine, PortfolioSummary, TradeResult
from src.paper.fees import FeeBreakdown, FeeConfig, calculate_fees
from src.paper.portfolio import PositionUpdate
from src.paper.shadow import ShadowFill

STARTING_CAPITAL = 1_000_000.0


@pytest.fixture()
def engine(tmp_path: Path) -> PaperTradingEngine:
    """Create a fresh PaperTradingEngine with default config."""
    db = str(tmp_path / "paper.db")
    return PaperTradingEngine(db_path=db, starting_capital=STARTING_CAPITAL)


@pytest.fixture()
def market() -> MarketSnapshot:
    """Typical mid-cap NSE snapshot."""
    return MarketSnapshot(
        ltp=1500.0,
        bid=1499.50,
        ask=1500.50,
        volume=500_000,
        avg_daily_volume=2_000_000,
    )


# ------------------------------------------------------------------
# 1. Full BUY flow
# ------------------------------------------------------------------


class TestBuyFlow:
    """Market BUY → position opened, cash reduced, fees deducted, shadow captured."""

    def test_market_buy_returns_filled_trade_result(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        assert isinstance(result, TradeResult)
        assert result.filled is True
        assert result.filled_qty == 100
        assert result.fill_price > 0.0
        assert result.order_id.startswith("PAPER-RELIANCE-BUY-100-")

    def test_market_buy_fill_price_is_above_ltp(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        """BUY market order fills at LTP + half-spread."""
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )
        # Half-spread = (1500.50 - 1499.50) / 2 = 0.50
        expected_price = 1500.0 + 0.50
        assert result.fill_price == pytest.approx(expected_price)

    def test_market_buy_opens_position(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        assert isinstance(result.position_update, PositionUpdate)
        assert result.position_update.action == "opened"
        assert result.position_update.old_qty == 0
        assert result.position_update.new_qty == 100

    def test_market_buy_deducts_cash_plus_fees(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        trade_value = result.fill_price * result.filled_qty
        fees = result.fees.total
        expected_cash = STARTING_CAPITAL - trade_value - fees

        summary = engine.get_summary(current_prices={"RELIANCE": market.ltp})
        assert summary.cash == pytest.approx(expected_cash)

    def test_market_buy_fees_are_nonzero(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        assert isinstance(result.fees, FeeBreakdown)
        assert result.fees.total > 0
        # BUY side: stamp duty should be charged
        assert result.fees.stamp_duty > 0
        # BUY side intraday: no STT
        assert result.fees.stt == 0.0

    def test_market_buy_captures_shadow_fill(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        assert result.shadow_fill is not None
        assert isinstance(result.shadow_fill, ShadowFill)
        assert result.shadow_fill.paper_fill_price == result.fill_price
        assert result.shadow_fill.market_ltp == market.ltp

    def test_market_buy_slippage_is_positive(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        assert result.slippage > 0


# ------------------------------------------------------------------
# 2. Sell to close — buy then sell → realized P&L
# ------------------------------------------------------------------


class TestSellToClose:
    """BUY then SELL: position fully closed, realized P&L computed."""

    def test_sell_closes_position(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        # Open position
        buy = engine.execute_order(
            symbol="TCS",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=50,
            market=market,
        )
        assert buy.filled is True

        # Sell at higher price
        sell_market = MarketSnapshot(
            ltp=1550.0,
            bid=1549.50,
            ask=1550.50,
            volume=500_000,
            avg_daily_volume=2_000_000,
        )
        sell = engine.execute_order(
            symbol="TCS",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=50,
            market=sell_market,
        )

        assert sell.filled is True
        assert sell.position_update.action == "closed"
        assert sell.position_update.new_qty == 0

    def test_sell_realized_pnl_positive_when_price_rises(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        # Buy at ~1500.50 (ltp + half-spread)
        buy = engine.execute_order(
            symbol="TCS",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=50,
            market=market,
        )

        # Sell at higher price → sell fills at ltp - half-spread
        sell_market = MarketSnapshot(
            ltp=1600.0,
            bid=1599.50,
            ask=1600.50,
            volume=500_000,
            avg_daily_volume=2_000_000,
        )
        sell = engine.execute_order(
            symbol="TCS",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=50,
            market=sell_market,
        )

        # Realized P&L = (sell_fill_price - buy_fill_price) * qty
        # sell_fill_price = 1600.0 - 0.50 = 1599.50
        # buy_fill_price  = 1500.0 + 0.50 = 1500.50
        # pnl = (1599.50 - 1500.50) * 50 = 4950.0
        assert sell.position_update.realized_pnl == pytest.approx(4950.0)

    def test_no_positions_remain_after_sell(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="INFY",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )
        engine.execute_order(
            symbol="INFY",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        summary = engine.get_summary(current_prices={})
        assert len(summary.positions) == 0

    def test_sell_deducts_sell_side_stt(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="HDFCBANK",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        sell = engine.execute_order(
            symbol="HDFCBANK",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        # Intraday SELL: STT should be charged
        assert sell.fees.stt > 0
        # SELL side: no stamp duty
        assert sell.fees.stamp_duty == 0.0


# ------------------------------------------------------------------
# 3. Unfilled limit order
# ------------------------------------------------------------------


class TestUnfilledLimitOrder:
    """Limit order not hit → filled=False, no state changes."""

    def test_limit_buy_above_ltp_is_unfilled(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        """Limit BUY at 1400 when LTP is 1500 → not filled."""
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        assert result.filled is False
        assert result.filled_qty == 0
        assert result.fill_price == 0.0

    def test_unfilled_order_zero_fees(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        assert result.fees.total == 0.0
        assert result.fees.stt == 0.0
        assert result.fees.exchange_charges == 0.0
        assert result.fees.stamp_duty == 0.0

    def test_unfilled_order_no_position_created(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="WIPRO",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        summary = engine.get_summary(current_prices={})
        assert len(summary.positions) == 0

    def test_unfilled_order_cash_unchanged(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="WIPRO",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        summary = engine.get_summary(current_prices={})
        assert summary.cash == pytest.approx(STARTING_CAPITAL)

    def test_unfilled_order_no_shadow_fill(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        assert result.shadow_fill is None

    def test_unfilled_order_position_update_sentinel(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="LIMIT",
            qty=100,
            market=market,
            limit_price=1400.0,
        )

        assert result.position_update.action == "no_fill"
        assert result.position_update.old_qty == 0
        assert result.position_update.new_qty == 0
        assert result.position_update.realized_pnl == 0.0


# ------------------------------------------------------------------
# 4. Portfolio summary with updated prices
# ------------------------------------------------------------------


class TestPortfolioSummary:
    """get_summary() returns correct totals with current prices."""

    def test_summary_after_buy(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        summary = engine.get_summary(current_prices={"RELIANCE": 1600.0})

        assert isinstance(summary, PortfolioSummary)
        assert len(summary.positions) == 1
        assert summary.positions[0]["symbol"] == "RELIANCE"
        assert summary.positions[0]["qty"] == 100

    def test_summary_unrealized_pnl_when_price_rises(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        buy = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        # Price rose from buy price (1500.50) to 1600
        summary = engine.get_summary(current_prices={"RELIANCE": 1600.0})

        expected_unrealized = (1600.0 - buy.fill_price) * 100
        assert summary.unrealized_pnl == pytest.approx(expected_unrealized)

    def test_summary_total_value(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        summary = engine.get_summary(current_prices={"RELIANCE": 1600.0})

        assert summary.total_value == pytest.approx(
            summary.cash + summary.unrealized_pnl
        )

    def test_summary_realized_pnl_after_round_trip(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        engine.execute_order(
            symbol="SBIN",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        sell_market = MarketSnapshot(
            ltp=1600.0,
            bid=1599.50,
            ask=1600.50,
            volume=500_000,
            avg_daily_volume=2_000_000,
        )
        engine.execute_order(
            symbol="SBIN",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=sell_market,
        )

        summary = engine.get_summary(current_prices={})
        assert summary.realized_pnl > 0  # price went up, should have profit

    def test_summary_total_fees_paid(
        self, engine: PaperTradingEngine, market: MarketSnapshot
    ) -> None:
        r1 = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        r2 = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=market,
        )

        summary = engine.get_summary(current_prices={})
        expected_fees = r1.fees.total + r2.fees.total
        assert summary.total_fees_paid == pytest.approx(expected_fees)

    def test_summary_empty_portfolio(
        self, engine: PaperTradingEngine
    ) -> None:
        summary = engine.get_summary(current_prices={})
        assert summary.cash == pytest.approx(STARTING_CAPITAL)
        assert len(summary.positions) == 0
        assert summary.unrealized_pnl == 0.0
        assert summary.realized_pnl == 0.0
        assert summary.total_value == pytest.approx(STARTING_CAPITAL)
        assert summary.total_fees_paid == 0.0
