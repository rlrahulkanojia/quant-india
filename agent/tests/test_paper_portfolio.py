"""Tests for Portfolio — position management and P&L tracking.

Covers:
  - Open new LONG position (BUY when no existing position)
  - Add to existing LONG (BUY more shares — verify weighted avg price)
  - Reduce LONG partially (SELL some shares — verify realized P&L)
  - Close LONG fully (SELL all shares — position deleted, P&L realized)
  - Open new SHORT position (SELL when no existing position)
  - Cash balance updates correctly (deducted on BUY, increased on SELL,
    fees always deducted)
"""

from __future__ import annotations

import pytest

from src.paper import OrderRequest
from src.paper.fees import FeeBreakdown
from src.paper.fill_simulator import FillResult
from src.paper.portfolio import Portfolio, Position, PositionUpdate
from src.paper.store import PaperStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fill(fill_price: float, filled_qty: int, slippage: float = 0.0) -> FillResult:
    """Create a FillResult for testing."""
    return FillResult(
        filled=True,
        fill_price=fill_price,
        filled_qty=filled_qty,
        slippage=slippage,
    )


def make_order(
    symbol: str, exchange: str, side: str, qty: int
) -> OrderRequest:
    """Create a MARKET OrderRequest for testing."""
    return OrderRequest(
        symbol=symbol,
        exchange=exchange,
        side=side,
        order_type="MARKET",
        qty=qty,
    )


def make_fees(
    total: float,
    stt: float = 0.0,
    exchange_charges: float = 0.0,
    sebi: float = 0.0,
    gst: float = 0.0,
    stamp_duty: float = 0.0,
    brokerage: float = 0.0,
) -> FeeBreakdown:
    """Create a FeeBreakdown for testing."""
    return FeeBreakdown(
        stt=stt,
        exchange_charges=exchange_charges,
        sebi=sebi,
        gst=gst,
        stamp_duty=stamp_duty,
        brokerage=brokerage,
        total=total,
    )


@pytest.fixture
def portfolio(tmp_path) -> Portfolio:
    """Create an initialised Portfolio with 1,000,000 starting capital."""
    store = PaperStore(tmp_path / "test.db")
    p = Portfolio(store)
    p.init(starting_capital=1_000_000.0)
    return p


# ---------------------------------------------------------------------------
# Open new LONG position
# ---------------------------------------------------------------------------


class TestOpenLong:
    """BUY when no existing position opens a new LONG."""

    def test_opens_long_position(self, portfolio: Portfolio):
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        fees = make_fees(total=10.0)

        update = portfolio.process_fill(fill, order, fees)

        assert update.action == "opened"
        assert update.new_qty == 50
        assert update.old_qty == 0
        assert update.realized_pnl == 0.0

    def test_position_stored_correctly(self, portfolio: Portfolio):
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        fees = make_fees(total=10.0)

        portfolio.process_fill(fill, order, fees)

        positions = portfolio.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "RELIANCE"
        assert pos.exchange == "NSE"
        assert pos.qty == 50
        assert pos.avg_price == 100.0
        assert pos.side == "LONG"

    def test_position_id_returned(self, portfolio: Portfolio):
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        fees = make_fees(total=10.0)

        update = portfolio.process_fill(fill, order, fees)

        assert update.position_id > 0


# ---------------------------------------------------------------------------
# Add to existing LONG
# ---------------------------------------------------------------------------


class TestAddToLong:
    """BUY more shares when already LONG — verify weighted average price."""

    def test_weighted_avg_price(self, portfolio: Portfolio):
        order = make_order("INFY", "NSE", "BUY", 100)
        fees = make_fees(total=5.0)

        # First buy: 100 @ 1500
        fill1 = make_fill(fill_price=1500.0, filled_qty=100)
        portfolio.process_fill(fill1, order, fees)

        # Second buy: 50 @ 1600
        fill2 = make_fill(fill_price=1600.0, filled_qty=50)
        order2 = make_order("INFY", "NSE", "BUY", 50)
        update = portfolio.process_fill(fill2, order2, fees)

        assert update.action == "added"
        assert update.old_qty == 100
        assert update.new_qty == 150
        assert update.realized_pnl == 0.0

        # Weighted avg: (1500*100 + 1600*50) / 150 = 230000/150 = 1533.33...
        positions = portfolio.get_positions()
        assert len(positions) == 1
        assert positions[0].avg_price == pytest.approx(1533.3333, rel=1e-3)
        assert positions[0].qty == 150

    def test_same_position_id(self, portfolio: Portfolio):
        """Adding to a LONG should keep the same position id."""
        order = make_order("INFY", "NSE", "BUY", 100)
        fees = make_fees(total=5.0)

        fill1 = make_fill(fill_price=1500.0, filled_qty=100)
        update1 = portfolio.process_fill(fill1, order, fees)

        fill2 = make_fill(fill_price=1600.0, filled_qty=50)
        order2 = make_order("INFY", "NSE", "BUY", 50)
        update2 = portfolio.process_fill(fill2, order2, fees)

        assert update2.position_id == update1.position_id


# ---------------------------------------------------------------------------
# Reduce LONG partially
# ---------------------------------------------------------------------------


class TestReduceLongPartially:
    """SELL some shares of a LONG position — verify realized P&L."""

    def test_partial_sell_reduces_qty(self, portfolio: Portfolio):
        # Open LONG: 100 @ 200
        fill_buy = make_fill(fill_price=200.0, filled_qty=100)
        order_buy = make_order("TCS", "NSE", "BUY", 100)
        fees = make_fees(total=5.0)
        portfolio.process_fill(fill_buy, order_buy, fees)

        # Sell 40 @ 250
        fill_sell = make_fill(fill_price=250.0, filled_qty=40)
        order_sell = make_order("TCS", "NSE", "SELL", 40)
        update = portfolio.process_fill(fill_sell, order_sell, fees)

        assert update.action == "reduced"
        assert update.old_qty == 100
        assert update.new_qty == 60
        # Realized P&L = (250 - 200) * 40 = 2000
        assert update.realized_pnl == pytest.approx(2000.0)

    def test_partial_sell_preserves_avg_price(self, portfolio: Portfolio):
        """Partial sell should NOT change avg_price of remaining shares."""
        fill_buy = make_fill(fill_price=200.0, filled_qty=100)
        order_buy = make_order("TCS", "NSE", "BUY", 100)
        fees = make_fees(total=5.0)
        portfolio.process_fill(fill_buy, order_buy, fees)

        fill_sell = make_fill(fill_price=250.0, filled_qty=40)
        order_sell = make_order("TCS", "NSE", "SELL", 40)
        portfolio.process_fill(fill_sell, order_sell, fees)

        positions = portfolio.get_positions()
        assert len(positions) == 1
        assert positions[0].avg_price == pytest.approx(200.0)
        assert positions[0].qty == 60

    def test_losing_partial_sell(self, portfolio: Portfolio):
        """Selling at a loss should produce negative realized P&L."""
        fill_buy = make_fill(fill_price=500.0, filled_qty=100)
        order_buy = make_order("HDFC", "NSE", "BUY", 100)
        fees = make_fees(total=5.0)
        portfolio.process_fill(fill_buy, order_buy, fees)

        # Sell 30 @ 450 (loss)
        fill_sell = make_fill(fill_price=450.0, filled_qty=30)
        order_sell = make_order("HDFC", "NSE", "SELL", 30)
        update = portfolio.process_fill(fill_sell, order_sell, fees)

        # Realized P&L = (450 - 500) * 30 = -1500
        assert update.realized_pnl == pytest.approx(-1500.0)


# ---------------------------------------------------------------------------
# Close LONG fully
# ---------------------------------------------------------------------------


class TestCloseLongFully:
    """SELL all shares — position deleted, P&L realized."""

    def test_full_close_deletes_position(self, portfolio: Portfolio):
        fill_buy = make_fill(fill_price=300.0, filled_qty=50)
        order_buy = make_order("WIPRO", "NSE", "BUY", 50)
        fees = make_fees(total=5.0)
        portfolio.process_fill(fill_buy, order_buy, fees)

        # Close: sell all 50 @ 350
        fill_sell = make_fill(fill_price=350.0, filled_qty=50)
        order_sell = make_order("WIPRO", "NSE", "SELL", 50)
        update = portfolio.process_fill(fill_sell, order_sell, fees)

        assert update.action == "closed"
        assert update.old_qty == 50
        assert update.new_qty == 0
        assert update.realized_pnl == pytest.approx(2500.0)  # (350-300)*50

        positions = portfolio.get_positions()
        assert len(positions) == 0

    def test_full_close_returns_position_id(self, portfolio: Portfolio):
        fill_buy = make_fill(fill_price=300.0, filled_qty=50)
        order_buy = make_order("WIPRO", "NSE", "BUY", 50)
        fees = make_fees(total=5.0)
        update_open = portfolio.process_fill(fill_buy, order_buy, fees)

        fill_sell = make_fill(fill_price=350.0, filled_qty=50)
        order_sell = make_order("WIPRO", "NSE", "SELL", 50)
        update_close = portfolio.process_fill(fill_sell, order_sell, fees)

        assert update_close.position_id == update_open.position_id


# ---------------------------------------------------------------------------
# Open new SHORT position
# ---------------------------------------------------------------------------


class TestOpenShort:
    """SELL when no existing position opens a new SHORT."""

    def test_opens_short_position(self, portfolio: Portfolio):
        fill = make_fill(fill_price=800.0, filled_qty=25)
        order = make_order("SBIN", "NSE", "SELL", 25)
        fees = make_fees(total=8.0)

        update = portfolio.process_fill(fill, order, fees)

        assert update.action == "opened"
        assert update.new_qty == 25
        assert update.old_qty == 0
        assert update.realized_pnl == 0.0

    def test_short_position_stored(self, portfolio: Portfolio):
        fill = make_fill(fill_price=800.0, filled_qty=25)
        order = make_order("SBIN", "NSE", "SELL", 25)
        fees = make_fees(total=8.0)

        portfolio.process_fill(fill, order, fees)

        positions = portfolio.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "SBIN"
        assert pos.side == "SHORT"
        assert pos.qty == 25
        assert pos.avg_price == 800.0

    def test_close_short_with_buy(self, portfolio: Portfolio):
        """BUY to cover a SHORT — realized P&L should be positive if price fell."""
        # Open SHORT: 25 @ 800
        fill_sell = make_fill(fill_price=800.0, filled_qty=25)
        order_sell = make_order("SBIN", "NSE", "SELL", 25)
        fees = make_fees(total=8.0)
        portfolio.process_fill(fill_sell, order_sell, fees)

        # Close SHORT: BUY 25 @ 750 (price fell = profit for short)
        fill_buy = make_fill(fill_price=750.0, filled_qty=25)
        order_buy = make_order("SBIN", "NSE", "BUY", 25)
        update = portfolio.process_fill(fill_buy, order_buy, fees)

        assert update.action == "closed"
        assert update.new_qty == 0
        # Short P&L = (avg_price - fill_price) * qty = (800 - 750) * 25 = 1250
        assert update.realized_pnl == pytest.approx(1250.0)
        assert len(portfolio.get_positions()) == 0

    def test_add_to_short(self, portfolio: Portfolio):
        """SELL more when already SHORT — verify weighted avg price."""
        fees = make_fees(total=5.0)

        # Open SHORT: 50 @ 400
        fill1 = make_fill(fill_price=400.0, filled_qty=50)
        order1 = make_order("ITC", "NSE", "SELL", 50)
        portfolio.process_fill(fill1, order1, fees)

        # Add to SHORT: 30 @ 420
        fill2 = make_fill(fill_price=420.0, filled_qty=30)
        order2 = make_order("ITC", "NSE", "SELL", 30)
        update = portfolio.process_fill(fill2, order2, fees)

        assert update.action == "added"
        assert update.old_qty == 50
        assert update.new_qty == 80

        positions = portfolio.get_positions()
        assert len(positions) == 1
        # Weighted avg: (400*50 + 420*30) / 80 = 32600/80 = 407.5
        assert positions[0].avg_price == pytest.approx(407.5)

    def test_reduce_short_partially(self, portfolio: Portfolio):
        """BUY some shares to partially cover SHORT."""
        fees = make_fees(total=5.0)

        # Open SHORT: 100 @ 600
        fill_sell = make_fill(fill_price=600.0, filled_qty=100)
        order_sell = make_order("TATAMOTORS", "NSE", "SELL", 100)
        portfolio.process_fill(fill_sell, order_sell, fees)

        # Partially cover: BUY 40 @ 580 (price fell, profit)
        fill_buy = make_fill(fill_price=580.0, filled_qty=40)
        order_buy = make_order("TATAMOTORS", "NSE", "BUY", 40)
        update = portfolio.process_fill(fill_buy, order_buy, fees)

        assert update.action == "reduced"
        assert update.old_qty == 100
        assert update.new_qty == 60
        # Short P&L = (600 - 580) * 40 = 800
        assert update.realized_pnl == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# Cash balance updates
# ---------------------------------------------------------------------------


class TestCashBalance:
    """Cash balance: deducted on BUY, increased on SELL, fees always deducted."""

    def test_initial_balance(self, portfolio: Portfolio):
        assert portfolio.get_cash_balance() == pytest.approx(1_000_000.0)

    def test_buy_deducts_cash(self, portfolio: Portfolio):
        """BUY should deduct (fill_price * qty + fees) from cash."""
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        fees = make_fees(total=10.0)

        portfolio.process_fill(fill, order, fees)

        # Cost = 100*50 + 10 = 5010
        expected = 1_000_000.0 - 5010.0
        assert portfolio.get_cash_balance() == pytest.approx(expected)

    def test_sell_to_close_long_increases_cash(self, portfolio: Portfolio):
        """SELL of LONG position should credit proceeds minus fees."""
        fees = make_fees(total=10.0)

        # Buy 50 @ 100 => cash -= 5010
        fill_buy = make_fill(fill_price=100.0, filled_qty=50)
        order_buy = make_order("RELIANCE", "NSE", "BUY", 50)
        portfolio.process_fill(fill_buy, order_buy, fees)

        # Sell 50 @ 120 => cash += (120*50 - 10) = 5990
        fill_sell = make_fill(fill_price=120.0, filled_qty=50)
        order_sell = make_order("RELIANCE", "NSE", "SELL", 50)
        portfolio.process_fill(fill_sell, order_sell, fees)

        # Net: 1_000_000 - 5010 + 5990 = 1_000_980
        expected = 1_000_000.0 - 5010.0 + 5990.0
        assert portfolio.get_cash_balance() == pytest.approx(expected)

    def test_open_short_credits_cash(self, portfolio: Portfolio):
        """Opening a SHORT should credit (fill_price * qty - fees) to cash."""
        fill = make_fill(fill_price=500.0, filled_qty=20)
        order = make_order("SBIN", "NSE", "SELL", 20)
        fees = make_fees(total=15.0)

        portfolio.process_fill(fill, order, fees)

        # Credit = 500*20 - 15 = 9985
        expected = 1_000_000.0 + 9985.0
        assert portfolio.get_cash_balance() == pytest.approx(expected)

    def test_close_short_deducts_cash(self, portfolio: Portfolio):
        """Closing a SHORT via BUY should deduct cost + fees."""
        fees = make_fees(total=10.0)

        # Open SHORT: 20 @ 500 => cash += 500*20 - 10 = 9990
        fill_sell = make_fill(fill_price=500.0, filled_qty=20)
        order_sell = make_order("SBIN", "NSE", "SELL", 20)
        portfolio.process_fill(fill_sell, order_sell, fees)

        # Close SHORT: BUY 20 @ 480 => cash -= 480*20 + 10 = 9610
        fill_buy = make_fill(fill_price=480.0, filled_qty=20)
        order_buy = make_order("SBIN", "NSE", "BUY", 20)
        portfolio.process_fill(fill_buy, order_buy, fees)

        # Net: 1_000_000 + 9990 - 9610 = 1_000_380
        expected = 1_000_000.0 + 9990.0 - 9610.0
        assert portfolio.get_cash_balance() == pytest.approx(expected)

    def test_fees_always_deducted(self, portfolio: Portfolio):
        """Two trades with different fees — total fees = sum of both."""
        fees1 = make_fees(total=25.0)
        fees2 = make_fees(total=35.0)

        fill1 = make_fill(fill_price=100.0, filled_qty=10)
        order1 = make_order("A", "NSE", "BUY", 10)
        portfolio.process_fill(fill1, order1, fees1)

        fill2 = make_fill(fill_price=200.0, filled_qty=5)
        order2 = make_order("B", "NSE", "BUY", 5)
        portfolio.process_fill(fill2, order2, fees2)

        # Cash = 1M - (100*10+25) - (200*5+35) = 1M - 1025 - 1035 = 997940
        expected = 1_000_000.0 - 1025.0 - 1035.0
        assert portfolio.get_cash_balance() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Unrealized P&L and total value
# ---------------------------------------------------------------------------


class TestUnrealizedPnl:
    """get_unrealized_pnl and get_total_value calculations."""

    def test_unrealized_pnl_long(self, portfolio: Portfolio):
        fees = make_fees(total=5.0)
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        portfolio.process_fill(fill, order, fees)

        # Current price 120 => unrealized = (120 - 100) * 50 = 1000
        pnl = portfolio.get_unrealized_pnl({"RELIANCE": 120.0})
        assert pnl == pytest.approx(1000.0)

    def test_unrealized_pnl_short(self, portfolio: Portfolio):
        fees = make_fees(total=5.0)
        fill = make_fill(fill_price=500.0, filled_qty=20)
        order = make_order("SBIN", "NSE", "SELL", 20)
        portfolio.process_fill(fill, order, fees)

        # Current price 480 => unrealized = (500 - 480) * 20 = 400
        pnl = portfolio.get_unrealized_pnl({"SBIN": 480.0})
        assert pnl == pytest.approx(400.0)

    def test_unrealized_pnl_multiple_positions(self, portfolio: Portfolio):
        fees = make_fees(total=5.0)

        fill1 = make_fill(fill_price=100.0, filled_qty=50)
        order1 = make_order("A", "NSE", "BUY", 50)
        portfolio.process_fill(fill1, order1, fees)

        fill2 = make_fill(fill_price=200.0, filled_qty=30)
        order2 = make_order("B", "NSE", "SELL", 30)
        portfolio.process_fill(fill2, order2, fees)

        # A LONG: (150 - 100)*50 = 2500
        # B SHORT: (200 - 180)*30 = 600
        pnl = portfolio.get_unrealized_pnl({"A": 150.0, "B": 180.0})
        assert pnl == pytest.approx(3100.0)

    def test_total_value(self, portfolio: Portfolio):
        fees = make_fees(total=5.0)
        fill = make_fill(fill_price=100.0, filled_qty=50)
        order = make_order("RELIANCE", "NSE", "BUY", 50)
        portfolio.process_fill(fill, order, fees)

        # Cash = 1M - (100*50 + 5) = 994995
        # Unrealized = (120-100)*50 = 1000
        # Total = 994995 + 1000 = 995995
        total = portfolio.get_total_value({"RELIANCE": 120.0})
        assert total == pytest.approx(995_995.0)
