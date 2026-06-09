"""Full pipeline integration test for the paper trading engine.

Exercises the entire paper trading flow end-to-end:
    BUY -> verify position -> SELL -> verify closed -> portfolio summary
    -> shadow report -> daily summary -> equity curve

Uses realistic RELIANCE share prices (LTP ~2850-2900).
Starting capital: 10,00,000 (10 lakh INR).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure agent/ is on sys.path
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import pytest

from src.paper import MarketSnapshot
from src.paper.engine import PaperTradingEngine, PortfolioSummary, TradeResult
from src.paper.fees import FeeConfig, calculate_fees
from src.paper.reports import daily_summary, equity_curve, shadow_comparison
from src.paper.shadow import DivergenceReport, ShadowFill

STARTING_CAPITAL = 10_00_000.0  # ₹10 lakh


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def engine(tmp_path: Path) -> PaperTradingEngine:
    """Create a PaperTradingEngine with 10 lakh starting capital."""
    db = str(tmp_path / "integration.db")
    return PaperTradingEngine(db_path=db, starting_capital=STARTING_CAPITAL)


@pytest.fixture()
def buy_market() -> MarketSnapshot:
    """Market snapshot for the BUY leg (LTP=2850)."""
    return MarketSnapshot(
        ltp=2850.0,
        bid=2849.0,
        ask=2851.0,
        volume=50_000,
        avg_daily_volume=5_000_000,
    )


@pytest.fixture()
def sell_market() -> MarketSnapshot:
    """Market snapshot for the SELL leg (LTP=2900)."""
    return MarketSnapshot(
        ltp=2900.0,
        bid=2899.0,
        ask=2901.0,
        volume=60_000,
        avg_daily_volume=5_000_000,
    )


# ------------------------------------------------------------------
# Full pipeline integration test
# ------------------------------------------------------------------


class TestPaperEnginePipeline:
    """End-to-end integration: BUY → position → SELL → close → reports."""

    def test_full_buy_sell_cycle(
        self,
        engine: PaperTradingEngine,
        buy_market: MarketSnapshot,
        sell_market: MarketSnapshot,
    ) -> None:
        """Execute a complete BUY-SELL cycle and verify all modules cooperate."""

        today = datetime.now().strftime("%Y-%m-%d")

        # =============================================================
        # Step 1: BUY RELIANCE-EQ 100 shares at market
        # =============================================================
        buy_result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=buy_market,
        )

        # -- TradeResult assertions --
        assert isinstance(buy_result, TradeResult)
        assert buy_result.filled is True
        assert buy_result.filled_qty == 100

        # Fill price should be ltp + half_spread = 2850 + 1 = 2851
        assert buy_result.fill_price == pytest.approx(2851.0, abs=0.01)

        # Slippage = fill_price - ltp = 1.0
        assert buy_result.slippage == pytest.approx(1.0, abs=0.01)

        # Fees must be positive
        assert buy_result.fees.total > 0

        # Position update: opened a new LONG
        assert buy_result.position_update.action == "opened"
        assert buy_result.position_update.new_qty == 100

        # Shadow fill captured with divergence ≈ (2851 - 2850) / 2850 * 100
        assert buy_result.shadow_fill is not None
        expected_buy_div = (2851.0 - 2850.0) / 2850.0 * 100
        assert buy_result.shadow_fill.divergence_pct == pytest.approx(
            expected_buy_div, abs=0.01
        )
        assert buy_result.shadow_fill.market_ltp == 2850.0

        # -- Portfolio state after BUY --
        buy_cost = buy_result.fill_price * buy_result.filled_qty
        buy_fees = buy_result.fees.total

        # Cash should be reduced by cost + fees
        summary_after_buy = engine.get_summary(
            current_prices={"RELIANCE": buy_market.ltp}
        )
        assert summary_after_buy.cash == pytest.approx(
            STARTING_CAPITAL - buy_cost - buy_fees, abs=0.01
        )
        assert summary_after_buy.cash < STARTING_CAPITAL - buy_cost

        # Should have exactly one position
        assert len(summary_after_buy.positions) == 1
        pos = summary_after_buy.positions[0]
        assert pos["symbol"] == "RELIANCE"
        assert pos["qty"] == 100
        assert pos["side"] == "LONG"

        # =============================================================
        # Step 2: SELL RELIANCE-EQ 100 shares at market (price rallied)
        # =============================================================
        sell_result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=sell_market,
        )

        # -- TradeResult assertions --
        assert isinstance(sell_result, TradeResult)
        assert sell_result.filled is True
        assert sell_result.filled_qty == 100

        # Fill price should be ltp - half_spread = 2900 - 1 = 2899
        assert sell_result.fill_price == pytest.approx(2899.0, abs=0.01)

        # Position update: closed the position
        assert sell_result.position_update.action == "closed"
        assert sell_result.position_update.new_qty == 0

        # Realized P&L from the trade = (2899 - 2851) * 100 = 4800
        expected_pnl = (sell_result.fill_price - buy_result.fill_price) * 100
        assert sell_result.position_update.realized_pnl == pytest.approx(
            expected_pnl, abs=0.01
        )
        assert sell_result.position_update.realized_pnl > 0  # profit

        # Fees on sell side
        assert sell_result.fees.total > 0

        # Shadow fill for SELL: divergence ≈ (2899 - 2900) / 2900 * 100
        assert sell_result.shadow_fill is not None
        expected_sell_div = (2899.0 - 2900.0) / 2900.0 * 100
        assert sell_result.shadow_fill.divergence_pct == pytest.approx(
            expected_sell_div, abs=0.01
        )

        # =============================================================
        # Step 3: Portfolio summary — position closed
        # =============================================================
        summary_final = engine.get_summary(current_prices={})

        # No positions remaining
        assert len(summary_final.positions) == 0

        # Unrealized P&L should be 0 (no positions)
        assert summary_final.unrealized_pnl == pytest.approx(0.0, abs=0.01)

        # Realized P&L > 0 (we made money on RELIANCE)
        assert summary_final.realized_pnl > 0
        assert summary_final.realized_pnl == pytest.approx(expected_pnl, abs=0.01)

        # Total value ≈ starting + profit - all fees
        all_fees = buy_result.fees.total + sell_result.fees.total
        assert summary_final.total_value == pytest.approx(
            STARTING_CAPITAL + expected_pnl - all_fees, abs=0.01
        )
        assert summary_final.total_fees_paid == pytest.approx(all_fees, abs=0.01)

        # Cash = starting - buy_cost - buy_fees + sell_proceeds - sell_fees
        sell_proceeds = sell_result.fill_price * sell_result.filled_qty
        expected_cash = STARTING_CAPITAL - buy_cost - buy_fees + sell_proceeds - sell_result.fees.total
        assert summary_final.cash == pytest.approx(expected_cash, abs=0.01)

        # =============================================================
        # Step 4: Shadow report
        # =============================================================
        shadow_report = engine.get_shadow_report(days=7)

        assert isinstance(shadow_report, DivergenceReport)
        assert shadow_report.fill_count == 2

        # Divergence data should be non-zero
        assert shadow_report.avg_divergence_pct != 0.0
        assert shadow_report.max_divergence_pct > 0.0

        # Total values should reflect both trades
        assert shadow_report.total_paper_value > 0
        assert shadow_report.total_market_value > 0

        # =============================================================
        # Step 5: Daily summary for today
        # =============================================================
        store = engine._store
        day_report = daily_summary(store, today)

        assert day_report.date == today
        assert day_report.trade_count == 2
        assert day_report.fees_paid > 0

        # top_winner and top_loser should be present
        assert day_report.top_winner is not None
        assert day_report.top_loser is not None

        # =============================================================
        # Step 6: Equity curve
        # =============================================================
        curve = equity_curve(store, STARTING_CAPITAL)

        # Must have at least 3 data points: start + BUY + SELL
        assert len(curve) >= 3

        # First entry is always ("start", starting_capital)
        assert curve[0] == ("start", STARTING_CAPITAL)

        # After BUY: capital - (fill_price * qty) - buy_fees
        after_buy_value = STARTING_CAPITAL - buy_cost - buy_fees
        assert curve[1][1] == pytest.approx(after_buy_value, abs=0.01)

        # After SELL: after_buy + sell_proceeds - sell_fees
        after_sell_value = after_buy_value + sell_proceeds - sell_result.fees.total
        assert curve[2][1] == pytest.approx(after_sell_value, abs=0.01)

        # Final equity should equal the cash balance (no positions left)
        assert curve[-1][1] == pytest.approx(summary_final.cash, abs=0.01)

        # =============================================================
        # Step 7: Shadow comparison from reports module
        # =============================================================
        shadow_cmp = shadow_comparison(store, days=7)

        assert shadow_cmp.fill_count == 2
        assert shadow_cmp.paper_total_value > 0
        assert shadow_cmp.market_total_value > 0
        # avg_divergence_pct should be non-zero (fills deviate from LTP)
        assert shadow_cmp.avg_divergence_pct != 0.0
        # Note: divergence_cost may be ~0 when BUY and SELL divergences cancel
        # (BUY fills above LTP, SELL fills below LTP by symmetric amounts)

    def test_fees_are_realistic_for_indian_market(
        self,
        engine: PaperTradingEngine,
        buy_market: MarketSnapshot,
    ) -> None:
        """Verify fees include STT, exchange charges, SEBI, GST, stamp duty, brokerage."""
        result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=buy_market,
        )

        fees = result.fees
        # All Indian-market fee components should exist
        assert fees.stt >= 0
        assert fees.exchange_charges >= 0
        assert fees.sebi >= 0
        assert fees.gst >= 0
        assert fees.stamp_duty >= 0
        assert fees.brokerage >= 0
        assert fees.total > 0

        # Total should equal sum of components
        component_sum = (
            fees.stt
            + fees.exchange_charges
            + fees.sebi
            + fees.gst
            + fees.stamp_duty
            + fees.brokerage
        )
        assert fees.total == pytest.approx(component_sum, abs=0.01)

        # For a ₹2,85,100 trade (2851 * 100), fees should be in a
        # reasonable range (roughly 0.01% - 0.5% of trade value)
        trade_value = result.fill_price * result.filled_qty
        fee_pct = fees.total / trade_value * 100
        assert 0.005 < fee_pct < 1.0, f"Fee % {fee_pct:.4f}% looks unrealistic"

    def test_multiple_buys_then_partial_sell(
        self,
        engine: PaperTradingEngine,
        buy_market: MarketSnapshot,
        sell_market: MarketSnapshot,
    ) -> None:
        """Verify engine handles multiple buys followed by a partial sell."""

        # Buy 100 shares twice
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=buy_market,
        )
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=100,
            market=buy_market,
        )

        # Verify 200 shares held
        summary = engine.get_summary(current_prices={"RELIANCE": 2850.0})
        assert len(summary.positions) == 1
        assert summary.positions[0]["qty"] == 200

        # Sell only 50 shares (partial exit)
        sell_result = engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=50,
            market=sell_market,
        )

        assert sell_result.filled is True
        assert sell_result.filled_qty == 50
        assert sell_result.position_update.action == "reduced"
        assert sell_result.position_update.new_qty == 150

        # Still holding 150
        summary2 = engine.get_summary(current_prices={"RELIANCE": 2900.0})
        assert len(summary2.positions) == 1
        assert summary2.positions[0]["qty"] == 150

        # Shadow report should show 3 fills (2 buys + 1 sell)
        shadow = engine.get_shadow_report(days=7)
        assert shadow.fill_count == 3

    def test_equity_curve_monotonic_labels(
        self,
        engine: PaperTradingEngine,
        buy_market: MarketSnapshot,
        sell_market: MarketSnapshot,
    ) -> None:
        """Equity curve labels should start with 'start' then dates."""
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=50,
            market=buy_market,
        )
        engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=50,
            market=sell_market,
        )

        curve = equity_curve(engine._store, STARTING_CAPITAL)

        assert curve[0][0] == "start"
        # Remaining entries should have date-like labels (YYYY-MM-DD)
        for label, value in curve[1:]:
            assert len(label) == 10, f"Expected date label, got {label}"
            assert label[4] == "-" and label[7] == "-"
            # Values should always be positive (can't go negative with 10L capital)
            assert value > 0
