"""Paper Trading Engine — orchestrator that composes all paper trading modules.

Wires together fees, fill simulator, store, portfolio, and shadow tracker
into a single entry point for executing paper trades and querying state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.paper import MarketSnapshot, OrderRequest
from src.paper.fees import FeeBreakdown, FeeConfig, calculate_fees
from src.paper.fill_simulator import FillResult, simulate_fill
from src.paper.portfolio import Portfolio, PositionUpdate
from src.paper.shadow import DivergenceReport, ShadowFill, ShadowTracker
from src.paper.store import PaperStore


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TradeResult:
    """Immutable result of an execute_order() call."""

    order_id: str
    filled: bool
    fill_price: float
    filled_qty: int
    slippage: float
    fees: FeeBreakdown
    position_update: PositionUpdate
    shadow_fill: Optional[ShadowFill] = None


@dataclass(frozen=True)
class PortfolioSummary:
    """Snapshot of the portfolio at a point in time."""

    cash: float
    positions: list[dict]
    unrealized_pnl: float
    realized_pnl: float
    total_value: float
    total_fees_paid: float


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class PaperTradingEngine:
    """Orchestrates paper trading: fill simulation, fees, portfolio, shadow.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    starting_capital:
        Initial cash balance in ₹ (default 1,000,000).
    fee_config:
        Optional fee rate overrides; uses :class:`FeeConfig` defaults when *None*.
    """

    def __init__(
        self,
        db_path: str,
        starting_capital: float = 1_000_000.0,
        fee_config: FeeConfig | None = None,
    ) -> None:
        self._store = PaperStore(db_path)
        self._portfolio = Portfolio(self._store)
        self._shadow = ShadowTracker(self._store)
        self._portfolio.init(starting_capital)
        self._fee_config = fee_config or FeeConfig()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        order_type: str,
        qty: int,
        market: MarketSnapshot,
        limit_price: float | None = None,
        trade_type: str = "equity_intraday",
    ) -> TradeResult:
        """Execute a paper order against the current market snapshot.

        Parameters
        ----------
        symbol:
            Instrument symbol (e.g. ``"RELIANCE"``).
        exchange:
            Exchange code (e.g. ``"NSE"``).
        side:
            ``"BUY"`` or ``"SELL"``.
        order_type:
            ``"MARKET"`` or ``"LIMIT"``.
        qty:
            Number of shares/lots.
        market:
            Current market snapshot for the symbol.
        limit_price:
            Required when *order_type* is ``"LIMIT"``.
        trade_type:
            Fee schedule key — ``"equity_intraday"`` (default),
            ``"equity_delivery"``, or ``"commodity"``.

        Returns
        -------
        TradeResult
            Frozen result with fill details, fees, position change, and shadow.
        """
        order_id = f"PAPER-{symbol}-{side}-{qty}-{int(time.time())}"
        now_iso = _now_iso()

        # 1. Build order request
        order = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
        )

        # 2. Simulate fill
        fill = simulate_fill(order, market)

        # 3. Unfilled → save as REJECTED, return early
        if not fill.filled:
            zero_fees = FeeBreakdown(
                stt=0.0,
                exchange_charges=0.0,
                sebi=0.0,
                gst=0.0,
                stamp_duty=0.0,
                brokerage=0.0,
                total=0.0,
            )
            no_fill_update = PositionUpdate(
                position_id=0,
                action="no_fill",
                old_qty=0,
                new_qty=0,
                realized_pnl=0.0,
            )
            self._store.save_order({
                "id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "side": side,
                "order_type": order_type,
                "qty": qty,
                "limit_price": limit_price,
                "fill_price": None,
                "slippage": None,
                "fees_total": 0.0,
                "fees_breakdown": _fee_breakdown_dict(zero_fees),
                "status": "REJECTED",
                "filled_qty": 0,
                "created_at": now_iso,
                "filled_at": None,
            })
            return TradeResult(
                order_id=order_id,
                filled=False,
                fill_price=0.0,
                filled_qty=0,
                slippage=0.0,
                fees=zero_fees,
                position_update=no_fill_update,
                shadow_fill=None,
            )

        # 4. Calculate fees
        trade_value = fill.fill_price * fill.filled_qty
        fees = calculate_fees(
            trade_value=trade_value,
            side=side,
            trade_type=trade_type,
            config=self._fee_config,
        )

        # 5. Process fill in portfolio
        pos_update = self._portfolio.process_fill(fill, order, fees)

        # 6. Save order as FILLED (must precede shadow capture for FK)
        self._store.save_order({
            "id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "limit_price": limit_price,
            "fill_price": fill.fill_price,
            "slippage": fill.slippage,
            "fees_total": fees.total,
            "fees_breakdown": _fee_breakdown_dict(fees),
            "status": "FILLED",
            "filled_qty": fill.filled_qty,
            "created_at": now_iso,
            "filled_at": now_iso,
        })

        # 7. Capture shadow fill (after order is persisted for FK constraint)
        shadow = self._shadow.capture(
            paper_order_id=order_id,
            paper_fill_price=fill.fill_price,
            qty=fill.filled_qty,
            side=side,
            market=market,
        )

        # 8. Return full result
        return TradeResult(
            order_id=order_id,
            filled=True,
            fill_price=fill.fill_price,
            filled_qty=fill.filled_qty,
            slippage=fill.slippage,
            fees=fees,
            position_update=pos_update,
            shadow_fill=shadow,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_summary(self, current_prices: dict[str, float]) -> PortfolioSummary:
        """Build a portfolio summary with current market prices.

        Parameters
        ----------
        current_prices:
            Mapping of symbol → current market price, used to compute
            unrealised P&L.

        Returns
        -------
        PortfolioSummary
            Frozen snapshot of the entire portfolio.
        """
        cash = self._portfolio.get_cash_balance()
        positions = self._store.get_positions()
        unrealized_pnl = self._portfolio.get_unrealized_pnl(current_prices)

        portfolio_row = self._store.get_portfolio()
        realized_pnl = portfolio_row["total_realized"] if portfolio_row else 0.0

        total_value = cash + unrealized_pnl

        # Sum fees from all filled orders
        orders = self._store.list_orders()
        total_fees_paid = sum(
            (o.get("fees_total") or 0.0)
            for o in orders
            if o["status"] == "FILLED"
        )

        return PortfolioSummary(
            cash=cash,
            positions=positions,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_value=total_value,
            total_fees_paid=total_fees_paid,
        )

    def get_shadow_report(self, days: int = 7) -> DivergenceReport:
        """Generate a divergence report for the specified lookback period.

        Parameters
        ----------
        days:
            Number of calendar days to look back (default 7).

        Returns
        -------
        DivergenceReport
            Aggregated divergence statistics.
        """
        return self._shadow.get_report(days=days)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _fee_breakdown_dict(fees: FeeBreakdown) -> dict:
    """Convert a FeeBreakdown to a dict for JSON serialisation."""
    return {
        "stt": fees.stt,
        "exchange_charges": fees.exchange_charges,
        "sebi": fees.sebi,
        "gst": fees.gst,
        "stamp_duty": fees.stamp_duty,
        "brokerage": fees.brokerage,
        "total": fees.total,
    }
