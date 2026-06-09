"""Portfolio tracker for paper trading.

Manages positions (LONG and SHORT), computes weighted-average prices,
tracks realized and unrealized P&L, and keeps the cash balance in sync
with every fill processed through the paper engine.

All state is persisted via :class:`~src.paper.store.PaperStore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.paper import OrderRequest
from src.paper.fees import FeeBreakdown
from src.paper.fill_simulator import FillResult
from src.paper.store import PaperStore


@dataclass(frozen=True)
class Position:
    """Immutable snapshot of an open position."""

    id: int
    symbol: str
    exchange: str
    qty: int
    avg_price: float
    side: str          # "LONG" | "SHORT"
    opened_at: str


@dataclass(frozen=True)
class PositionUpdate:
    """Result of processing a fill against the portfolio."""

    position_id: int
    action: str        # "opened" | "added" | "reduced" | "closed"
    old_qty: int
    new_qty: int
    realized_pnl: float


def _now_iso() -> str:
    return datetime.now().isoformat()


class Portfolio:
    """Portfolio tracker backed by PaperStore.

    Processes fills to open/add/reduce/close positions, adjusting cash
    and realized P&L on every trade.
    """

    def __init__(self, store: PaperStore) -> None:
        self._store = store

    def init(self, starting_capital: float) -> None:
        """Initialise the portfolio with the given starting cash."""
        self._store.init_portfolio(starting_capital)

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def process_fill(
        self,
        fill: FillResult,
        order: OrderRequest,
        fees: FeeBreakdown,
    ) -> PositionUpdate:
        """Apply a fill to the portfolio, updating positions and cash.

        Returns a :class:`PositionUpdate` describing what changed.
        """
        existing = self._store.get_position_by_symbol(order.symbol, order.exchange)

        if order.side == "BUY":
            return self._process_buy(fill, order, fees, existing)
        return self._process_sell(fill, order, fees, existing)

    # ------------------------------------------------------------------
    # BUY logic
    # ------------------------------------------------------------------

    def _process_buy(
        self,
        fill: FillResult,
        order: OrderRequest,
        fees: FeeBreakdown,
        existing: dict | None,
    ) -> PositionUpdate:
        trade_value = fill.fill_price * fill.filled_qty

        if existing is None:
            # Open new LONG
            pos_id = self._store.save_position({
                "symbol": order.symbol,
                "exchange": order.exchange,
                "qty": fill.filled_qty,
                "avg_price": fill.fill_price,
                "side": "LONG",
                "opened_at": _now_iso(),
            })
            self._store.update_cash(-(trade_value + fees.total))
            return PositionUpdate(
                position_id=pos_id,
                action="opened",
                old_qty=0,
                new_qty=fill.filled_qty,
                realized_pnl=0.0,
            )

        if existing["side"] == "LONG":
            # Add to existing LONG — weighted average price
            old_qty = existing["qty"]
            old_avg = existing["avg_price"]
            new_qty = old_qty + fill.filled_qty
            new_avg = (old_avg * old_qty + fill.fill_price * fill.filled_qty) / new_qty

            self._store.update_position(existing["id"], {
                "qty": new_qty,
                "avg_price": new_avg,
            })
            self._store.update_cash(-(trade_value + fees.total))
            return PositionUpdate(
                position_id=existing["id"],
                action="added",
                old_qty=old_qty,
                new_qty=new_qty,
                realized_pnl=0.0,
            )

        # existing SHORT — BUY to reduce/close
        old_qty = existing["qty"]
        avg_price = existing["avg_price"]
        realized_pnl = (avg_price - fill.fill_price) * fill.filled_qty
        remaining = old_qty - fill.filled_qty

        # Cash: deduct cost of buying to cover, credit realized P&L
        self._store.update_cash(
            -(trade_value + fees.total),
            realized_delta=realized_pnl,
        )

        if remaining <= 0:
            # Fully closed
            self._store.delete_position(existing["id"])
            return PositionUpdate(
                position_id=existing["id"],
                action="closed",
                old_qty=old_qty,
                new_qty=0,
                realized_pnl=realized_pnl,
            )

        # Partially covered
        self._store.update_position(existing["id"], {"qty": remaining})
        return PositionUpdate(
            position_id=existing["id"],
            action="reduced",
            old_qty=old_qty,
            new_qty=remaining,
            realized_pnl=realized_pnl,
        )

    # ------------------------------------------------------------------
    # SELL logic
    # ------------------------------------------------------------------

    def _process_sell(
        self,
        fill: FillResult,
        order: OrderRequest,
        fees: FeeBreakdown,
        existing: dict | None,
    ) -> PositionUpdate:
        trade_value = fill.fill_price * fill.filled_qty

        if existing is None:
            # Open new SHORT
            pos_id = self._store.save_position({
                "symbol": order.symbol,
                "exchange": order.exchange,
                "qty": fill.filled_qty,
                "avg_price": fill.fill_price,
                "side": "SHORT",
                "opened_at": _now_iso(),
            })
            self._store.update_cash(trade_value - fees.total)
            return PositionUpdate(
                position_id=pos_id,
                action="opened",
                old_qty=0,
                new_qty=fill.filled_qty,
                realized_pnl=0.0,
            )

        if existing["side"] == "SHORT":
            # Add to existing SHORT — weighted average price
            old_qty = existing["qty"]
            old_avg = existing["avg_price"]
            new_qty = old_qty + fill.filled_qty
            new_avg = (old_avg * old_qty + fill.fill_price * fill.filled_qty) / new_qty

            self._store.update_position(existing["id"], {
                "qty": new_qty,
                "avg_price": new_avg,
            })
            self._store.update_cash(trade_value - fees.total)
            return PositionUpdate(
                position_id=existing["id"],
                action="added",
                old_qty=old_qty,
                new_qty=new_qty,
                realized_pnl=0.0,
            )

        # existing LONG — SELL to reduce/close
        old_qty = existing["qty"]
        avg_price = existing["avg_price"]
        realized_pnl = (fill.fill_price - avg_price) * fill.filled_qty
        remaining = old_qty - fill.filled_qty

        # Cash: credit proceeds, deduct fees
        self._store.update_cash(
            trade_value - fees.total,
            realized_delta=realized_pnl,
        )

        if remaining <= 0:
            # Fully closed
            self._store.delete_position(existing["id"])
            return PositionUpdate(
                position_id=existing["id"],
                action="closed",
                old_qty=old_qty,
                new_qty=0,
                realized_pnl=realized_pnl,
            )

        # Partially sold
        self._store.update_position(existing["id"], {"qty": remaining})
        return PositionUpdate(
            position_id=existing["id"],
            action="reduced",
            old_qty=old_qty,
            new_qty=remaining,
            realized_pnl=realized_pnl,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """Return all open positions as immutable Position objects."""
        rows = self._store.get_positions()
        return [
            Position(
                id=r["id"],
                symbol=r["symbol"],
                exchange=r["exchange"],
                qty=r["qty"],
                avg_price=r["avg_price"],
                side=r["side"],
                opened_at=r["opened_at"],
            )
            for r in rows
        ]

    def get_cash_balance(self) -> float:
        """Return the current cash balance."""
        portfolio = self._store.get_portfolio()
        if portfolio is None:
            return 0.0
        return portfolio["cash_balance"]

    def get_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Compute unrealised P&L across all positions.

        Args:
            prices: Mapping of symbol -> current market price.

        Returns:
            Sum of unrealised P&L.  LONG positions gain when price rises;
            SHORT positions gain when price falls.
        """
        positions = self.get_positions()
        total = 0.0
        for pos in positions:
            current = prices.get(pos.symbol, pos.avg_price)
            if pos.side == "LONG":
                total += (current - pos.avg_price) * pos.qty
            else:  # SHORT
                total += (pos.avg_price - current) * pos.qty
        return total

    def get_total_value(self, prices: dict[str, float]) -> float:
        """Return total portfolio value: cash + unrealised P&L."""
        return self.get_cash_balance() + self.get_unrealized_pnl(prices)
