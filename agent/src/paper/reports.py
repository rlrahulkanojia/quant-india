"""Pure-function reports for paper trading analysis.

Provides daily summaries, equity curves, and shadow-fill comparison
without any mutable state — each function takes a PaperStore and
returns a frozen dataclass or plain list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.paper.store import PaperStore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailySummary:
    """Aggregate statistics for a single trading day."""

    date: str
    trade_count: int
    realized_pnl: float
    fees_paid: float
    top_winner: Optional[tuple[str, float]]
    top_loser: Optional[tuple[str, float]]


@dataclass(frozen=True)
class ShadowComparison:
    """Comparison of paper fills vs real market prices over a period."""

    period_days: int
    fill_count: int
    avg_divergence_pct: float
    paper_total_value: float
    market_total_value: float
    divergence_cost: float


# ---------------------------------------------------------------------------
# daily_summary
# ---------------------------------------------------------------------------


def daily_summary(store: PaperStore, date: str) -> DailySummary:
    """Aggregate filled orders for a single date.

    Parameters
    ----------
    store : PaperStore
        The paper trading store to query.
    date : str
        ISO date string (e.g. "2025-06-10") to filter on.

    Returns
    -------
    DailySummary
        Frozen dataclass with trade_count, fees_paid, realized_pnl,
        top_winner, and top_loser.
    """
    all_orders = store.list_orders()

    # Filter to filled orders whose filled_at date matches
    day_orders = [
        o for o in all_orders
        if o.get("filled_at")
        and o["status"] == "FILLED"
        and o["filled_at"][:10] == date
    ]

    if not day_orders:
        return DailySummary(
            date=date,
            trade_count=0,
            realized_pnl=0.0,
            fees_paid=0.0,
            top_winner=None,
            top_loser=None,
        )

    trade_count = len(day_orders)
    fees_paid = sum(o.get("fees_total", 0.0) or 0.0 for o in day_orders)

    # Simplified realized P&L: negative fee impact per trade
    # (full P&L tracking lives in portfolio module)
    pnl_per_order: list[tuple[str, float]] = []
    for o in day_orders:
        fee = o.get("fees_total", 0.0) or 0.0
        pnl_per_order.append((o["symbol"], -fee))

    realized_pnl = sum(pnl for _, pnl in pnl_per_order)

    # Sort by P&L: best (least negative) first
    pnl_per_order.sort(key=lambda x: x[1], reverse=True)
    top_winner = pnl_per_order[0] if pnl_per_order else None
    top_loser = pnl_per_order[-1] if len(pnl_per_order) > 1 else top_winner

    return DailySummary(
        date=date,
        trade_count=trade_count,
        realized_pnl=realized_pnl,
        fees_paid=fees_paid,
        top_winner=top_winner,
        top_loser=top_loser,
    )


# ---------------------------------------------------------------------------
# equity_curve
# ---------------------------------------------------------------------------


def equity_curve(
    store: PaperStore, starting_capital: float
) -> list[tuple[str, float]]:
    """Build a running cash-value curve from filled orders.

    Parameters
    ----------
    store : PaperStore
        The paper trading store to query.
    starting_capital : float
        Initial cash value.

    Returns
    -------
    list[tuple[str, float]]
        List of (date_or_label, running_value) pairs.  The first entry
        is always ``("start", starting_capital)``.
    """
    all_orders = store.list_orders()

    # Keep only filled orders with valid fill data
    filled = [
        o for o in all_orders
        if o["status"] == "FILLED"
        and o.get("filled_at")
        and o.get("fill_price") is not None
        and o.get("filled_qty")
    ]

    # Sort chronologically by filled_at
    filled.sort(key=lambda o: o["filled_at"])

    curve: list[tuple[str, float]] = [("start", starting_capital)]
    running = starting_capital

    for o in filled:
        price = o["fill_price"]
        qty = o["filled_qty"]
        fees = o.get("fees_total", 0.0) or 0.0
        date_str = o["filled_at"][:10]

        if o["side"] == "BUY":
            running -= price * qty
        else:  # SELL
            running += price * qty

        running -= fees
        curve.append((date_str, running))

    return curve


# ---------------------------------------------------------------------------
# shadow_comparison
# ---------------------------------------------------------------------------


def shadow_comparison(
    store: PaperStore,
    days: int = 7,
    *,
    as_of: str | None = None,
) -> ShadowComparison:
    """Compare paper fills to market prices over a recent window.

    Parameters
    ----------
    store : PaperStore
        The paper trading store to query.
    days : int
        Lookback window in days.
    as_of : str, optional
        ISO-8601 timestamp to use as "now".  Defaults to actual now.

    Returns
    -------
    ShadowComparison
        Frozen dataclass with fill_count, avg_divergence_pct,
        paper_total_value, market_total_value, and divergence_cost.
    """
    if as_of:
        ref = datetime.fromisoformat(as_of)
    else:
        ref = datetime.now()

    since = ref - timedelta(days=days)
    since_iso = since.isoformat()

    fills = store.get_all_shadow_fills_since(since_iso)

    if not fills:
        return ShadowComparison(
            period_days=days,
            fill_count=0,
            avg_divergence_pct=0.0,
            paper_total_value=0.0,
            market_total_value=0.0,
            divergence_cost=0.0,
        )

    fill_count = len(fills)
    avg_divergence_pct = sum(f["divergence_pct"] for f in fills) / fill_count

    paper_total_value = sum(
        f["paper_fill_price"] * f["qty"] for f in fills
    )
    market_total_value = sum(
        f["market_ltp"] * f["qty"] for f in fills
    )
    divergence_cost = paper_total_value - market_total_value

    return ShadowComparison(
        period_days=days,
        fill_count=fill_count,
        avg_divergence_pct=avg_divergence_pct,
        paper_total_value=paper_total_value,
        market_total_value=market_total_value,
        divergence_cost=divergence_cost,
    )
