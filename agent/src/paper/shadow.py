"""Shadow account — tracks divergence between paper fills and live market.

Captures every paper fill alongside the real-time market snapshot so we
can measure how realistic the paper engine's fills are over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.paper import MarketSnapshot
from src.paper.store import PaperStore


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowFill:
    """Single point-in-time comparison between a paper fill and the market."""

    paper_order_id: str
    paper_fill_price: float
    market_ltp: float
    market_bid: float
    market_ask: float
    divergence_pct: float
    captured_at: str


@dataclass(frozen=True)
class DivergenceReport:
    """Aggregated divergence stats over a time window."""

    avg_divergence_pct: float
    max_divergence_pct: float
    fill_count: int
    total_paper_value: float
    total_market_value: float
    period_start: str
    period_end: str


# ------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------


class ShadowTracker:
    """Captures paper-vs-market divergence and produces summary reports."""

    def __init__(self, store: PaperStore) -> None:
        self._store = store

    # ---- capture ---------------------------------------------------

    def capture(
        self,
        paper_order_id: str,
        paper_fill_price: float,
        qty: int,
        side: str,
        market: MarketSnapshot,
    ) -> ShadowFill:
        """Record a single fill against its market snapshot.

        Parameters
        ----------
        paper_order_id:
            ID of the parent paper order.
        paper_fill_price:
            Price the paper engine assigned for the fill.
        qty:
            Number of shares/lots filled.
        side:
            ``"BUY"`` or ``"SELL"``.
        market:
            Real-time market data at the moment of the fill.

        Returns
        -------
        ShadowFill
            Frozen record with computed divergence.
        """
        divergence_pct = (paper_fill_price - market.ltp) / market.ltp * 100
        now_iso = datetime.now().isoformat()

        self._store.save_shadow_fill(
            {
                "paper_order_id": paper_order_id,
                "paper_fill_price": paper_fill_price,
                "market_ltp": market.ltp,
                "market_bid": market.bid,
                "market_ask": market.ask,
                "divergence_pct": divergence_pct,
                "qty": qty,
                "captured_at": now_iso,
            }
        )

        return ShadowFill(
            paper_order_id=paper_order_id,
            paper_fill_price=paper_fill_price,
            market_ltp=market.ltp,
            market_bid=market.bid,
            market_ask=market.ask,
            divergence_pct=divergence_pct,
            captured_at=now_iso,
        )

    # ---- report ----------------------------------------------------

    def get_report(self, days: int = 7) -> DivergenceReport:
        """Aggregate divergence statistics over the last *days* days.

        Parameters
        ----------
        days:
            Look-back window in calendar days (default 7).

        Returns
        -------
        DivergenceReport
            Summary with averages, max (absolute), and value totals.
        """
        now = datetime.now()
        since = now - timedelta(days=days)
        since_iso = since.isoformat()
        now_iso = now.isoformat()

        fills = self._store.get_all_shadow_fills_since(since_iso)

        if not fills:
            return DivergenceReport(
                avg_divergence_pct=0.0,
                max_divergence_pct=0.0,
                fill_count=0,
                total_paper_value=0.0,
                total_market_value=0.0,
                period_start=since_iso,
                period_end=now_iso,
            )

        divergences = [f["divergence_pct"] for f in fills]
        avg_div = sum(divergences) / len(divergences)
        max_div = max(abs(d) for d in divergences)

        total_paper = sum(f["paper_fill_price"] * f["qty"] for f in fills)
        total_market = sum(f["market_ltp"] * f["qty"] for f in fills)

        captured_times = [f["captured_at"] for f in fills]

        return DivergenceReport(
            avg_divergence_pct=avg_div,
            max_divergence_pct=max_div,
            fill_count=len(fills),
            total_paper_value=total_paper,
            total_market_value=total_market,
            period_start=min(captured_times),
            period_end=max(captured_times),
        )
