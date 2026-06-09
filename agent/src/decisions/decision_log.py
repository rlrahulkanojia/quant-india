"""High-level decision logger that wraps DecisionStore.

Provides a clean API for the trading pipeline to:
- log new trade decisions (with debate history and analysis context),
- close trades when positions are exited,
- query recent decisions,
- calculate win rates for strategy evaluation.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

from src.debate.schemas import AnalysisReport, TradeDecision
from src.debate.scoring import ScoringResult
from src.decisions.store import DecisionStore


class DecisionLogger:
    """Logs trade decisions and computes trading performance metrics."""

    def __init__(self, store: DecisionStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        decision: TradeDecision,
        scoring: ScoringResult,
        analysis: AnalysisReport,
        paper_order_id: str | None = None,
    ) -> int:
        """Serialize a trade decision with its debate context and persist it.

        Returns the auto-generated decision ID.
        """
        data = {
            "symbol": decision.symbol,
            "action": decision.action,
            "confidence": decision.confidence,
            "debate_log": dataclasses.asdict(decision)["debate_rounds"],
            "analysis_report": dataclasses.asdict(analysis),
            "signal_type": decision.signal_type,
            "entry_price": decision.suggested_price,
            "paper_order_id": paper_order_id,
        }
        return self._store.save_decision(data)

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def close_trade(
        self,
        decision_id: int,
        exit_price: float,
        pnl_amount: float,
        pnl_pct: float,
        hold_duration_minutes: int,
        exit_reason: str,
    ) -> None:
        """Mark a trade decision as closed with exit metrics."""
        self._store.close_decision(
            decision_id=decision_id,
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            pnl_pct=pnl_pct,
            hold_duration=hold_duration_minutes,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recent_decisions(self, days: int = 7) -> list[dict]:
        """Return decisions from the last *days* days, newest first."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        return self._store.list_decisions(since_date=since)

    def get_win_rate(
        self, days: int = 30, signal_type: str | None = None
    ) -> float:
        """Calculate the win rate over closed trades in the last *days* days.

        A *win* is defined as ``pnl_amount > 0``.  Only closed trades
        (``closed_at IS NOT NULL``) are counted.  Returns ``0.0`` when
        there are no qualifying trades.
        """
        since = (datetime.now() - timedelta(days=days)).isoformat()
        decisions = self._store.list_decisions(since_date=since, limit=10_000)

        closed = [d for d in decisions if d.get("closed_at") is not None]

        if signal_type is not None:
            closed = [d for d in closed if d.get("signal_type") == signal_type]

        if not closed:
            return 0.0

        wins = sum(1 for d in closed if (d.get("pnl_amount") or 0) > 0)
        return wins / len(closed)
