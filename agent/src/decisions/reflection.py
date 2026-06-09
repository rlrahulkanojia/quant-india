"""Weekly reflection service — reviews past trades with Claude.

Queries closed trades from :class:`DecisionStore`, computes performance
metrics (win rate, average P&L, Sharpe ratio), sends the trade summary
to Claude for pattern analysis, and persists the resulting
:class:`ReflectionReport`.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

import anthropic

from src.debate.prompts import REFLECTION_PROMPT
from src.decisions.store import DecisionStore


@dataclass(frozen=True)
class ReflectionReport:
    """Immutable summary of a weekly performance review."""

    period_start: str
    period_end: str
    total_trades: int
    win_rate: float
    avg_pnl_pct: float
    sharpe_ratio: float
    findings: list[str]
    recommendations: list[dict]


class ReflectionService:
    """Runs weekly trade reflections using Claude to identify patterns."""

    def __init__(
        self,
        store: DecisionStore,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._store = store
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_weekly_reflection(
        self, as_of: str | None = None
    ) -> ReflectionReport:
        """Analyse the last 7 days of closed trades and return a report.

        Parameters
        ----------
        as_of:
            ISO-8601 datetime marking the end of the period.
            Defaults to *now*.
        """
        period_end = as_of or datetime.now().isoformat()
        period_start = (
            datetime.fromisoformat(period_end) - timedelta(days=7)
        ).isoformat()

        # 1 — Fetch closed decisions in the window
        closed = self._get_closed_in_period(period_start, period_end)

        # 2 — Calculate metrics
        total_trades = len(closed)
        win_rate = self._calc_win_rate(closed)
        avg_pnl_pct = self._calc_avg_pnl_pct(closed)
        sharpe_ratio = self._calc_sharpe(closed)

        # 3 — Ask Claude for insights (even on empty to get a baseline)
        findings, recommendations = self._ask_claude(closed)

        # 4 — Persist the reflection report
        report = ReflectionReport(
            period_start=period_start,
            period_end=period_end,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl_pct,
            sharpe_ratio=sharpe_ratio,
            findings=findings,
            recommendations=recommendations,
        )
        self._store.save_reflection({
            "period_start": report.period_start,
            "period_end": report.period_end,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "avg_pnl_pct": report.avg_pnl_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "findings": report.findings,
            "recommendations": report.recommendations,
        })

        return report

    # ------------------------------------------------------------------
    # Internals — data retrieval
    # ------------------------------------------------------------------

    def _get_closed_in_period(
        self, period_start: str, period_end: str
    ) -> list[dict]:
        """Return only closed decisions whose closed_at falls within range."""
        all_decisions = self._store.list_decisions(
            since_date=period_start, limit=10_000
        )
        return [
            d for d in all_decisions
            if d.get("closed_at") is not None
            and d["closed_at"] >= period_start
            and d["closed_at"] <= period_end
        ]

    # ------------------------------------------------------------------
    # Internals — metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_win_rate(closed: list[dict]) -> float:
        if not closed:
            return 0.0
        wins = sum(1 for d in closed if (d.get("pnl_amount") or 0) > 0)
        return wins / len(closed)

    @staticmethod
    def _calc_avg_pnl_pct(closed: list[dict]) -> float:
        if not closed:
            return 0.0
        pnl_pcts = [d.get("pnl_pct", 0.0) or 0.0 for d in closed]
        return sum(pnl_pcts) / len(pnl_pcts)

    @staticmethod
    def _calc_sharpe(closed: list[dict]) -> float:
        """Sharpe ratio: mean(pnl_pct) / std(pnl_pct).

        Returns 0.0 when there are fewer than 2 trades or when
        standard deviation is zero.
        """
        if len(closed) < 2:
            return 0.0
        pnl_pcts = [d.get("pnl_pct", 0.0) or 0.0 for d in closed]
        if all(p == 0.0 for p in pnl_pcts):
            return 0.0
        mean = statistics.mean(pnl_pcts)
        std = statistics.pstdev(pnl_pcts)
        if std == 0:
            return 0.0
        return mean / std

    # ------------------------------------------------------------------
    # Internals — Claude call
    # ------------------------------------------------------------------

    def _ask_claude(
        self, closed: list[dict]
    ) -> tuple[list[str], list[dict]]:
        """Send trade summary to Claude and parse findings/recommendations."""
        trade_summary = self._build_trade_summary(closed)

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{REFLECTION_PROMPT}\n\n"
                        f"## Trade Data\n\n{trade_summary}"
                    ),
                },
            ],
        )

        raw = response.content[0].text
        parsed = json.loads(raw)

        findings: list[str] = parsed.get("patterns_identified", [])
        if parsed.get("bias_flags"):
            findings.extend(parsed["bias_flags"])
        if parsed.get("summary"):
            findings.append(parsed["summary"])

        recommendations: list[dict] = parsed.get(
            "threshold_recommendations", []
        )

        return findings, recommendations

    @staticmethod
    def _build_trade_summary(closed: list[dict]) -> str:
        """Format closed trades into a readable summary for Claude."""
        if not closed:
            return "No closed trades in this period."
        lines = []
        for d in closed:
            lines.append(
                f"- {d['symbol']} {d['action']} | "
                f"confidence={d['confidence']} signal={d.get('signal_type', '?')} | "
                f"entry={d.get('entry_price')} exit={d.get('exit_price')} | "
                f"pnl={d.get('pnl_pct', 0):.2f}% ({d.get('pnl_amount', 0):.0f}) | "
                f"hold={d.get('hold_duration', 0)}min | "
                f"exit_reason={d.get('exit_reason', '?')}"
            )
        return "\n".join(lines)
