"""Confidence scoring — pure logic, no LLM calls.

Maps a :class:`TradeDecision` confidence value to an action bucket:

* **TRADE** — confidence ≥ ``trade_above`` → execute the order.
* **LOG**   — confidence in ``[skip_below, trade_above)`` → record for
  review but do *not* trade.
* **SKIP**  — confidence < ``skip_below`` → discard silently.

All thresholds come from :class:`DebateConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.debate.schemas import DebateConfig, TradeDecision


@dataclass(frozen=True)
class ScoringResult:
    """Immutable outcome of the confidence scoring step."""

    action: str        # "TRADE", "LOG", or "SKIP"
    confidence: int
    reason: str


def score_decision(
    decision: TradeDecision,
    config: DebateConfig,
) -> ScoringResult:
    """Bucket *decision* into TRADE / LOG / SKIP based on *config* thresholds."""

    if decision.confidence >= config.confidence_trade_above:
        return ScoringResult(
            action="TRADE",
            confidence=decision.confidence,
            reason=(
                f"Confidence {decision.confidence} >= trade threshold "
                f"{config.confidence_trade_above}"
            ),
        )

    if decision.confidence >= config.confidence_skip_below:
        return ScoringResult(
            action="LOG",
            confidence=decision.confidence,
            reason=(
                f"Confidence {decision.confidence} in log range "
                f"[{config.confidence_skip_below}, "
                f"{config.confidence_trade_above})"
            ),
        )

    return ScoringResult(
        action="SKIP",
        confidence=decision.confidence,
        reason=(
            f"Confidence {decision.confidence} < skip threshold "
            f"{config.confidence_skip_below}"
        ),
    )


def should_execute(scoring_result: ScoringResult) -> bool:
    """Return *True* only when the scoring result calls for a trade."""
    return scoring_result.action == "TRADE"
