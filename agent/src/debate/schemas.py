"""Frozen dataclass schemas for the adversarial debate pipeline.

These immutable value objects flow through the debate engine:

  AnalysisReport  ->  DebateRound[]  ->  RiskAssessment  ->  TradeDecision

:class:`DebateConfig` controls engine parameters (rounds, confidence
thresholds, model selection).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnalysisReport:
    """Combined analysis output fed into the debate engine.

    Aggregates technical, fundamental, and sentiment signals into a
    single report with a combined signal direction and strength.
    """

    symbol: str
    exchange: str
    timestamp: str
    technical: dict
    fundamental: dict
    sentiment: dict
    combined_signal: str        # "bullish" | "bearish" | "neutral"
    signal_strength: float      # 0-1


@dataclass(frozen=True)
class DebateRound:
    """One round of bull-vs-bear argumentation."""

    round_number: int
    bull_argument: str
    bear_argument: str


@dataclass(frozen=True)
class RiskAssessment:
    """Position sizing and exposure limits for a potential trade."""

    volatility_score: float     # 0-1
    position_size_pct: float
    max_exposure_pct: float
    correlation_risk: str
    sizing_method: str          # "atr" | "kelly" | "fixed"


@dataclass(frozen=True)
class TradeDecision:
    """Final output of the debate engine: act, skip, or log-only."""

    symbol: str
    exchange: str
    action: str                 # "BUY" | "SELL" | "HOLD" | "SKIP"
    confidence: int             # 0-100
    signal_type: str
    debate_rounds: list
    risk_assessment: RiskAssessment
    reasoning: str
    suggested_qty: int
    suggested_price: float


@dataclass(frozen=True)
class DebateConfig:
    """Tunable parameters for the debate engine."""

    rounds: int = 3
    confidence_skip_below: int = 60
    confidence_log_below: int = 80
    confidence_trade_above: int = 80
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
