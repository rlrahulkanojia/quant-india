"""Debate engine orchestrator — composes analysis, debate, and scoring.

Orchestrates the full adversarial debate pipeline:

  1. **analyze** — runs technical, fundamental, and sentiment agents →
     :class:`AnalysisReport`
  2. **debate**  — multiple bull-vs-bear rounds, then risk assessment and
     portfolio decision → :class:`TradeDecision`
  3. **evaluate** — end-to-end pipeline: analyse → debate → score →
     ``(TradeDecision, ScoringResult)``

All LLM calls are delegated to the agent functions in
:mod:`src.debate.agents`; this module contains only orchestration logic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.debate.agents import (
    run_bear_case,
    run_bull_case,
    run_fundamental_analysis,
    run_portfolio_decision,
    run_risk_assessment,
    run_sentiment_analysis,
    run_technical_analysis,
)
from src.debate.schemas import (
    AnalysisReport,
    DebateConfig,
    DebateRound,
    TradeDecision,
)
from src.debate.scoring import ScoringResult, score_decision


class DebateEngine:
    """Orchestrates the full adversarial debate pipeline.

    Parameters
    ----------
    config:
        Engine parameters (rounds, confidence thresholds, model).
        Uses :class:`DebateConfig` defaults when *None*.
    """

    def __init__(self, config: DebateConfig | None = None) -> None:
        self._config = config or DebateConfig()

    # ------------------------------------------------------------------
    # Stage 1 — Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        exchange: str,
        market_data: dict,
        financial_data: dict,
        sentiment_data: dict,
    ) -> AnalysisReport:
        """Run three analysis agents and combine into a single report.

        Combined signal is determined by majority vote:
        * 2+ bullish → ``"bullish"``
        * 2+ bearish → ``"bearish"``
        * otherwise  → ``"neutral"``

        Signal strength is the average of individual ``"strength"`` values
        when present, falling back to ``0.5`` when all are missing.
        """
        model = self._config.model

        technical = run_technical_analysis(symbol, market_data, model=model)
        fundamental = run_fundamental_analysis(symbol, financial_data, model=model)
        sentiment = run_sentiment_analysis(symbol, sentiment_data, model=model)

        # -- Majority vote for combined signal --------------------------------
        signals = [
            technical.get("signal", "neutral"),
            fundamental.get("signal", "neutral"),
            sentiment.get("signal", "neutral"),
        ]
        bullish_count = sum(1 for s in signals if s == "bullish")
        bearish_count = sum(1 for s in signals if s == "bearish")

        if bullish_count >= 2:
            combined_signal = "bullish"
        elif bearish_count >= 2:
            combined_signal = "bearish"
        else:
            combined_signal = "neutral"

        # -- Average strength (fallback 0.5 when keys are absent) -------------
        strengths = [
            r.get("strength")
            for r in (technical, fundamental, sentiment)
        ]
        present = [s for s in strengths if s is not None]
        signal_strength = sum(present) / len(present) if present else 0.5

        return AnalysisReport(
            symbol=symbol,
            exchange=exchange,
            timestamp=datetime.now(timezone.utc).isoformat(),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            combined_signal=combined_signal,
            signal_strength=signal_strength,
        )

    # ------------------------------------------------------------------
    # Stage 2 — Debate
    # ------------------------------------------------------------------

    def debate(self, analysis_report: AnalysisReport) -> TradeDecision:
        """Run multiple bull-vs-bear debate rounds, then decide.

        For each round:
        * **Bull** argues using the analysis report (round 1) or rebuts
          the bear's previous argument (rounds 2+).
        * **Bear** rebuts the bull's current argument.

        After all rounds the risk manager and portfolio manager produce
        the final :class:`TradeDecision`.
        """
        model = self._config.model
        rounds: list[DebateRound] = []

        for round_num in range(1, self._config.rounds + 1):
            # Bull argues
            bull_argument = run_bull_case(
                analysis_report=analysis_report, model=model,
            )

            # Bear rebuts
            bear_argument = run_bear_case(
                analysis_report=analysis_report,
                bull_argument=bull_argument,
                model=model,
            )

            rounds.append(
                DebateRound(
                    round_number=round_num,
                    bull_argument=bull_argument,
                    bear_argument=bear_argument,
                )
            )

        # Risk assessment over all rounds
        risk = run_risk_assessment(
            analysis_report=analysis_report,
            debate_rounds=rounds,
            model=model,
        )

        # Portfolio manager final decision
        decision = run_portfolio_decision(
            analysis_report=analysis_report,
            debate_rounds=rounds,
            risk=risk,
            model=model,
        )

        return decision

    # ------------------------------------------------------------------
    # Stage 3 — Full pipeline
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        exchange: str,
        market_data: dict,
        financial_data: dict,
        sentiment_data: dict,
    ) -> tuple[TradeDecision, ScoringResult]:
        """End-to-end pipeline: analyze → debate → score.

        Returns
        -------
        tuple[TradeDecision, ScoringResult]
            The trade decision and its confidence scoring bucket.
        """
        report = self.analyze(symbol, exchange, market_data, financial_data, sentiment_data)
        decision = self.debate(report)
        scoring = score_decision(decision, self._config)
        return decision, scoring
