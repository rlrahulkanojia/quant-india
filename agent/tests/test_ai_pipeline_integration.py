"""Full AI pipeline integration test — end-to-end from debate to calibration.

Exercises the entire decision lifecycle:
  DebateEngine.evaluate() → DecisionLogger.log_decision()
  → PaperTradingEngine.execute_order() → DecisionLogger.close_trade()
  → ReflectionService.run_weekly_reflection()
  → CalibrationService.apply_reflection()
  → CalibrationService.get_current_config()

ALL Claude API calls are mocked — no real LLM traffic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.debate.engine import DebateEngine
from src.debate.schemas import DebateConfig, TradeDecision
from src.debate.scoring import ScoringResult
from src.decisions.calibration import CalibrationService
from src.decisions.decision_log import DecisionLogger
from src.decisions.reflection import ReflectionService
from src.decisions.store import DecisionStore
from src.paper import MarketSnapshot
from src.paper.engine import PaperTradingEngine


# ---------------------------------------------------------------------------
# Shared realistic RELIANCE data fixtures
# ---------------------------------------------------------------------------

MARKET_DATA = {
    "symbol": "RELIANCE",
    "ltp": 2850.0,
    "open": 2820.0,
    "high": 2865.0,
    "low": 2810.0,
    "close": 2845.0,
    "volume": 12_500_000,
    "avg_volume": 10_000_000,
    "52w_high": 3000.0,
    "52w_low": 2200.0,
}

FINANCIAL_DATA = {
    "symbol": "RELIANCE",
    "pe_ratio": 28.5,
    "eps": 100.0,
    "market_cap": "19.2L Cr",
    "revenue_growth": 12.3,
    "profit_margin": 10.5,
    "debt_to_equity": 0.45,
    "roe": 12.8,
}

SENTIMENT_DATA = {
    "symbol": "RELIANCE",
    "fii_flow": "positive",
    "dii_flow": "neutral",
    "pcr": 1.2,
    "max_pain": 2850,
    "news_headlines": ["Reliance Jio reports strong subscriber growth"],
    "rbi_policy": "neutral",
}


# -- Mock Claude responses for each agent role -----------------------------

MOCK_TECHNICAL = json.dumps({
    "rsi": 62,
    "macd": "bullish",
    "trend": "uptrend",
    "support": 2800,
    "resistance": 2950,
    "volume_profile": "increasing",
    "summary": "Bullish momentum",
    "signal": "bullish",
    "strength": 0.7,
})

MOCK_FUNDAMENTAL = json.dumps({
    "pe_ratio": 28.5,
    "margin_trend": "expanding",
    "sector_signal": "bullish",
    "relative_valuation": "fair",
    "summary": "Strong fundamentals",
    "signal": "bullish",
    "strength": 0.65,
})

MOCK_SENTIMENT = json.dumps({
    "fii_flow": "positive",
    "dii_flow": "neutral",
    "pcr": 1.2,
    "max_pain": 2850,
    "news_sentiment": "positive",
    "rbi_impact": "neutral",
    "summary": "Positive flows",
    "signal": "bullish",
    "strength": 0.6,
})

MOCK_BULL_CASE = (
    "Strong momentum with expanding margins and positive FII flows "
    "support upside to 2950."
)

MOCK_BEAR_CASE = (
    "Valuation stretched at 28.5x PE with resistance at 2950."
)

MOCK_RISK = json.dumps({
    "volatility_score": 0.35,
    "position_size_pct": 2.0,
    "max_exposure_pct": 5.0,
    "correlation_risk": "low",
    "sizing_method": "atr",
    "rationale": "Moderate volatility",
})

MOCK_PORTFOLIO_BUY = json.dumps({
    "action": "BUY",
    "confidence": 85,
    "signal_type": "momentum_breakout",
    "reasoning": "Strong bullish consensus",
    "suggested_qty": 100,
    "suggested_price": 2850.0,
})

MOCK_PORTFOLIO_SKIP = json.dumps({
    "action": "HOLD",
    "confidence": 40,
    "signal_type": "weak_signal",
    "reasoning": "Insufficient conviction to trade",
    "suggested_qty": 0,
    "suggested_price": 2850.0,
})

MOCK_REFLECTION = json.dumps({
    "patterns_identified": [
        "Momentum signals outperformed in current regime",
    ],
    "threshold_recommendations": [
        {
            "parameter": "confidence_trade_above",
            "current_value": 80,
            "recommended_value": 75,
            "reason": "High win rate on momentum",
        },
    ],
    "bias_flags": [],
})

# Market snapshot for paper trades
RELIANCE_SNAPSHOT = MarketSnapshot(
    ltp=2850.0,
    bid=2849.0,
    ask=2851.0,
    volume=12_500_000,
    avg_daily_volume=10_000_000,
)

RELIANCE_EXIT_SNAPSHOT = MarketSnapshot(
    ltp=2900.0,
    bid=2899.0,
    ask=2901.0,
    volume=11_000_000,
    avg_daily_volume=10_000_000,
)


# ---------------------------------------------------------------------------
# Helper: build a mock Anthropic client that returns canned responses
# ---------------------------------------------------------------------------

def _make_mock_anthropic_client(responses: list[str]) -> MagicMock:
    """Create a mock Anthropic client cycling through *responses* in order.

    Each call to ``client.messages.create(...)`` returns the next response.
    """
    client = MagicMock()
    side_effects = []
    for text in responses:
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = text
        mock_response.content = [mock_content_block]
        side_effects.append(mock_response)
    client.messages.create.side_effect = side_effects
    return client


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of the entire AI decision pipeline."""

    def test_evaluate_to_calibration(self, tmp_path):
        """Full flow: evaluate → log → paper trade → close → reflect → calibrate."""
        # ── 1. Set up services ──────────────────────────────────────────
        decision_db = str(tmp_path / "decisions.db")
        paper_db = str(tmp_path / "paper.db")

        store = DecisionStore(decision_db)
        logger = DecisionLogger(store)
        paper_engine = PaperTradingEngine(paper_db, starting_capital=1_000_000.0)

        config = DebateConfig(
            rounds=1,  # single round for speed
            confidence_skip_below=60,
            confidence_log_below=80,
            confidence_trade_above=80,
        )
        engine = DebateEngine(config)

        # ── 2. Mock all Claude calls for the debate pipeline ────────────
        # With 1 round: tech, fund, sent, bull, bear, risk, portfolio = 7 calls
        debate_responses = [
            MOCK_TECHNICAL,
            MOCK_FUNDAMENTAL,
            MOCK_SENTIMENT,
            MOCK_BULL_CASE,
            MOCK_BEAR_CASE,
            MOCK_RISK,
            MOCK_PORTFOLIO_BUY,
        ]

        mock_client = _make_mock_anthropic_client(debate_responses)

        with patch("src.debate.agents.anthropic.Anthropic", return_value=mock_client):
            # ── 3. Run DebateEngine.evaluate() ──────────────────────────
            decision, scoring = engine.evaluate(
                symbol="RELIANCE",
                exchange="NSE",
                market_data=MARKET_DATA,
                financial_data=FINANCIAL_DATA,
                sentiment_data=SENTIMENT_DATA,
            )

        # ── 4. Assert decision ──────────────────────────────────────────
        assert isinstance(decision, TradeDecision)
        assert decision.action == "BUY"
        assert decision.confidence == 85
        assert decision.symbol == "RELIANCE"
        assert decision.exchange == "NSE"
        assert decision.signal_type == "momentum_breakout"
        assert decision.suggested_qty == 100
        assert decision.suggested_price == 2850.0

        # ── 5. Assert scoring ──────────────────────────────────────────
        assert isinstance(scoring, ScoringResult)
        assert scoring.action == "TRADE"
        assert scoring.confidence == 85

        # ── 6. Log decision ────────────────────────────────────────────
        # First execute the paper trade to get an order_id
        buy_result = paper_engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="BUY",
            order_type="MARKET",
            qty=decision.suggested_qty,
            market=RELIANCE_SNAPSHOT,
        )
        assert buy_result.filled is True
        assert buy_result.filled_qty == 100

        # Build a minimal AnalysisReport for logging
        from src.debate.schemas import AnalysisReport

        analysis = AnalysisReport(
            symbol="RELIANCE",
            exchange="NSE",
            timestamp=datetime.now().isoformat(),
            technical=json.loads(MOCK_TECHNICAL),
            fundamental=json.loads(MOCK_FUNDAMENTAL),
            sentiment=json.loads(MOCK_SENTIMENT),
            combined_signal="bullish",
            signal_strength=0.65,
        )

        decision_id = logger.log_decision(
            decision=decision,
            scoring=scoring,
            analysis=analysis,
            paper_order_id=buy_result.order_id,
        )
        assert isinstance(decision_id, int)
        assert decision_id >= 1

        # ── 7. Execute sell to close the paper trade ───────────────────
        sell_result = paper_engine.execute_order(
            symbol="RELIANCE",
            exchange="NSE",
            side="SELL",
            order_type="MARKET",
            qty=100,
            market=RELIANCE_EXIT_SNAPSHOT,
        )
        assert sell_result.filled is True

        # ── 8. Close trade with profit ─────────────────────────────────
        entry_price = buy_result.fill_price
        exit_price = sell_result.fill_price
        pnl_amount = (exit_price - entry_price) * 100  # approximate
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        logger.close_trade(
            decision_id=decision_id,
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            pnl_pct=pnl_pct,
            hold_duration_minutes=60,
            exit_reason="target_reached",
        )

        # Verify trade is closed in the store
        recent = logger.get_recent_decisions(days=1)
        closed_trade = next(
            (d for d in recent if d["id"] == decision_id), None
        )
        assert closed_trade is not None
        assert closed_trade["closed_at"] is not None
        assert closed_trade["pnl_amount"] is not None

        # ── 9. Run reflection ──────────────────────────────────────────
        reflection_service = ReflectionService(store)

        mock_reflection_client = _make_mock_anthropic_client([MOCK_REFLECTION])

        with patch(
            "src.decisions.reflection.anthropic.Anthropic",
            return_value=mock_reflection_client,
        ):
            report = reflection_service.run_weekly_reflection(
                as_of=datetime.now().isoformat(),
            )

        assert report.total_trades == 1
        assert report.win_rate == 1.0  # 1 winning trade out of 1
        assert len(report.findings) >= 1
        assert len(report.recommendations) == 1
        assert report.recommendations[0]["parameter"] == "confidence_trade_above"

        # ── 10. Apply calibration ──────────────────────────────────────
        calibration = CalibrationService(store, base_config=config)

        # We need the reflection_id — get the latest from the store
        latest_reflection = store.get_latest_reflection()
        assert latest_reflection is not None
        reflection_id = latest_reflection["id"]

        updates = calibration.apply_reflection(report, reflection_id)
        assert len(updates) == 1
        assert updates[0].parameter == "confidence_trade_above"
        assert updates[0].old_value == 80
        assert updates[0].new_value == 75

        # ── 11. Verify thresholds changed ──────────────────────────────
        new_config = calibration.get_current_config()
        assert new_config.confidence_trade_above == 75  # changed from 80
        # Other thresholds should remain unchanged
        assert new_config.confidence_skip_below == 60
        assert new_config.confidence_log_below == 80
        assert new_config.rounds == 1

    def test_skip_decision_not_executed(self, tmp_path):
        """Low-confidence decision → SKIP → no paper trade executed."""
        decision_db = str(tmp_path / "decisions.db")

        store = DecisionStore(decision_db)
        logger = DecisionLogger(store)

        config = DebateConfig(
            rounds=1,
            confidence_skip_below=60,
            confidence_log_below=80,
            confidence_trade_above=80,
        )
        engine = DebateEngine(config)

        # Mock: portfolio returns low-confidence HOLD
        debate_responses = [
            MOCK_TECHNICAL,
            MOCK_FUNDAMENTAL,
            MOCK_SENTIMENT,
            MOCK_BULL_CASE,
            MOCK_BEAR_CASE,
            MOCK_RISK,
            MOCK_PORTFOLIO_SKIP,
        ]

        mock_client = _make_mock_anthropic_client(debate_responses)

        with patch("src.debate.agents.anthropic.Anthropic", return_value=mock_client):
            decision, scoring = engine.evaluate(
                symbol="RELIANCE",
                exchange="NSE",
                market_data=MARKET_DATA,
                financial_data=FINANCIAL_DATA,
                sentiment_data=SENTIMENT_DATA,
            )

        # Decision should be HOLD with low confidence
        assert decision.action == "HOLD"
        assert decision.confidence == 40

        # Scoring should be SKIP (40 < 60 threshold)
        assert scoring.action == "SKIP"

        # In a SKIP scenario, no paper trade should be placed
        # and no decision should be logged (business rule)
        recent = store.list_decisions(since_date="2000-01-01")
        assert len(recent) == 0, "SKIP decisions should not be logged"

    def test_imports_clean(self):
        """All pipeline modules import without circular dependencies."""
        # These imports should work independently and together
        import src.debate.schemas
        import src.debate.agents
        import src.debate.engine
        import src.debate.scoring
        import src.debate.prompts
        import src.decisions.store
        import src.decisions.decision_log
        import src.decisions.reflection
        import src.decisions.calibration
        import src.paper
        import src.paper.engine
        import src.paper.fees
        import src.paper.fill_simulator
        import src.paper.portfolio
        import src.paper.shadow
        import src.paper.store

        # Verify key classes/functions are accessible
        assert hasattr(src.debate.engine, "DebateEngine")
        assert hasattr(src.debate.scoring, "score_decision")
        assert hasattr(src.decisions.decision_log, "DecisionLogger")
        assert hasattr(src.decisions.reflection, "ReflectionService")
        assert hasattr(src.decisions.calibration, "CalibrationService")
        assert hasattr(src.paper.engine, "PaperTradingEngine")
