"""Tests for Claude API agent wrappers in the adversarial debate pipeline.

All tests mock the Anthropic client so no real API calls are made.

Covers:
  - _call_claude returns text from mocked response
  - _call_claude with response_schema appends schema to system prompt
  - run_technical_analysis calls Claude and returns parsed dict
  - run_bull_case returns plain text from Claude
  - run_bear_case passes both analysis report AND bull argument to Claude
  - run_risk_assessment parses JSON response into RiskAssessment dataclass
  - run_portfolio_decision parses JSON response into TradeDecision dataclass
  - Invalid JSON from Claude raises ValueError
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.debate.agents import (
    _call_claude,
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
    DebateRound,
    RiskAssessment,
    TradeDecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(response_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns *response_text*."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def _sample_report() -> AnalysisReport:
    """Reusable AnalysisReport fixture."""
    return AnalysisReport(
        symbol="RELIANCE",
        exchange="NSE",
        timestamp="2026-06-09T10:00:00",
        technical={"rsi": 55.0, "macd": "bullish", "trend": "bullish"},
        fundamental={"pe_ratio": 25.3, "margin_trend": "expanding"},
        sentiment={"fii_flow": "buying", "pcr": 1.1},
        combined_signal="bullish",
        signal_strength=0.75,
    )


def _sample_debate_rounds() -> list[DebateRound]:
    """Reusable debate rounds fixture."""
    return [
        DebateRound(
            round_number=1,
            bull_argument="Strong momentum and sector tailwinds.",
            bear_argument="Overvalued on PE basis, FII outflows rising.",
        ),
    ]


MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# _call_claude
# ---------------------------------------------------------------------------

class TestCallClaude:
    """Low-level helper that wraps the Anthropic SDK call."""

    @patch("src.debate.agents.anthropic.Anthropic")
    def test_returns_text_from_mocked_response(self, mock_cls: MagicMock):
        client = _mock_client("Hello from Claude")
        mock_cls.return_value = client

        result = _call_claude(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            model=MODEL,
        )

        assert result == "Hello from Claude"
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == MODEL
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello."}]

    @patch("src.debate.agents.anthropic.Anthropic")
    def test_appends_schema_to_system_prompt(self, mock_cls: MagicMock):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        client = _mock_client('{"name": "test"}')
        mock_cls.return_value = client

        _call_claude(
            system_prompt="You are helpful.",
            user_message="Give me JSON.",
            model=MODEL,
            response_schema=schema,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        system_sent = call_kwargs["system"]
        assert "You are helpful." in system_sent
        assert json.dumps(schema) in system_sent
        assert "Respond ONLY with valid JSON matching this schema" in system_sent


# ---------------------------------------------------------------------------
# Analysis agents
# ---------------------------------------------------------------------------

class TestTechnicalAnalysis:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_returns_parsed_dict(self, mock_cls: MagicMock):
        response_json = {
            "rsi": 55.0,
            "macd": "bullish",
            "trend": "bullish",
            "support": 2400.0,
            "resistance": 2600.0,
            "volume_profile": "confirming",
            "fibonacci_levels": {"0.236": 2450, "0.382": 2480, "0.5": 2500, "0.618": 2520},
            "summary": "Bullish momentum with strong volume.",
        }
        client = _mock_client(json.dumps(response_json))
        mock_cls.return_value = client

        result = run_technical_analysis(
            symbol="RELIANCE",
            market_data={"close": [2400, 2450, 2500]},
            model=MODEL,
        )

        assert isinstance(result, dict)
        assert result["rsi"] == 55.0
        assert result["trend"] == "bullish"


class TestFundamentalAnalysis:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_returns_parsed_dict(self, mock_cls: MagicMock):
        response_json = {
            "pe_ratio": 25.3,
            "margin_trend": "expanding",
            "sector_signal": "inflow",
            "relative_valuation": "fairly_valued",
            "summary": "Solid fundamentals with expanding margins.",
        }
        client = _mock_client(json.dumps(response_json))
        mock_cls.return_value = client

        result = run_fundamental_analysis(
            symbol="RELIANCE",
            financial_data={"revenue": [100, 110, 120]},
            model=MODEL,
        )

        assert isinstance(result, dict)
        assert result["pe_ratio"] == 25.3


class TestSentimentAnalysis:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_returns_parsed_dict(self, mock_cls: MagicMock):
        response_json = {
            "fii_flow": "buying",
            "dii_flow": "neutral",
            "pcr": 1.15,
            "max_pain": 2500.0,
            "news_sentiment": "positive",
            "rbi_impact": "neutral",
            "summary": "FII inflows supportive, neutral RBI stance.",
        }
        client = _mock_client(json.dumps(response_json))
        mock_cls.return_value = client

        result = run_sentiment_analysis(
            symbol="RELIANCE",
            sentiment_data={"news": ["positive headline"]},
            model=MODEL,
        )

        assert isinstance(result, dict)
        assert result["fii_flow"] == "buying"
        assert result["pcr"] == 1.15


# ---------------------------------------------------------------------------
# Debate agents
# ---------------------------------------------------------------------------

class TestBullCase:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_returns_plain_text(self, mock_cls: MagicMock):
        bull_text = "RELIANCE shows strong bullish momentum with sector tailwinds."
        client = _mock_client(bull_text)
        mock_cls.return_value = client

        result = run_bull_case(
            analysis_report=_sample_report(),
            model=MODEL,
        )

        assert result == bull_text
        assert isinstance(result, str)


class TestBearCase:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_passes_report_and_bull_argument(self, mock_cls: MagicMock):
        bear_text = "The bullish case ignores rising FII outflows and margin compression."
        client = _mock_client(bear_text)
        mock_cls.return_value = client

        bull_arg = "RELIANCE has strong momentum and expanding margins."
        result = run_bear_case(
            analysis_report=_sample_report(),
            bull_argument=bull_arg,
            model=MODEL,
        )

        assert result == bear_text
        # Verify both analysis report and bull argument were sent to Claude
        call_kwargs = client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "RELIANCE" in user_message
        assert bull_arg in user_message


class TestRiskAssessment:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_parses_json_into_dataclass(self, mock_cls: MagicMock):
        risk_json = {
            "volatility_score": 0.65,
            "position_size_pct": 2.5,
            "max_exposure_pct": 10.0,
            "correlation_risk": "moderate",
            "sizing_method": "atr",
        }
        client = _mock_client(json.dumps(risk_json))
        mock_cls.return_value = client

        result = run_risk_assessment(
            analysis_report=_sample_report(),
            debate_rounds=_sample_debate_rounds(),
            model=MODEL,
        )

        assert isinstance(result, RiskAssessment)
        assert result.volatility_score == 0.65
        assert result.position_size_pct == 2.5
        assert result.max_exposure_pct == 10.0
        assert result.correlation_risk == "moderate"
        assert result.sizing_method == "atr"

    @patch("src.debate.agents.anthropic.Anthropic")
    def test_handles_markdown_code_fences(self, mock_cls: MagicMock):
        risk_json = {
            "volatility_score": 0.4,
            "position_size_pct": 1.5,
            "max_exposure_pct": 5.0,
            "correlation_risk": "low",
            "sizing_method": "kelly",
        }
        fenced_response = f"```json\n{json.dumps(risk_json)}\n```"
        client = _mock_client(fenced_response)
        mock_cls.return_value = client

        result = run_risk_assessment(
            analysis_report=_sample_report(),
            debate_rounds=_sample_debate_rounds(),
            model=MODEL,
        )

        assert isinstance(result, RiskAssessment)
        assert result.sizing_method == "kelly"


class TestPortfolioDecision:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_parses_json_into_dataclass(self, mock_cls: MagicMock):
        decision_json = {
            "action": "BUY",
            "confidence": 82,
            "signal_type": "technical",
            "reasoning": "Strong bullish momentum with manageable risk.",
            "suggested_qty": 10,
            "suggested_price": 2450.50,
        }
        client = _mock_client(json.dumps(decision_json))
        mock_cls.return_value = client

        risk = RiskAssessment(
            volatility_score=0.65,
            position_size_pct=2.5,
            max_exposure_pct=10.0,
            correlation_risk="moderate",
            sizing_method="atr",
        )

        result = run_portfolio_decision(
            analysis_report=_sample_report(),
            debate_rounds=_sample_debate_rounds(),
            risk=risk,
            model=MODEL,
        )

        assert isinstance(result, TradeDecision)
        assert result.action == "BUY"
        assert result.confidence == 82
        assert result.signal_type == "technical"
        assert result.reasoning == "Strong bullish momentum with manageable risk."
        assert result.suggested_qty == 10
        assert result.suggested_price == 2450.50
        assert result.symbol == "RELIANCE"
        assert result.exchange == "NSE"
        assert isinstance(result.risk_assessment, RiskAssessment)
        assert len(result.debate_rounds) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("src.debate.agents.anthropic.Anthropic")
    def test_invalid_json_raises_value_error(self, mock_cls: MagicMock):
        client = _mock_client("This is not valid JSON at all {{{")
        mock_cls.return_value = client

        with pytest.raises(ValueError, match="Failed to parse Claude response as JSON"):
            run_technical_analysis(
                symbol="RELIANCE",
                market_data={"close": [2400]},
                model=MODEL,
            )

    @patch("src.debate.agents.anthropic.Anthropic")
    def test_partial_json_raises_value_error(self, mock_cls: MagicMock):
        client = _mock_client('{"rsi": 55.0, "macd": "bullish"')  # truncated
        mock_cls.return_value = client

        with pytest.raises(ValueError, match="Failed to parse Claude response as JSON"):
            run_technical_analysis(
                symbol="RELIANCE",
                market_data={"close": [2400]},
                model=MODEL,
            )
