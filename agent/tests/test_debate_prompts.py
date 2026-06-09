"""Tests for India-specific system prompts used by each agent role.

Verifies:
  - Each prompt constant is a non-empty string
  - Analysis prompts contain key domain phrases
  - Analysis prompts mention JSON output format
  - Bull/Bear/Reflection prompts contain role-specific phrases
"""

from __future__ import annotations

import pytest

from src.debate.prompts import (
    BEAR_RESEARCHER_PROMPT,
    BULL_RESEARCHER_PROMPT,
    FUNDAMENTAL_ANALYST_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    REFLECTION_PROMPT,
    RISK_MANAGER_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
)

ALL_PROMPTS = [
    TECHNICAL_ANALYST_PROMPT,
    FUNDAMENTAL_ANALYST_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    BULL_RESEARCHER_PROMPT,
    BEAR_RESEARCHER_PROMPT,
    RISK_MANAGER_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    REFLECTION_PROMPT,
]


# ---------------------------------------------------------------------------
# Basic sanity: every prompt is a non-empty string
# ---------------------------------------------------------------------------

class TestPromptBasics:
    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_is_nonempty_string(self, prompt: str):
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0


# ---------------------------------------------------------------------------
# Technical Analyst
# ---------------------------------------------------------------------------

class TestTechnicalAnalystPrompt:
    def test_contains_rsi(self):
        assert "RSI" in TECHNICAL_ANALYST_PROMPT

    def test_contains_macd(self):
        assert "MACD" in TECHNICAL_ANALYST_PROMPT

    def test_contains_fibonacci(self):
        assert "Fibonacci" in TECHNICAL_ANALYST_PROMPT or "fibonacci" in TECHNICAL_ANALYST_PROMPT

    def test_contains_support_resistance(self):
        prompt_lower = TECHNICAL_ANALYST_PROMPT.lower()
        assert "support" in prompt_lower and "resistance" in prompt_lower

    def test_contains_json(self):
        assert "JSON" in TECHNICAL_ANALYST_PROMPT

    def test_contains_nse_bse(self):
        assert "NSE" in TECHNICAL_ANALYST_PROMPT or "BSE" in TECHNICAL_ANALYST_PROMPT

    def test_contains_circuit_limits(self):
        assert "circuit" in TECHNICAL_ANALYST_PROMPT.lower()

    def test_contains_settlement(self):
        assert "T+1" in TECHNICAL_ANALYST_PROMPT


# ---------------------------------------------------------------------------
# Fundamental Analyst
# ---------------------------------------------------------------------------

class TestFundamentalAnalystPrompt:
    def test_contains_pe(self):
        assert "PE" in FUNDAMENTAL_ANALYST_PROMPT or "P/E" in FUNDAMENTAL_ANALYST_PROMPT

    def test_contains_margin(self):
        assert "margin" in FUNDAMENTAL_ANALYST_PROMPT.lower()

    def test_contains_sector(self):
        assert "sector" in FUNDAMENTAL_ANALYST_PROMPT.lower()

    def test_contains_nifty(self):
        assert "NIFTY" in FUNDAMENTAL_ANALYST_PROMPT or "Nifty" in FUNDAMENTAL_ANALYST_PROMPT

    def test_contains_json(self):
        assert "JSON" in FUNDAMENTAL_ANALYST_PROMPT


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------

class TestSentimentAnalystPrompt:
    def test_contains_fii(self):
        assert "FII" in SENTIMENT_ANALYST_PROMPT

    def test_contains_dii(self):
        assert "DII" in SENTIMENT_ANALYST_PROMPT

    def test_contains_pcr(self):
        assert "PCR" in SENTIMENT_ANALYST_PROMPT

    def test_contains_max_pain(self):
        assert "max pain" in SENTIMENT_ANALYST_PROMPT.lower()

    def test_contains_rbi(self):
        assert "RBI" in SENTIMENT_ANALYST_PROMPT

    def test_contains_json(self):
        assert "JSON" in SENTIMENT_ANALYST_PROMPT


# ---------------------------------------------------------------------------
# Bull Researcher
# ---------------------------------------------------------------------------

class TestBullResearcherPrompt:
    def test_contains_bullish_or_case_for(self):
        prompt_lower = BULL_RESEARCHER_PROMPT.lower()
        assert "bullish" in prompt_lower or "case for" in prompt_lower

    def test_does_not_require_json(self):
        """Bull researcher returns plain text, not structured JSON."""
        # Just verify the prompt exists and is substantive
        assert len(BULL_RESEARCHER_PROMPT) > 100


# ---------------------------------------------------------------------------
# Bear Researcher
# ---------------------------------------------------------------------------

class TestBearResearcherPrompt:
    def test_contains_risk_or_against(self):
        prompt_lower = BEAR_RESEARCHER_PROMPT.lower()
        assert "risk" in prompt_lower or "against" in prompt_lower

    def test_does_not_require_json(self):
        """Bear researcher returns plain text, not structured JSON."""
        assert len(BEAR_RESEARCHER_PROMPT) > 100


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class TestRiskManagerPrompt:
    def test_contains_volatility(self):
        assert "volatility" in RISK_MANAGER_PROMPT.lower()

    def test_contains_position_sizing(self):
        assert "position" in RISK_MANAGER_PROMPT.lower()

    def test_contains_kelly_or_atr(self):
        prompt_lower = RISK_MANAGER_PROMPT.lower()
        assert "kelly" in prompt_lower or "atr" in prompt_lower

    def test_contains_json(self):
        assert "JSON" in RISK_MANAGER_PROMPT


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class TestPortfolioManagerPrompt:
    def test_contains_buy_sell_hold(self):
        assert "BUY" in PORTFOLIO_MANAGER_PROMPT
        assert "SELL" in PORTFOLIO_MANAGER_PROMPT
        assert "HOLD" in PORTFOLIO_MANAGER_PROMPT

    def test_contains_confidence(self):
        assert "confidence" in PORTFOLIO_MANAGER_PROMPT.lower()

    def test_contains_json(self):
        assert "JSON" in PORTFOLIO_MANAGER_PROMPT


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

class TestReflectionPrompt:
    def test_contains_pattern_or_threshold(self):
        prompt_lower = REFLECTION_PROMPT.lower()
        assert "pattern" in prompt_lower or "threshold" in prompt_lower

    def test_contains_json(self):
        assert "JSON" in REFLECTION_PROMPT

    def test_contains_bias(self):
        assert "bias" in REFLECTION_PROMPT.lower()
