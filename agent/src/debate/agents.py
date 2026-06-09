"""Claude API agent wrappers for the adversarial debate pipeline.

Each function calls Claude via the Anthropic SDK with a role-specific
system prompt and returns either plain text (advocacy roles) or parsed
JSON (analysis/decision roles).

Agents:
  Analysis  : run_technical_analysis, run_fundamental_analysis, run_sentiment_analysis
  Advocacy  : run_bull_case, run_bear_case
  Decision  : run_risk_assessment, run_portfolio_decision
"""

from __future__ import annotations

import json
import re

import anthropic

from src.debate.prompts import (
    BEAR_RESEARCHER_PROMPT,
    BULL_RESEARCHER_PROMPT,
    FUNDAMENTAL_ANALYST_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    RISK_MANAGER_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
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

def _call_claude(
    system_prompt: str,
    user_message: str,
    model: str,
    response_schema: dict | None = None,
) -> str:
    """Send a single message to Claude and return the text response.

    Parameters
    ----------
    system_prompt:
        The system-level instruction for the Claude model.
    user_message:
        The user-turn content.
    model:
        Anthropic model identifier (e.g. ``claude-sonnet-4-20250514``).
    response_schema:
        If provided, appended to the system prompt as a JSON-output
        instruction so Claude returns structured data.

    Returns
    -------
    str
        The text content of Claude's first content block.
    """
    if response_schema is not None:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema: "
            f"{json.dumps(response_schema)}"
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response, stripping markdown code fences if present.

    Raises
    ------
    ValueError
        If the response cannot be parsed as valid JSON.
    """
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    stripped = re.sub(
        r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
        r"\1",
        text.strip(),
        flags=re.DOTALL,
    )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse Claude response as JSON: {text[:200]}"
        )


# ---------------------------------------------------------------------------
# Analysis agents
# ---------------------------------------------------------------------------

def run_technical_analysis(
    symbol: str,
    market_data: dict,
    model: str,
) -> dict:
    """Run the technical analyst agent and return parsed JSON output."""
    user_message = (
        f"Analyse the following market data for {symbol}:\n\n"
        f"{json.dumps(market_data, indent=2)}"
    )
    response = _call_claude(
        system_prompt=TECHNICAL_ANALYST_PROMPT,
        user_message=user_message,
        model=model,
    )
    return _parse_json_response(response)


def run_fundamental_analysis(
    symbol: str,
    financial_data: dict,
    model: str,
) -> dict:
    """Run the fundamental analyst agent and return parsed JSON output."""
    user_message = (
        f"Analyse the following financial data for {symbol}:\n\n"
        f"{json.dumps(financial_data, indent=2)}"
    )
    response = _call_claude(
        system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
        user_message=user_message,
        model=model,
    )
    return _parse_json_response(response)


def run_sentiment_analysis(
    symbol: str,
    sentiment_data: dict,
    model: str,
) -> dict:
    """Run the sentiment analyst agent and return parsed JSON output."""
    user_message = (
        f"Analyse the following sentiment data for {symbol}:\n\n"
        f"{json.dumps(sentiment_data, indent=2)}"
    )
    response = _call_claude(
        system_prompt=SENTIMENT_ANALYST_PROMPT,
        user_message=user_message,
        model=model,
    )
    return _parse_json_response(response)


# ---------------------------------------------------------------------------
# Debate agents
# ---------------------------------------------------------------------------

def run_bull_case(
    analysis_report: AnalysisReport,
    model: str,
) -> str:
    """Run the bull researcher agent — returns plain-text argument."""
    user_message = (
        f"Build the bullish case for {analysis_report.symbol} "
        f"on {analysis_report.exchange}.\n\n"
        f"Analysis Report:\n"
        f"  Technical: {json.dumps(analysis_report.technical)}\n"
        f"  Fundamental: {json.dumps(analysis_report.fundamental)}\n"
        f"  Sentiment: {json.dumps(analysis_report.sentiment)}\n"
        f"  Combined signal: {analysis_report.combined_signal} "
        f"(strength: {analysis_report.signal_strength})"
    )
    return _call_claude(
        system_prompt=BULL_RESEARCHER_PROMPT,
        user_message=user_message,
        model=model,
    )


def run_bear_case(
    analysis_report: AnalysisReport,
    bull_argument: str,
    model: str,
) -> str:
    """Run the bear researcher agent — returns plain-text rebuttal.

    Receives both the analysis report and the bull argument to rebut.
    """
    user_message = (
        f"Argue against the proposed trade for {analysis_report.symbol} "
        f"on {analysis_report.exchange}.\n\n"
        f"Analysis Report:\n"
        f"  Technical: {json.dumps(analysis_report.technical)}\n"
        f"  Fundamental: {json.dumps(analysis_report.fundamental)}\n"
        f"  Sentiment: {json.dumps(analysis_report.sentiment)}\n"
        f"  Combined signal: {analysis_report.combined_signal} "
        f"(strength: {analysis_report.signal_strength})\n\n"
        f"Bull Researcher's Argument:\n{bull_argument}"
    )
    return _call_claude(
        system_prompt=BEAR_RESEARCHER_PROMPT,
        user_message=user_message,
        model=model,
    )


# ---------------------------------------------------------------------------
# Decision agents
# ---------------------------------------------------------------------------

def run_risk_assessment(
    analysis_report: AnalysisReport,
    debate_rounds: list[DebateRound],
    model: str,
) -> RiskAssessment:
    """Run the risk manager agent and parse response into RiskAssessment."""
    rounds_text = "\n".join(
        f"  Round {r.round_number}:\n"
        f"    Bull: {r.bull_argument}\n"
        f"    Bear: {r.bear_argument}"
        for r in debate_rounds
    )
    user_message = (
        f"Assess risk for a potential trade in {analysis_report.symbol} "
        f"on {analysis_report.exchange}.\n\n"
        f"Analysis Report:\n"
        f"  Technical: {json.dumps(analysis_report.technical)}\n"
        f"  Fundamental: {json.dumps(analysis_report.fundamental)}\n"
        f"  Sentiment: {json.dumps(analysis_report.sentiment)}\n"
        f"  Combined signal: {analysis_report.combined_signal} "
        f"(strength: {analysis_report.signal_strength})\n\n"
        f"Debate Rounds:\n{rounds_text}"
    )
    response = _call_claude(
        system_prompt=RISK_MANAGER_PROMPT,
        user_message=user_message,
        model=model,
    )
    data = _parse_json_response(response)
    # Extract only the fields the RiskAssessment dataclass expects
    return RiskAssessment(
        volatility_score=data["volatility_score"],
        position_size_pct=data["position_size_pct"],
        max_exposure_pct=data["max_exposure_pct"],
        correlation_risk=data["correlation_risk"],
        sizing_method=data["sizing_method"],
    )


def run_portfolio_decision(
    analysis_report: AnalysisReport,
    debate_rounds: list[DebateRound],
    risk: RiskAssessment,
    model: str,
) -> TradeDecision:
    """Run the portfolio manager agent and parse response into TradeDecision."""
    rounds_text = "\n".join(
        f"  Round {r.round_number}:\n"
        f"    Bull: {r.bull_argument}\n"
        f"    Bear: {r.bear_argument}"
        for r in debate_rounds
    )
    user_message = (
        f"Make a final trade decision for {analysis_report.symbol} "
        f"on {analysis_report.exchange}.\n\n"
        f"Analysis Report:\n"
        f"  Technical: {json.dumps(analysis_report.technical)}\n"
        f"  Fundamental: {json.dumps(analysis_report.fundamental)}\n"
        f"  Sentiment: {json.dumps(analysis_report.sentiment)}\n"
        f"  Combined signal: {analysis_report.combined_signal} "
        f"(strength: {analysis_report.signal_strength})\n\n"
        f"Debate Rounds:\n{rounds_text}\n\n"
        f"Risk Assessment:\n"
        f"  Volatility: {risk.volatility_score}\n"
        f"  Position size: {risk.position_size_pct}%\n"
        f"  Max exposure: {risk.max_exposure_pct}%\n"
        f"  Correlation risk: {risk.correlation_risk}\n"
        f"  Sizing method: {risk.sizing_method}"
    )
    response = _call_claude(
        system_prompt=PORTFOLIO_MANAGER_PROMPT,
        user_message=user_message,
        model=model,
    )
    data = _parse_json_response(response)
    return TradeDecision(
        symbol=analysis_report.symbol,
        exchange=analysis_report.exchange,
        action=data["action"],
        confidence=data["confidence"],
        signal_type=data["signal_type"],
        debate_rounds=list(debate_rounds),
        risk_assessment=risk,
        reasoning=data["reasoning"],
        suggested_qty=data["suggested_qty"],
        suggested_price=data["suggested_price"],
    )
