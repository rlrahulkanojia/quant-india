#!/usr/bin/env python3
"""Paper Trading Runner — fetch live data, run AI debate, execute paper trades.

Usage:
    python run_paper_trading.py                    # Default watchlist
    python run_paper_trading.py --symbols RELIANCE INFY TCS
    python run_paper_trading.py --capital 500000   # ₹5 lakh starting capital
    python run_paper_trading.py --dry-run          # Analyze only, no trades
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add agent/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paper import MarketSnapshot, OrderRequest
from src.paper.engine import PaperTradingEngine
from src.paper.fees import FeeConfig
from src.debate.schemas import DebateConfig, AnalysisReport, TradeDecision
from src.debate.scoring import ScoringResult, score_decision, should_execute
from src.decisions.store import DecisionStore
from src.decisions.decision_log import DecisionLogger

# ── AngelOne helpers ──────────────────────────────────────────────────────────

def _get_angelone_client():
    """Create authenticated AngelOne SmartConnect session."""
    from SmartApi import SmartConnect
    import pyotp

    api_key = os.environ["ANGELONE_API_KEY"]
    client_id = os.environ["ANGELONE_CLIENT_ID"]
    password = os.environ["ANGELONE_PASSWORD"]
    totp_secret = os.environ["ANGELONE_TOTP_SECRET"]

    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    data = obj.generateSession(client_id, password, totp)
    if not data.get("status"):
        raise RuntimeError(f"AngelOne login failed: {data}")
    print(f"  ✓ AngelOne authenticated as {client_id}")
    return obj


# Symbol token mapping (AngelOne needs numeric tokens)
SYMBOL_TOKENS = {
    "RELIANCE": "2885",
    "INFY": "1594",
    "TCS": "11536",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "SBIN": "3045",
    "TATAMOTORS": "3456",
    "WIPRO": "3787",
    "BAJFINANCE": "317",
    "LT": "11483",
    "SUNPHARMA": "3351",
    "MARUTI": "10999",
    "ASIANPAINT": "236",
    "TITAN": "3506",
    "AXISBANK": "5900",
    "KOTAKBANK": "1922",
    "ADANIENT": "25",
    "HINDUNILVR": "1394",
    "BHARTIARTL": "10604",
    "ITC": "1660",
}


def fetch_market_data(client, symbol: str) -> tuple[MarketSnapshot, dict] | None:
    """Fetch live market data from AngelOne. Returns (MarketSnapshot, ohlc_dict) or None."""
    token = SYMBOL_TOKENS.get(symbol)
    if not token:
        print(f"  ⚠ No token mapping for {symbol}, skipping")
        return None

    try:
        ltp_data = client.ltpData("NSE", f"{symbol}-EQ", token)
        if not ltp_data.get("status") or not ltp_data.get("data"):
            print(f"  ⚠ No data for {symbol}: {ltp_data.get('message', 'unknown')}")
            return None

        d = ltp_data["data"]
        ltp = d.get("ltp", 0)
        open_price = d.get("open", ltp)
        high = d.get("high", ltp)
        low = d.get("low", ltp)
        close = d.get("close", ltp)  # Previous close

        # Estimate bid/ask from day range
        day_range = high - low if high > low else ltp * 0.01
        spread_est = max(day_range * 0.01, ltp * 0.0005)  # 1% of day range or 0.05%

        ohlc = {"open": open_price, "high": high, "low": low, "close": close, "ltp": ltp}

        return MarketSnapshot(
            ltp=ltp,
            bid=round(ltp - spread_est / 2, 2),
            ask=round(ltp + spread_est / 2, 2),
            volume=50000,
            avg_daily_volume=5_000_000,
        ), ohlc
    except Exception as e:
        print(f"  ⚠ Error fetching {symbol}: {e}")
        return None


# ── AI Analysis (Claude) ─────────────────────────────────────────────────────

def run_ai_analysis(symbol: str, market: MarketSnapshot, config: DebateConfig) -> tuple[TradeDecision, ScoringResult] | None:
    """Run full AI debate pipeline on a symbol."""
    from src.debate.engine import DebateEngine

    engine = DebateEngine(config=config)

    market_data = {
        "ltp": market.ltp,
        "bid": market.bid,
        "ask": market.ask,
        "volume": market.volume,
        "spread_pct": round((market.ask - market.bid) / market.ltp * 100, 4),
    }
    financial_data = {"symbol": symbol, "exchange": "NSE"}
    sentiment_data = {"symbol": symbol, "exchange": "NSE"}

    try:
        decision, scoring = engine.evaluate(
            symbol=symbol,
            exchange="NSE",
            market_data=market_data,
            financial_data=financial_data,
            sentiment_data=sentiment_data,
        )
        return decision, scoring
    except Exception as e:
        print(f"  ⚠ AI analysis failed for {symbol}: {e}")
        return None


# ── Simple Technical Strategy (no LLM needed) ────────────────────────────────

def simple_momentum_strategy(symbol: str, market: MarketSnapshot, ohlc: dict) -> tuple[TradeDecision, ScoringResult]:
    """OHLC-based momentum strategy — no LLM needed.

    Signals:
    - BUY: LTP near day high, trading above open (bullish momentum)
    - SELL: LTP near day low, trading below open (bearish momentum)
    - HOLD: LTP in middle of day range (no clear direction)
    """
    from src.debate.schemas import RiskAssessment, DebateRound

    ltp = market.ltp
    open_price = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    prev_close = ohlc["close"]

    day_range = high - low if high > low else 1
    day_change_pct = ((ltp - prev_close) / prev_close) * 100
    position_in_range = (ltp - low) / day_range  # 0 = at low, 1 = at high
    above_open = ltp > open_price

    # ── Signal logic ──────────────────────────────────────────────────────
    if position_in_range >= 0.75 and above_open and day_change_pct > 0.3:
        action = "BUY"
        confidence = min(90, int(65 + position_in_range * 25 + day_change_pct * 3))
        signal_type = "momentum_breakout"
        reasoning = (
            f"{symbol} bullish momentum — LTP ₹{ltp:,.2f} near day high ₹{high:,.2f} "
            f"({position_in_range:.0%} of range), up {day_change_pct:+.2f}% from prev close ₹{prev_close:,.2f}, "
            f"trading above open ₹{open_price:,.2f}"
        )

    elif position_in_range <= 0.25 and not above_open and day_change_pct < -0.3:
        action = "SELL"
        confidence = min(85, int(65 + (1 - position_in_range) * 20 + abs(day_change_pct) * 3))
        signal_type = "momentum_breakdown"
        reasoning = (
            f"{symbol} bearish momentum — LTP ₹{ltp:,.2f} near day low ₹{low:,.2f} "
            f"({position_in_range:.0%} of range), down {day_change_pct:+.2f}% from prev close ₹{prev_close:,.2f}, "
            f"trading below open ₹{open_price:,.2f}"
        )

    elif position_in_range >= 0.6 and day_change_pct > 0:
        action = "BUY"
        confidence = int(55 + position_in_range * 15 + day_change_pct * 2)
        signal_type = "mild_bullish"
        reasoning = (
            f"{symbol} mild bullish — LTP ₹{ltp:,.2f} in upper half of range "
            f"(₹{low:,.2f}–₹{high:,.2f}), up {day_change_pct:+.2f}%"
        )

    elif position_in_range <= 0.4 and day_change_pct < 0:
        action = "SELL"
        confidence = int(55 + (1 - position_in_range) * 15 + abs(day_change_pct) * 2)
        signal_type = "mild_bearish"
        reasoning = (
            f"{symbol} mild bearish — LTP ₹{ltp:,.2f} in lower half of range "
            f"(₹{low:,.2f}–₹{high:,.2f}), down {day_change_pct:+.2f}%"
        )

    else:
        action = "HOLD"
        confidence = 40
        signal_type = "range_bound"
        reasoning = (
            f"{symbol} range-bound — LTP ₹{ltp:,.2f} at {position_in_range:.0%} of day range "
            f"(₹{low:,.2f}–₹{high:,.2f}), change {day_change_pct:+.2f}%"
        )

    risk = RiskAssessment(
        volatility_score=round(day_range / ltp, 3),  # Day range as % of price
        position_size_pct=2.0,
        max_exposure_pct=5.0,
        correlation_risk="low",
        sizing_method="fixed",
    )

    decision = TradeDecision(
        symbol=symbol,
        exchange="NSE",
        action=action,
        confidence=min(confidence, 95),  # Cap at 95
        signal_type=signal_type,
        debate_rounds=[DebateRound(
            round_number=1,
            bull_argument=f"Day range position: {position_in_range:.0%}, change: {day_change_pct:+.2f}%, above open: {above_open}",
            bear_argument=f"Simple strategy — single timeframe, no volume confirmation",
        )],
        risk_assessment=risk,
        reasoning=reasoning,
        suggested_qty=max(1, int(50000 / ltp)),  # ~₹50K per position
        suggested_price=ltp,
    )

    config = DebateConfig()
    scoring = score_decision(decision, config)

    return decision, scoring


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Stock symbols to trade (default: random 5 from NIFTY50)")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Starting capital in ₹ (default: 10,00,000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only — don't execute paper trades")
    parser.add_argument("--use-ai", action="store_true",
                        help="Use Claude AI for analysis (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of random stocks to pick (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Quant India — Paper Trading Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 60)

    # ── Pick symbols ──────────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        import random
        all_symbols = list(SYMBOL_TOKENS.keys())
        symbols = random.sample(all_symbols, min(args.count, len(all_symbols)))

    print(f"\n📋 Watchlist: {', '.join(symbols)}")
    print(f"💰 Starting capital: ₹{args.capital:,.0f}")
    print(f"🤖 Strategy: {'AI Debate (Claude)' if args.use_ai else 'Simple Momentum'}")
    print(f"{'🔍 DRY RUN — no trades will be executed' if args.dry_run else '📈 PAPER TRADING — simulated orders'}")

    # ── Connect to AngelOne ───────────────────────────────────────────────
    print("\n🔌 Connecting to AngelOne SmartAPI...")
    client = _get_angelone_client()

    # ── Initialize paper trading engine ───────────────────────────────────
    db_path = os.path.expanduser("~/.quant-india/paper.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    paper_engine = PaperTradingEngine(
        db_path=db_path,
        starting_capital=args.capital,
    )

    decision_db_path = os.path.expanduser("~/.quant-india/decisions.db")
    decision_store = DecisionStore(decision_db_path)
    logger = DecisionLogger(decision_store)

    # ── Fetch market data + analyze ───────────────────────────────────────
    print("\n📊 Fetching live market data...\n")

    results = []
    for symbol in symbols:
        print(f"  {'─' * 50}")
        print(f"  📈 {symbol}")

        fetch_result = fetch_market_data(client, symbol)
        if fetch_result is None:
            continue
        market, ohlc = fetch_result

        day_change = ((ohlc["ltp"] - ohlc["close"]) / ohlc["close"]) * 100
        print(f"     LTP: ₹{market.ltp:,.2f}  |  Open: ₹{ohlc['open']:,.2f}  |  High: ₹{ohlc['high']:,.2f}  |  Low: ₹{ohlc['low']:,.2f}  |  Chg: {day_change:+.2f}%")

        # Run analysis
        if args.use_ai:
            result = run_ai_analysis(symbol, market, DebateConfig())
            if result is None:
                continue
            decision, scoring = result
        else:
            decision, scoring = simple_momentum_strategy(symbol, market, ohlc)

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "SKIP": "⚪"}.get(decision.action, "⚪")
        score_emoji = {"TRADE": "✅", "LOG": "📝", "SKIP": "⏭️"}.get(scoring.action, "❓")

        print(f"     Signal: {action_emoji} {decision.action}  |  Confidence: {decision.confidence}/100  |  Score: {score_emoji} {scoring.action}")
        print(f"     Reasoning: {decision.reasoning[:80]}...")

        results.append((symbol, market, decision, scoring))

        # Small delay between API calls
        time.sleep(0.5)

    # ── Execute trades ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TRADE EXECUTION")
    print(f"{'=' * 60}\n")

    trades_executed = 0
    for symbol, market, decision, scoring in results:
        if not should_execute(scoring):
            print(f"  ⏭️  {symbol}: {scoring.action} — {scoring.reason}")
            continue

        if args.dry_run:
            print(f"  🔍 {symbol}: Would {decision.action} {decision.suggested_qty} shares @ ₹{market.ltp:,.2f} (dry run)")
            continue

        if decision.action not in ("BUY", "SELL"):
            continue

        print(f"  📝 {symbol}: Executing {decision.action} {decision.suggested_qty} shares @ market...")

        trade_result = paper_engine.execute_order(
            symbol=symbol,
            exchange="NSE",
            side=decision.action,
            order_type="MARKET",
            qty=decision.suggested_qty,
            market=market,
            trade_type="equity_intraday",
        )

        if trade_result.filled:
            print(f"     ✅ Filled: {trade_result.filled_qty} @ ₹{trade_result.fill_price:,.2f}")
            print(f"     💸 Fees: ₹{trade_result.fees.total:,.2f}  |  Slippage: ₹{trade_result.slippage:,.2f}")
            trades_executed += 1

            # Log decision
            analysis = AnalysisReport(
                symbol=symbol, exchange="NSE",
                timestamp=datetime.now().isoformat(),
                technical={"ltp": market.ltp},
                fundamental={},
                sentiment={},
                combined_signal="bullish" if decision.action == "BUY" else "bearish",
                signal_strength=decision.confidence / 100,
            )
            logger.log_decision(decision, scoring, analysis, paper_order_id=trade_result.order_id)
        else:
            print(f"     ❌ Not filled: {trade_result.position_update.action}")

    # ── Portfolio Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'=' * 60}\n")

    current_prices = {}
    for symbol, market, _, _ in results:
        current_prices[symbol] = market.ltp

    summary = paper_engine.get_summary(current_prices)
    print(f"  💰 Cash Balance:    ₹{summary.cash:>12,.2f}")
    print(f"  📊 Unrealized P&L:  ₹{summary.unrealized_pnl:>12,.2f}")
    print(f"  📈 Realized P&L:    ₹{summary.realized_pnl:>12,.2f}")
    print(f"  💸 Total Fees:      ₹{summary.total_fees_paid:>12,.2f}")
    print(f"  🏦 Total Value:     ₹{summary.total_value:>12,.2f}")

    if summary.positions:
        print(f"\n  Open Positions:")
        print(f"  {'Symbol':<15} {'Side':<6} {'Qty':>6} {'Avg Price':>12} {'Current':>12} {'P&L':>12}")
        print(f"  {'─' * 65}")
        for pos in summary.positions:
            pnl = (pos.get('current_price', 0) - pos['avg_price']) * pos['qty']
            if pos['side'] == 'SHORT':
                pnl = -pnl
            print(f"  {pos['symbol']:<15} {pos['side']:<6} {pos['qty']:>6} ₹{pos['avg_price']:>10,.2f} ₹{pos.get('current_price', 0):>10,.2f} ₹{pnl:>10,.2f}")

    print(f"\n  Trades executed: {trades_executed}")
    print(f"  Paper DB: {db_path}")
    print(f"  Decision DB: {decision_db_path}")

    # ── Shadow Report ─────────────────────────────────────────────────────
    shadow = paper_engine.get_shadow_report(days=1)
    if shadow.fill_count > 0:
        print(f"\n  📊 Shadow Account (today):")
        print(f"     Fills: {shadow.fill_count}")
        print(f"     Avg Divergence: {shadow.avg_divergence_pct:.4f}%")

    print(f"\n{'=' * 60}")
    print("  Done! Run again anytime to analyze and trade.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
