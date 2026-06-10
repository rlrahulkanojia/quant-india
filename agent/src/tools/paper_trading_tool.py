"""Paper trading tools — execute paper trades, view portfolio, analyze symbols.

These tools bridge the Agent chat with the PaperTradingEngine, letting users
trade via natural language:

  "Buy 100 shares of RELIANCE"
  "Show my portfolio"
  "Analyze INFY, TCS, HDFCBANK for momentum signals"
  "Run paper trades on the best signals"

All trades use fake money (paper mode) with live AngelOne market data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.agent.tools import BaseTool

# ── Shared helpers ────────────────────────────────────────────────────────────

_PAPER_DB = os.path.expanduser(os.environ.get("DATA_DIR", "~/.quant-india") + "/paper.db")
_DECISION_DB = os.path.expanduser(os.environ.get("DATA_DIR", "~/.quant-india") + "/decisions.db")

SYMBOL_TOKENS = {
    "RELIANCE": "2885", "INFY": "1594", "TCS": "11536", "HDFCBANK": "1333",
    "ICICIBANK": "4963", "SBIN": "3045", "TATAMOTORS": "3456", "WIPRO": "3787",
    "BAJFINANCE": "317", "LT": "11483", "SUNPHARMA": "3351", "MARUTI": "10999",
    "ASIANPAINT": "236", "TITAN": "3506", "AXISBANK": "5900", "KOTAKBANK": "1922",
    "ADANIENT": "25", "HINDUNILVR": "1394", "BHARTIARTL": "10604", "ITC": "1660",
}


def _ensure_dirs():
    os.makedirs(os.path.dirname(_PAPER_DB), exist_ok=True)


def _get_angelone_client():
    """Create authenticated AngelOne SmartConnect session."""
    from SmartApi import SmartConnect
    import pyotp

    api_key = os.environ.get("ANGELONE_API_KEY", "")
    client_id = os.environ.get("ANGELONE_CLIENT_ID", "")
    password = os.environ.get("ANGELONE_PASSWORD", "")
    totp_secret = os.environ.get("ANGELONE_TOTP_SECRET", "")

    if not api_key or not client_id:
        raise RuntimeError("AngelOne credentials not configured — set ANGELONE_API_KEY and ANGELONE_CLIENT_ID in .env")

    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now() if totp_secret else ""
    obj.generateSession(client_id, password, totp)
    return obj


def _fetch_market_snapshot(client, symbol: str):
    """Fetch live OHLC data from AngelOne and return (MarketSnapshot, ohlc_dict)."""
    from src.paper import MarketSnapshot

    token = SYMBOL_TOKENS.get(symbol.upper())
    if not token:
        return None, None

    data = client.ltpData("NSE", f"{symbol.upper()}-EQ", token)
    if not data.get("status") or not data.get("data"):
        return None, None

    d = data["data"]
    ltp = d.get("ltp", 0)
    high = d.get("high", ltp)
    low = d.get("low", ltp)
    day_range = high - low if high > low else ltp * 0.01
    spread = max(day_range * 0.01, ltp * 0.0005)

    snapshot = MarketSnapshot(
        ltp=ltp,
        bid=round(ltp - spread / 2, 2),
        ask=round(ltp + spread / 2, 2),
        volume=50000,
        avg_daily_volume=5_000_000,
    )
    ohlc = {"open": d.get("open", ltp), "high": high, "low": low, "close": d.get("close", ltp), "ltp": ltp}
    return snapshot, ohlc


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# ── Tool 1: Execute Paper Trade ───────────────────────────────────────────────


class PaperTradeExecuteTool(BaseTool):
    """Execute a paper trade using live AngelOne market data."""

    name = "paper_trade_execute"
    description = (
        "Execute a paper trade (fake money) on an Indian stock using live market data from AngelOne. "
        "Supports BUY and SELL for NSE stocks. Returns fill price, fees, slippage, and position update. "
        "Available symbols: RELIANCE, INFY, TCS, HDFCBANK, ICICIBANK, SBIN, TATAMOTORS, WIPRO, "
        "BAJFINANCE, LT, SUNPHARMA, MARUTI, ASIANPAINT, TITAN, AXISBANK, KOTAKBANK, ADANIENT, "
        "HINDUNILVR, BHARTIARTL, ITC."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock symbol (e.g. RELIANCE, INFY, TCS)",
            },
            "side": {
                "type": "string",
                "enum": ["BUY", "SELL"],
                "description": "Trade direction — BUY or SELL",
            },
            "qty": {
                "type": "integer",
                "description": "Number of shares to trade",
            },
            "order_type": {
                "type": "string",
                "enum": ["MARKET", "LIMIT"],
                "description": "Order type — MARKET (default) or LIMIT",
                "default": "MARKET",
            },
            "limit_price": {
                "type": "number",
                "description": "Limit price (required only for LIMIT orders)",
            },
        },
        "required": ["symbol", "side", "qty"],
    }
    is_readonly = False

    @classmethod
    def check_available(cls) -> bool:
        return bool(os.environ.get("ANGELONE_API_KEY"))

    def execute(self, symbol: str, side: str, qty: int, order_type: str = "MARKET", limit_price: float | None = None, **_: Any) -> str:
        from src.paper.engine import PaperTradingEngine

        symbol = symbol.upper().replace("-EQ", "")
        if symbol not in SYMBOL_TOKENS:
            return _json({"status": "error", "error": f"Unknown symbol '{symbol}'. Supported: {', '.join(sorted(SYMBOL_TOKENS))}"})

        try:
            _ensure_dirs()
            client = _get_angelone_client()
            snapshot, ohlc = _fetch_market_snapshot(client, symbol)
            if snapshot is None:
                return _json({"status": "error", "error": f"Could not fetch market data for {symbol}"})

            engine = PaperTradingEngine(db_path=_PAPER_DB, starting_capital=1_000_000.0)

            result = engine.execute_order(
                symbol=symbol,
                exchange="NSE",
                side=side.upper(),
                order_type=order_type.upper(),
                qty=int(qty),
                market=snapshot,
                limit_price=limit_price,
                trade_type="equity_intraday",
            )

            return _json({
                "status": "ok",
                "filled": result.filled,
                "order_id": result.order_id,
                "symbol": symbol,
                "side": side.upper(),
                "qty": result.filled_qty,
                "fill_price": result.fill_price,
                "slippage": result.slippage,
                "fees_total": result.fees.total,
                "fees_breakdown": {
                    "stt": result.fees.stt,
                    "exchange": result.fees.exchange_charges,
                    "sebi": result.fees.sebi,
                    "gst": result.fees.gst,
                    "stamp": result.fees.stamp_duty,
                },
                "position_action": result.position_update.action,
                "market_ltp": snapshot.ltp,
                "market_open": ohlc["open"],
                "market_high": ohlc["high"],
                "market_low": ohlc["low"],
                "day_change_pct": round(((ohlc["ltp"] - ohlc["close"]) / ohlc["close"]) * 100, 2),
            })
        except Exception as e:
            return _json({"status": "error", "error": str(e)})


# ── Tool 2: Portfolio Summary ─────────────────────────────────────────────────


class PaperTradePortfolioTool(BaseTool):
    """Show current paper trading portfolio — cash, positions, P&L."""

    name = "paper_trade_portfolio"
    description = (
        "Show the current paper trading portfolio including cash balance, open positions, "
        "unrealized P&L, realized P&L, and total fees paid. Use this to check portfolio "
        "status before or after trades."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, **_: Any) -> str:
        from src.paper.store import PaperStore

        if not os.path.exists(_PAPER_DB):
            return _json({"status": "ok", "initialized": False, "message": "No paper portfolio yet. Execute a trade first."})

        try:
            store = PaperStore(_PAPER_DB)
            portfolio = store.get_portfolio()
            if portfolio is None:
                return _json({"status": "ok", "initialized": False, "message": "No paper portfolio yet."})

            positions = store.get_positions()
            orders = store.list_orders()
            total_fees = sum(o.get("fees_total", 0) or 0 for o in orders if o.get("status") == "FILLED")

            return _json({
                "status": "ok",
                "initialized": True,
                "cash_balance": portfolio["cash_balance"],
                "realized_pnl": portfolio.get("total_realized", 0),
                "total_fees_paid": total_fees,
                "positions": [
                    {
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "qty": p["qty"],
                        "avg_price": p["avg_price"],
                    }
                    for p in positions
                ],
                "position_count": len(positions),
                "total_trades": len([o for o in orders if o.get("status") == "FILLED"]),
            })
        except Exception as e:
            return _json({"status": "error", "error": str(e)})


# ── Tool 3: Analyze Symbol ────────────────────────────────────────────────────


class PaperTradeAnalyzeTool(BaseTool):
    """Analyze one or more stocks using live AngelOne data — returns momentum signals."""

    name = "paper_trade_analyze"
    description = (
        "Analyze Indian stocks using live market data from AngelOne. Returns OHLC data, "
        "day change %, position in day range, and a momentum signal (BUY/SELL/HOLD with confidence). "
        "Use this to decide which stocks to trade before calling paper_trade_execute. "
        "Pass a comma-separated list of symbols to analyze multiple stocks at once."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "string",
                "description": "Comma-separated stock symbols (e.g. 'RELIANCE,INFY,TCS')",
            },
        },
        "required": ["symbols"],
    }

    @classmethod
    def check_available(cls) -> bool:
        return bool(os.environ.get("ANGELONE_API_KEY"))

    def execute(self, symbols: str, **_: Any) -> str:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not symbol_list:
            return _json({"status": "error", "error": "No symbols provided"})

        try:
            client = _get_angelone_client()
        except Exception as e:
            return _json({"status": "error", "error": f"AngelOne connection failed: {e}"})

        results = []
        for symbol in symbol_list:
            if symbol not in SYMBOL_TOKENS:
                results.append({"symbol": symbol, "status": "error", "error": "Unknown symbol"})
                continue

            snapshot, ohlc = _fetch_market_snapshot(client, symbol)
            if snapshot is None:
                results.append({"symbol": symbol, "status": "error", "error": "No market data"})
                continue

            ltp = ohlc["ltp"]
            open_price = ohlc["open"]
            high = ohlc["high"]
            low = ohlc["low"]
            prev_close = ohlc["close"]
            day_range = high - low if high > low else 1
            day_change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            position_in_range = round((ltp - low) / day_range, 2)
            above_open = ltp > open_price

            # Momentum signal
            if position_in_range >= 0.75 and above_open and day_change_pct > 0.3:
                signal = "BUY"
                confidence = min(90, int(65 + position_in_range * 25 + day_change_pct * 3))
                signal_type = "momentum_breakout"
            elif position_in_range <= 0.25 and not above_open and day_change_pct < -0.3:
                signal = "SELL"
                confidence = min(85, int(65 + (1 - position_in_range) * 20 + abs(day_change_pct) * 3))
                signal_type = "momentum_breakdown"
            elif position_in_range >= 0.6 and day_change_pct > 0:
                signal = "BUY"
                confidence = int(55 + position_in_range * 15 + day_change_pct * 2)
                signal_type = "mild_bullish"
            elif position_in_range <= 0.4 and day_change_pct < 0:
                signal = "SELL"
                confidence = int(55 + (1 - position_in_range) * 15 + abs(day_change_pct) * 2)
                signal_type = "mild_bearish"
            else:
                signal = "HOLD"
                confidence = 40
                signal_type = "range_bound"

            suggested_qty = max(1, int(50000 / ltp))

            results.append({
                "symbol": symbol,
                "status": "ok",
                "ltp": ltp,
                "open": open_price,
                "high": high,
                "low": low,
                "prev_close": prev_close,
                "day_change_pct": day_change_pct,
                "position_in_range": position_in_range,
                "above_open": above_open,
                "signal": signal,
                "confidence": min(confidence, 95),
                "signal_type": signal_type,
                "suggested_qty": suggested_qty,
                "suggested_value": round(suggested_qty * ltp, 2),
            })

            time.sleep(0.3)  # Rate limiting

        # Summary
        tradeable = [r for r in results if r.get("signal") in ("BUY", "SELL") and r.get("confidence", 0) >= 60]

        return _json({
            "status": "ok",
            "analyzed": len(results),
            "tradeable_signals": len(tradeable),
            "results": results,
            "summary": (
                f"Analyzed {len(results)} stocks. "
                f"{len(tradeable)} have tradeable signals (confidence >= 60). "
                + (
                    "Top signals: " + ", ".join(
                        f"{r['symbol']} {r['signal']} ({r['confidence']}%)"
                        for r in sorted(tradeable, key=lambda x: -x.get("confidence", 0))[:5]
                    )
                    if tradeable
                    else "No strong signals found."
                )
            ),
        })
