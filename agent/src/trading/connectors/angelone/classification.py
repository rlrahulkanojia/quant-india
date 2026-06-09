"""Curated read/write classification for AngelOne SmartAPI SDK operations.

Keys are the SmartAPI SDK method names. Order-mutating SDK calls are pinned
WRITE so the live gate never treats them as plain reads; anything unlisted and
not a known read is treated as WRITE (fail-closed) by the gate.
"""

from __future__ import annotations

from src.live.classification import ToolClass

#: AngelOne SmartAPI SDK operation read/write catalog.
ANGELONE_TOOL_CLASS: dict[str, ToolClass] = {
    # ── READ ───────────────────────────────────────────
    "ltpData":       ToolClass.READ,
    "getMarketData": ToolClass.READ,
    "getCandleData": ToolClass.READ,
    "orderBook":     ToolClass.READ,
    "tradeBook":     ToolClass.READ,
    "position":      ToolClass.READ,
    "holding":       ToolClass.READ,
    "rmsLimit":      ToolClass.READ,
    # ── WRITE ──────────────────────────────────────────
    "placeOrder":    ToolClass.WRITE,
    "modifyOrder":   ToolClass.WRITE,
    "cancelOrder":   ToolClass.WRITE,
}
