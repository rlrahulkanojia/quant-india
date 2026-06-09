"""Paper trading engine — shared dataclasses used across all modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time market data for a single symbol."""

    ltp: float
    bid: float
    ask: float
    volume: int
    avg_daily_volume: int


@dataclass(frozen=True)
class OrderRequest:
    """Immutable order intent submitted to the paper engine."""

    symbol: str
    exchange: str
    side: str       # "BUY" | "SELL"
    order_type: str  # "MARKET" | "LIMIT"
    qty: int
    limit_price: Optional[float] = None
