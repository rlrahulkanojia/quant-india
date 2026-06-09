"""Paper trading fill simulator.

Simulates realistic order fills for paper trading by modelling:
  - Bid-ask spread impact on market orders
  - Limit order price-level checks
  - Partial fills for large orders (>5% avg daily volume)

All prices are in INR. Slippage is always reported as a positive number
representing the distance from LTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.paper import MarketSnapshot, OrderRequest

# Fraction of LTP used as half-spread when bid/ask are missing or zero.
_FALLBACK_HALF_SPREAD_PCT = 0.0005

# Maximum fillable fraction of average daily volume per order.
_MAX_ADV_FILL_PCT = 0.05


@dataclass(frozen=True)
class FillResult:
    """Immutable result of a simulated fill attempt."""

    filled: bool
    fill_price: float
    filled_qty: int
    slippage: float
    reason: Optional[str] = None


def _half_spread(market: MarketSnapshot) -> float:
    """Compute half the bid-ask spread, with a fallback for missing quotes."""
    if market.bid > 0 and market.ask > 0:
        return (market.ask - market.bid) / 2
    return market.ltp * _FALLBACK_HALF_SPREAD_PCT


def _clamp_qty(requested_qty: int, avg_daily_volume: int) -> int:
    """Cap fill quantity to 5% of average daily volume."""
    max_fillable = int(avg_daily_volume * _MAX_ADV_FILL_PCT)
    return min(requested_qty, max_fillable)


def simulate_fill(order: OrderRequest, market: MarketSnapshot) -> FillResult:
    """Simulate filling *order* against the current *market* snapshot.

    Returns a ``FillResult`` describing whether the order filled,
    at what price, the filled quantity, and the slippage incurred.
    """
    if order.order_type == "MARKET":
        return _fill_market(order, market)
    if order.order_type == "LIMIT":
        return _fill_limit(order, market)
    return FillResult(
        filled=False,
        fill_price=0.0,
        filled_qty=0,
        slippage=0.0,
        reason=f"unsupported order_type: {order.order_type}",
    )


# ------------------------------------------------------------------
# Market orders
# ------------------------------------------------------------------

def _fill_market(order: OrderRequest, market: MarketSnapshot) -> FillResult:
    hs = _half_spread(market)
    filled_qty = _clamp_qty(order.qty, market.avg_daily_volume)

    if order.side == "BUY":
        fill_price = market.ltp + hs
        slippage = fill_price - market.ltp  # always positive
    else:  # SELL
        fill_price = market.ltp - hs
        slippage = market.ltp - fill_price  # always positive

    return FillResult(
        filled=True,
        fill_price=fill_price,
        filled_qty=filled_qty,
        slippage=slippage,
    )


# ------------------------------------------------------------------
# Limit orders
# ------------------------------------------------------------------

def _fill_limit(order: OrderRequest, market: MarketSnapshot) -> FillResult:
    assert order.limit_price is not None, "LIMIT order requires limit_price"

    filled_qty = _clamp_qty(order.qty, market.avg_daily_volume)

    if order.side == "BUY":
        if market.ltp <= order.limit_price:
            return FillResult(
                filled=True,
                fill_price=order.limit_price,
                filled_qty=filled_qty,
                slippage=0.0,
            )
        return FillResult(
            filled=False,
            fill_price=0.0,
            filled_qty=0,
            slippage=0.0,
            reason=f"ltp {market.ltp} > limit {order.limit_price}",
        )

    # SELL
    if market.ltp >= order.limit_price:
        return FillResult(
            filled=True,
            fill_price=order.limit_price,
            filled_qty=filled_qty,
            slippage=0.0,
        )
    return FillResult(
        filled=False,
        fill_price=0.0,
        filled_qty=0,
        slippage=0.0,
        reason=f"ltp {market.ltp} < limit {order.limit_price}",
    )
