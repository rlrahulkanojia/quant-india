"""Indian fee calculator for paper trading.

Computes realistic transaction costs for equity intraday, equity delivery,
and commodity trades on Indian exchanges (NSE/BSE/MCX).

All rates are configurable via FeeConfig; defaults reflect 2024-25 SEBI/exchange
schedules.  The calculator is pure — no I/O, no state — suitable for both
backtesting and live shadow accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_CRORE = 1_00_00_000  # 10^7


@dataclass(frozen=True)
class FeeConfig:
    """Fee rates for Indian market transactions.

    All rates are expressed as fractions of trade value unless noted otherwise.
    """

    # Securities Transaction Tax (equity)
    stt_intraday_sell: float = 0.00025   # 0.025% on sell side only
    stt_delivery: float = 0.001          # 0.1% on both sides

    # Commodity Transaction Tax (commodity segment)
    ctt_sell: float = 0.0001             # 0.01% on sell side only

    # Exchange transaction charges
    exchange_equity: float = 0.0000345   # 0.00345%
    exchange_commodity: float = 0.000026 # 0.0026%

    # SEBI turnover fee
    sebi_per_crore: float = 10.0         # ₹10 per crore of turnover

    # GST (on brokerage only)
    gst_rate: float = 0.18               # 18%

    # Stamp duty (on BUY side only)
    stamp_buy_intraday: float = 0.00003  # 0.003%
    stamp_buy_delivery: float = 0.00015  # 0.015%
    stamp_buy_commodity: float = 0.00002 # 0.002%


@dataclass(frozen=True)
class FeeBreakdown:
    """Itemised transaction cost breakdown."""

    stt: float             # STT or CTT (commodity uses this field for CTT)
    exchange_charges: float
    sebi: float
    gst: float             # 18% on brokerage only
    stamp_duty: float      # charged on BUY side only
    brokerage: float
    total: float


def calculate_fees(
    trade_value: float,
    side: str,
    trade_type: str,
    brokerage: float = 0.0,
    config: Optional[FeeConfig] = None,
) -> FeeBreakdown:
    """Compute realistic Indian transaction fees.

    Args:
        trade_value: Absolute trade value in ₹ (price * qty).
        side: ``"BUY"`` or ``"SELL"``.
        trade_type: One of ``"equity_intraday"``, ``"equity_delivery"``,
            ``"commodity"``.
        brokerage: Flat brokerage amount in ₹ (default 0).
        config: Optional rate overrides; uses :class:`FeeConfig` defaults
            when *None*.

    Returns:
        A frozen :class:`FeeBreakdown` with all cost components and total.
    """
    if config is None:
        config = FeeConfig()

    is_buy = side.upper() == "BUY"

    # --- Short-circuit on zero value ---
    if trade_value == 0:
        return FeeBreakdown(
            stt=0.0,
            exchange_charges=0.0,
            sebi=0.0,
            gst=0.0,
            stamp_duty=0.0,
            brokerage=0.0,
            total=0.0,
        )

    # --- STT / CTT ---
    stt = 0.0
    if trade_type == "equity_intraday":
        if not is_buy:
            stt = trade_value * config.stt_intraday_sell
    elif trade_type == "equity_delivery":
        stt = trade_value * config.stt_delivery  # both sides
    elif trade_type == "commodity":
        if not is_buy:
            stt = trade_value * config.ctt_sell   # CTT stored in stt field

    # --- Exchange transaction charges ---
    if trade_type in ("equity_intraday", "equity_delivery"):
        exchange_charges = trade_value * config.exchange_equity
    else:  # commodity
        exchange_charges = trade_value * config.exchange_commodity

    # --- SEBI turnover fee ---
    sebi = trade_value * config.sebi_per_crore / _CRORE

    # --- GST (on brokerage only) ---
    gst = brokerage * config.gst_rate

    # --- Stamp duty (BUY side only) ---
    stamp_duty = 0.0
    if is_buy:
        if trade_type == "equity_intraday":
            stamp_duty = trade_value * config.stamp_buy_intraday
        elif trade_type == "equity_delivery":
            stamp_duty = trade_value * config.stamp_buy_delivery
        elif trade_type == "commodity":
            stamp_duty = trade_value * config.stamp_buy_commodity

    total = stt + exchange_charges + sebi + gst + stamp_duty + brokerage

    return FeeBreakdown(
        stt=stt,
        exchange_charges=exchange_charges,
        sebi=sebi,
        gst=gst,
        stamp_duty=stamp_duty,
        brokerage=brokerage,
        total=total,
    )
