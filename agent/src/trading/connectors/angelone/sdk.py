"""Read-only + order AngelOne connector via the official ``smartapi-python`` SDK.

Wraps ``SmartApi.SmartConnect`` for market data (LTP, candles) and order
placement. Supports NSE/BSE equities, F&O, and MCX commodities.

Paper-vs-live: AngelOne has no sandbox environment. Paper mode uses the same
API for market data reads but simulates orders locally. Live mode would place
real orders through AngelOne's production SmartAPI endpoints — but this
connector is structurally capped at paper-only for safety.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Config:
    """AngelOne connector connection settings loaded from environment."""

    api_key: str = ""
    client_id: str = ""
    password: str = ""
    totp_secret: str = ""
    profile: str = "paper"

    @property
    def is_paper(self) -> bool:
        return self.profile == "paper"


def _load_config() -> _Config:
    """Build config from ANGELONE_* environment variables.

    Defaults profile to ``paper`` — the only supported mode. All order
    methods check ``is_paper`` before proceeding.
    """
    return _Config(
        api_key=os.environ.get("ANGELONE_API_KEY", ""),
        client_id=os.environ.get("ANGELONE_CLIENT_ID", ""),
        password=os.environ.get("ANGELONE_PASSWORD", ""),
        totp_secret=os.environ.get("ANGELONE_TOTP_SECRET", ""),
        profile=os.environ.get("ANGELONE_PROFILE", "paper"),
    )


#: Returned by order methods when a non-paper config reaches them. AngelOne
#: exposes no runtime paper/live discriminator, so — following the Dhan
#: precedent — the connector is structurally capped at paper and never opens
#: a live order path.
_PAPER_ONLY_ERROR = (
    "AngelOne connector is paper-only: live order placement is not "
    "supported. Set ANGELONE_PROFILE=paper to use simulated orders."
)


# ---------------------------------------------------------------------------
# Market data (reads — work in any mode)
# ---------------------------------------------------------------------------

def get_ltp(symbol: str, exchange: str = "NSE") -> dict[str, Any]:
    """Fetch last traded price from AngelOne SmartAPI.

    Uses real AngelOne API — works regardless of paper/live profile since
    this is a read-only operation. Lazy-imports ``SmartApi`` and ``pyotp``
    to avoid hard dependency at module level.
    """
    cfg = _load_config()
    client = _smart_connect(cfg)

    try:
        data = client.ltpData(exchange, symbol, symbol)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "symbol": symbol}

    ltp = None
    if isinstance(data, dict) and data.get("data"):
        ltp = data["data"].get("ltp")

    return {
        "status": "ok",
        "symbol": symbol,
        "exchange": exchange,
        "ltp": ltp,
    }


# ---------------------------------------------------------------------------
# Order placement (paper-only, simulated locally)
# ---------------------------------------------------------------------------

def place_order(
    *,
    symbol: str,
    side: str,
    order_type: str = "MARKET",
    qty: int,
    limit_price: float | None = None,
) -> dict[str, Any]:
    """Place a PAPER-ONLY order on AngelOne (simulated locally).

    AngelOne exposes no sandbox, so paper orders are simulated. The very
    first check refuses any config whose ``is_paper`` is not True. There
    is therefore no live order path here, by design.

    Args:
        symbol: Trading symbol (e.g. ``RELIANCE-EQ``).
        side: ``BUY`` or ``SELL``.
        order_type: ``MARKET`` or ``LIMIT``.
        qty: Number of shares/lots (must be > 0).
        limit_price: Required for LIMIT orders.
    """
    cfg = _load_config()

    # ---- HARD GUARD: structurally paper-only (must run before anything) ----
    if not cfg.is_paper:
        return {"status": "error", "error": _PAPER_ONLY_ERROR}

    # ---- Input validation ----
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        return {"status": "error", "error": "symbol is required"}

    side_token = str(side or "").strip().upper()
    if side_token not in ("BUY", "SELL"):
        return {"status": "error", "error": "side must be 'BUY' or 'SELL'"}

    type_token = str(order_type or "").strip().upper()
    if type_token not in ("MARKET", "LIMIT"):
        return {"status": "error", "error": "order_type must be 'MARKET' or 'LIMIT'"}

    if qty is None or int(qty) <= 0:
        return {"status": "error", "error": "qty must be positive"}

    clean_qty = int(qty)

    if type_token == "LIMIT" and limit_price is None:
        return {"status": "error", "error": "limit order requires limit_price"}

    price = float(limit_price) if limit_price is not None else 0

    ts = int(time.time())

    # Paper-only: simulate locally (AngelOne has no sandbox).
    return {
        "status": "ok",
        "order_id": f"PAPER-{clean_symbol}-{side_token}-{clean_qty}-{ts}",
        "symbol": clean_symbol,
        "side": side_token.lower(),
        "profile": cfg.profile,
        "is_paper": True,
        "paper_guard": "simulated_locally",
        "order_type": type_token.lower(),
        "quantity": clean_qty,
        "limit_price": price if type_token == "LIMIT" else None,
        "order_status": "simulated_fill",
    }


def cancel_order(*, order_id: str) -> dict[str, Any]:
    """Cancel a PAPER-ONLY order on AngelOne (simulated locally).

    Like :func:`place_order`, the first check refuses any non-paper config —
    this connector never reaches a live order, so it never cancels one.
    """
    cfg = _load_config()

    # ---- HARD GUARD: structurally paper-only (must run before anything) ----
    if not cfg.is_paper:
        return {"status": "error", "error": _PAPER_ONLY_ERROR}

    clean_id = str(order_id or "").strip()
    if not clean_id:
        return {"status": "error", "error": "order_id is required"}

    return {
        "status": "ok",
        "order_id": clean_id,
        "profile": cfg.profile,
        "is_paper": True,
        "cancelled": True,
    }


# ---------------------------------------------------------------------------
# SDK plumbing
# ---------------------------------------------------------------------------

def _smart_connect(cfg: _Config):
    """Create an authenticated SmartConnect session.

    Lazy-imports ``SmartApi`` and ``pyotp`` so the module loads even when
    these optional packages are not installed.
    """
    from SmartApi import SmartConnect  # type: ignore
    import pyotp  # type: ignore

    if not cfg.api_key or not cfg.client_id:
        raise RuntimeError(
            "AngelOne connector not configured: set ANGELONE_API_KEY and "
            "ANGELONE_CLIENT_ID environment variables."
        )

    obj = SmartConnect(api_key=cfg.api_key)
    totp = pyotp.TOTP(cfg.totp_secret).now() if cfg.totp_secret else ""
    obj.generateSession(cfg.client_id, cfg.password, totp)
    return obj
