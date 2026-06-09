"""Built-in AngelOne connector profiles.

AngelOne (https://angelone.in, formerly Angel Broking) is an Indian full-service
broker with SmartAPI access. Supports NSE/BSE equities, F&O (NIFTY/BANKNIFTY/
FINNIFTY options), currency, and commodity segments.

Paper-only by design: AngelOne's SmartAPI exposes no sandbox environment and no
runtime paper/live discriminator — a single API key hits the same live account.
Following the Dhan/Longbridge precedent, this connector ships read-only
paper/live profiles plus a locally simulated paper-trade profile, and exposes
NO live order placement. Paper vs live is operator-declared (config-trust);
the connector's order path is structurally capped at paper.
"""

from __future__ import annotations

from src.trading.types import READ_CAPABILITIES, TradingProfile

ANGELONE_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="angelone-paper-sdk",
        connector="angelone",
        label="AngelOne Paper · SmartAPI (India)",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "paper"},
        notes=(
            "Reads real-time Indian market data (NSE/BSE) via AngelOne SmartAPI. "
            "Paper vs live is operator-declared (the API exposes no runtime "
            "discriminator). Supports equities, F&O (NIFTY/BANKNIFTY/FINNIFTY "
            "options), currency, commodity."
        ),
    ),
    TradingProfile(
        id="angelone-paper-trade",
        connector="angelone",
        label="AngelOne Paper · SmartAPI Trade (India)",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper"},
        notes=(
            "Places PAPER orders simulated locally using real AngelOne market "
            "data — no real money at risk. Paper-only by design: AngelOne "
            "SmartAPI exposes no runtime paper/live discriminator, so live "
            "order placement is not supported. Supports NSE equities and F&O "
            "(NIFTY/BANKNIFTY/FINNIFTY options)."
        ),
    ),
    TradingProfile(
        id="angelone-live-readonly",
        connector="angelone",
        label="AngelOne Live · SmartAPI Read-Only (India)",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "live-readonly"},
        notes=(
            "Reads a live AngelOne account (account, positions, orders, quotes, "
            "history). Order placement is not exposed in this profile."
        ),
    ),
)
