"""Tests for connector-first trading profile operations (India-only fork)."""

from __future__ import annotations

import json

import pytest

from src.trading import profiles, service
from src.tools.trading_connector_tool import TradingSelectConnectionTool

pytestmark = pytest.mark.unit


def test_connector_profile_id_for_broker_fallback() -> None:
    """Broker on-ramps for unknown brokers should generate a default id."""
    assert service.connector_profile_id_for_broker("futurebroker") == "futurebroker-live-mcp"


def test_select_connection_tool_returns_canonical_profile_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Selecting a profile should persist and return the canonical id."""
    monkeypatch.setattr(profiles, "get_runtime_root", lambda: tmp_path)

    result = TradingSelectConnectionTool().execute(connection="DHAN-PAPER-SDK")

    assert result
    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["selected_profile"] == "dhan-paper-sdk"
    assert profiles.load_selected_profile_id() == "dhan-paper-sdk"


def test_no_dhan_live_trade_profile() -> None:
    """Dhan must NOT expose a live order-placing profile (no runtime discriminator)."""
    ids = {p.id for p in profiles.list_profiles()}
    assert "dhan-paper-trade" in ids
    assert "dhan-live-trade" not in ids

    for p in profiles.list_profiles():
        if p.connector == "dhan" and p.environment == "live":
            assert not any(".place" in cap or "requires_mandate" in cap for cap in p.capabilities)


def test_no_shoonya_live_trade_profile() -> None:
    """Shoonya must NOT expose a live order-placing profile (no runtime discriminator)."""
    ids = {p.id for p in profiles.list_profiles()}
    assert "shoonya-paper-trade" in ids
    assert "shoonya-live-trade" not in ids

    for p in profiles.list_profiles():
        if p.connector == "shoonya" and p.environment == "live":
            assert not any(".place" in cap or "requires_mandate" in cap for cap in p.capabilities)
