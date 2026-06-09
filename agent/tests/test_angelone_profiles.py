"""Tests for AngelOne connector profiles."""

from __future__ import annotations

import pytest

from src.trading.connectors.angelone.profiles import ANGELONE_PROFILES

pytestmark = pytest.mark.unit


def test_three_profiles_defined():
    """Must have exactly 3 profiles."""
    assert len(ANGELONE_PROFILES) == 3


def test_paper_sdk_is_readonly():
    """Paper SDK profile must be read-only."""
    paper_sdk = [p for p in ANGELONE_PROFILES if p.id == "angelone-paper-sdk"][0]
    assert paper_sdk.environment == "paper"
    assert paper_sdk.readonly is True
    assert "orders.place" not in paper_sdk.capabilities


def test_paper_trade_allows_orders():
    """Paper trade profile must allow simulated order placement."""
    paper_trade = [p for p in ANGELONE_PROFILES if p.id == "angelone-paper-trade"][0]
    assert paper_trade.environment == "paper"
    assert paper_trade.readonly is False
    assert "orders.place" in paper_trade.capabilities


def test_live_readonly():
    """Live readonly profile must not allow order placement."""
    live_ro = [p for p in ANGELONE_PROFILES if p.id == "angelone-live-readonly"][0]
    assert live_ro.environment == "live"
    assert live_ro.readonly is True
    assert "orders.place" not in live_ro.capabilities


def test_all_profiles_are_angelone_connector():
    """Every profile must declare connector='angelone'."""
    for p in ANGELONE_PROFILES:
        assert p.connector == "angelone"


def test_all_profiles_use_broker_sdk_transport():
    """Every profile must use broker_sdk transport."""
    for p in ANGELONE_PROFILES:
        assert p.transport == "broker_sdk"


def test_read_capabilities_present_in_all():
    """All profiles must include standard read capabilities."""
    for p in ANGELONE_PROFILES:
        assert "account.read" in p.capabilities
        assert "positions.read" in p.capabilities
        assert "orders.read" in p.capabilities
        assert "quotes.read" in p.capabilities
        assert "history.read" in p.capabilities
