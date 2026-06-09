"""Tests for AngelOne tool classification — fail-closed design."""

from __future__ import annotations

from src.live.classification import ToolClass
from src.trading.connectors.angelone.classification import ANGELONE_TOOL_CLASS


def test_read_ops_classified() -> None:
    """All read operations must be classified as READ."""
    read_ops = [
        "ltpData",
        "getMarketData",
        "getCandleData",
        "orderBook",
        "tradeBook",
        "position",
        "holding",
        "rmsLimit",
    ]
    for op in read_ops:
        assert ANGELONE_TOOL_CLASS[op] is ToolClass.READ, f"{op} should be READ"


def test_write_ops_classified() -> None:
    """All write operations must be classified as WRITE."""
    write_ops = ["placeOrder", "modifyOrder", "cancelOrder"]
    for op in write_ops:
        assert ANGELONE_TOOL_CLASS[op] is ToolClass.WRITE, f"{op} should be WRITE"


def test_unknown_op_not_in_registry() -> None:
    """Unknown ops must not be in the registry — gate treats them as WRITE (fail-closed)."""
    assert "someRandomCall" not in ANGELONE_TOOL_CLASS
