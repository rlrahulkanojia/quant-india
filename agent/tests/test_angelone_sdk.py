"""Tests for AngelOne SDK wrapper — paper-only order simulation."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.trading.connectors.angelone.sdk import (
    get_ltp,
    place_order,
    cancel_order,
)


class TestPaperOrderGuard:
    """Order methods must be structurally paper-only."""

    def test_place_order_rejects_live(self):
        """place_order must reject when config is not paper."""
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=False)
            result = place_order(symbol="RELIANCE-EQ", side="BUY",
                                 order_type="MARKET", qty=10)
            assert "error" in result
            assert "paper" in result["error"].lower()

    def test_place_order_simulates_fill_in_paper(self):
        """place_order in paper mode must return simulated fill."""
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=True)
            result = place_order(symbol="RELIANCE-EQ", side="BUY",
                                 order_type="MARKET", qty=10)
            assert result["order_status"] == "simulated_fill"
            assert result["paper_guard"] == "simulated_locally"
            assert "PAPER-" in result["order_id"]

    def test_cancel_order_rejects_live(self):
        """cancel_order must reject when config is not paper."""
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=False)
            result = cancel_order(order_id="123456")
            assert "error" in result


class TestPlaceOrderValidation:
    """Input validation for place_order."""

    def test_rejects_missing_symbol(self):
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=True)
            result = place_order(symbol="", side="BUY",
                                 order_type="MARKET", qty=10)
            assert "error" in result

    def test_rejects_invalid_side(self):
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=True)
            result = place_order(symbol="RELIANCE-EQ", side="YOLO",
                                 order_type="MARKET", qty=10)
            assert "error" in result

    def test_rejects_zero_qty(self):
        with patch("src.trading.connectors.angelone.sdk._load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_paper=True)
            result = place_order(symbol="RELIANCE-EQ", side="BUY",
                                 order_type="MARKET", qty=0)
            assert "error" in result
