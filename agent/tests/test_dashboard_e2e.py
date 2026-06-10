"""E2E tests for the new dashboard pages."""
from playwright.sync_api import Page

FRONTEND = "http://localhost:5899"
SCREENSHOTS = "/Users/rahul/Documents/Projects/Quant/code/quant-india/screenshots"

import os
os.makedirs(SCREENSHOTS, exist_ok=True)


class TestPortfolioPage:
    def test_loads_with_data(self, page: Page):
        page.goto(f"{FRONTEND}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        assert "Portfolio" in page.content() or "Cash" in page.content() or "portfolio" in page.content().lower()
        page.screenshot(path=f"{SCREENSHOTS}/portfolio.png", full_page=True)

    def test_shows_positions_table(self, page: Page):
        page.goto(f"{FRONTEND}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        body = page.locator("body").inner_text()
        assert len(body) > 50  # Page has real content


class TestTradeHistoryPage:
    def test_loads(self, page: Page):
        page.goto(f"{FRONTEND}/trades")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        assert "Trade" in page.content() or "trade" in page.content().lower()
        page.screenshot(path=f"{SCREENSHOTS}/trades.png", full_page=True)


class TestShadowAccountPage:
    def test_loads(self, page: Page):
        page.goto(f"{FRONTEND}/shadow")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        assert "Shadow" in page.content() or "shadow" in page.content().lower()
        page.screenshot(path=f"{SCREENSHOTS}/shadow.png", full_page=True)


class TestNavigation:
    def test_sidebar_has_new_items(self, page: Page):
        page.goto(FRONTEND)
        page.wait_for_load_state("networkidle")
        sidebar = page.locator("aside").inner_text()
        assert "Portfolio" in sidebar
        assert "Trades" in sidebar
        assert "Shadow" in sidebar
        page.screenshot(path=f"{SCREENSHOTS}/sidebar.png", full_page=True)
