"""
SEC EDGAR browser agent using Playwright.

Searches the SEC EDGAR full-text search interface for a given company ticker
and year, then extracts 10-Q and 10-K filing dates and document URLs.

Usage (sync):
    from browser_agent.sec_scraper import scrape_sec_filings
    result = scrape_sec_filings("NVDA", 2024)
    print(result)

Usage (async):
    from browser_agent.sec_scraper import scrape_sec_filings_async
    result = await scrape_sec_filings_async("NVDA", 2024)
"""

import asyncio
import re
from typing import Any

# Base URL for SEC EDGAR full-text search
EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&forms={form}"
EDGAR_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"


async def scrape_sec_filings_async(ticker: str, year: int) -> dict[str, Any]:
    """
    Async Playwright-based scraper for SEC EDGAR 10-Q / 10-K filings.

    Opens the EDGAR company search page, searches for the ticker, navigates
    to the filing list, and extracts relevant filings for the given year.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA"
        year:   Calendar year to search, e.g. 2024

    Returns:
        {
            "ticker": "NVDA",
            "year": 2024,
            "filings": [
                {"type": "10-Q", "date": "2024-08-28", "url": "https://..."},
                ...
            ]
        }
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        return {
            "ticker": ticker,
            "year": year,
            "filings": [],
            "error": "playwright is not installed.  Run: pip install playwright && python -m playwright install chromium",
        }

    filings: list[dict[str, str]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to EDGAR company search
        search_url = (
            f"{EDGAR_SEARCH_URL}"
            f"?action=getcompany"
            f"&company={ticker}"
            f"&type=10-"
            f"&dateb=&owner=include"
            f"&count=40"
            f"&search_text="
        )
        await page.goto(search_url, wait_until="domcontentloaded", timeout=30_000)

        # Try to find and click the first company result that matches the ticker
        try:
            # Look for a link whose text contains the ticker exactly
            ticker_link = page.locator(f"a:has-text('{ticker}')")
            count = await ticker_link.count()
            if count > 0:
                await ticker_link.first.click()
                await page.wait_for_load_state("domcontentloaded")
        except Exception:
            pass  # Continue anyway — we may already be on the right page

        # Scrape the filing table
        rows = page.locator("table.tableFile2 tr, table tr")
        row_count = await rows.count()

        for i in range(row_count):
            row = rows.nth(i)
            cells = row.locator("td")
            cell_count = await cells.count()
            if cell_count < 4:
                continue

            filing_type = (await cells.nth(0).inner_text()).strip()
            if filing_type not in ("10-K", "10-Q"):
                continue

            date_text = (await cells.nth(3).inner_text()).strip()
            # Check year matches (date format: YYYY-MM-DD)
            if not date_text.startswith(str(year)):
                continue

            # Extract the filing URL from the link in the first cell
            link = cells.nth(1).locator("a").first
            href = await link.get_attribute("href")
            full_url = f"https://www.sec.gov{href}" if href and href.startswith("/") else (href or "")

            filings.append(
                {
                    "type": filing_type,
                    "date": date_text,
                    "url": full_url,
                }
            )

        await browser.close()

    return {
        "ticker": ticker.upper(),
        "year": year,
        "filings": filings,
    }


def scrape_sec_filings(ticker: str, year: int) -> dict[str, Any]:
    """
    Synchronous wrapper around scrape_sec_filings_async.

    Use this from synchronous code (e.g., the MCP server tool handler).
    """
    return asyncio.run(scrape_sec_filings_async(ticker=ticker, year=year))


if __name__ == "__main__":
    import json

    result = scrape_sec_filings("NVDA", 2024)
    print(json.dumps(result, indent=2))
