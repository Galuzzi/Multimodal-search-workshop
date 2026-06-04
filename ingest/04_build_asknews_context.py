"""
Step 4: Pre-fetch AskNews context for each ticker+date in the collection.

For every unique (ticker, date) pair found in Qdrant, this script queries
the AskNews API and saves the results to:

    data/asknews_cache/{ticker}_{date}.json

If ASKNEWS_CLIENT_ID is not set, the script exits gracefully so the rest of
the workshop can proceed in offline mode.

Usage:
    python ingest/04_build_asknews_context.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "earnings_calls")
ASKNEWS_API_KEY: Optional[str] = os.getenv("ASKNEWS_API_KEY") or None
ASKNEWS_CLIENT_ID: Optional[str] = os.getenv("ASKNEWS_CLIENT_ID") or None
ASKNEWS_CLIENT_SECRET: Optional[str] = os.getenv(
    "ASKNEWS_CLIENT_SECRET") or None
CACHE_DIR = Path("./data/asknews_cache")

# How many news articles to fetch per ticker+date
MAX_ARTICLES = 5

TICKER_KEYWORDS = ["earnings", "tariffs", "economy"]
SPEAKER_KEYWORDS = ["earnings", "tariffs", "economy"]


def get_ticker_dates_and_speaker(client: Any) -> list[tuple[str, str, str]]:
    """Scroll through the collection and collect unique (ticker, date) pairs."""
    seen: set[tuple[str, str, str]] = set()
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["ticker", "date", "speaker"],
            with_vectors=False,
        )
        for point in result:
            ticker = point.payload.get("ticker", "")
            date = point.payload.get("date", "")
            speaker = point.payload.get("speaker", "")
            if ticker and date:
                seen.add((ticker, date, speaker))
        if next_offset is None:
            break
        offset = next_offset

    return sorted(seen)


def fetch_asknews(ticker: str, date: str, speaker: str,
                  ticker_keywords: list[str] = [], speaker_keywords: list[str] = []
                  ) -> dict[str, Any]:
    """
    Fetch news published in the 7 days leading up to *date* for *ticker*.

    Returns a dict matching the cache format:
        {ticker, date, window, articles: [{title, summary, source, url, published_at}]}
    """
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    try:
        from asknews_sdk import AskNewsSDK  # type: ignore
    except ImportError:
        raise RuntimeError(
            "asknews SDK not installed.  Run: pip install asknews")

    if ASKNEWS_API_KEY:
        sdk = AskNewsSDK(api_key=ASKNEWS_API_KEY)
    else:
        sdk = AskNewsSDK(client_id=ASKNEWS_CLIENT_ID,
                         client_secret=ASKNEWS_CLIENT_SECRET)

    call_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    is_recent = abs((datetime.now(timezone.utc) -
                    call_dt).total_seconds()) <= 48 * 3600

    start_ts = int((call_dt - timedelta(days=7)).timestamp())
    end_ts = int((call_dt + timedelta(days=1)).timestamp())

    ticker_query = " ".join([ticker] + ticker_keywords)
    ticker_response = sdk.news.search_news(
        query=ticker_query,
        n_articles=MAX_ARTICLES,
        start_timestamp=start_ts if not is_recent else None,
        end_timestamp=end_ts if not is_recent else None,
        time_filter="pub_date",
        historical=True if not is_recent else False,
        method="kw",
        return_type="dicts",
        categories=["Finance", "Business", "Politics", "Technology", "World"],
    )

    responses: list = []
    responses.extend(getattr(ticker_response, "as_dicts", []))

    if "speaker" not in speaker.lower() and speaker.lower() not in ["operator", "analyst"]:
        speaker_query = " ".join([speaker] + speaker_keywords)
        speaker_response = sdk.news.search_news(
            query=speaker_query,
            string_guarantee=[speaker],
            n_articles=MAX_ARTICLES,
            start_timestamp=start_ts if not is_recent else None,
            end_timestamp=end_ts if not is_recent else None,
            time_filter="pub_date",
            historical=True if not is_recent else False,
            method="kw",
            return_type="dicts",
            categories=["Finance", "Business",
                        "Politics", "Technology", "World"],
        )

        responses.extend(getattr(speaker_response, "as_dicts", []))

    entity_types = ["Person", "Organization", "Location", "Event", "Money", "Law",
                    "Politics", "Product", "Technology", "Science"]

    articles: list[dict[str, Any]] = []
    seen_article_ids = set()
    for item in responses:
        if item.article_id in seen_article_ids:
            continue

        raw_entities = getattr(item, "entities", None)
        entities = {k: v for k, v in raw_entities.model_dump(
        ).items() if k in entity_types and v} if raw_entities else {}
        articles.append(
            {
                "title": getattr(item, "eng_title", None) or getattr(item, "title", ""),
                "summary": getattr(item, "summary", ""),
                "sentiment": getattr(item, "sentiment", ""),
                "entities": entities,
                "language": getattr(item, "language", ""),
                "bias": getattr(item, "bias", ""),
                "reporting_voice": getattr(item, "reporting_voice", ""),
                "source": getattr(item, "source_id", ""),
                "authors": [a.model_dump() for a in (getattr(item, "authors", None) or [])],
                "content_type": getattr(item, "content_type", ""),
                "url": str(getattr(item, "article_url", "") or ""),
                "image_url": str(getattr(item, "image_url", "") or ""),
                "image_description": getattr(item, "image_description", ""),
                "published_at": str(getattr(item, "pub_date", "")),
            }
        )
        seen_article_ids.add(item.article_id)

    return {
        "ticker": ticker,
        "date": date,
        "window": f"{(call_dt - timedelta(days=7)).date()} → {call_dt.date()}",
        "articles": articles,
    }


def main() -> None:
    if not ASKNEWS_API_KEY and not ASKNEWS_CLIENT_ID:
        print(
            "ASKNEWS_CLIENT_ID is not set in .env — skipping AskNews cache.\n"
            "The MCP server will return empty news context without this.\n"
            "To enable: add your AskNews credentials to .env and re-run this script."
        )
        sys.exit(0)

    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        print("ERROR: qdrant-client not installed.")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        path = QDRANT_PATH or "./data/qdrant_storage"
        print(f"Using local Qdrant at {path} ...")
        client = QdrantClient(path=path)

    ticker_dates_speaker = get_ticker_dates_and_speaker(client)
    print(
        f"Found {len(ticker_dates_speaker)} unique (ticker, date, speaker) pairs\n")

    for ticker, date, speaker in ticker_dates_speaker:
        cache_path = CACHE_DIR / f"{ticker}_{date}.json"
        if cache_path.exists():
            print(f"  [skip] {cache_path.name} already cached")
            continue

        print(f"  Fetching news for {ticker} on {date} ...")
        try:
            payload = fetch_asknews(
                ticker, date, speaker, TICKER_KEYWORDS, SPEAKER_KEYWORDS)
            cache_path.write_text(json.dumps(payload, indent=2))
            n = len(payload.get("articles", []))
            window = payload.get("window", "")
            print(f"    Saved {n} articles ({window}) → {cache_path.name}")
        except Exception as exc:
            print(f"    [error] {exc}")

    print("\nDone.  AskNews cache is ready.")
    print("Next step: python cli/setup_mcp.py install")


if __name__ == "__main__":
    main()
