"""
Step 4: Pre-fetch AskNews deep-research context for each chunk in the collection.

For every point found in Qdrant, this script queries the AskNews DeepNews API
and saves the results to:

    data/asknews_cache/{ticker}_{date}_{point_id}.json

If ASKNEWS_API_KEY is not set, the script exits gracefully so the rest of
the workshop can proceed in offline mode.

Usage:
    python ingest/04_build_asknews_context.py
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "earnings_calls")
ASKNEWS_API_KEY: Optional[str] = os.getenv("ASKNEWS_API_KEY") or None
CACHE_DIR = Path("./data/asknews_cache")


def get_all_points(client: Any) -> list[dict[str, Any]]:
    """Scroll through the collection and return every point's payload + id."""
    points = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            p = point.payload or {}
            points.append(
                {
                    "point_id": str(point.id),
                    "ticker": p.get("ticker", ""),
                    "company": p.get("company", ""),
                    "quarter": p.get("quarter", ""),
                    "year": p.get("year", 0),
                    "date": p.get("date", ""),
                    "speaker": p.get("speaker", ""),
                    "chunk_text": p.get("chunk_text", ""),
                }
            )
        if next_offset is None:
            break
        offset = next_offset

    return points


def fetch_deep_news_context(
    ticker: str,
    company: str,
    quarter: str,
    year: int,
    date: str,
    speaker: str,
    chunk_text: str,
) -> dict[str, Any]:
    """
    Run a DeepNews deep-research query for a single transcript chunk.

    Returns a dict matching the cache format:
        {ticker, date, window, analysis, articles}
    """
    from asknews_sdk import AskNewsSDK  # type: ignore
    from asknews_sdk.dto.deepnews import (  # type: ignore
        AnthropicTextDelta,
        ContentBlockDeltaEvent,
        CreateDeepNewsResponseStreamChunkV2,
        CreateDeepNewsResponseStreamSource,
        CreateDeepNewsResponseStreamSourcesNewsSource,
        CreateDeepNewsResponseStreamSourcesWebSource,
    )

    ask = AskNewsSDK(api_key=ASKNEWS_API_KEY)

    call_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    query = (
        f"Use the search_news, search_x_twitter, search_wikipedia, and search_google tools to "
        f"search for information relevant to this specific moment from the "
        f"{company} ({ticker}) {quarter} {year} earnings call on {date}.\n\n"
        f"The speaker is {speaker}, and they said:\n\"{chunk_text}\"\n\n"
        f"Search for news/tweets ±7 days around {call_dt.date()} that explains the macro events, "
        f"market conditions, or company-specific news that provides context for what "
        f"{speaker} was discussing. Also search for any relevant background in the news, "
        f"google, wikipedia, and twitter from the prior couple of months."
    )

    response = ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=1,
        max_depth=6,
        sources=["asknews", "google", "x", "wiki"],
        stream=True,
        return_sources=True,
        model="claude-sonnet-4-6",
        engine="v2.0",
        only_cited_sources=True,
    )

    entity_types = {
        "Person", "Organization", "Location", "Event", "Money",
        "Law", "Politics", "Product", "Technology", "Science",
    }

    full_text_parts: list[str] = []
    articles: list[dict[str, Any]] = []
    seen_article_ids: set[str] = set()

    for message in response:
        if isinstance(message, CreateDeepNewsResponseStreamChunkV2):
            event = message.choices[0].delta
            if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, AnthropicTextDelta):
                full_text_parts.append(event.delta.text)
            continue

        if not isinstance(message, CreateDeepNewsResponseStreamSource):
            continue

        if isinstance(message.source, CreateDeepNewsResponseStreamSourcesNewsSource):
            item = message.source.data  # SearchResponseDictItem (extends Article)
            article_id = str(item.article_id)
            if article_id in seen_article_ids:
                continue
            entities = {
                k: v for k, v in item.entities.model_dump().items()
                if k in entity_types and v
            }
            articles.append(
                {
                    "title": item.eng_title or item.title,
                    "summary": item.summary,
                    "sentiment": item.sentiment,
                    "entities": entities,
                    "language": item.language,
                    "bias": item.bias,
                    "reporting_voice": item.reporting_voice,
                    "source": item.source_id,
                    "authors": [a.model_dump() for a in (item.authors or [])],
                    "content_type": item.content_type,
                    "url": str(item.article_url),
                    "image_url": str(item.image_url or ""),
                    "image_description": item.image_description or "",
                    "published_at": str(item.pub_date),
                }
            )
            seen_article_ids.add(article_id)

        elif isinstance(message.source, CreateDeepNewsResponseStreamSourcesWebSource):
            item = message.source.data  # WebSearchResult
            article_id = str(item.url)
            if article_id in seen_article_ids:
                continue
            articles.append(
                {
                    "title": item.title,
                    "summary": " ".join(item.key_points) if item.key_points else item.raw_text,
                    "sentiment": None,
                    "entities": {},
                    "language": "",
                    "bias": None,
                    "reporting_voice": "",
                    "source": item.source,
                    "authors": [],
                    "content_type": "web",
                    "url": str(item.url),
                    "image_url": "",
                    "image_description": "",
                    "published_at": item.published,
                }
            )
            seen_article_ids.add(article_id)

    full_text = "".join(full_text_parts)
    tag_open = "<final_answer>"
    tag_close = "</final_answer>"
    start = full_text.find(tag_open)
    end = full_text.find(tag_close)
    analysis = full_text[start + len(tag_open):end].strip() if start != -1 and end != -1 else full_text.strip()

    return {
        "ticker": ticker,
        "date": date,
        "window": f"{(call_dt - timedelta(days=7)).date()} → {call_dt.date()}",
        "analysis": analysis,
        "articles": articles,
    }


def main() -> None:
    if not ASKNEWS_API_KEY:
        print(
            "ASKNEWS_API_KEY is not set in .env — skipping AskNews cache.\n"
            "The MCP server will return empty news context without this.\n"
            "To enable: add your ASKNEWS_API_KEY to .env and re-run this script."
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

    all_points = get_all_points(client)
    print(f"Found {len(all_points)} chunks to process\n")

    for pt in all_points:
        point_id = pt["point_id"]
        ticker = pt["ticker"]
        date = pt["date"]
        cache_path = CACHE_DIR / f"{ticker}_{date}_{point_id}.json"

        if cache_path.exists():
            print(f"  [skip] {cache_path.name} already cached")
            continue

        print(f"  Fetching deep news for {ticker} {date} chunk {point_id[:8]}... ({pt['speaker']})")
        try:
            payload = fetch_deep_news_context(
                ticker=pt["ticker"],
                company=pt["company"],
                quarter=pt["quarter"],
                year=pt["year"],
                date=pt["date"],
                speaker=pt["speaker"],
                chunk_text=pt["chunk_text"],
            )
            cache_path.write_text(json.dumps(payload, indent=2))
            n = len(payload.get("articles", []))
            print(f"    Saved {n} articles → {cache_path.name}")
        except Exception as exc:
            print(f"    [error] {exc}")

    print("\nDone.  AskNews cache is ready.")
    print("Next step: python cli/setup_mcp.py install")


if __name__ == "__main__":
    main()
