# INSTRUCTOR REFERENCE: Full working solution
"""
Earnings Call MCP Server — complete implementation.

This file is the reference solution for the workshop.  Participants work in
server.py; this file demonstrates what a finished implementation looks like.

Run with:
    python mcp_server/server_solution.py
"""

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure ffmpeg/ffprobe are available for audio slicing
if not shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue, Range

from mcp_server.embeddings import embed_query

# ── Configuration ─────────────────────────────────────────────────────────────
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
ASKNEWS_CACHE_DIR: Path = Path("./data/asknews_cache")
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", "./data/audio"))

# ── Qdrant client ─────────────────────────────────────────────────────────────
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(path=QDRANT_PATH or "./data/qdrant_storage")

# ── FastMCP app ────────────────────────────────────────────────────────────────
mcp = FastMCP("earnings-call-server")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: search_earnings
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_earnings(
    query: str,
    ticker: Optional[str] = None,
    date_range: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over earnings call transcripts stored in Qdrant.

    Args:
        query:      Natural-language question, e.g. "data center demand outlook"
        ticker:     Optional stock ticker to restrict results, e.g. "NVDA"
        date_range: Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD"

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    try:
        # Step 1: Embed the query
        query_vector = embed_query(query)

        # Step 2: Build Qdrant filter
        conditions: list[FieldCondition] = []

        if ticker:
            conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
            )

        if date_range:
            parts = date_range.split(":")
            if len(parts) == 2:
                start_date, end_date = parts[0].strip(), parts[1].strip()
                conditions.append(
                    FieldCondition(
                        key="date",
                        range=Range(gte=start_date, lte=end_date),
                    )
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Step 3: Run the vector search against the `text` named vector.
        # The collection stores two named vectors per chunk (text + audio)
        # in the same multimodal space, so we must pick which one to query.
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="text",
            query_filter=qdrant_filter,
            limit=5,
            with_payload=True,
        )

        # Step 4: Format results
        return [
            {
                "point_id": str(r.id),
                "ticker": r.payload.get("ticker"),
                "company": r.payload.get("company"),
                "quarter": r.payload.get("quarter"),
                "year": r.payload.get("year"),
                "chunk_text": r.payload.get("chunk_text"),
                "speaker": r.payload.get("speaker"),
                "start_time": r.payload.get("start_time"),
                "score": r.score,
            }
            for r in results.points
        ]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_audio_clip
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_audio_clip(point_id: str) -> dict[str, Any]:
    """
    Retrieve a base64-encoded audio clip for a specific transcript chunk.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with audio_base64 and metadata, or {"error": "..."} on failure.
    """
    try:
        # Step 1: Retrieve point from Qdrant
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found in collection"}

        payload = points[0].payload
        ticker = payload.get("ticker", "")
        start_time: float = payload.get("start_time", 0.0)
        end_time: float = payload.get("end_time", 0.0)

        # Step 2: Check for a pre-sliced clip
        clip_path = CLIPS_DIR / f"{point_id}.mp3"
        if clip_path.exists():
            audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()
            return {
                "point_id": point_id,
                "ticker": ticker,
                "audio_base64": audio_b64,
                "start_time": start_time,
                "end_time": end_time,
                "format": "mp3",
            }

        # Step 3: Try to slice from the full audio file using pydub
        audio_file = payload.get("audio_file", "")
        full_audio_path = AUDIO_DIR / audio_file

        if full_audio_path.exists():
            try:
                from pydub import AudioSegment  # type: ignore

                audio = AudioSegment.from_mp3(str(full_audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                clip = audio[start_ms:end_ms]

                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                clip.export(str(clip_path), format="mp3")
                audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()

                return {
                    "point_id": point_id,
                    "ticker": ticker,
                    "audio_base64": audio_b64,
                    "start_time": start_time,
                    "end_time": end_time,
                    "format": "mp3",
                }
            except ImportError:
                pass  # pydub not available; fall through to error

        return {
            "error": (
                f"No pre-sliced clip found for point {point_id} "
                f"(expected: {clip_path}). "
                "Run the ingestion pipeline to generate clips, or ensure "
                "pydub is installed for on-the-fly slicing."
            )
        }

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: get_news_context
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_news_context(point_id: str) -> dict[str, Any]:
    """
    Return AskNews articles published around the time of the earnings call chunk.

    Searches a ±7-day window centred on the call date so the news reflects
    what was actually happening in the world when management spoke, not today.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with ticker, date, window, and articles list.
    """
    try:
        from datetime import datetime, timedelta, timezone  # noqa: PLC0415

        # Step 1: Retrieve point
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}

        payload = points[0].payload
        ticker: str = payload.get("ticker", "")
        date: str = payload.get("date", "")      # "YYYY-MM-DD"
        speaker: str = payload.get("speaker", "")

        # Step 2: Check disk cache
        cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        # Step 3: Live AskNews call bounded to ±7 days around the call date
        asknews_key = os.getenv("ASKNEWS_API_KEY", "")
        asknews_id = os.getenv("ASKNEWS_CLIENT_ID", "")
        asknews_secret = os.getenv("ASKNEWS_CLIENT_SECRET", "")
        if not asknews_key and not (asknews_id and asknews_secret):
            return {
                "ticker": ticker,
                "date": date,
                "articles": [],
                "note": (
                    "No AskNews cache found and credentials not set. "
                    "Run ingest/04_build_asknews_context.py to pre-populate the cache."
                ),
            }

        from asknews_sdk import AskNewsSDK  # type: ignore

        if asknews_key:
            sdk = AskNewsSDK(api_key=asknews_key)
        else:
            sdk = AskNewsSDK(client_id=asknews_id, client_secret=asknews_secret)

        call_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        is_recent = abs((datetime.now(timezone.utc) - call_dt).total_seconds()) <= 48 * 3600

        start_ts = int((call_dt - timedelta(days=7)).timestamp())
        end_ts   = int((call_dt + timedelta(days=1)).timestamp())

        # Ticker-based keyword search (matches 04_build_asknews_context.py)
        ticker_keywords = ["earnings", "tariffs", "economy"]
        ticker_query = " ".join([ticker] + ticker_keywords)

        ticker_response = sdk.news.search_news(
            query=ticker_query,
            n_articles=5,
            start_timestamp=start_ts if not is_recent else None,
            end_timestamp=end_ts if not is_recent else None,
            time_filter="pub_date",
            historical=not is_recent,
            method="kw",
            return_type="dicts",
            categories=["Finance", "Business", "Politics", "Technology", "World"],
        )

        responses: list = []
        responses.extend(getattr(ticker_response, "as_dicts", []))

        # Speaker-based search for named speakers
        speaker_keywords = ["earnings", "tariffs", "economy"]
        if "speaker" not in speaker.lower() and speaker.lower() not in ["operator", "analyst"]:
            speaker_query = " ".join([speaker] + speaker_keywords)
            speaker_response = sdk.news.search_news(
                query=speaker_query,
                string_guarantee=[speaker],
                n_articles=5,
                start_timestamp=start_ts if not is_recent else None,
                end_timestamp=end_ts if not is_recent else None,
                time_filter="pub_date",
                historical=not is_recent,
                method="kw",
                return_type="dicts",
                categories=["Finance", "Business", "Politics", "Technology", "World"],
            )
            responses.extend(getattr(speaker_response, "as_dicts", []))

        entity_types = [
            "Person", "Organization", "Location", "Event", "Money",
            "Law", "Politics", "Product", "Technology", "Science",
        ]

        articles: list[dict[str, Any]] = []
        seen_article_ids: set[str] = set()
        for item in responses:
            if item.article_id in seen_article_ids:
                continue

            raw_entities = getattr(item, "entities", None)
            entities = {
                k: v for k, v in raw_entities.model_dump().items()
                if k in entity_types and v
            } if raw_entities else {}

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

        result = {
            "ticker": ticker,
            "date": date,
            "window": f"{(call_dt - timedelta(days=7)).date()} → {call_dt.date()}",
            "articles": articles,
        }

        # Cache for offline use
        ASKNEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2))

        return result

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: recommend_similar
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def recommend_similar(point_id: str) -> list[dict[str, Any]]:
    """
    Find transcript chunks semantically similar to a given chunk.

    Uses Qdrant's recommendation API with the given point as a positive example.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks.
    """
    try:
        # Fetch the seed point's vectors. For named-vector collections
        # `retrieve(..., with_vectors=True)` returns pts[0].vector as a dict
        # like {"text": [...], "audio": [...]}. Use the text vector for
        # cross-call topical similarity.
        pts = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True,
        )
        if not pts:
            return [{"error": f"Point {point_id} not found"}]

        seed_vectors = pts[0].vector
        seed_text = (
            seed_vectors["text"] if isinstance(seed_vectors, dict) else seed_vectors
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=seed_text,
            using="text",
            query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
            limit=5,
            with_payload=True,
        )

        return [
            {
                "point_id": str(r.id),
                "ticker": r.payload.get("ticker"),
                "chunk_text": r.payload.get("chunk_text"),
                "score": r.score,
                "quarter": r.payload.get("quarter"),
                "year": r.payload.get("year"),
            }
            for r in results.points
        ]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Tool 5: scrape_sec_filings (browser agent demo)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def scrape_sec_filings(ticker: str, year: int) -> dict[str, Any]:
    """
    Scrape recent SEC filings (10-Q, 10-K) for a given ticker using Playwright.

    This tool demonstrates how a browser agent can be wired into an MCP server.

    Args:
        ticker: Stock ticker, e.g. "NVDA"
        year:   Calendar year to search, e.g. 2024

    Returns:
        Dict with ticker and a list of filing dicts (type, date, url).
    """
    try:
        from browser_agent.sec_scraper import scrape_sec_filings as _scrape

        return _scrape(ticker=ticker, year=year)
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
