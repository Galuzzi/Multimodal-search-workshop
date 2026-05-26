"""
Step 3: Embed transcript chunks and upsert them into Qdrant.

For each transcript JSON in data/transcripts/:
  - Embeds each chunk with Gemini text-embedding-004 (vector size 768)
  - Uses a local cache (data/embedding_cache.json) keyed by sha256(text) to
    avoid re-computing embeddings for unchanged chunks
  - Creates the Qdrant collection "earnings_calls" (cosine distance) if it
    doesn't already exist
  - Upserts points with UUID IDs and the full payload schema
  - Saves a point_id → audio_offset mapping to data/transcripts/point_map.json

Usage:
    python ingest/03_embed_and_index.py
"""

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTS_DIR = Path("./data/transcripts")
CACHE_FILE = Path("./data/embedding_cache.json")
POINT_MAP_FILE = TRANSCRIPTS_DIR / "point_map.json"

QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "earnings_calls")
VECTOR_SIZE = 3072
EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def embed_text(text: str, cache: dict[str, list[float]]) -> list[float]:
    """Return the embedding for *text*, using cache when available."""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in cache:
        return cache[key]

    try:
        from google import genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        vec: list[float] = list(result.embeddings[0].values)
        cache[key] = vec
        return vec
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: Any) -> None:
    """Create the collection and payload indexes if they don't already exist."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams  # type: ignore

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created collection '{COLLECTION_NAME}'")
    else:
        print(f"  Collection '{COLLECTION_NAME}' already exists")

    for field, schema in [
        ("ticker", PayloadSchemaType.KEYWORD),
        ("date",   PayloadSchemaType.KEYWORD),
        ("year",   PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
        except Exception:
            pass  # index already exists


def upsert_batch(
    client: Any,
    points: list[Any],
) -> None:
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def process_transcript(
    transcript_path: Path,
    client: Any,
    cache: dict[str, list[float]],
    point_map: dict[str, Any],
) -> int:
    """
    Embed and index all chunks in a single transcript file.
    Returns the number of points upserted.
    """
    from qdrant_client.models import PointStruct  # type: ignore

    data = json.loads(transcript_path.read_text())
    chunks: list[dict[str, Any]] = data.get("chunks", [])
    if not chunks:
        print(f"  [warn] {transcript_path.name} has no chunks, skipping")
        return 0

    ticker = data.get("ticker", "UNKNOWN")
    company = data.get("company", "")
    quarter = data.get("quarter", "")
    year = data.get("year", 0)
    date = data.get("date", "")
    audio_file = data.get("audio_file", "")
    youtube_id = data.get("youtube_id", "")

    batch: list[PointStruct] = []
    total = 0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        # Deterministic ID: re-running is idempotent
        stable_key = f"{ticker}_{quarter}_{year}_{chunk['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_key))

        try:
            vector = embed_text(text, cache)
        except RuntimeError as exc:
            print(f"    [error] chunk {chunk['chunk_index']}: {exc}")
            continue

        payload: dict[str, Any] = {
            "ticker": ticker,
            "company": company,
            "quarter": quarter,
            "year": year,
            "chunk_index": chunk["chunk_index"],
            "chunk_text": text,
            "speaker": chunk.get("speaker", "unknown"),
            "start_time": chunk.get("start", 0.0),
            "end_time": chunk.get("end", 0.0),
            "audio_file": audio_file,
            "youtube_id": youtube_id,
            "date": date,
        }

        batch.append(PointStruct(id=point_id, vector=vector, payload=payload))
        point_map[point_id] = {
            "audio_file": audio_file,
            "start_time": chunk.get("start", 0.0),
            "end_time": chunk.get("end", 0.0),
        }

        if len(batch) >= BATCH_SIZE:
            upsert_batch(client, batch)
            total += len(batch)
            batch = []

    if batch:
        upsert_batch(client, batch)
        total += len(batch)

    return total


def main() -> None:
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        print("ERROR: qdrant-client is not installed.  Run: pip install qdrant-client")
        sys.exit(1)

    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    # Exclude the point map file itself
    transcript_files = [f for f in transcript_files if f.name != "point_map.json"]

    if not transcript_files:
        print(f"No transcript JSONs found in {TRANSCRIPTS_DIR.resolve()}")
        print("Run 02_transcribe.py first.")
        sys.exit(0)

    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        path = QDRANT_PATH or "./data/qdrant_storage"
        print(f"Using local Qdrant at {path} ...")
        client = QdrantClient(path=path)
    ensure_collection(client)

    cache = _load_cache()
    point_map: dict[str, Any] = {}
    if POINT_MAP_FILE.exists():
        point_map = json.loads(POINT_MAP_FILE.read_text())

    grand_total = 0
    for transcript_path in sorted(transcript_files):
        print(f"\nProcessing {transcript_path.name} ...")
        n = process_transcript(transcript_path, client, cache, point_map)
        print(f"  Upserted {n} points")
        grand_total += n

    _save_cache(cache)
    POINT_MAP_FILE.write_text(json.dumps(point_map, indent=2))

    print(f"\nTotal points upserted: {grand_total}")
    print(f"Embedding cache saved: {CACHE_FILE.resolve()}")
    print(f"Point map saved:       {POINT_MAP_FILE.resolve()}")
    print("\nDone.  Next step: python ingest/04_cache_asknews.py")


if __name__ == "__main__":
    main()
