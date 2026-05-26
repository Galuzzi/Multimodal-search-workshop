"""
Embedding helper with local cache fallback.

Used by both the ingestion pipeline and the MCP server so that the server
can answer queries without a live Gemini API key as long as the query text
was already cached during ingestion.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

CACHE_FILE = Path(__file__).parent.parent / "data" / "embedding_cache.json"


EMBEDDING_MODEL = "models/gemini-embedding-001"


def embed_query(text: str, api_key: Optional[str] = None) -> list[float]:
    """
    Embed *text* using Gemini gemini-embedding-001 (3072 dimensions).

    Falls back to the local embedding cache if the API is unavailable or
    the key is not set.  Raises RuntimeError if both fail.
    """
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache = _load_cache()

    if cache_key in cache:
        return cache[cache_key]

    # Try live Gemini call
    try:
        from google import genai  # type: ignore

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=resolved_key)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        vec: list[float] = list(result.embeddings[0].values)
        # Persist to cache for future offline use
        cache[cache_key] = vec
        _save_cache(cache)
        return vec
    except Exception as exc:
        raise RuntimeError(f"Embedding failed and no cache hit: {exc}") from exc


def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))
