"""
Step 4b: Refresh the `speaker` payload field on already-indexed Qdrant points.

Use this after re-running ingest/benzinga_youtube/02b_diarize.py to overwrite the placeholder
"Speaker A/B" labels in transcript JSONs without re-embedding 577 chunks.

For each transcript JSON in data/transcripts/, the script:
  - Computes each chunk's deterministic point_id (must match the formula
    used by ingest/benzinga_youtube/03_embed_and_index.py)
  - Groups point_ids by their (now-resolved) speaker name
  - Calls qdrant client.set_payload once per (speaker name) group to patch
    only the `speaker` field — vectors and other payload fields are
    untouched

Usage:
    python ingest/benzinga_youtube/04b_update_speakers.py
"""

import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTS_DIR = Path("./data/transcripts")
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "earnings_calls")
QDRANT_TIMEOUT_S = 60


def chunks_with_ids(transcript_path: Path) -> list[tuple[str, str]]:
    """Return [(point_id, speaker), ...] for every chunk in *transcript_path*."""
    data = json.loads(transcript_path.read_text())
    ticker = data.get("ticker", "UNKNOWN")
    quarter = data.get("quarter", "")
    year = data.get("year", 0)
    out: list[tuple[str, str]] = []
    for chunk in data.get("chunks", []):
        stable_key = f"{ticker}_{quarter}_{year}_{chunk['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_key))
        out.append((point_id, chunk.get("speaker", "unknown")))
    return out


def main() -> None:
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        print("ERROR: qdrant-client is not installed.  Run: pip install qdrant-client")
        sys.exit(1)

    transcript_files = sorted(
        f for f in TRANSCRIPTS_DIR.glob("*.json") if f.name != "point_map.json"
    )
    if not transcript_files:
        print(f"No transcript JSONs found in {TRANSCRIPTS_DIR.resolve()}")
        sys.exit(0)

    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        client = QdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT_S
        )
    else:
        path = QDRANT_PATH or "./data/qdrant_storage"
        print(f"Using local Qdrant at {path} ...")
        client = QdrantClient(path=path)

    grand_total = 0
    for transcript_path in transcript_files:
        pairs = chunks_with_ids(transcript_path)
        if not pairs:
            print(f"  [skip] {transcript_path.name} — no chunks")
            continue

        by_speaker: dict[str, list[str]] = defaultdict(list)
        for pid, speaker in pairs:
            by_speaker[speaker].append(pid)

        print(f"\n{transcript_path.name}: {len(pairs)} chunks, {len(by_speaker)} unique speakers")
        for speaker, ids in sorted(by_speaker.items()):
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"speaker": speaker},
                points=ids,
            )
            print(f"  {len(ids):4d}  {speaker}")
            grand_total += len(ids)

    print(f"\nUpdated speaker field on {grand_total} points.")


if __name__ == "__main__":
    main()
