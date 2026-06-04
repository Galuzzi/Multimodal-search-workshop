"""
Step 0: Fetch earnings call transcripts and audio from the Benzinga API.

Downloads transcript text and audio files for earnings calls, saving them in
the same format expected by downstream pipeline steps.

Requires:
    - BENZINGA_API_KEY environment variable
    - requests package

Usage:
    python3 ingest/01b_fetch_benzinga.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EARNINGS_CALLS = 3  # Number of calls with audio to collect
PAGE_SIZE = 50  # Results per API page

BASE_URL = "https://api.benzinga.com/api/v1/transcripts"
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "./data/transcripts"))


def get_api_key() -> str:
    key = os.getenv("BENZINGA_API_KEY")
    if not key:
        print("ERROR: BENZINGA_API_KEY environment variable is not set.")
        sys.exit(1)
    return key


def fetch_calls(
    token: str,
    page: int = 1,
    page_size: int = PAGE_SIZE,
) -> dict[str, Any]:
    """Fetch a page of completed earnings calls from Benzinga."""
    resp = requests.get(
        f"{BASE_URL}/calls",
        params={
            "token": token,
            "page": page,
            "page_size": page_size,
            "status": "COMPLETED",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_call_detail(token: str, call_id: str) -> dict[str, Any]:
    """Fetch detailed call info with audio pre-signed URLs."""
    resp = requests.get(
        f"{BASE_URL}/calls/{call_id}",
        params={"token": token, "audio": "true", "format": "json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def download_audio_file(url: str, dest: Path) -> None:
    """Download audio from a pre-signed URL."""
    if dest.exists():
        print(f"    [skip] {dest.name} already exists")
        return
    print(f"    Downloading audio -> {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(
        f"    Saved: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def find_audio_url(call_data: dict[str, Any]) -> str | None:
    """Extract pre-signed audio URL from the call detail response.

    Prefers audio/mpeg format over HLS streams.
    """
    recordings = call_data.get("recordings") or []
    if isinstance(recordings, list):
        for rec in recordings:
            if not isinstance(rec, dict):
                continue
            formats = rec.get("formats") or []
            # First pass: prefer audio/mpeg
            for fmt in formats:
                if fmt.get("content_type") == "audio/mpeg":
                    link = fmt.get("file_link")
                    if link:
                        return link
            # Second pass: any format with a file_link
            for fmt in formats:
                link = fmt.get("file_link")
                if link:
                    return link

    return None


def process_call(token: str, call: dict[str, Any]) -> bool:
    """Process a single call: fetch details, download audio, save metadata.

    Returns True if the call had audio and was saved, False otherwise.
    """
    call_id = call["call_id"]
    symbol = call.get("symbol", "UNKNOWN")
    period = call.get("period", "earnings")
    year = call.get("year", "")
    call_title = call.get("call_title", "")

    filename_stem = f"{symbol.replace('/', '-').lower()}_{period.lower()}_{year}"

    print(f"\n  {symbol} {period} {year} — {call_title[:60]}")

    # Fetch detail with audio=true
    detail_resp = fetch_call_detail(token, call_id)
    raw_data = detail_resp.get("data", detail_resp)
    if isinstance(raw_data, list) and raw_data:
        raw_data = raw_data[0]
    call_data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}

    # Check for audio first — skip calls without it
    audio_url = find_audio_url(call_data)
    if not audio_url:
        print("    [skip] No audio available")
        return False

    # Download audio
    audio_path = AUDIO_DIR / f"{filename_stem}.mp3"
    download_audio_file(audio_url, audio_path)

    # Audio sidecar
    audio_meta = {
        "ticker": symbol,
        "company": call.get("company", ""),
        "quarter": call.get("period", ""),
        "year": year,
        "date": call.get("created_at", ""),
        "youtube_id": call.get("youtube_url", ""),
        "title": call.get("call_title", ""),
        "duration": call.get("duration", 0),
        "audio_file": f"{filename_stem}.mp3",
    }

    sidecar_path = AUDIO_DIR / f"{filename_stem}.json"
    sidecar_path.write_text(json.dumps(audio_meta, indent=2, default=str))

    # Transcript file
    transcript_meta = {k: v for k, v in call_data.items()}
    transcript_meta["audio_file"] = f"{filename_stem}.mp3"
    transcript_path = TRANSCRIPT_DIR / f"{filename_stem}.json"
    transcript_path.write_text(json.dumps(
        {"api_response": transcript_meta}, indent=2, default=str))

    print("    Audio ready for transcription pipeline")
    return True


def main() -> None:
    token = get_api_key()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Audio directory:      {AUDIO_DIR.resolve()}")
    print(f"Transcript directory: {TRANSCRIPT_DIR.resolve()}")
    print(f"Collecting {N_EARNINGS_CALLS} calls with audio ...\n")

    total_saved = 0
    total_skipped = 0
    page = 1

    while total_saved < N_EARNINGS_CALLS:
        print(f"--- Page {page} ---")
        result = fetch_calls(token, page=page)
        calls = result.get("data", [])

        if not calls:
            print("  No more calls available.")
            break

        for call in calls:
            if total_saved >= N_EARNINGS_CALLS:
                break
            try:
                if process_call(token, call):
                    total_saved += 1
                else:
                    total_skipped += 1
            except Exception as exc:
                symbol = call.get("symbol", "?")
                print(f"    [error] {symbol}: {exc}")
                total_skipped += 1

        page += 1

    print(
        f"\nDone. Saved {total_saved} calls with audio, skipped {total_skipped}.")


if __name__ == "__main__":
    main()
