"""
Step 1: Download earnings call audio from YouTube.

Downloads each configured earnings call as an mp3 file and saves a metadata
sidecar JSON alongside it.  Requires yt-dlp to be installed.

Usage:
    python ingest/01_download_audio.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Earnings call catalogue
# ---------------------------------------------------------------------------
EARNINGS_CALLS: list[dict[str, Any]] = [
    {
        "url": "https://www.youtube.com/watch?v=qNpyWGj-Kro",
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
        "quarter": "Q3",
        "year": 2024,
        "date": "2023-11-21",
    },
    # ── Q1 2025 (calendar) — tariff-heavy calls ──────────────────────────────
    {
        "url": "https://www.youtube.com/watch?v=PgTTJDzPADA",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "quarter": "Q2FY",
        "year": 2025,
        "date": "2025-04-30",
    },
    {
        "url": "https://www.youtube.com/watch?v=tad_DWtWpoU",
        "ticker": "AMZN",
        "company": "Amazon.com Inc.",
        "quarter": "Q1",
        "year": 2025,
        "date": "2025-05-01",
    },
    {
        "url": "https://www.youtube.com/watch?v=hnPLczgEcR8",
        "ticker": "WMT",
        "company": "Walmart Inc.",
        "quarter": "Q1FY26",
        "year": 2025,
        "date": "2025-05-15",
    },
    {
        "url": "https://www.youtube.com/watch?v=vs4cfyyMWhQ",
        "ticker": "TSLA",
        "company": "Tesla Inc.",
        "quarter": "Q1",
        "year": 2025,
        "date": "2025-04-22",
    },
]

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))


def _ensure_ffmpeg() -> None:
    """Add static-ffmpeg binaries to PATH if ffmpeg isn't already available."""
    import shutil
    if not shutil.which("ffmpeg"):
        try:
            import static_ffmpeg  # type: ignore
            static_ffmpeg.add_paths()
        except ImportError:
            pass


def download_audio(call: dict[str, Any], output_dir: Path) -> Path:
    """Download a single earnings call as mp3 using yt-dlp Python API."""
    _ensure_ffmpeg()
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        print("ERROR: yt-dlp is not installed.  Run: pip install yt-dlp")
        sys.exit(1)

    ticker = call["ticker"]
    quarter = call["quarter"]
    year = call["year"]
    filename_stem = f"{ticker.lower()}_{quarter.lower()}_{year}"
    output_path = output_dir / f"{filename_stem}.mp3"
    sidecar_path = output_dir / f"{filename_stem}.json"

    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return output_path

    print(f"  Downloading {ticker} {quarter} {year} ...")
    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{filename_stem}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(call["url"], download=True)

    # Write metadata sidecar
    meta = {
        "ticker": call["ticker"],
        "company": call["company"],
        "quarter": call["quarter"],
        "year": call["year"],
        "date": call["date"],
        "youtube_id": info.get("id", ""),
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "audio_file": output_path.name,
    }
    sidecar_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved: {output_path.name}  ({meta['duration']}s)")
    return output_path


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {AUDIO_DIR.resolve()}")
    print(f"Downloading {len(EARNINGS_CALLS)} earnings calls...\n")

    for call in EARNINGS_CALLS:
        # Skip placeholders
        if "PLACEHOLDER" in call["url"]:
            print(
                f"  [skip] {call['ticker']} {call['quarter']} {call['year']}"
                " — placeholder URL, add a real YouTube link to the catalogue"
            )
            continue
        try:
            download_audio(call, AUDIO_DIR)
        except Exception as exc:
            print(f"  [error] {call['ticker']} {call['quarter']} {call['year']}: {exc}")

    print("\nDone.  Next step: python ingest/02_transcribe.py")


if __name__ == "__main__":
    main()
