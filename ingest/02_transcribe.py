"""
Step 2: Transcribe downloaded audio files with OpenAI Whisper.

For each .mp3 in data/audio/ (that has a matching .json sidecar) the script:
  - Loads the Whisper "base" model
  - Transcribes with word-level timestamps enabled
  - Groups words into ~30-second chunks
  - Applies a naive speaker heuristic (alternating "Speaker A" / "Speaker B"
    on long pauses, useful as a placeholder until diarization is added)
  - Saves a structured JSON to data/transcripts/{ticker}_{quarter}_{year}.json

Output format:
  {
    "ticker": "NVDA",
    "company": "NVIDIA Corporation",
    "quarter": "Q3",
    "year": 2024,
    "date": "2023-11-21",
    "audio_file": "nvda_q3_2024.mp3",
    "youtube_id": "qNpyWGj-Kro",
    "chunks": [
      {
        "chunk_index": 0,
        "text": "...",
        "start": 0.0,
        "end": 30.1,
        "speaker": "Speaker A"
      },
      ...
    ]
  }

Usage:
    python ingest/02_transcribe.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Ensure ffmpeg is on PATH (uses static binary if system ffmpeg is absent)
import shutil as _shutil
if not _shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
TRANSCRIPTS_DIR = Path("./data/transcripts")
CHUNK_DURATION = 30.0          # seconds per chunk
SPEAKER_PAUSE_THRESHOLD = 2.0  # seconds — pause longer than this → new speaker


def group_words_into_chunks(
    words: list[dict[str, Any]], chunk_duration: float = CHUNK_DURATION
) -> list[dict[str, Any]]:
    """
    Group Whisper word-level segments into fixed-duration chunks.

    Each returned dict has keys: text, start, end, speaker, chunk_index.
    """
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    current_words: list[str] = []
    chunk_start: float = words[0].get("start", 0.0)
    chunk_end: float = chunk_start
    speaker = "Speaker A"
    speaker_index = 0
    chunk_index = 0
    prev_end: float = chunk_start

    for word_info in words:
        word_text: str = word_info.get("word", "")
        w_start: float = word_info.get("start", prev_end)
        w_end: float = word_info.get("end", w_start)

        # Speaker heuristic: long pause → flip speaker
        if w_start - prev_end > SPEAKER_PAUSE_THRESHOLD and current_words:
            speaker_index = (speaker_index + 1) % 2
            speaker = f"Speaker {'A' if speaker_index == 0 else 'B'}"

        # If this word would push us past the chunk boundary, flush current chunk
        if w_end - chunk_start >= chunk_duration and current_words:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": " ".join(current_words).strip(),
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "speaker": speaker,
                }
            )
            chunk_index += 1
            current_words = []
            chunk_start = w_start
            # Reset speaker at chunk boundary (keep last speaker)
        current_words.append(word_text)
        chunk_end = w_end
        prev_end = w_end

    # Flush the final chunk
    if current_words:
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": " ".join(current_words).strip(),
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "speaker": speaker,
            }
        )

    return chunks


def transcribe_file(mp3_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Transcribe a single mp3 file and return structured transcript data."""
    try:
        import whisper  # type: ignore
    except ImportError:
        print("ERROR: openai-whisper is not installed.  Run: pip install openai-whisper")
        sys.exit(1)

    print(f"  Loading Whisper model 'base'...")
    model = whisper.load_model("base")

    print(f"  Transcribing {mp3_path.name} ...")
    result = model.transcribe(
        str(mp3_path),
        word_timestamps=True,
        verbose=False,
    )

    # Flatten word-level segments from Whisper output
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
            )

    chunks = group_words_into_chunks(words)

    transcript = {
        "ticker": meta["ticker"],
        "company": meta.get("company", ""),
        "quarter": meta["quarter"],
        "year": meta["year"],
        "date": meta.get("date", ""),
        "audio_file": meta["audio_file"],
        "youtube_id": meta.get("youtube_id", ""),
        "chunks": chunks,
    }
    return transcript


def main() -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {AUDIO_DIR.resolve()}")
        print("Run 01_download_audio.py first.")
        sys.exit(0)

    print(f"Found {len(mp3_files)} audio file(s) in {AUDIO_DIR.resolve()}\n")

    for mp3_path in sorted(mp3_files):
        sidecar_path = mp3_path.with_suffix(".json")
        if not sidecar_path.exists():
            print(f"  [skip] {mp3_path.name} — no sidecar JSON found")
            continue

        meta = json.loads(sidecar_path.read_text())
        ticker = meta["ticker"]
        quarter = meta["quarter"]
        year = meta["year"]
        out_name = f"{ticker.lower()}_{quarter.lower()}_{year}.json"
        out_path = TRANSCRIPTS_DIR / out_name

        if out_path.exists():
            print(f"  [skip] {out_name} already exists")
            continue

        try:
            transcript = transcribe_file(mp3_path, meta)
            out_path.write_text(json.dumps(transcript, indent=2))
            print(
                f"  Saved {out_name} — {len(transcript['chunks'])} chunks"
            )
        except Exception as exc:
            print(f"  [error] {mp3_path.name}: {exc}")

    print("\nDone.  Next step: python ingest/03_embed_and_index.py")


if __name__ == "__main__":
    main()
