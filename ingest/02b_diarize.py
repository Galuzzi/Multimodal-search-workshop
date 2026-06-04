"""
Step 2b: Diarize transcripts and identify speakers.

Upgrades the naive "Speaker A" / "Speaker B" labels produced by step 02 with:
  1. Real speaker diarization via pyannote/speaker-diarization-3.1
     → anonymous turns like SPEAKER_00, SPEAKER_01, ...
  2. Audio-based speaker identification via Gemini multimodal
     → resolves each anonymous label to "Name (Role)" using a per-ticker
       roster of known executives

For each transcript JSON in data/transcripts/ that still has placeholder
speakers, the script:
  - Runs pyannote on the matching .mp3 in data/audio/
  - Re-assigns every chunk's `speaker` by majority temporal overlap with
    the diarized turns
  - For each unique anonymous label, splices a short representative audio
    sample and asks Gemini who is speaking, given the ticker roster
  - Substitutes the resolved name into all chunks
  - Writes the transcript JSON in place (idempotent — already-diarized
    files are skipped)

Requires:
  HF_TOKEN        HuggingFace token with access to:
                    - pyannote/speaker-diarization-3.1
                    - pyannote/segmentation-3.0
                    - pyannote/speaker-diarization-community-1
                  (all three gated; accept terms on huggingface.co first)
  GEMINI_API_KEY  for the multimodal speaker-ID call

Usage:
    python3 ingest/02b_diarize.py
"""

import shutil as _shutil
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Ensure ffmpeg is on PATH (uses static binary if system ffmpeg is absent)
if not _shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
TRANSCRIPTS_DIR = Path("./data/transcripts")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GEMINI_MODEL = "gemini-2.5-flash"

# Per-speaker audio sample sent to Gemini: concatenation of the longest
# diarized turns for that speaker, capped at this many seconds.
SAMPLE_DURATION_S = 60.0
# Minimum useful sample length — below this, ID is unreliable.
MIN_SAMPLE_DURATION_S = 4.0

PLACEHOLDER_SPEAKER = re.compile(r"^Speaker [A-Z]$")


# ---------------------------------------------------------------------------
# Per-ticker roster of known voices on the call.
# Extend this dict to support more companies. Roles are free-text; Gemini
# echoes them back in the "Name (Role)" answer.
# ---------------------------------------------------------------------------
ROSTERS: dict[str, list[str]] = {
    "NVDA": [
        "Jensen Huang (CEO)",
        "Colette Kress (CFO)",
        "Simona Jankowski (IR)",
    ],
    "AAPL": [
        "Tim Cook (CEO)",
        "Luca Maestri (CFO)",
        "Suhasini Chandramouli (IR)",
    ],
    "AMZN": [
        "Andy Jassy (CEO)",
        "Brian Olsavsky (CFO)",
        "Dave Fildes (IR)",
    ],
    "TSLA": [
        "Elon Musk (CEO)",
        "Vaibhav Taneja (CFO)",
        "Travis Axelrod (IR)",
    ],
}

GENERIC_ROLES = [
    "Operator (conference operator)",
    "Analyst (external — name unknown)",
]


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def _load_waveform(audio_path: Path) -> dict[str, Any]:
    """
    Load *audio_path* into a {waveform, sample_rate} dict pyannote can ingest
    directly. We go via pydub + the bundled static-ffmpeg CLI so we don't
    depend on torchcodec's runtime FFmpeg shared libraries (which are often
    missing on macOS).
    """
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from pydub import AudioSegment  # type: ignore

    seg = AudioSegment.from_file(str(audio_path))
    # pyannote-3.1 expects 16 kHz mono
    seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(1 << 15)  # int16 → [-1, 1]
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (channel=1, time)
    return {"waveform": waveform, "sample_rate": 16000}


def run_diarization(
    audio_path: Path, hf_token: str
) -> list[tuple[float, float, str]]:
    """Run pyannote on *audio_path*. Returns (start, end, label) turns."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        print(
            "ERROR: pyannote.audio is not installed.  "
            "Run: pip install pyannote.audio"
        )
        sys.exit(1)

    print(f"  Loading {DIARIZATION_MODEL} ...")
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

    print(f"  Preloading audio {audio_path.name} ...")
    audio_input = _load_waveform(audio_path)

    print(f"  Diarizing ...")
    diarization = pipeline(audio_input)
    # pyannote 4.x wraps the Annotation in a DiarizeOutput dataclass;
    # 3.x returns the Annotation directly.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    turns: list[tuple[float, float, str]] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(label)))
    return turns


def assign_speaker_to_chunk(
    chunk_start: float,
    chunk_end: float,
    turns: list[tuple[float, float, str]],
) -> str | None:
    """Return the speaker label with greatest temporal overlap, or None."""
    best_label: str | None = None
    best_overlap = 0.0
    for t_start, t_end, label in turns:
        overlap = max(0.0, min(chunk_end, t_end) - max(chunk_start, t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

def extract_sample(
    audio_path: Path,
    turns: list[tuple[float, float, str]],
    label: str,
    out_path: Path,
) -> float:
    """
    Build a short audio sample of *label* by concatenating their longest
    contiguous turns up to SAMPLE_DURATION_S. Writes mp3 to *out_path*.
    Returns the total duration written (seconds).
    """
    from pydub import AudioSegment  # type: ignore

    speaker_turns = sorted(
        ((s, e) for s, e, lbl in turns if lbl == label),
        key=lambda se: se[1] - se[0],
        reverse=True,
    )
    if not speaker_turns:
        return 0.0

    audio = AudioSegment.from_file(str(audio_path))
    combined = AudioSegment.empty()
    total_s = 0.0
    for s, e in speaker_turns:
        if total_s >= SAMPLE_DURATION_S:
            break
        remaining = SAMPLE_DURATION_S - total_s
        take_s = min(e - s, remaining)
        combined += audio[int(s * 1000): int((s + take_s) * 1000)]
        total_s += take_s

    if total_s < MIN_SAMPLE_DURATION_S:
        return 0.0

    combined.export(str(out_path), format="mp3")
    return total_s


# ---------------------------------------------------------------------------
# Gemini speaker ID
# ---------------------------------------------------------------------------

def identify_speaker(
    sample_path: Path,
    ticker: str,
    company: str,
    quarter: str,
    year: int,
    date: str,
) -> str:
    """Ask Gemini who is speaking in *sample_path*. Returns a name string."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)

    roster = ROSTERS.get(ticker.upper(), [])
    roster_lines = "\n".join(f"  - {r}" for r in roster + GENERIC_ROLES)

    prompt = (
        f"You are listening to a short audio sample from the {company} "
        f"({ticker}) {quarter} {year} earnings conference call held on {date}.\n\n"
        f"Identify the single speaker in this audio. Pick the most likely "
        f"candidate from this roster:\n{roster_lines}\n\n"
        f"Base your answer on what they say (self-introductions, role-specific "
        f"phrasing, references to prepared remarks vs. questions), and on the "
        f"speaker's voice characteristics.\n\n"
        f"Respond with EXACTLY ONE line in the form: Name (Role)\n"
        f"If you cannot tell, respond: Unknown\n"
        f"Do not add any other text, punctuation, or explanation."
    )

    audio_bytes = sample_path.read_bytes()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
            prompt,
        ],
    )
    answer = (response.text or "").strip().splitlines()[0].strip()
    return answer or "Unknown"


# ---------------------------------------------------------------------------
# Per-transcript processing
# ---------------------------------------------------------------------------

def needs_diarization(chunks: list[dict[str, Any]]) -> bool:
    """True if every chunk's speaker still matches the placeholder pattern."""
    return all(PLACEHOLDER_SPEAKER.match(c.get("speaker", "")) for c in chunks)


def process_transcript(transcript_path: Path, hf_token: str) -> None:
    data = json.loads(transcript_path.read_text())
    chunks: list[dict[str, Any]] = data.get("chunks", [])
    if not chunks:
        print(f"  [skip] {transcript_path.name} has no chunks")
        return
    if not needs_diarization(chunks):
        print(f"  [skip] {transcript_path.name} already diarized")
        return

    audio_file = data.get("audio_file", "")
    audio_path = AUDIO_DIR / audio_file
    if not audio_path.exists():
        print(f"  [skip] audio file missing: {audio_path}")
        return

    turns = run_diarization(audio_path, hf_token)
    if not turns:
        print(f"  [warn] no speaker turns detected — leaving placeholders")
        return

    # Step 1: assign anonymous labels by overlap
    unique_labels: set[str] = set()
    for chunk in chunks:
        label = assign_speaker_to_chunk(
            float(chunk.get("start", 0.0)),
            float(chunk.get("end", 0.0)),
            turns,
        )
        if label:
            chunk["speaker"] = label
            unique_labels.add(label)

    print(
        f"  Diarized {len(turns)} turns → {len(unique_labels)} unique speakers"
    )

    # Step 2: resolve each anonymous label via Gemini
    ticker = data.get("ticker", "")
    company = data.get("company", "")
    quarter = data.get("quarter", "")
    year = int(data.get("year", 0) or 0)
    date = data.get("date", "")
    label_to_name: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for label in sorted(unique_labels):
            sample_path = tmp_dir / f"{label}.mp3"
            duration = extract_sample(audio_path, turns, label, sample_path)
            if duration <= 0:
                print(f"    {label}: sample too short, marking Unknown")
                label_to_name[label] = "Unknown"
                continue
            try:
                name = identify_speaker(
                    sample_path, ticker, company, quarter, year, date
                )
            except Exception as exc:
                print(f"    {label}: Gemini error ({exc}); marking Unknown")
                name = "Unknown"
            print(f"    {label} ({duration:.1f}s) → {name}")
            label_to_name[label] = name

    # Step 3: rewrite chunks with resolved names
    for chunk in chunks:
        label = chunk.get("speaker", "")
        if label in label_to_name:
            chunk["speaker"] = label_to_name[label]

    transcript_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved {transcript_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "  1. Create a read token at https://huggingface.co/settings/tokens\n"
            "  2. Accept terms at all three:\n"
            "       https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "       https://huggingface.co/pyannote/segmentation-3.0\n"
            "       https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  3. Add HF_TOKEN=... to .env"
        )
        sys.exit(1)

    transcript_files = sorted(
        f for f in TRANSCRIPTS_DIR.glob("*.json") if f.name != "point_map.json"
    )
    if not transcript_files:
        print(f"No transcript JSONs found in {TRANSCRIPTS_DIR.resolve()}")
        print("Run 02_transcribe.py first.")
        sys.exit(0)

    print(f"Found {len(transcript_files)} transcript(s) in {TRANSCRIPTS_DIR}\n")
    for transcript_path in transcript_files:
        print(f"Processing {transcript_path.name} ...")
        try:
            process_transcript(transcript_path, hf_token)
        except Exception as exc:
            print(f"  [error] {transcript_path.name}: {exc}")
        print()

    print("Done.  Next step: python3 ingest/03_embed_and_index.py")


if __name__ == "__main__":
    main()
