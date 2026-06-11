"""
Step 3: Transcribe downloaded audio files with OpenAI Whisper, diarize
speaker turns using pyannote.audio, then identify named speakers with Gemini.

For each .mp3 in data/audio/ (that has a matching .json sidecar) the script:
  - Loads the Whisper "base" model
  - Transcribes with word-level timestamps enabled
  - Runs pyannote.audio speaker diarization on the audio to get speaker segments
  - Assigns each word to a speaker segment, then groups words into ~30-second chunks
  - Groups chunks into larger paragraphs for Gemini context
  - Sends each paragraph to Gemini 2.5 Flash-Lite to resolve speaker labels to
    real names (e.g. "Jensen Huang") — falls back to "Speaker 0", "Speaker 1", etc.
  - Saves a structured JSON to data/transcripts/{ticker}_{quarter}_{year}.json

Requirements:
  - HF_TOKEN env var: required to download pyannote models.
    Accept the pyannote/speaker-diarization-3.1 license at:
    https://huggingface.co/pyannote/speaker-diarization-3.1
  - GEMINI_API_KEY env var: required for speaker name identification.
  - pip install pyannote.audio

Output format:
  {
    "ticker": "NVDA",
    "company": "NVIDIA Corporation",
    "quarter": "Q3",
    "year": 2024,
    "date": "2023-11-21",
    "audio_file": "nvda_q3_2024.mp3",
    "youtube_id": "qNpyWGj-Kro",
    "speakers": {
      "Speaker 0": "Jensen Huang",
      "Speaker 1": "Colette Kress"
    },
    "chunks": [
      {
        "chunk_index": 0,
        "text": "...",
        "start": 0.0,
        "end": 30.1,
        "speaker": "Speaker 0"
      },
      ...
    ]
  }

Usage:
    python3 ingest/02_transcribe_and_diarize.py
"""

import shutil as _shutil
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Ensure ffmpeg is on PATH (uses static binary if system ffmpeg is absent)
if not _shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "./data/transcripts"))

CHUNK_DURATION = 30.0  # seconds per chunk

# Speaker identification model
IDENTIFICATION_MODEL = "gemini-2.5-flash-lite"

# Maximum tokens per paragraph sent to the speaker identification model.
# Uses tiktoken cl100k_base as an offline approximation (~5-10% off for Gemini).
IDENTIFY_TOKENS_PER_PARAGRAPH = 2048


# ---------------------------------------------------------------------------
# Structured output schema for Gemini speaker identification
# ---------------------------------------------------------------------------

class SpeakerAssignment(BaseModel):
    """Speaker label and optional resolved name for one chunk."""
    speaker_label: str          # e.g. "Speaker 0"
    speaker_name: Optional[str] = None  # e.g. "Jensen Huang", None if unknown


class ParagraphDiarization(BaseModel):
    """Speaker identification result for one paragraph (group of chunks)."""
    assignments: list[SpeakerAssignment]


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_file(mp3_path: Path) -> list[dict[str, Any]]:
    """
    Transcribe a single mp3 with Whisper.
    Returns a flat list of word dicts: {word, start, end}.
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        print("ERROR: openai-whisper is not installed.  Run: pip install openai-whisper")
        sys.exit(1)

    print("  Loading Whisper model 'base'...")
    model = whisper.load_model("base")

    print(f"  Transcribing {mp3_path.name} ...")
    result = model.transcribe(
        str(mp3_path),
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,  # prevents hallucination loops
        no_speech_threshold=0.6,           # suppress segments likely to be silence
        logprob_threshold=-1.0,            # discard low-confidence segments
        language="en",
    )

    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        # Skip segments flagged as no-speech by Whisper
        if segment.get("no_speech_prob", 0.0) > 0.6:
            continue
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
            )
    return words


# ---------------------------------------------------------------------------
# pyannote.audio diarization
# ---------------------------------------------------------------------------

def diarize_audio(mp3_path: Path) -> list[dict[str, Any]]:
    """
    Run pyannote.audio speaker diarization on *mp3_path*.

    Returns a list of segments sorted by start time:
        [{"start": float, "end": float, "speaker": str}, ...]

    Speaker labels are pyannote's internal IDs, e.g. "SPEAKER_00".
    Requires HF_TOKEN env var and accepted model license at:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if not hf_token:
        print(
            "  [warn] HF_TOKEN not set — skipping diarization, speakers will be unknown")
        return []

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        print("  [warn] pyannote.audio not installed — skipping diarization")
        print("         Run: pip install pyannote.audio")
        return []

    import torch  # type: ignore

    print("  Loading pyannote speaker-diarization-3.1 pipeline ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(torch.device(device))
    print(f"  Running diarization on {mp3_path.name} (device={device}) ...")

    # Convert MP3 to WAV first — pyannote has a known issue with MP3 sample
    # count mismatches that causes a crash mid-file.
    import tempfile
    import subprocess
    from tqdm import tqdm  # type: ignore

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path),
             "-ar", "16000", "-ac", "1", wav_path],
            check=True, capture_output=True,
        )

        # Progress bar driven by pyannote's hook callback.
        # The hook is called per step (segmentation, embeddings, clustering);
        # we show one bar per step, updating as batches complete.
        bars: dict[str, tqdm] = {}

        def _hook(step_name: str, _chunk: Any, total: int = 1,
                  completed: int = 0, **kwargs: Any) -> None:
            if step_name not in bars:
                bars[step_name] = tqdm(
                    total=total,
                    desc=f"    {step_name}",
                    unit="batch",
                    leave=True,
                )
            bar = bars[step_name]
            bar.total = total
            bar.n = completed
            bar.refresh()

        try:
            diarization = pipeline(wav_path, hook=_hook)
        finally:
            for bar in bars.values():
                bar.close()
    finally:
        Path(wav_path).unlink(missing_ok=True)

    # DiarizeOutput wraps the Annotation; fall back gracefully for older versions
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,  # e.g. "SPEAKER_00"
            }
        )

    # Normalise pyannote speaker IDs to "Speaker 0", "Speaker 1", ...
    seen: dict[str, str] = {}
    for seg in segments:
        raw = seg["speaker"]
        if raw not in seen:
            seen[raw] = f"Speaker {len(seen)}"
        seg["speaker"] = seen[raw]

    print(
        f"  Diarization found {len(seen)} speaker(s) across {len(segments)} segment(s)")
    return segments


# ---------------------------------------------------------------------------
# Word → speaker assignment and chunking
# ---------------------------------------------------------------------------

def _speaker_for_word(
    w_start: float, w_end: float, segments: list[dict[str, Any]]
) -> str:
    """
    Return the speaker label for a word by finding the diarization segment
    with maximum overlap with [w_start, w_end].
    Falls back to the nearest segment (by midpoint distance) if no overlap.
    """
    w_mid = (w_start + w_end) / 2
    best_speaker = "Speaker 0"
    best_overlap = -1.0
    best_dist = float("inf")

    for seg in segments:
        overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
        if overlap <= 0:
            dist = min(abs(w_mid - seg["start"]), abs(w_mid - seg["end"]))
            if dist < best_dist:
                best_dist = dist
                if best_overlap <= 0:
                    best_speaker = seg["speaker"]

    return best_speaker


def build_chunks_from_diarization(
    words: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
    chunk_duration: float = CHUNK_DURATION,
) -> list[dict[str, Any]]:
    """
    Build chunks from words and pyannote diarization segments.

    Chunk boundaries are driven by speaker changes: a new chunk starts whenever
    the speaker changes. Long single-speaker turns are also split at
    chunk_duration to keep chunks a manageable size for Gemini.

    Each returned dict: chunk_index, text, start, end, speaker.
    """
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    current_words: list[str] = []
    chunk_start: float = words[0].get("start", 0.0)
    chunk_end: float = chunk_start
    current_speaker: str = _speaker_for_word(
        words[0].get("start", 0.0), words[0].get(
            "end", 0.0), diarization_segments
    )
    chunk_index = 0

    def flush(end: float) -> None:
        nonlocal chunk_index
        if not current_words:
            return
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": " ".join(w.strip() for w in current_words),
                "start": round(chunk_start, 3),
                "end": round(end, 3),
                "speaker": current_speaker,
            }
        )
        chunk_index += 1

    for word_info in words:
        word_text: str = word_info.get("word", "")
        w_start: float = word_info.get("start", chunk_end)
        w_end: float = word_info.get("end", w_start)
        spk = _speaker_for_word(w_start, w_end, diarization_segments)

        speaker_changed = spk != current_speaker
        too_long = current_words and (w_end - chunk_start >= chunk_duration)

        if speaker_changed or too_long:
            flush(chunk_end)
            current_words = []
            chunk_start = w_start
            current_speaker = spk

        current_words.append(word_text)
        chunk_end = w_end

    flush(chunk_end)
    return chunks


# ---------------------------------------------------------------------------
# Gemini speaker identification
# ---------------------------------------------------------------------------

def _try_identify_speakers(
    unknown_labels: list[str],
    label_texts: dict[str, str],
    known_speakers: dict[str, str],
    client: Any,
    context_chunks: list[dict[str, Any]] | None = None,
    participant_list: list[dict] = [{}],
) -> ParagraphDiarization:
    """
    Ask Gemini to identify real names for *unknown_labels* only.

    *label_texts* maps each unknown label to a sample of its speech.
    *known_speakers* maps already-resolved labels to names, for context.
    *context_chunks* are preceding chunks shown to Gemini as surrounding context
    (e.g. operator introductions just before an analyst speaks).
    """

    participants = [
        f'{p.get("name", "")} ({p.get("role", "")})' for p in participant_list]
    known_prts = ""
    if participants:
        participants = ", ".join(participants)
        known_prts = f"Known participants: {participants}.\n\n"

    known_ctx = ""
    if known_speakers:
        entries = ", ".join(f"{k} = {v}" for k, v in known_speakers.items())
        known_ctx = f"Already identified speakers: {entries}.\n\n"

    context_ctx = ""
    if context_chunks:
        lines = "\n".join(
            f"{c['speaker']}: {c['text']}" for c in context_chunks)
        context_ctx = f"Preceding context (for reference only):\n{lines}\n\n"

    speaker_samples = "\n".join(
        f"{label}: {label_texts[label]}" for label in unknown_labels
    )

    prompt = f"""You are a financial transcript analyst identifying speakers on an earnings call.

{known_prts}{known_ctx}{context_ctx}For each speaker label below, identify the real person's name 
if you can determine it from the transcript text or the preceding context (e.g. from introductions, 
self-references, or how others address them).

Rules:
- Only assign a name when you are confident from the text itself or the context above.
- Do NOT guess. Use null when unsure.
- Each label is a DIFFERENT person. Never assign the same name to two labels.
- Keep speaker_label exactly as shown.

Transcript samples:

{speaker_samples}
"""

    response = client.models.generate_content(
        model=IDENTIFICATION_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ParagraphDiarization,
        },
    )

    return response.parsed


def _split_into_paragraphs(
    chunks: list[dict[str, Any]],
    max_tokens: int = IDENTIFY_TOKENS_PER_PARAGRAPH,
) -> list[list[dict[str, Any]]]:
    """
    Group chunks into paragraphs whose rendered text stays under *max_tokens*.
    Uses tiktoken cl100k_base as a fast offline approximation for Gemini token counts.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text): return len(enc.encode(text))
    except ImportError:
        def count_tokens(text): return len(text) // 4

    paragraphs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if current and current_tokens + chunk_tokens > max_tokens:
            paragraphs.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        paragraphs.append(current)

    return paragraphs


def identify_speakers(
    chunks: list[dict[str, Any]],
    participant_list: list[dict] = [{}],
    max_tokens: int = IDENTIFY_TOKENS_PER_PARAGRAPH,
) -> tuple[list[dict[str, Any]], dict[str, Optional[str]]]:
    """
    Run Gemini speaker identification over all chunks and return:
      - updated chunks list (speaker field replaced with resolved name or label)
      - speakers dict mapping label → resolved name (None if unresolved)
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  [warn] GEMINI_API_KEY not set — skipping speaker identification")
        return chunks, {}

    try:
        from google import genai  # type: ignore
    except ImportError:
        print("  [warn] google-genai not installed — skipping speaker identification")
        return chunks, {}

    client = genai.Client(api_key=api_key)

    speakers: dict[str, Optional[str]] = {c["speaker"]: None for c in chunks}
    paragraphs = _split_into_paragraphs(chunks, max_tokens)

    print(
        f"  Identifying speakers across {len(paragraphs)} paragraph(s) with "
        f"{IDENTIFICATION_MODEL} ...")

    for para_idx, para_chunks in enumerate(paragraphs):
        known = {k: v for k, v in speakers.items() if v is not None}
        unknown = [label for label in dict.fromkeys(c["speaker"] for c in para_chunks)
                   if speakers.get(label) is None]
        if not unknown:
            continue

        # Build a text sample per unknown label.
        # Include the last chunk of the previous paragraph as a prefix so that
        # introductions like "Our next question is from [Name]" are visible
        # even when the speaker's first chunk is at a paragraph boundary.
        prev_chunks = paragraphs[para_idx - 1] if para_idx > 0 else []
        context_chunks = prev_chunks[-1:] + para_chunks

        label_texts: dict[str, list[str]] = {label: [] for label in unknown}
        for chunk in context_chunks:
            label = chunk["speaker"]
            if label in label_texts:
                label_texts[label].append(chunk["text"])
        label_samples = {label: " ".join(
            texts[:3]) for label, texts in label_texts.items()}

        result: Optional[ParagraphDiarization] = None
        for attempt in range(3):
            try:
                result = _try_identify_speakers(
                    unknown, label_samples, known, client, prev_chunks[-3:], participant_list)
                break
            except Exception as exc:
                if attempt < 2 and ("503" in str(exc) or "429" in str(exc)):
                    wait = 10 * (attempt + 1)
                    print(
                        f"  [retry] Paragraph {para_idx} failed ({exc.__class__.__name__}), "
                        f"retrying in {wait}s ...")
                    import time
                    time.sleep(wait)
                else:
                    print(
                        f"  [warn] Speaker identification failed for paragraph {para_idx}: {exc}")
                    break
        if result is None:
            continue

        for assignment in result.assignments:
            label = assignment.speaker_label
            name = assignment.speaker_name
            if not name or label not in speakers:
                continue
            existing = speakers.get(label)
            if existing is None:
                speakers[label] = name
            else:
                # Prefer the more complete name — if one contains the other, keep the longer.
                # e.g. "Tim" -> "Tim Cook" wins; unrelated names keep the existing.
                if existing.lower() in name.lower():
                    speakers[label] = name

    unresolved = [label for label, name in speakers.items() if name is None]
    if unresolved:
        print(
            f"  [warn] Could not identify name(s) for: {', '.join(sorted(unresolved))}")

    updated_chunks = []
    for chunk in chunks:
        c = dict(chunk)
        label = c["speaker"]
        c["speaker"] = speakers[label] if speakers.get(
            label) is not None else label
        updated_chunks.append(c)

    return updated_chunks, speakers


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _base_transcript(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": meta["ticker"],
        "company": meta.get("company", ""),
        "quarter": meta["quarter"],
        "year": meta["year"],
        "date": meta.get("date", ""),
        "audio_file": meta["audio_file"],
        "youtube_id": meta.get("youtube_id", ""),
        "speakers": None,
        "_words": [],   # populated after Whisper, removed after diarization
        "chunks": [],   # populated after diarization
    }


def main() -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {AUDIO_DIR.resolve()}")
        print("Run 01_download_audio.py or 01b_fetch_benzinga.py first.")
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
        filename_stem = f"{ticker.replace('/', '-').lower()}_{quarter.lower()}_{year}"
        out_name = f"{filename_stem}.json"
        out_path = TRANSCRIPT_DIR / out_name

        existing: dict[str, Any] | None = None
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing and existing.get("speakers") is not None:
                print(
                    f"  [skip] {out_name} already diarized ({len(existing['speakers'])} "
                    "speaker(s))")
                continue

        print(f"Processing {mp3_path.name} ...")
        try:
            transcript = existing or _base_transcript(meta)

            # Step 1: transcribe — save words immediately so we can resume if diarization fails
            if not transcript.get("_words"):
                transcript["_words"] = transcribe_file(mp3_path)
                out_path.write_text(json.dumps(transcript, indent=2))
                print(
                    f"  Saved {len(transcript['_words'])} words from transcription")
            else:
                print(
                    f"  [resume] {out_name} has {len(transcript['_words'])} words — "
                    "skipping transcription")

            # Step 2: diarize and build speaker-boundary chunks
            # _words is kept in the file so we never need to re-transcribe
            if not transcript.get("chunks"):
                diarization_segments = diarize_audio(mp3_path)
                transcript["chunks"] = build_chunks_from_diarization(
                    transcript["_words"], diarization_segments)
                out_path.write_text(json.dumps(transcript, indent=2))
                print(
                    f"  Diarization complete — {len(transcript['chunks'])} chunks")
            else:
                print(
                    f"  [resume] {out_name} has {len(transcript['chunks'])} chunks — "
                    "skipping diarization")

            # Step 3: speaker identification
            participant_list = transcript.get(
                "api_response", {}).get("participants", [{}])
            chunks, speakers = identify_speakers(
                transcript["chunks"], participant_list)
            transcript["chunks"] = chunks
            transcript["speakers"] = speakers
            out_path.write_text(json.dumps(transcript, indent=2))

            n_speakers = len(transcript["speakers"])
            print(
                f"  Saved {out_name} — {len(transcript['chunks'])} chunks, "
                f"{n_speakers} speaker(s): {transcript['speakers']}"
            )
        except Exception as exc:
            print(f"  [error] {mp3_path.name}: {exc}")

    print("\nDone.  Next step: python3 ingest/03_embed_and_index.py")


if __name__ == "__main__":
    main()
