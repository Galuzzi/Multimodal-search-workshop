# Earnings Call MCP Server — Berlin Workshop

Build an **MCP (Model Context Protocol) server** that lets Claude search earnings call
transcripts by meaning, play back the exact audio moment, and see what was happening
in the news when management spoke.

---

## Architecture

```
┌─────────────────────────── INGESTION PIPELINE (pre-built, run once) ───────────────────────────┐
│                                                                                                  │
│  YouTube ──► yt-dlp ──► MP3 ──► Whisper + pyannote ──► JSON chunks ──► Gemini Embedding 2 ──► Qdrant │
│                                                          │                                       │
│                                              AskNews API (historical)                            │
│                                                          │                                       │
│                                              data/asknews_cache/*.json                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────────────────────────────────────┐
                          │            Qdrant Cloud (vector database)           │
                          │  collection: earnings_calls                         │
                          │  named vectors: text, audio  (both 3072-dim cosine) │
                          │  payload: ticker · date · speaker · timestamps · …  │
                          └──────────────────────┬──────────────────────────────┘
                                                 │
                          ┌──────────────────────▼──────────────────────────────┐
                          │          MCP Server  mcp_server/server.py           │
                          │                                                      │
                          │  ► search_earnings(query, ticker?, date_range?)     │◄── YOU BUILD
                          │  ► get_audio_clip(point_id) → base64 MP3            │◄── YOU BUILD
                          │  ► get_news_context(point_id) → AskNews articles    │◄── YOU BUILD
                          │  ► recommend_similar(point_id) → similar chunks     │◄── YOU BUILD
                          └─────────┬──────────────────────────┬────────────────┘
                                    │                          │
              ┌─────────────────────▼──────┐    ┌─────────────▼────────────────┐
              │     Claude Desktop /       │    │    Web Demo  app.py          │
              │     Claude Code (CLI)      │    │    http://localhost:8000      │
              │  "Which CEOs mentioned     │    │  search box + audio players  │
              │   tariffs in Q1 2025?"     │    │  + historical news cards     │
              └────────────────────────────┘    └──────────────────────────────┘
```

**Data flow for a query:**
1. User types a natural-language question
2. MCP server embeds it with Gemini Embedding 2 (`models/gemini-embedding-2`, 3072 dims, multimodal)
3. Qdrant searches the `text` named vector (`using="text"`) for top-5 matches.
   Audio→audio search is also possible via `using="audio"` since both
   modalities share the same embedding space.
4. Server fetches a base64-encoded audio clip for each chunk (pre-sliced by the ingest pipeline)
5. Server loads historically-bounded AskNews articles (from ±7 days around the call date)
6. Claude (or the web UI) presents chunks, playable audio, and world context together

---

## What You Build vs. What's Pre-built

| Component | Who builds it | Notes |
|---|---|---|
| Ingestion pipeline (`ingest/01–04`) | Pre-built | Run once to populate the DB |
| Qdrant collection | Pre-built via pipeline | 577 points across AAPL, AMZN, NVDA, TSLA — each carries named vectors `text` (3072-dim) and `audio` (3072-dim) |
| AskNews cache (`data/asknews_cache/`) | Pre-built via pipeline | Historical news per ticker+date |
| Audio clips (`data/audio_clips/`) | Pre-built via pipeline | 30-second MP3 slices per point, also fed into the audio embedding |
| `mcp_server/embeddings.py` | Pre-built | Gemini Embedding 2 text-side embed + disk cache fallback |
| `mcp_server/server.py` — **`search_earnings`** | **You build** | Exercise 1 — core vector search |
| `mcp_server/server.py` — **`get_audio_clip`** | **You build** | Exercise 2 — retrieve + encode audio |
| `mcp_server/server.py` — **`get_news_context`** | **You build** | Exercise 3 — read AskNews cache |
| `mcp_server/server.py` — **`recommend_similar`** | **You build** | Exercise 4 (bonus) |
| `mcp_server/server_solution.py` | Reference only | Full working solution — peek if stuck |
| Web demo (`app.py`) | Pre-built | FastAPI app at localhost:8000 |
| CLI (`cli/setup_mcp.py`) | Pre-built | Registers server with Claude Desktop |
| Browser agent (`browser_agent/`) | Pre-built | Bonus: Playwright + SEC EDGAR |

### Recommended scope for a 90-minute workshop

- **Exercises 1–3** are the core and fit comfortably in 90 minutes.
- **Exercise 4** (`recommend_similar`) is a good stretch goal — it shows off Qdrant's
  nearest-neighbour API and only takes ~10 extra lines.
- The ingestion pipeline, web app, and browser agent are intentionally pre-built so
  participants can focus on the MCP/Qdrant interaction rather than boilerplate.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | `python3 --version`; use `uv` to install if needed |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — free tier |
| `QDRANT_URL` + `QDRANT_API_KEY` | Qdrant Cloud cluster (pre-provisioned for workshop) |
| `ASKNEWS_API_KEY` | [asknews.app](https://asknews.app) — optional; cache works offline |
| `HF_TOKEN` | Only for step 02b (diarization). Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), and [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). |
| ffmpeg | Bundled via `static-ffmpeg` — no system install needed |

---

## Quick Start

```bash
# 1. Clone / open the project
cd BerlinWorkshop

# 2. Install dependencies
uv venv --python 3.12 && source .venv/bin/activate
python -m ensurepip && python -m pip install -r requirements.txt  # or: uv pip install -r requirements.txt

# 3. Copy env template and fill in your API keys
cp .env.example .env
nano .env   # set GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 4. (Instructor only) Run the ingestion pipeline
python3 ingest/01_download_audio.py   # download earnings calls from YouTube
python3 ingest/01b_fetch_benzinga.py   # fetch transcripts and audio from Benzinga API
python3 ingest/02_transcribe_and_diarize.py # Whisper transcription → 30s chunks + pyannote diarization + Gemini speaker ID
python3 ingest/02_transcribe.py       # Whisper transcription → 30s chunks
python3 ingest/02b_diarize.py         # pyannote diarization + Gemini speaker ID
python3 ingest/03_embed_and_index.py  # Gemini embeddings → Qdrant Cloud
python3 ingest/04_build_asknews_context.py                # pre-fetch historical news (optional)

# 5. Register the MCP server with Claude Desktop
python3 cli/setup_mcp.py install
python3 cli/setup_mcp.py status           # verify

# 6. Open the exercises and start building
open workshop/exercises.md

# 7. Run the web demo to verify your implementation
python3 app.py                            # http://localhost:8000
```

---

## Project Structure

```
BerlinWorkshop/
├── README.md                         ← you are here
├── requirements.txt
├── setup.sh
├── .env.example                      ← copy to .env and fill in keys
├── app.py                            ← FastAPI web demo (pre-built)
├── demo.py                           ← CLI demo with rich tables
├── data/
│   ├── audio/                        ← downloaded .mp3 files (AAPL, AMZN, NVDA, TSLA)
│   ├── transcripts/                  ← Whisper JSON chunks
│   ├── audio_clips/                  ← pre-sliced clips keyed by Qdrant point_id
│   ├── asknews_cache/                ← {TICKER}_{DATE}.json, one per earnings call
│   └── embedding_cache.json          ← offline embedding fallback (sha256 keyed)
├── ingest/
│   ├── 01_download_audio.py          ← yt-dlp → MP3
│   ├── 01b_fetch_benzinga.py         ← api → MP3 + transcript
│   ├── 02_transcribe_and_diarize.py  ← OpenAI Whisper → word-timestamped chunks
│   ├── 02_transcribe.py              ← OpenAI Whisper → word-timestamped chunks
│   ├── 02b_diarize.py                ← pyannote diarization + Gemini speaker ID
│   ├── 03_embed_and_index.py         ← Gemini Embedding 2 (text + audio) → Qdrant upsert
│   ├── 04_cache_asknews.py           ← AskNews context → JSON cache
│   └── 04b_update_speakers.py        ← payload-only refresh of `speaker` after re-diarize
├── mcp_server/
│   ├── server.py                     ← SKELETON — participants complete this
│   ├── server_solution.py            ← full working solution (instructor reference)
│   └── embeddings.py                 ← embed_query() with disk cache fallback
├── browser_agent/
│   └── sec_scraper.py                ← Playwright + SEC EDGAR (bonus exercise)
├── cli/
│   └── setup_mcp.py                  ← installs server into Claude Desktop config
└── workshop/
    └── exercises.md                  ← step-by-step workshop guide
```

---

## Qdrant Payload Schema

Each point represents a ~30-second transcript chunk and carries TWO
named vectors in the same multimodal space produced by Gemini Embedding 2:

```json
{
  "id": "<uuid v5, stable per chunk>",
  "vector": {
    "text":  [3072 floats],   // gemini-embedding-2 over chunk_text
    "audio": [3072 floats]    // gemini-embedding-2 over the audio clip
  },
  "payload": {
    "ticker":      "TSLA",
    "company":     "Tesla Inc.",
    "quarter":     "Q1",
    "year":        2025,
    "chunk_index": 12,
    "chunk_text":  "We are navigating the tariff environment carefully...",
    "speaker":     "Elon Musk",
    "start_time":  342.1,
    "end_time":    372.8,
    "audio_file":  "tsla_q1_2025.mp3",
    "youtube_id":  "vs4cfyyMWhQ",
    "date":        "2025-04-22"
  }
}
```

Queries must specify which named vector to search (`using="text"` or
`using="audio"`). Because both vectors live in the same shared space, a
text query can rank against `audio` and vice versa.

**Payload indexes** (required for filtered search):

| Field | Type | Used by |
|---|---|---|
| `ticker` | KEYWORD | `search_earnings(ticker=...)` |
| `date` | KEYWORD | `search_earnings(date_range=...)` |
| `year` | INTEGER | range queries |

---

## Earnings Calls in the Dataset

| Ticker | Company | Quarter | Date | Duration |
|---|---|---|---|---|
| NVDA | NVIDIA Corporation | Q3 FY2024 | 2023-11-21 | 121 chunks |
| AAPL | Apple Inc. | Q2 FY2025 | 2025-04-30 | 121 chunks |
| AMZN | Amazon.com Inc. | Q1 2025 | 2025-05-01 | 161 chunks |
| TSLA | Tesla Inc. | Q1 2025 | 2025-04-22 | 174 chunks |

---

## Test Questions for Claude

Once the server is running and registered:

```
Which CEOs mentioned tariffs in Q1 2025 earnings calls?
What did NVDA say about data center demand in Q3 2024?
How did Apple describe the impact of tariffs on its supply chain?
What was Amazon's tone on consumer spending in Q1 2025?
Find all mentions of AI infrastructure investment
Compare how TSLA and AAPL described macroeconomic uncertainty
Play the audio for that last transcript chunk
Find other earnings chunks similar to that one
```

---

## Troubleshooting

**MCP server not appearing in Claude:**
```bash
python cli/setup_mcp.py install
# then restart Claude Desktop (Cmd+Q, reopen)
```

**Embedding errors:**
- Check `GEMINI_API_KEY` in `.env`
- The `data/embedding_cache_v2.json` provides offline fallback once populated.
  Old caches from the text-only `gemini-embedding-001` pipeline live in
  `data/embedding_cache.json` and are not reused (different vector space)

**Qdrant connection errors:**
- Check `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- For local fallback: set `QDRANT_PATH=./data/qdrant_storage` and leave `QDRANT_URL` empty

**Audio not playing:**
- Check that `data/audio_clips/` contains `.mp3` files
- If empty, `pydub` will slice on demand from `data/audio/` (needs `static-ffmpeg` installed)

**Whisper takes too long (if re-transcribing):**
- Edit `ingest/benzinga_youtube/02_transcribe.py` and change `"base"` to `"tiny"` for speed
