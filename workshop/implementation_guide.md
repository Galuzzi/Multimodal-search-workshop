# Implementation Guide — Building the Earnings Call MCP Server

This is a **step-by-step coding guide** for completing the four MCP tools in
`mcp_server/server.py`. Follow it top to bottom. Every step shows the exact code
to write, explains *why*, tells you how to test it, and lists common mistakes.

> You only edit **one file**: `mcp_server/server.py`.
> If you get truly stuck, `mcp_server/server_solution.py` is the full reference —
> but try each step yourself first.

---

## Where the data comes from (pre-built, source-agnostic)

The collection you query was populated by the ingestion pipeline. The
**showcased data source is the Benzinga Earnings Call Transcripts API**
(`ingest/00_fetch_benzinga.py`), which delivers, per completed call:

- the **transcript** (timestamped text segments), and
- a direct **audio MP3** (the API also offers an HLS `.m3u8` stream; the fetcher
  prefers the direct `audio/mpeg` link).

**But the pipeline is not limited to Benzinga.** Everything after step 00 operates
on a plain MP3 + metadata JSON, so any audio source works — the repo also ships
`ingest/01_download_audio.py` (YouTube via `yt-dlp`). Benzinga is the *default
showcase*, not a hard dependency.

**Why diarization exists (`ingest/02b_diarize.py`):** Benzinga transcript segments
include a `speaker` field, but it arrives **empty** — the API gives you *what* was
said and *when*, not *who* said it. So the pipeline recovers speaker identity from
the audio itself: `pyannote` diarizes the call into anonymous turns
(`SPEAKER_00`, `SPEAKER_01`, …), then Gemini multimodal listens to a sample of each
turn and resolves it to a real `"Name (Role)"` from a per-ticker executive roster.
This is the bridge that makes a source-agnostic audio file usable for
speaker-attributed search — and it's why a `speaker` payload index exists in Qdrant.

To (re)fetch source data you need `BENZINGA_API_KEY` in `.env`. You do **not** need
it to do the exercises below — the collection is already populated.

---

## 0. Before you write any code

### 0.1 Activate the environment

```bash
cd BerlinWorkshop
source .venv/bin/activate          # create with: uv venv --python 3.12 && source .venv/bin/activate
pip install -r requirements.txt    # if not already installed
```

### 0.2 Check your `.env`

You need at minimum these three keys. Open `.env` and confirm they are filled in:

```bash
GEMINI_API_KEY=...          # required — embeds your query text
QDRANT_URL=...              # required — your Qdrant Cloud cluster URL
QDRANT_API_KEY=...          # required — Qdrant Cloud key
COLLECTION_NAME=earnings_calls
```

> **Offline fallback:** if you have no `GEMINI_API_KEY`, embedding still works for
> *previously cached* queries via `data/embedding_cache_v2.json`. For new queries
> you need the key.

### 0.3 Sanity-check the connection

Run this one-liner. It should print the point count (≈577), not an error:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
print('points:', c.count('earnings_calls').count)
"
```

If this fails, fix your `.env` before continuing — none of the tools will work otherwise.

### 0.4 Understand what you're editing

Open `mcp_server/server.py`. Notice:

- Lines 32–43: config + the `client` (Qdrant) are **already built for you**.
- Line 46: `mcp = FastMCP("earnings-call-server")` — the MCP app.
- Each tool is a function decorated with `@mcp.tool()`. **The decorator is what
  exposes the function to Claude.** FastMCP reads your type hints + docstring to
  build the tool schema automatically — so keep the signatures and docstrings.
- Each tool currently returns a placeholder `{"error": "... not yet implemented"}`.
  Your job is to replace that placeholder with real logic.

---

## 1. Exercise 1 — `search_earnings`  (the core tool)

**Goal:** take a natural-language question, embed it, search Qdrant, return the
top-5 matching transcript chunks.

**Location:** `server.py`, function `search_earnings` (starts ~line 53). Replace
the four `# TODO` blocks and the final placeholder `return`.

### Step 1.1 — Add the import you'll need

At the **top** of `server.py`, next to the existing imports, add the Qdrant
filter models:

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
```

(`embed_query` is already imported on line 29 — you do not need to add it.)

### Step 1.2 — Embed the query

Inside `search_earnings`, the first thing you do is turn the text question into a
vector:

```python
    query_vector = embed_query(query)
```

`embed_query` returns a `list[float]` of length **3072** from Gemini Embedding 2.
This vector lives in the same space as the stored `text` and `audio` vectors.

### Step 1.3 — Build the optional filter

`ticker` and `date_range` are optional. Build a list of conditions and only wrap
it in a `Filter` if there's at least one:

```python
    conditions: list[FieldCondition] = []

    if ticker:
        conditions.append(
            FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
        )

    if date_range:
        start_date, end_date = date_range.split(":")
        conditions.append(
            FieldCondition(key="date", range=Range(gte=start_date, lte=end_date))
        )

    qdrant_filter = Filter(must=conditions) if conditions else None
```

- `.upper()` on the ticker matters — payloads store `"NVDA"`, not `"nvda"`.
- `date` is stored as a string `"YYYY-MM-DD"`; a string `Range` sorts correctly.
- `must=[...]` means **all** conditions must hold (logical AND).

### Step 1.4 — Run the vector search

```python
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="text",                 # search the TEXT named vector
        query_filter=qdrant_filter,   # None = no filter
        limit=5,
        with_payload=True,            # we need the metadata back
    )
```

**The most important line is `using="text"`.** Each point stores *two* named
vectors (`text` and `audio`). You must tell Qdrant which one to compare against.
Omitting `using=` raises an error on a named-vector collection.

### Step 1.5 — Format the results

`query_points` returns an object; the hits are on `.points`. Each hit `r` has
`r.id`, `r.score`, and `r.payload` (a dict):

```python
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
```

Wrap the whole body in `try/except` returning `[{"error": str(exc)}]` so Claude
gets a readable message instead of a crash. **Delete the placeholder `return`.**

### Step 1.6 — Test it

Run the standalone test snippet (no MCP client needed yet):

```bash
python -c "
from mcp_server.server import search_earnings
import json
print(json.dumps(search_earnings('data center demand'), indent=2)[:800])
"
```

Expect 5 results with `chunk_text`, a `score`, and a `point_id`. **Copy one
`point_id`** — you'll reuse it in Exercises 2–4.

Try the filter too:

```bash
python -c "from mcp_server.server import search_earnings; print(search_earnings('tariffs', ticker='tsla')[0]['ticker'])"
# → TSLA
```

**Common mistakes**
- `Unknown vector name` → you forgot `using="text"`.
- Empty results with a ticker filter → you didn't `.upper()` the ticker.
- `AttributeError: 'list' object has no attribute 'points'` → you iterated
  `results` instead of `results.points`.

---

## 2. Exercise 2 — `get_audio_clip`

**Goal:** given a `point_id`, return the 30-second MP3 for that chunk as base64,
so a UI (or Claude) can play it.

**Location:** function `get_audio_clip` (~line 139).

### Step 2.1 — Add the base64 import

At the top of `server.py`:

```python
import base64
```

### Step 2.2 — Retrieve the point

You need the payload to get the ticker and timestamps:

```python
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        return {"error": f"Point {point_id} not found in collection"}

    payload = points[0].payload
    ticker = payload.get("ticker", "")
    start_time = payload.get("start_time", 0.0)
    end_time = payload.get("end_time", 0.0)
```

`retrieve` takes a **list** of ids and returns a **list** of points — hence
`points[0]`. Always guard the empty case.

### Step 2.3 — Read the pre-sliced clip and encode it

The ingestion pipeline already sliced every chunk into
`data/audio_clips/{point_id}.mp3`. Read those bytes and base64-encode:

```python
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
```

`CLIPS_DIR` is already defined in the config block (line 36).

### Step 2.4 — Return a clear error if the clip is missing

```python
    return {
        "error": (
            f"No pre-sliced clip found for point {point_id} "
            f"(expected: {clip_path}). Run the ingestion pipeline to generate clips."
        )
    }
```

> **Stretch:** the reference solution also slices on-the-fly with `pydub` from the
> full MP3 in `data/audio/` when the clip is missing. Optional — the workshop data
> ships with all clips pre-sliced, so Step 2.3 will hit.

Wrap in `try/except` returning `{"error": str(exc)}`. Delete the placeholder.

### Step 2.5 — Test it

```bash
python -c "
from mcp_server.server import get_audio_clip
r = get_audio_clip('PASTE_A_POINT_ID_FROM_EXERCISE_1')
print({k: (v[:40]+'...' if k=='audio_base64' else v) for k,v in r.items()})
"
```

Expect `format: mp3`, real `start_time`/`end_time`, and a long base64 string.
A quick way to confirm the base64 is valid audio:

```bash
python -c "
import base64
from mcp_server.server import get_audio_clip
b = base64.b64decode(get_audio_clip('PASTE_POINT_ID')['audio_base64'])
open('/tmp/clip.mp3','wb').write(b); print('wrote', len(b), 'bytes')
"
# then: open /tmp/clip.mp3
```

**Common mistakes**
- Forgetting `.decode()` → you return raw `bytes`, which isn't JSON-serializable.
- Passing `point_id` (a string) instead of `[point_id]` to `retrieve`.

---

## 3. Exercise 3 — `get_news_context`

**Goal:** given a `point_id`, return news articles from the week *around* that
earnings call — what the world looked like when management spoke.

**Location:** function `get_news_context` (~line 195).

### Step 3.1 — Add the json import

```python
import json
```

### Step 3.2 — Retrieve the point and pull ticker + date

Same pattern as Exercise 2:

```python
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        return {"error": f"Point {point_id} not found"}

    payload = points[0].payload
    ticker = payload.get("ticker", "")
    date = payload.get("date", "")        # "YYYY-MM-DD"
```

### Step 3.3 — Read the pre-fetched cache (the happy path)

The pipeline saved one file per call as `data/asknews_cache/{TICKER}_{DATE}.json`.
For the workshop, **this is all you need** — just read and return it:

```python
    cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
```

`ASKNEWS_CACHE_DIR` is already defined (line 37).

### Step 3.4 — Graceful fallback when there's no cache

If there's no cache file and no AskNews credentials, return an empty-but-valid
result rather than crashing:

```python
    if not os.getenv("ASKNEWS_CLIENT_ID"):
        return {
            "ticker": ticker,
            "date": date,
            "articles": [],
            "note": "No cache found and AskNews credentials not set.",
        }
```

> **Stretch (live call):** with `ASKNEWS_CLIENT_ID` / `ASKNEWS_CLIENT_SECRET` set,
> you can call the live API and bound it to ±7 days around `date` so results are
> *historical*, then write the result to `cache_path` for next time. See
> `server_solution.py` lines 273–321 for the exact `sdk.news.search_news(...)`
> call (`historical=True`, `start_timestamp`/`end_timestamp`). Optional.

Wrap in `try/except`. Delete the placeholder.

### Step 3.5 — Test it

```bash
python -c "
from mcp_server.server import get_news_context
r = get_news_context('PASTE_POINT_ID')
print(r.get('ticker'), r.get('date'), '→', len(r.get('articles', [])), 'articles')
"
```

Expect a handful of articles (or the `note` fallback if that call has no cache).

**Common mistakes**
- Building the filename with the wrong case/format — it must be exactly
  `{ticker}_{date}.json`, e.g. `NVDA_2023-11-21.json`.
- Returning the raw file path instead of the parsed JSON (`json.loads(...)`).

---

## 4. Exercise 4 (stretch) — `recommend_similar`

**Goal:** given a `point_id`, find the 5 most similar chunks (possibly from other
companies/quarters) — "more like this."

**Location:** function `recommend_similar` (~line 244).

> Note: older tutorials use `client.recommend()`, but qdrant-client ≥1.9 removed
> it. You'll fetch the seed point's own vector and search with it instead.

### Step 4.1 — Add the HasIdCondition import

Extend your Qdrant models import from Exercise 1:

```python
from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue, Range
```

### Step 4.2 — Fetch the seed point's stored vector

This time pass `with_vectors=True`. On a named-vector collection the vector comes
back as a dict `{"text": [...], "audio": [...]}`:

```python
    pts = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=True,
    )
    if not pts:
        return [{"error": f"Point {point_id} not found"}]

    seed_vectors = pts[0].vector
    seed_text = seed_vectors["text"] if isinstance(seed_vectors, dict) else seed_vectors
```

We use the **text** side for topical similarity.

### Step 4.3 — Search with that vector, excluding the seed itself

If you don't exclude the seed, the top result will be the point itself (score 1.0):

```python
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=seed_text,
        using="text",
        query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
        limit=5,
        with_payload=True,
    )
```

### Step 4.4 — Format and return

```python
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
```

Wrap in `try/except` returning `[{"error": str(exc)}]`. Delete the placeholder.

### Step 4.5 — Test it

```bash
python -c "
from mcp_server.server import recommend_similar
for r in recommend_similar('PASTE_POINT_ID'):
    print(r['ticker'], r['quarter'], r['year'], round(r['score'],3))
"
```

Expect 5 rows and **none** of them equal to your seed `point_id`.

**Common mistakes**
- Forgetting `with_vectors=True` → `seed_vectors` is `None`.
- Treating the vector as a flat list → on this collection it's a dict; use
  `seed_vectors["text"]`.
- Skipping the `must_not` filter → the seed point ranks #1 against itself.

---

## 5. Run the server and register it with Claude

Once your tools pass the standalone tests above:

### 5.1 — Smoke-test the server process

```bash
python mcp_server/server.py
```

It should start and wait silently (it speaks MCP over stdio). Press `Ctrl-C` to stop.

### 5.2 — Register with Claude Desktop / Claude Code

```bash
python cli/setup_mcp.py install
python cli/setup_mcp.py status     # verify it's registered + deps present
```

Then **fully restart Claude Desktop** (`Cmd-Q`, reopen). Claude Code picks it up
automatically.

### 5.3 — Ask Claude

| Ask this | Exercises the tool |
|---|---|
| "What did NVDA say about data center demand in Q3 2024?" | `search_earnings` |
| "Play the audio for that last chunk" | `get_audio_clip` |
| "What was happening in the news around NVDA's Q3 earnings?" | `get_news_context` |
| "Find other chunks similar to that one" | `recommend_similar` |
| "Which CEOs mentioned tariffs in Q1 2025?" | `search_earnings` (×ticker) |

---

## 6. Verify in the web demo (optional, fast feedback loop)

The FastAPI demo calls the **solution** tools and renders results with audio
players and news cards:

```bash
python app.py        # http://localhost:8000
```

Type a query (e.g. "tariffs") and confirm you see ranked cards, an inline audio
player per result, and news context. This is a good visual confirmation that the
same operations your tools perform are wired correctly end-to-end.

---

## 7. Bonus — add the SEC filings tool (5th tool)

In `server.py`, add a new tool that wraps the pre-built Playwright scraper:

```python
@mcp.tool()
def get_sec_filings(ticker: str, year: int) -> dict[str, Any]:
    """Retrieve SEC 10-Q and 10-K filings for a ticker and year."""
    from browser_agent.sec_scraper import scrape_sec_filings
    return scrape_sec_filings(ticker=ticker, year=year)
```

Install the browser binary once (`playwright install chromium`), restart the
server, then ask Claude: *"What 10-Q filings did NVDA submit in 2024?"*

---

## Checklist

- [ ] `.env` has GEMINI + Qdrant keys; 0.3 prints ~577 points
- [ ] Ex1 `search_earnings` returns 5 scored chunks; ticker filter works
- [ ] Ex2 `get_audio_clip` returns valid base64 mp3 + timestamps
- [ ] Ex3 `get_news_context` returns cached articles for a point
- [ ] Ex4 `recommend_similar` returns 5 chunks, excludes the seed
- [ ] `setup_mcp.py status` shows the server registered
- [ ] Claude answers all five test questions
