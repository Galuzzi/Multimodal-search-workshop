# Workshop Exercises: Build an Earnings Call MCP Server

Welcome!  By the end of this workshop you will have a working MCP server that
lets Claude query a Qdrant vector database of earnings call transcripts and
audio clips.

> This is the **overview sheet**. For detailed, line-by-line steps see
> [`implementation_guide.md`](./implementation_guide.md). The exercise numbers
> here match the guide exactly.

---

## 0. Prerequisites

- Python 3.9+ installed
- `git clone` of this repository
- API keys / connection details (see `.env.example`):
  - **GEMINI_API_KEY** — your own, from [Google AI Studio](https://aistudio.google.com); embeds your query text
  - **QDRANT_URL** — the workshop Qdrant Cloud cluster endpoint *(provided by the instructor)*
  - **QDRANT_API_KEY** — the workshop cluster key *(provided by the instructor; read-only is fine for Ex 1–6)*
  - **ASKNEWS_API_KEY** — [AskNews](https://my.asknews.app/en/settings/api-credentials) *(optional — only for live news)*

> **The data is pre-built.** The shared workshop cluster is already loaded with
> ~577 earnings-call chunks (with the filterable-HNSW config from Exercise 5
> already applied). You do **not** need to run the ingestion pipeline — that's
> an optional appendix at the end for anyone who wants to rebuild it locally.

### Setup

```bash
# Install dependencies and configure environment
bash setup.sh

# Edit the generated .env file with the keys above
nano .env   # (or use your preferred editor)
```

Verify the connection is healthy (should report ~577 points):
```bash
python cli/setup_mcp.py status
```

---

## 1. Exercise 1 — Implement `search_earnings`

Open `mcp_server/server.py`.  Find the `search_earnings` function and follow
the TODO comments to implement it.

**What it should do:**
1. Embed the `query` string with `embed_query(query)` → 768-dimensional vector
2. Optionally build a Qdrant filter for `ticker` and/or `date_range`
3. Call `client.query_points()` with the vector and filter
4. Return a list of result dicts

**Hint — Qdrant filter for ticker:**
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

qdrant_filter = Filter(must=[
    FieldCondition(key="ticker", match=MatchValue(value="NVDA"))
])
```

**Hint — date range filter:**

The `date` payload field is indexed as **DATETIME**, so use `DatetimeRange`
(not the numeric `Range`). It accepts ISO date strings:
```python
from qdrant_client.models import DatetimeRange

FieldCondition(key="date", range=DatetimeRange(gte="2023-01-01", lte="2024-01-01"))
```

**Test it** by running the server and asking Claude:
> "What did NVDA say about data center demand?"

---

## 2. Exercise 2 — Implement `get_audio_clip`

Still in `mcp_server/server.py`, implement `get_audio_clip`.

**What it should do:**
1. Retrieve the Qdrant point by ID with `client.retrieve()`
2. Check if `data/audio_clips/{point_id}.mp3` exists
3. If yes: read the file, base64-encode it, return with metadata
4. If no: return a helpful error message

**Hint — base64 encoding:**
```python
import base64
audio_b64 = base64.b64encode(Path("file.mp3").read_bytes()).decode()
```

**Test it** by taking a `point_id` from Exercise 1 results and calling:
> "Play the audio clip for point \<id\>"

---

## 3. Exercise 3 — Implement `get_news_context`

Implement `get_news_context` in `mcp_server/server.py`.

**What it should do:**
1. Retrieve the point to get `ticker` and `date`
2. Check for `data/asknews_cache/{ticker}_{date}.json`
3. If cached: return the JSON contents
4. If not cached + credentials available: call AskNews live and save to cache
5. If neither: return `{"articles": [], "note": "No news available"}`

**Test it:**
> "What was the news around NVDA's Q3 2024 earnings?"

---

## 4. Exercise 4 (stretch) — Implement `recommend_similar`

Implement `recommend_similar` in `mcp_server/server.py`. The pattern is:
fetch the seed point's stored `text` vector, then search with it while
excluding the seed itself.

**What it should do:**
- Retrieve the seed point's `text` vector
- Search with that vector, excluding the seed `point_id`
- Return up to 5 similar chunks

**Hint — exclude the seed with a `HasIdCondition`:**
```python
from qdrant_client.models import Filter, HasIdCondition

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=seed_vector,
    using="text",
    query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
    limit=5,
    with_payload=True,
)
```

**Test it:**
> "Find other transcript chunks similar to this one about data centers"

---

## 5. Exercise 5 (stretch) — Filterable HNSW + payload indexes

**Goal:** understand *why* the collection is configured so that **filtered**
search stays fast and accurate — and be able to (re)build it that way.

This is **collection/ingestion configuration**, not a query tool — it lives in
`ingest/03_embed_and_index.py` (`ensure_collection`) and is already applied to
the workshop cluster, so you don't need to run it.

**The two ingredients:**
1. **Filterable HNSW** — `HnswConfigDiff(m=16, payload_m=16)`. The `payload_m`
   edges give each indexed payload value its own well-connected sub-graph, so a
   restrictive filter (e.g. `ticker="NVDA"` AND a tight date range) doesn't
   dead-end the search or fall back to a brute-force scan.
2. **Payload indexes** — the extra edges are only built for *indexed* fields,
   and only for points inserted *after* the index exists. Note `date` is indexed
   as **DATETIME** (not KEYWORD) — that one choice powers both the `DatetimeRange`
   filter (Ex 1) and the recency boost (Ex 6).

**Verify it on the cluster:**
```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from qdrant_client import QdrantClient
c=QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
i=c.get_collection('earnings_calls')
print('payload_m:', i.config.hnsw_config.payload_m)
print('indexes:', {k:str(v.data_type) for k,v in i.payload_schema.items()})
"
# payload_m: 16
# indexes: {'ticker': 'keyword', 'date': 'datetime', 'year': 'integer', 'speaker': 'keyword'}
```

See `implementation_guide.md` §4b for the full explanation.

---

## 6. Exercise 6 (stretch) — Time-based score boosting

**Goal:** rerank results so a more *recent* earnings call surfaces higher, while
still respecting semantic relevance. Query-time only — no reindexing.

**Location:** `search_earnings` in `server.py`.

**What it should do:**
- Add a `boost_recency: bool = False` argument to `search_earnings`
- When `True`, **prefetch then rerank**: pull a wider candidate pool by pure
  similarity, then score with `final = $score + 0.3 * exp_decay(now − date)`

**Hint — prefetch + FormulaQuery:**
```python
from datetime import datetime, timezone
from qdrant_client.models import (
    Prefetch, FormulaQuery, SumExpression, MultExpression,
    ExpDecayExpression, DecayParamsExpression,
    DatetimeKeyExpression, DatetimeExpression,
)

now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=Prefetch(query=query_vector, using="text", filter=qdrant_filter, limit=30),
    query=FormulaQuery(formula=SumExpression(sum=[
        "$score",                                 # original similarity
        MultExpression(mult=[
            0.3,                                  # boost weight
            ExpDecayExpression(exp_decay=DecayParamsExpression(
                x=DatetimeKeyExpression(datetime_key="date"),
                target=DatetimeExpression(datetime=now_iso),
                scale=180 * 86400,                # half-life ≈ 180 days (seconds)
                midpoint=0.5,
            )),
        ]),
    ])),
    limit=5,
    with_payload=True,
)
```

**Common mistakes:** `Prefetch` uses `filter=` (not `query_filter=`); the prefetch
`limit` must exceed your final `limit` or there's nothing to rerank; the `date`
field must be DATETIME-indexed (Ex 5).

**Test it:**
> "Find AI-investment mentions, prefer the most recent calls"

See `implementation_guide.md` §4c for the full walk-through.

---

## 7. Bonus — Add the SEC Scraper as a 5th MCP Tool

The `browser_agent/sec_scraper.py` file contains a Playwright-based scraper
for SEC EDGAR filings.  Add it as a new MCP tool in `server.py`.

**Steps:**
1. Import the scraper at the top of `server.py`:
   ```python
   from browser_agent.sec_scraper import scrape_sec_filings
   ```
2. Add a new `@mcp.tool()` function called `get_sec_filings`:
   ```python
   @mcp.tool()
   def get_sec_filings(ticker: str, year: int) -> dict:
       """Retrieve SEC 10-Q and 10-K filings for a ticker and year."""
       return scrape_sec_filings(ticker=ticker, year=year)
   ```
3. Restart the MCP server
4. Ask Claude: "What 10-Q filings did NVDA submit in 2024?"

---

## 8. Register and Test with Claude

Once all tools are implemented:

```bash
# Register the MCP server
python cli/setup_mcp.py install

# If using Claude Desktop: restart the app
# If using Claude Code: the server registers automatically
```

**Test questions to ask Claude:**

| Question | Tools used |
|---|---|
| "What did NVDA say about data center demand in Q3 2024?" | `search_earnings` |
| "What did $NVDA say about gross margins in their most recent call?" | `search_earnings` |
| "Find AI-investment mentions, prefer the most recent calls" | `search_earnings` (`boost_recency=True`) |
| "Play the audio for that last chunk" | `get_audio_clip` |
| "What was the news around NVDA's Q3 earnings?" | `get_news_context` |
| "Find other earnings chunks that discuss similar topics" | `recommend_similar` |
| "What 10-K filings did NVDA submit in 2024?" | `get_sec_filings` (bonus) |
| "Compare NVDA and MSFT's language around AI investment" | `search_earnings` ×2 |

---

## Reference: Qdrant Payload Schema

Each point in the `earnings_calls` collection has this payload:

```json
{
  "ticker": "NVDA",
  "company": "NVIDIA Corporation",
  "quarter": "Q3",
  "year": 2024,
  "chunk_index": 5,
  "chunk_text": "Data center revenue grew 279% year-over-year...",
  "speaker": "Colette Kress",
  "start_time": 612.4,
  "end_time": 643.1,
  "audio_file": "nvda_q3_2024.mp3",
  "youtube_id": "qNpyWGj-Kro",
  "date": "2023-11-21"
}
```

---

## Reference: MCP Tool Summary

| Tool | Args | Returns |
|---|---|---|
| `search_earnings` | `query`, `ticker?`, `date_range?`, `boost_recency?` | List of matching chunks |
| `get_audio_clip` | `point_id` | Base64 audio + metadata |
| `get_news_context` | `point_id` | AskNews articles |
| `recommend_similar` | `point_id` | Similar chunks |
| `get_sec_filings` *(bonus)* | `ticker`, `year` | SEC filing list |

---

## Appendix — (Optional) Run the Ingestion Pipeline Yourself

The workshop cluster is **pre-loaded**, so you don't need this. But if you want
to rebuild the data against your own (local or cloud) Qdrant, run:

Start a local Qdrant instance (or point `QDRANT_URL` at your own cluster):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then run the pipeline:
```bash
# Download audio (NVDA earnings call)
python ingest/01_download_audio.py

# Transcribe (Whisper) + diarize (pyannote) + identify speakers (Gemini)
python ingest/02_transcribe_and_diarize.py

# Embed with Gemini and upsert to Qdrant
# (also configures filterable HNSW + payload indexes — see Exercise 5)
python ingest/03_embed_and_index.py

# Pre-fetch AskNews context (requires credentials)
python ingest/04_build_asknews_context.py
```

Check that Qdrant has data:
```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
print(client.get_collection("earnings_calls"))
```

> ⚠️ Against the **shared** workshop cluster, do not run
> `ingest/03_embed_and_index.py` — it recreates the collection and would wipe
> everyone's data. That's why the instructor hands out a **read-only** key.

---

## Troubleshooting

**Qdrant not reachable:**
- Check `QDRANT_URL` / `QDRANT_API_KEY` in `.env` (the shared cluster).
- Running locally instead? `docker run -p 6333:6333 qdrant/qdrant`

**Embedding fails:**
- Ensure `GEMINI_API_KEY` is set in `.env`
- The cache at `data/embedding_cache_v2.json` is used for offline fallback

**`Index required ... of types: [datetime]`:**
- The `date` field isn't DATETIME-indexed — see Exercise 5.

**MCP server not visible in Claude:**
- Re-run `python cli/setup_mcp.py install`
- Restart Claude Desktop
- Check `python cli/setup_mcp.py status`

**Whisper is slow:**
- Use `whisper.load_model("tiny")` in `02_transcribe_and_diarize.py` for faster (less accurate) transcription
- For production, use `"medium"` or `"large"` for better accuracy
