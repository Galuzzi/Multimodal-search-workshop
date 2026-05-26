# Workshop Exercises: Build an Earnings Call MCP Server

Welcome!  By the end of this workshop you will have a working MCP server that
lets Claude query a Qdrant vector database of earnings call transcripts and
audio clips.

---

## 0. Prerequisites

- Python 3.9+ installed
- `git clone` of this repository
- API keys ready (see `.env.example`):
  - **GEMINI_API_KEY** — [Google AI Studio](https://aistudio.google.com)
  - **ASKNEWS_CLIENT_ID / ASKNEWS_CLIENT_SECRET** — [AskNews](https://asknews.app) *(optional)*
  - **QDRANT_URL** — defaults to `http://localhost:6333` (run Qdrant locally with Docker)

Start a local Qdrant instance:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 1. Setup

```bash
# Install dependencies and configure environment
bash setup.sh

# Edit the generated .env file with your API keys
nano .env   # (or use your preferred editor)
```

Verify everything looks healthy:
```bash
python cli/setup_mcp.py status
```

---

## 2. Exercise 1 — Run the Ingestion Pipeline

The ingestion pipeline downloads an NVDA earnings call, transcribes it, embeds
the chunks, and loads them into Qdrant.

```bash
# Download audio (NVDA Q3 2024 from YouTube)
python ingest/01_download_audio.py

# Transcribe with Whisper
python ingest/02_transcribe.py

# Embed with Gemini and upsert to Qdrant
python ingest/03_embed_and_index.py

# Pre-fetch AskNews context (requires credentials)
python ingest/04_cache_asknews.py
```

After running, check that Qdrant has data:
```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
print(client.get_collection("earnings_calls"))
```

---

## 3. Exercise 2 — Implement `search_earnings`

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
```python
from qdrant_client.models import Range

FieldCondition(key="date", range=Range(gte="2023-01-01", lte="2024-01-01"))
```

**Test it** by running the server and asking Claude:
> "What did NVDA say about data center demand?"

---

## 4. Exercise 3 — Implement `get_audio_clip`

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

**Test it** by taking a `point_id` from Exercise 2 results and calling:
> "Play the audio clip for point \<id\>"

---

## 5. Exercise 4 — Implement `get_news_context`

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

## 6. Exercise 5 — Implement `recommend_similar`

Implement `recommend_similar` in `mcp_server/server.py`.

**What it should do:**
- Use `client.recommend()` with `positive=[point_id]` to find similar chunks
- Return up to 5 results

**Hint:**
```python
results = client.recommend(
    collection_name=COLLECTION_NAME,
    positive=[point_id],
    limit=5,
    with_payload=True,
)
# Note: results is a list, not an object with .points
```

**Test it:**
> "Find other transcript chunks similar to this one about data centers"

---

## 7. Exercise 6 (Bonus) — Add the SEC Scraper as a 5th MCP Tool

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
| `search_earnings` | `query`, `ticker?`, `date_range?` | List of matching chunks |
| `get_audio_clip` | `point_id` | Base64 audio + metadata |
| `get_news_context` | `point_id` | AskNews articles |
| `recommend_similar` | `point_id` | Similar chunks |
| `get_sec_filings` *(bonus)* | `ticker`, `year` | SEC filing list |

---

## Troubleshooting

**Qdrant not reachable:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Embedding fails:**
- Ensure `GEMINI_API_KEY` is set in `.env`
- The cache at `data/embedding_cache.json` is used for offline fallback

**MCP server not visible in Claude:**
- Re-run `python cli/setup_mcp.py install`
- Restart Claude Desktop
- Check `python cli/setup_mcp.py status`

**Whisper is slow:**
- Use `whisper.load_model("tiny")` in `02_transcribe.py` for faster (less accurate) transcription
- For production, use `"medium"` or `"large"` for better accuracy
