"""
Earnings Call Audio Search — web demo.

Run with:  .venv/bin/python app.py
Then open: http://localhost:8000
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from mcp_server.server_solution import search_earnings, get_audio_clip, get_news_context

app = FastAPI()

HTML_PAGE = (
    "<!doctype html><html lang='en'><head>"
    "<meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Earnings Call Search</title>"
    "<style>"
    "*{box-sizing:border-box;margin:0;padding:0}"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "background:#0f1117;color:#e0e0e0;min-height:100vh;padding:32px 16px}"
    ".wrap{max-width:860px;margin:0 auto}"
    "h1{font-size:1.6rem;font-weight:700;margin-bottom:4px;color:#fff}"
    ".sub{color:#888;font-size:.9rem;margin-bottom:28px}"
    ".search-row{display:flex;gap:10px;margin-bottom:32px}"
    "input[type=text]{flex:1;padding:12px 16px;border-radius:8px;border:1px solid #333;"
    "background:#1a1d27;color:#fff;font-size:1rem;outline:none}"
    "input[type=text]:focus{border-color:#5b6af0}"
    "button{padding:12px 24px;border-radius:8px;border:none;background:#5b6af0;"
    "color:#fff;font-size:1rem;font-weight:600;cursor:pointer;white-space:nowrap}"
    "button:hover{background:#6b7af8}"
    "button:disabled{opacity:.5;cursor:not-allowed}"
    ".spinner{display:none;margin:40px auto;text-align:center;color:#888}"
    ".results{display:flex;flex-direction:column;gap:20px}"
    ".card{background:#1a1d27;border:1px solid #2a2d3d;border-radius:12px;"
    "padding:20px;transition:border-color .2s}"
    ".card:hover{border-color:#5b6af0}"
    ".card-header{display:flex;align-items:center;gap:12px;margin-bottom:12px;"
    "flex-wrap:wrap}"
    ".badge{background:#5b6af0;color:#fff;font-size:.75rem;font-weight:700;"
    "padding:3px 10px;border-radius:20px;white-space:nowrap}"
    ".badge.aapl{background:#555}"
    ".badge.amzn{background:#e47911}"
    ".badge.wmt{background:#007dc6}"
    ".badge.tsla{background:#cc0000}"
    ".badge.nvda{background:#76b900}"
    ".meta{font-size:.82rem;color:#888}"
    ".score{margin-left:auto;font-size:.82rem;color:#aaa}"
    ".quote{font-size:.95rem;line-height:1.7;color:#ccc;margin-bottom:14px;"
    "border-left:3px solid #5b6af0;padding-left:12px;white-space:pre-wrap}"
    "audio{width:100%;border-radius:6px;margin-top:4px;accent-color:#5b6af0}"
    ".audio-label{font-size:.78rem;color:#666;margin-bottom:4px}"
    ".no-clip{font-size:.82rem;color:#555;font-style:italic}"
    ".news-section{margin-top:16px;border-top:1px solid #2a2d3d;padding-top:14px}"
    ".news-label{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:#888;"
    "text-transform:uppercase;margin-bottom:10px}"
    ".news-window{font-size:.75rem;color:#555;margin-left:8px;font-weight:400}"
    ".article{margin-bottom:10px;padding:10px 12px;background:#13151f;"
    "border-radius:8px;border-left:2px solid #2a2d3d}"
    ".article:hover{border-left-color:#5b6af0}"
    ".article-title{font-size:.88rem;font-weight:600;color:#ddd;margin-bottom:4px}"
    ".article-title a{color:#8b9cf8;text-decoration:none}"
    ".article-title a:hover{text-decoration:underline}"
    ".article-meta{font-size:.75rem;color:#555}"
    ".article-summary{font-size:.82rem;color:#999;margin-top:4px;line-height:1.5}"
    ".error{color:#f87171;text-align:center;padding:20px}"
    ".empty{color:#666;text-align:center;padding:40px}"
    "</style></head><body>"
    "<div class='wrap'>"
    "<h1>Earnings Call Search</h1>"
    "<p class='sub'>Semantic search over earnings call transcripts — hear the exact moment.</p>"
    "<form class='search-row' id='form'>"
    "<input type='text' id='q' name='q' "
    "placeholder='e.g. Which CEOs mentioned tariffs in Q1 2025?' "
    "value='QUERY_PLACEHOLDER' autocomplete='off' autofocus>"
    "<button type='submit' id='btn'>Search</button>"
    "</form>"
    "<div class='spinner' id='spinner'>Searching &amp; fetching audio clips&#8230;</div>"
    "<div class='results' id='results'>RESULTS_PLACEHOLDER</div>"
    "</div>"
    "<script>"
    "const form=document.getElementById('form'),"
    "qInput=document.getElementById('q'),"
    "btn=document.getElementById('btn'),"
    "spin=document.getElementById('spinner'),"
    "res=document.getElementById('results');"
    "form.addEventListener('submit',async e=>{"
    "e.preventDefault();"
    "const q=qInput.value.trim();if(!q)return;"
    "btn.disabled=true;spin.style.display='block';res.innerHTML='';"
    "try{"
    "const r=await fetch('/search?q='+encodeURIComponent(q));"
    "res.innerHTML=await r.text();"
    "}catch(err){res.innerHTML='<p class=error>'+err+'</p>';}"
    "finally{btn.disabled=false;spin.style.display='none';}"
    "});"
    "</script></body></html>"
)

def _news_block(point_id: str) -> str:
    news = get_news_context(point_id)
    if "error" in news:
        return ""
    articles = news.get("articles", [])
    if not articles:
        return ""
    window = news.get("window", "")
    rows = []
    for a in articles:
        title = a.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
        url   = a.get("url", "")
        src   = a.get("source", "")
        pub   = str(a.get("published_at", ""))[:10]
        summ  = a.get("summary", "")[:180].replace("<", "&lt;").replace(">", "&gt;")
        title_html = f'<a href="{url}" target="_blank" rel="noopener">{title}</a>' if url else title
        rows.append(
            f'<div class="article">'
            f'<div class="article-title">{title_html}</div>'
            f'<div class="article-meta">{src} &middot; {pub}</div>'
            + (f'<div class="article-summary">{summ}…</div>' if summ else "") +
            f'</div>'
        )
    window_span = f'<span class="news-window">{window}</span>' if window else ""
    return (
        f'<div class="news-section">'
        f'<div class="news-label">World context {window_span}</div>'
        + "\n".join(rows) +
        f'</div>'
    )


def _ticker_class(ticker: str) -> str:
    return ticker.lower()


def _audio_block(point_id: str) -> str:
    clip = get_audio_clip(point_id)
    if "error" in clip:
        return f'<p class="no-clip">⚠ {clip["error"]}</p>'
    b64 = clip["audio_base64"]
    t0, t1 = clip.get("start_time", 0), clip.get("end_time", 0)
    label = f"{t0:.1f}s – {t1:.1f}s"
    return (
        f'<div class="audio-label">{label}</div>'
        f'<audio controls preload="auto">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
        f'</audio>'
    )


def _render_results(query: str, ticker: str | None = None) -> str:
    results = search_earnings(query=query, ticker=ticker)
    if not results or "error" in results[0]:
        msg = results[0].get("error", "No results") if results else "No results"
        return f'<p class="empty">{msg}</p>'

    cards = []
    for r in results:
        pid    = r["point_id"]
        audio  = _audio_block(pid)
        news   = _news_block(pid)
        ticker = r.get("ticker", "?")
        card = (
            f'<div class="card">'
            f'<div class="card-header">'
            f'<span class="badge {_ticker_class(ticker)}">{ticker}</span>'
            f'<span>{r.get("company","")}</span>'
            f'<span class="meta">{r.get("quarter","")} {r.get("year","")} &middot; {r.get("date","")}</span>'
            f'<span class="score">score {r.get("score",0):.3f}</span>'
            f'</div>'
            f'<div class="quote">{r.get("chunk_text","").replace("  "," ")}</div>'
            f'{audio}'
            f'{news}'
            f'</div>'
        )
        cards.append(card)
    return "\n".join(cards)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("QUERY_PLACEHOLDER", "").replace("RESULTS_PLACEHOLDER", "")


@app.get("/search", response_class=HTMLResponse)
def search(q: str = "", ticker: str = ""):
    if not q:
        return '<p class="empty">Enter a query above.</p>'
    html = _render_results(q, ticker=ticker or None)
    return html


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
