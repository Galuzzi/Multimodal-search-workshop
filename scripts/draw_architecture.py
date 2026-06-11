"""Render the project's current architecture as architecture.png at repo root."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "architecture.png"


# ── Theme ────────────────────────────────────────────────────────────────────
BG = "#0b0d14"
EDGE = "#5b6af0"
TEXT = "#e7e9f3"
MUTED = "#8b90a6"
PANEL = "#141826"
ACCENT = "#22d3ee"
GOOD = "#5ee0a8"
WARN = "#f6c177"

WIDTH, HEIGHT = 22, 14
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=130)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.set_axis_off()


def box(x, y, w, h, title, lines=None, color=EDGE, fill=PANEL, title_color=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.6,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h - 0.32, title,
        color=title_color or TEXT, ha="center", va="center",
        fontsize=11, fontweight="bold", family="DejaVu Sans",
    )
    if lines:
        for i, line in enumerate(lines):
            ax.text(
                x + w / 2, y + h - 0.72 - 0.32 * i, line,
                color=MUTED, ha="center", va="center",
                fontsize=8.5, family="DejaVu Sans",
            )


def arrow(x1, y1, x2, y2, color=EDGE, style="->", lw=1.4, label=None,
          label_offset=(0, 0.18), label_color=None):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, shrinkA=4, shrinkB=4,
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label, color=label_color or MUTED, ha="center", va="center",
            fontsize=8, style="italic",
        )


# ── Title ────────────────────────────────────────────────────────────────────
ax.text(
    WIDTH / 2, HEIGHT - 0.4,
    "Earnings Call MCP Server — Architecture",
    color=TEXT, ha="center", va="center", fontsize=16, fontweight="bold",
)
ax.text(
    WIDTH / 2, HEIGHT - 0.85,
    "Multimodal embeddings (gemini-embedding-2) over diarized earnings calls",
    color=MUTED, ha="center", va="center", fontsize=10, style="italic",
)


# ── Source (top) ─────────────────────────────────────────────────────────────
box(8.5, HEIGHT - 2.4, 5, 1.1, "YouTube",
    ["4 earnings calls: AAPL · AMZN · NVDA · TSLA"], color=ACCENT)


# ── Ingestion pipeline (5 steps, left → right) ───────────────────────────────
PIPE_Y = HEIGHT - 5.4
PIPE_H = 1.7
PIPE_W = 3.95
PIPE_GAP = 0.35
PIPE_X0 = 0.4

steps = [
    ("01_download_audio.py",
     ["yt-dlp", "→ data/audio/*.mp3", "(+ sidecar JSON)"], EDGE),
    ("02_transcribe_and_diarize.py",
     ["Whisper (word ts) + pyannote 3.1", "Gemini 2.5 Flash speaker ID", "→ data/transcripts/*.json"], GOOD),
    ("03_embed_and_index.py",
     ["slice clip → audio_clips/", "Gemini Embedding 2 (text+audio)", "named vectors → Qdrant"], GOOD),
    ("04_build_asknews_context.py",
     ["AskNews DeepNews", "per-chunk context", "→ data/asknews_cache/"], EDGE),
]

for i, (title, lines, c) in enumerate(steps):
    x = PIPE_X0 + i * (PIPE_W + PIPE_GAP)
    box(x, PIPE_Y, PIPE_W, PIPE_H, title, lines, color=c)
    if i < len(steps) - 1:
        arrow(
            x + PIPE_W, PIPE_Y + PIPE_H / 2,
            x + PIPE_W + PIPE_GAP, PIPE_Y + PIPE_H / 2,
            color=MUTED,
        )

# Arrow YouTube → step 01
arrow(11, HEIGHT - 2.4, PIPE_X0 + PIPE_W / 2, PIPE_Y + PIPE_H, color=ACCENT)


# ── Storage row ──────────────────────────────────────────────────────────────
STORE_Y = HEIGHT - 8.8
STORE_H = 2.0

# Qdrant (centre) — wider, the star of the show
box(
    7.0, STORE_Y, 8.0, STORE_H,
    "Qdrant Cloud · earnings_calls",
    [
        "577 points (AAPL 121 · AMZN 161 · NVDA 121 · TSLA 174)",
        "named vectors:  text [3072]  ·  audio [3072]   (cosine)",
        "payload indexes: ticker · date · year · speaker",
        "shared multimodal space → text query can rank audio",
    ],
    color=GOOD,
)

# Local artifact stores around it
box(
    0.4, STORE_Y, 6.2, STORE_H,
    "Local artifact stores",
    [
        "data/audio/                  full MP3 per call",
        "data/audio_clips/*.mp3       pre-sliced per point_id",
        "data/transcripts/*.json      chunks + speaker",
        "data/embedding_cache_v2.json multimodal cache",
        "data/asknews_cache/*.json    historical news bundles",
    ],
    color=EDGE,
)

# Arrows pipeline → storage
arrow(PIPE_X0 + 3 * (PIPE_W + PIPE_GAP) + PIPE_W / 2, PIPE_Y,
      7.0 + 8.0 / 2 - 1.5, STORE_Y + STORE_H, color=GOOD,
      label="upsert", label_offset=(0, 0.2), label_color=GOOD)
arrow(PIPE_X0 + 2 * (PIPE_W + PIPE_GAP) + PIPE_W / 2, PIPE_Y,
      3.5, STORE_Y + STORE_H, color=MUTED,
      label="transcripts + clips", label_color=MUTED)
arrow(PIPE_X0 + 4 * (PIPE_W + PIPE_GAP) + PIPE_W / 2, PIPE_Y,
      3.5 + 2.5, STORE_Y + STORE_H, color=MUTED,
      label="asknews", label_color=MUTED)


# ── Serving layer ────────────────────────────────────────────────────────────
SERVE_Y = HEIGHT - 11.6
SERVE_H = 2.1

# MCP server
mcp_x, mcp_w = 0.4, 10.6
box(
    mcp_x, SERVE_Y, mcp_w, SERVE_H,
    "MCP Server  ·  mcp_server/server.py",
    [
        "search_earnings(query, ticker?, date_range?)  →  query_points(using='text', limit=5)",
        "get_audio_clip(point_id)  →  read data/audio_clips/{point_id}.mp3  →  base64",
        "get_news_context(point_id)  →  data/asknews_cache + live AskNews fallback",
        "recommend_similar(point_id)  →  retrieve seed.text vector + query_points(using='text')",
    ],
    color=ACCENT,
)

# Web app
web_x, web_w = 11.4, 4.5
box(
    web_x, SERVE_Y, web_w, SERVE_H,
    "Web Demo  ·  app.py",
    [
        "FastAPI on http://localhost:8000",
        "calls same Qdrant + clips",
        "inline base64 audio players",
        "AskNews cards",
    ],
    color=ACCENT,
)

# Embeddings module
emb_x, emb_w = 16.4, 5.2
box(
    emb_x, SERVE_Y, emb_w, SERVE_H,
    "mcp_server/embeddings.py",
    [
        "embed_query(text)  →  gemini-embedding-2",
        "→ 3072-dim text vector",
        "disk fallback: embedding_cache_v2.json",
        "(shared space with audio side)",
    ],
    color=EDGE,
)

# Arrows storage → serving
arrow(11.0, STORE_Y, mcp_x + mcp_w * 0.45, SERVE_Y + SERVE_H, color=GOOD,
      label="query_points", label_color=GOOD)
arrow(3.5, STORE_Y, mcp_x + mcp_w * 0.2, SERVE_Y + SERVE_H, color=MUTED)
arrow(15.0, STORE_Y + STORE_H / 2, web_x, SERVE_Y + SERVE_H * 0.6, color=GOOD)
arrow(emb_x + emb_w * 0.05, SERVE_Y + SERVE_H, 11.0, STORE_Y,
      color=EDGE, style="<-", label="embed text", label_color=EDGE)


# ── Clients (bottom) ─────────────────────────────────────────────────────────
CLI_Y = 0.3
CLI_H = 1.4

box(0.4, CLI_Y, 5.0, CLI_H, "Claude Desktop / Claude Code",
    ['"Which CEOs mentioned tariffs in Q1 2025?"', "stdio MCP transport"], color=EDGE)
box(6.0, CLI_Y, 5.0, CLI_H, "Browser  ·  localhost:8000",
    ["search box + audio players", "news cards"], color=EDGE)
box(11.6, CLI_Y, 4.4, CLI_H, "External (optional)",
    ["AskNews API", "HuggingFace (pyannote)"], color=MUTED)

arrow(2.8, SERVE_Y, 2.8, CLI_Y + CLI_H, color=ACCENT, style="<->")
arrow(13.6, SERVE_Y, 8.5, CLI_Y + CLI_H, color=ACCENT, style="<->")
arrow(13.6, SERVE_Y, 13.5, CLI_Y + CLI_H, color=MUTED, style="<->")


# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
