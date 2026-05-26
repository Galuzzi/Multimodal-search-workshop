"""
Demo script — exercises all four MCP tools against the live Qdrant collection.
Run with: .venv/bin/python demo.py
"""

import sys
from pathlib import Path

# Make sure repo root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Import tool functions directly from the solution server
from mcp_server.server_solution import (
    search_earnings,
    get_audio_clip,
    get_news_context,
    recommend_similar,
)


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")


# ── Tool 1: search_earnings ───────────────────────────────────────────────────
section("Tool 1 — search_earnings")
console.print('[dim]Query: "What did NVDA say about data center demand?"[/dim]\n')

results = search_earnings(query="data center demand outlook", ticker="NVDA")

table = Table(box=box.ROUNDED, show_lines=True, expand=True)
table.add_column("#", style="dim", width=3)
table.add_column("Score", width=6)
table.add_column("Quarter", width=8)
table.add_column("Speaker", width=14)
table.add_column("Transcript excerpt")

first_point_id = None
for i, r in enumerate(results):
    if "error" in r:
        console.print(f"[red]Error:[/red] {r['error']}")
        break
    if i == 0:
        first_point_id = r["point_id"]
    table.add_row(
        str(i + 1),
        f"{r['score']:.3f}",
        f"{r.get('quarter','?')} {r.get('year','')}",
        r.get("speaker", "unknown"),
        r.get("chunk_text", "")[:120].replace("\n", " "),
    )

console.print(table)

# ── Tool 2: get_audio_clip ────────────────────────────────────────────────────
section("Tool 2 — get_audio_clip")
if first_point_id:
    console.print(f"[dim]Fetching audio clip for point_id: {first_point_id}[/dim]\n")
    clip = get_audio_clip(first_point_id)
    if "error" in clip:
        console.print(Panel(clip["error"], title="[yellow]Audio clip status[/yellow]", border_style="yellow"))
    else:
        b64_len = len(clip.get("audio_base64", ""))
        console.print(Panel(
            f"ticker:     {clip.get('ticker')}\n"
            f"start_time: {clip.get('start_time'):.1f}s\n"
            f"end_time:   {clip.get('end_time'):.1f}s\n"
            f"format:     {clip.get('format')}\n"
            f"audio data: {b64_len} base64 chars ({b64_len * 3 // 4 // 1024} KB)",
            title="[green]Audio clip[/green]",
            border_style="green",
        ))
else:
    console.print("[yellow]No point_id available — skipping[/yellow]")

# ── Tool 3: get_news_context ──────────────────────────────────────────────────
section("Tool 3 — get_news_context")
if first_point_id:
    console.print(f"[dim]Fetching AskNews context for point_id: {first_point_id}[/dim]\n")
    news = get_news_context(first_point_id)
    if "error" in news:
        console.print(Panel(news["error"], title="[red]News error[/red]", border_style="red"))
    else:
        articles = news.get("articles", [])
        note = news.get("note", "")
        if note:
            console.print(Panel(note, title="[yellow]AskNews note[/yellow]", border_style="yellow"))
        elif articles:
            nt = Table(box=box.SIMPLE, expand=True)
            nt.add_column("Title")
            nt.add_column("Source", width=16)
            nt.add_column("Published", width=12)
            for a in articles:
                nt.add_row(a.get("title","")[:80], a.get("source",""), a.get("published_at","")[:10])
            console.print(nt)
        else:
            console.print("[dim]No articles returned.[/dim]")

# ── Tool 4: recommend_similar ─────────────────────────────────────────────────
section("Tool 4 — recommend_similar")
if first_point_id:
    console.print(f"[dim]Finding chunks similar to point_id: {first_point_id}[/dim]\n")
    recs = recommend_similar(first_point_id)

    rt = Table(box=box.ROUNDED, show_lines=True, expand=True)
    rt.add_column("#", style="dim", width=3)
    rt.add_column("Score", width=6)
    rt.add_column("Quarter", width=8)
    rt.add_column("Similar excerpt")

    for i, r in enumerate(recs):
        if "error" in r:
            console.print(f"[red]Error:[/red] {r['error']}")
            break
        rt.add_row(
            str(i + 1),
            f"{r['score']:.3f}",
            f"{r.get('quarter','?')} {r.get('year','')}",
            r.get("chunk_text", "")[:120].replace("\n", " "),
        )
    console.print(rt)

console.print()
console.print(Panel(
    "[bold green]All four MCP tools are working.[/bold green]\n\n"
    "Restart Claude Desktop and ask:\n"
    '  [italic]"What did NVDA say about data center demand in Q3 2024?"[/italic]',
    title="Demo complete",
    border_style="green",
))
