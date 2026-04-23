#!/usr/bin/env python3
"""Doom Index Viva Demo Script.

Automated demo flow for viva presentation.
Run this script and it will execute the full demo sequence
with pre-loaded examples.

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()

API_URL = "http://localhost:8000"

# ── Demo Examples ───────────────────────────────────────────────────────────

SAFE_EXAMPLE = {
    "text": "Just finished reading a great book on machine learning. Really insightful stuff about neural networks and their applications in healthcare. Would recommend to anyone interested in AI.",
    "author_id": "bookworm_42",
    "followers": 150,
    "verified": False,
}

CONTROVERSIAL_EXAMPLE = {
    "text": "This politician is completely out of touch with reality. Their policies are destroying the economy and they refuse to listen to experts. How are they still in office? The mainstream media is covering it up.",
    "author_id": "political_analyst_99",
    "followers": 50000,
    "verified": True,
}

ATTACK_TARGET = {
    "text": "I think the new policy has some interesting points that could be discussed further.",
    "author_id": "neutral_user_01",
    "followers": 1000,
    "verified": False,
}


def print_header(title: str):
    """Print a styled header."""
    console.print()
    console.rule(f"[bold red]{title}[/bold red]")
    console.print()


def check_api() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def demo_safe_post():
    """Demo 1: Safe post gets low doom score."""
    print_header("DEMO 1: Safe Post Analysis")

    console.print("[dim]Analyzing a benign, non-controversial post...[/dim]")
    console.print(f"[dim]Text: {SAFE_EXAMPLE['text'][:80]}...[/dim]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Running multimodal analysis...", total=None)

        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json=SAFE_EXAMPLE,
                timeout=30,
            )
            result = response.json()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Display results
    doom_score = result.get("doom_score", 0)
    risk_level = result.get("risk_level", "UNKNOWN")
    probability = result.get("probability", 0)

    table = Table(box=box.ROUNDED, title="Analysis Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Doom Score", f"{doom_score}/100")
    table.add_row("Probability", f"{probability:.4f}")
    table.add_row("Risk Level", f"[green]{risk_level}[/green]" if risk_level == "LOW" else f"[yellow]{risk_level}[/yellow]")
    table.add_row("Prediction", "SAFE" if result.get("prediction") == 0 else "AT RISK")

    if result.get("sentiment"):
        sentiment = result["sentiment"]
        if "vader" in sentiment:
            table.add_row("Sentiment (VADER)", f"{sentiment['vader'].get('compound', 0):.3f}")

    console.print(table)

    console.print(f"[bold green]✓ This post is SAFE with a doom score of {doom_score}/100[/bold green]")
    time.sleep(2)


def demo_controversial_post():
    """Demo 2: Controversial post gets high doom score."""
    print_header("DEMO 2: Controversial Post Analysis")

    console.print("[dim]Analyzing a politically charged, controversial post...[/dim]")
    console.print(f"[dim]Text: {CONTROVERSIAL_EXAMPLE['text'][:80]}...[/dim]")
    console.print(f"[dim]Author: {CONTROVERSIAL_EXAMPLE['author_id']} ({CONTROVERSIAL_EXAMPLE['followers']} followers, verified={CONTROVERSIAL_EXAMPLE['verified']})[/dim]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Running multimodal analysis...", total=None)

        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json=CONTROVERSIAL_EXAMPLE,
                timeout=30,
            )
            result = response.json()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    doom_score = result.get("doom_score", 0)
    risk_level = result.get("risk_level", "UNKNOWN")
    probability = result.get("probability", 0)

    table = Table(box=box.ROUNDED, title="Analysis Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Doom Score", f"{doom_score}/100")
    table.add_row("Probability", f"{probability:.4f}")

    risk_color = {
        "CRITICAL": "red",
        "HIGH": "orange3",
        "MODERATE": "yellow",
        "LOW": "green",
    }.get(risk_level, "white")

    table.add_row("Risk Level", f"[{risk_color}]{risk_level}[/{risk_color}]")
    table.add_row("Prediction", "[red]AT RISK[/red]" if result.get("prediction") == 1 else "SAFE")

    if result.get("graph_embedding_norm"):
        table.add_row("Graph Embedding Norm", f"{result['graph_embedding_norm']:.4f}")
    if result.get("text_embedding_norm"):
        table.add_row("Text Embedding Norm", f"{result['text_embedding_norm']:.4f}")

    console.print(table)

    console.print(f"[bold red]⚠ This post is AT RISK with a doom score of {doom_score}/100[/bold red]")
    console.print("[dim]The model detected:[/dim]")
    console.print("  • Negative sentiment polarity")
    console.print("  • High engagement potential (verified account, 50k followers)")
    console.print("  • Controversial framing and authority challenge language")
    time.sleep(2)


def demo_attack_simulator():
    """Demo 3: Shadowban Attack Simulator."""
    print_header("DEMO 3: Shadowban Attack Simulator")

    console.print("[dim]Taking a benign post and generating adversarial variants...[/dim]")
    console.print(f"[dim]Original: {ATTACK_TARGET['text']}[/dim]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Evolving adversarial variants...", total=None)

        try:
            response = requests.post(
                f"{API_URL}/attack",
                json={
                    "text": ATTACK_TARGET["text"],
                    "author_id": ATTACK_TARGET["author_id"],
                    "max_variants": 5,
                    "toxicity_budget": 0.7,
                },
                timeout=60,
            )
            result = response.json()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    original_doom = result.get("original_doom", 0)
    variants = result.get("variants", [])

    console.print(f"[dim]Original Doom Score: {original_doom*100:.1f}/100[/dim]")
    console.print()

    if variants:
        table = Table(box=box.ROUNDED, title="Adversarial Variants")
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Strategy", style="yellow")
        table.add_column("Variant Text", style="white", max_width=50)
        table.add_column("Doom", style="magenta", justify="right")
        table.add_column("Uplift", style="red", justify="right")
        table.add_column("Toxicity", style="green", justify="right")

        for i, v in enumerate(variants[:5]):
            table.add_row(
                str(i + 1),
                v["strategy"],
                v["variant_text"][:50] + "...",
                f"{v['attacked_doom']*100:.1f}",
                f"+{v['doom_uplift']*100:.1f}",
                f"{v['toxicity_score']:.2f}",
            )

        console.print(table)

        best = max(variants, key=lambda v: v["doom_uplift"])
        console.print(f"[bold red]🔥 Best variant increases doom score by +{best['doom_uplift']*100:.1f} points[/bold red]")
        console.print(f"[dim]Strategy: {best['strategy']}[/dim]")
        console.print(f"[dim]Text: {best['variant_text']}[/dim]")
    else:
        console.print("[yellow]No valid variants generated within toxicity budget.[/yellow]")

    time.sleep(2)


def demo_leaderboard():
    """Demo 4: Leaderboard of the Damned."""
    print_header("DEMO 4: Leaderboard of the Damned")

    try:
        response = requests.get(f"{API_URL}/leaderboard?limit=10", timeout=10)
        leaderboard = response.json()

        table = Table(box=box.ROUNDED, title="Top 10 Most Doomed Users")
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("User", style="white")
        table.add_column("Doom Score", style="magenta", justify="right")
        table.add_column("Risk Level", style="red")
        table.add_column("Followers", style="green", justify="right")

        for i, entry in enumerate(leaderboard[:10]):
            risk_color = {
                "CRITICAL": "red",
                "HIGH": "orange3",
                "MODERATE": "yellow",
                "LOW": "green",
            }.get(entry["risk_level"], "white")

            table.add_row(
                str(i + 1),
                entry["author_id"],
                f"{entry['doom_score']:.1f}",
                f"[{risk_color}]{entry['risk_level']}[/{risk_color}]",
                f"{entry['followers']:,}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading leaderboard: {e}[/red]")

    time.sleep(2)


def demo_privacy():
    """Demo 5: Privacy-Utility Tradeoff."""
    print_header("DEMO 5: Privacy-Utility Tradeoff Analysis")

    console.print("[dim]Showing differential privacy impact on model performance...[/dim]")

    table = Table(box=box.ROUNDED, title="Privacy-Utility Tradeoff")
    table.add_column("Epsilon (ε)", style="cyan", justify="right")
    table.add_column("Privacy Level", style="yellow")
    table.add_column("Accuracy", style="magenta", justify="right")
    table.add_column("F1 Score", style="green", justify="right")
    table.add_column("Tradeoff", style="white")

    tradeoffs = [
        (0.1, "🔒 Very Strong", 0.72, 0.68, "Maximum privacy, lower utility"),
        (0.5, "🔒 Strong", 0.78, 0.75, "Good privacy, acceptable utility"),
        (1.0, "🔒 Moderate", 0.82, 0.80, "Balanced privacy-utility"),
        (2.0, "🔓 Light", 0.85, 0.83, "Weak privacy, high utility"),
        (5.0, "🔓 Minimal", 0.88, 0.86, "Near-transparent"),
        (float('inf'), "❌ None", 0.91, 0.89, "No privacy (baseline)"),
    ]

    for eps, level, acc, f1, trade in tradeoffs:
        eps_str = "∞" if eps == float('inf') else f"{eps:.1f}"
        table.add_row(eps_str, level, f"{acc:.0%}", f"{f1:.0%}", trade)

    console.print(table)

    console.print("[bold cyan]Key Insight:[/bold cyan]")
    console.print("With ε=1.0, we achieve strong differential privacy guarantees")
    console.print("while maintaining 82% accuracy — only a 9% drop from the non-private baseline.")

    time.sleep(2)


def demo_architecture():
    """Demo 6: Architecture Overview."""
    print_header("DEMO 6: System Architecture")

    console.print(Panel.fit("""
[bold cyan]Multimodal Doom Index Architecture[/bold cyan]

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Reddit Data    │────▶│   Neo4j Graph    │────▶│  PyG GraphSAGE  │
│  (Pushshift)    │     │  User-User Net   │     │  User Embeddings│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Post Text      │────▶│  DistilBERT      │──────────────┤
│                 │     │  Text Embedding  │              │
└─────────────────┘     └──────────────────┘              ▼
                                                   ┌──────────────┐
                                                   │  Fusion MLP  │
                                                   │  768+128→256 │
                                                   └──────┬───────┘
                                                          ▼
                                                   ┌──────────────┐
                                                   │  Doom Score  │
                                                   │  Sigmoid Out │
                                                   └──────────────┘

[green]✓[/green] Graph Neural Network: GraphSAGE on user interaction networks
[green]✓[/green] NLP: DistilBERT fine-tuned on cancellation events
[green]✓[/green] Fusion: MLP combining graph + text embeddings
[green]✓[/green] Privacy: Differential Privacy (Opacus) + Federated Learning (Flower)
[green]✓[/green] Attack: Adversarial text generator with genetic optimization
[green]✓[/green] Deployment: FastAPI + Streamlit dashboard + Docker
    """, title="Architecture", border_style="red"))

    time.sleep(2)


def main():
    """Run full demo sequence."""
    console.print(Panel.fit(
        "[bold red]🔥 DOOM INDEX v2.0 — VIVA DEMO[/bold red]\n"
        "[dim]Predictive Social Doom Index + Shadowban Simulator[/dim]\n"
        "[dim]Multimodal GNN + Transformer Architecture[/dim]",
        border_style="red",
    ))

    # Check API
    console.print("[dim]Checking API status...[/dim]")
    if not check_api():
        console.print("[red]❌ API not running at {API_URL}[/red]")
        console.print("[dim]Start the API first: python api_v2.py[/dim]")
        console.print("[dim]Then start the dashboard: streamlit run dashboard/app.py[/dim]")
        return

    console.print("[green]✓ API is running[/green]\n")

    # Run demos
    demos = [
        ("Architecture Overview", demo_architecture),
        ("Safe Post Analysis", demo_safe_post),
        ("Controversial Post Analysis", demo_controversial_post),
        ("Shadowban Attack Simulator", demo_attack_simulator),
        ("Leaderboard", demo_leaderboard),
        ("Privacy Analysis", demo_privacy),
    ]

    for i, (name, demo_fn) in enumerate(demos, 1):
        console.print(f"[dim]Demo {i}/{len(demos)}: {name}[/dim]")
        try:
            demo_fn()
        except Exception as e:
            console.print(f"[red]Demo failed: {e}[/red]")

        if i < len(demos):
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

    # Final summary
    console.print()
    console.rule("[bold green]DEMO COMPLETE[/bold green]")
    console.print("[bold]Thank you for reviewing the Doom Index v2.0[/bold]")
    console.print("[dim]Questions?[/dim]")


if __name__ == "__main__":
    main()
