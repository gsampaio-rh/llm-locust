"""
Race summary display and export functionality.

Provides final results visualization and export to CSV/JSON.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from llm_locust.race.config import RaceConfig
    from llm_locust.race.state import RaceState


def show_race_summary(
    config: "RaceConfig",
    state: "RaceState",
    console: Console | None = None,
) -> None:
    """
    Display final race summary with winner and statistics.

    Args:
        config: Race configuration
        state: Final race state
        console: Rich console (creates new if None)
    """
    if console is None:
        console = Console()

    console.print()
    console.print("=" * 80, style="cyan")
    console.print("🏁 [bold cyan]RACE COMPLETE![/bold cyan]", justify="center")
    console.print("=" * 80, style="cyan")
    console.print()

    # Winner announcement
    engine_rankings = []
    for engine in config.engines:
        engine_state = state.get_engine_state(engine.name)
        if engine_state:
            engine_rankings.append((engine, engine_state))

    # Sort by request count
    engine_rankings.sort(key=lambda x: x[1].request_count, reverse=True)

    if engine_rankings:
        winner_engine, winner_state = engine_rankings[0]
        console.print(
            f"🎉 [bold yellow]WINNER: {winner_engine.emoji} {winner_engine.name}[/bold yellow]",
            justify="center",
        )
        console.print(
            f"   [dim]Completed {winner_state.request_count} requests "
            f"with {winner_state.success_rate:.1f}% success rate[/dim]",
            justify="center",
        )
        console.print()

    # Results table
    table = Table(title="📊 Final Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center", style="bold", width=6)
    table.add_column("Engine", style="cyan", width=15)
    table.add_column("Requests", justify="right", style="green")
    table.add_column("Failures", justify="right", style="red")
    table.add_column("Success Rate", justify="right")
    table.add_column("Total Tokens", justify="right", style="blue")

    medals = ["🥇", "🥈", "🥉"]

    for i, (engine, engine_state) in enumerate(engine_rankings):
        rank_display = medals[i] if i < len(medals) else f"{i+1}th"

        table.add_row(
            rank_display,
            f"{engine.emoji} {engine.name}",
            str(engine_state.request_count),
            str(engine_state.failure_count),
            f"{engine_state.success_rate:.1f}%",
            str(engine_state.total_tokens),
        )

    console.print(table)
    console.print()

    # Race statistics
    total_reqs = state.total_requests
    total_fails = state.total_failures
    overall_success = ((total_reqs / (total_reqs + total_fails)) * 100) if (total_reqs + total_fails) > 0 else 0

    stats_text = Text()
    stats_text.append("📈 Race Statistics:\n", style="bold")
    stats_text.append(f"   Total Requests: {total_reqs}\n")
    stats_text.append(f"   Total Failures: {total_fails}\n")
    stats_text.append(f"   Overall Success Rate: {overall_success:.1f}%\n")
    stats_text.append(f"   Duration: {config.duration}s ({config.duration // 60} min)\n")

    console.print(Panel(stats_text, border_style="green"))
    console.print()


def show_export_options(console: Console | None = None) -> None:
    """
    Show export options after race completion.

    Args:
        console: Rich console (creates new if None)
    """
    if console is None:
        console = Console()

    console.print("💾 [bold]Results saved to:[/bold]")
    console.print("   Individual CSVs available in results/ directory")
    console.print()
    console.print("📊 [bold]Next Steps:[/bold]")
    console.print("   1. Analyze results in Streamlit dashboard:")
    console.print("      [cyan]streamlit run streamlit_app/app.py[/cyan]")
    console.print()
    console.print("   2. Compare with previous runs")
    console.print()
    console.print("   3. Run another race:")
    console.print("      [cyan]llm-locust race --config configs/races/your-race.yaml[/cyan]")
    console.print()
    console.print("=" * 80, style="cyan")
    console.print()

