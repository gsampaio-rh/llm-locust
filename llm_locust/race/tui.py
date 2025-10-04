"""
Terminal UI for race visualization using Rich.

Provides live, real-time visualization of multi-engine races with:
- Full-screen terminal mode
- Live progress updates
- Status indicators
- Clean rendering (no flicker)
"""

import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from llm_locust.race.config import RaceConfig
    from llm_locust.race.state import RaceState

# Minimum terminal size
MIN_WIDTH = 80
MIN_HEIGHT = 24


class RaceTUI:
    """
    Terminal UI for race visualization.

    Manages the live display of race progress, engine status, and metrics
    using Rich's Live display system.
    """

    def __init__(self, config: "RaceConfig", state: "RaceState | None" = None) -> None:
        """
        Initialize race TUI.

        Args:
            config: Race configuration
            state: Race state tracker (optional, for live metrics)
        """
        self.config = config
        self.state = state
        self.console = Console()
        self.layout = Layout()
        self.running = False
        self.start_time = 0.0

        # Check terminal size
        terminal_size = self.console.size
        if terminal_size.width < MIN_WIDTH or terminal_size.height < MIN_HEIGHT:
            self.console.print(
                f"[yellow]⚠️  Terminal size ({terminal_size.width}x{terminal_size.height}) "
                f"is smaller than recommended ({MIN_WIDTH}x{MIN_HEIGHT})[/yellow]"
            )
            self.console.print("[yellow]   Some content may not display correctly[/yellow]")
            self.console.print()

    def setup_layout(self) -> None:
        """Set up the layout structure."""
        # Create main layout sections
        self.layout.split(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Split body into engines and leaderboard
        self.layout["body"].split_row(
            Layout(name="engines", ratio=2),
            Layout(name="leaderboard", ratio=1),
        )

    def render_header(self) -> Panel:
        """Render the header section with race info."""
        elapsed = int(time.time() - self.start_time) if self.start_time > 0 else 0
        remaining = max(0, self.config.duration - elapsed)
        progress_pct = (elapsed / self.config.duration * 100) if self.config.duration > 0 else 0

        header_text = Text()
        header_text.append("🏁 ", style="bold")
        header_text.append(self.config.name, style="bold cyan")
        header_text.append("\n")
        header_text.append(
            f"⏱️  {elapsed}s / {self.config.duration}s  "
            f"({progress_pct:.1f}% complete, {remaining}s remaining)",
            style="yellow" if remaining < 10 else "green",
        )

        # Add loading status if state shows no requests yet
        if self.state and self.state.total_requests == 0 and elapsed > 5 and elapsed < 90:
            header_text.append("\n")
            header_text.append(
                f"📦 Engines loading datasets... (~{90-elapsed}s remaining)",
                style="yellow dim"
            )

        return Panel(
            header_text,
            title="Race Status",
            border_style="cyan",
        )

    def render_engines(self) -> Panel:
        """Render the engines section with status and progress for each engine."""
        from rich.table import Table

        # Create a simple table for better visibility
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Engine", style="cyan", width=20)
        table.add_column("Requests", justify="right", style="green")
        table.add_column("Failures", justify="right", style="red")
        table.add_column("Success Rate", justify="right")
        table.add_column("Status", style="yellow")

        for engine in self.config.engines:
            # Get current state if available
            engine_state = self.state.get_engine_state(engine.name) if self.state else None

            if engine_state and engine_state.request_count > 0:
                # Show real data
                table.add_row(
                    f"{engine.emoji} {engine.name}",
                    str(engine_state.request_count),
                    str(engine_state.failure_count),
                    f"{engine_state.success_rate:.1f}%",
                    f"[green]Racing! ({engine_state.requests_per_second:.1f} req/s)[/green]",
                )
            else:
                # Show loading/waiting state based on elapsed time
                elapsed = self.state.elapsed_time if self.state else 0
                if elapsed < 90:
                    # Still in typical loading window
                    status = "[yellow]Loading datasets...[/yellow]"
                else:
                    # Been waiting a long time, might be stuck
                    status = "[red]Waiting (check logs)...[/red]"

                table.add_row(
                    f"{engine.emoji} {engine.name}",
                    "0",
                    "0",
                    "—",
                    status,
                )

        return Panel(
            table,
            title=f"Engines ({len(self.config.engines)})",
            border_style="blue",
        )

    def render_footer(self) -> Panel:
        """Render the footer with controls."""
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to stop race", style="dim")

        return Panel(
            footer_text,
            border_style="dim",
        )

    def render_leaderboard(self) -> Panel:
        """Render the leaderboard with rankings."""
        from rich.table import Table

        table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 1))
        table.add_column("Rank", justify="center", style="bold", width=6)
        table.add_column("Engine", style="cyan")
        table.add_column("Requests", justify="right", style="green")

        # Get engine states and sort by request count
        engine_rankings = []
        for engine in self.config.engines:
            engine_state = self.state.get_engine_state(engine.name) if self.state else None
            if engine_state:
                engine_rankings.append((engine, engine_state))

        # Sort by request count (descending)
        engine_rankings.sort(key=lambda x: x[1].request_count, reverse=True)

        # Medals for top 3
        medals = ["🥇", "🥈", "🥉"]

        for i, (engine, engine_state) in enumerate(engine_rankings):
            rank_num = i + 1
            rank_display = medals[i] if i < len(medals) else f"{rank_num}th"

            # Color code based on rank
            if i == 0:
                name_style = "bold yellow"
            elif i == 1:
                name_style = "bold white"
            elif i == 2:
                name_style = "bold rgb(205,127,50)"  # Bronze
            else:
                name_style = "dim"

            table.add_row(
                rank_display,
                f"[{name_style}]{engine.emoji} {engine.name}[/{name_style}]",
                str(engine_state.request_count),
            )

        # If no requests yet, show all engines as tied
        if not engine_rankings:
            for engine in self.config.engines:
                table.add_row("—", f"{engine.emoji} {engine.name}", "0")

        return Panel(
            table,
            title="🏆 Leaderboard",
            border_style="yellow",
        )

    def render(self) -> Layout:
        """Render the complete TUI."""
        self.layout["header"].update(self.render_header())
        self.layout["engines"].update(self.render_engines())
        self.layout["leaderboard"].update(self.render_leaderboard())
        self.layout["footer"].update(self.render_footer())
        return self.layout

    def start(self) -> None:
        """Start the live TUI display."""
        self.setup_layout()
        self.running = True
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop the TUI display."""
        self.running = False

    def run_live(self, duration: float = 0) -> None:
        """
        Run the TUI in live mode with automatic refresh.

        Args:
            duration: How long to run (0 = until stopped)
        """
        self.start()

        try:
            with Live(
                self.render(),
                console=self.console,
                refresh_per_second=10,  # 10 FPS
                screen=False,  # Don't take over full screen (for now)
            ) as live:
                end_time = time.time() + duration if duration > 0 else float("inf")

                while self.running and time.time() < end_time:
                    live.update(self.render())
                    time.sleep(0.1)  # 10 FPS

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        finally:
            self.stop()


def show_race_header(config: "RaceConfig", console: Console | None = None) -> None:
    """
    Show static race header before starting live display.

    Args:
        config: Race configuration
        console: Rich console (creates new if None)
    """
    if console is None:
        console = Console()

    console.print()
    console.print("=" * 80, style="cyan")
    console.print(f"🏁 [bold cyan]{config.name}[/bold cyan]", justify="center")
    console.print("=" * 80, style="cyan")
    console.print()

    # Engine list
    console.print(f"[bold]Engines:[/bold] {len(config.engines)}")
    for engine in config.engines:
        console.print(
            f"   {engine.emoji} [bold {engine.color}]{engine.name}[/bold {engine.color}] - {engine.model}"
        )

    console.print()
    console.print(f"[bold]Duration:[/bold] {config.duration}s ({config.duration // 60} minutes)")
    console.print(f"[bold]Users:[/bold] {config.users} per engine")
    console.print(f"[bold]Spawn Rate:[/bold] {config.spawn_rate} users/second")
    console.print(f"[bold]Dataset:[/bold] {config.dataset.upper()}")
    console.print()
    console.print("=" * 80, style="cyan")
    console.print()

