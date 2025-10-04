"""
Race orchestrator - coordinates multi-process benchmark execution.

The orchestrator manages the lifecycle of racing multiple engines:
- Spawns separate process per engine
- Coordinates synchronized start (countdown)
- Collects metrics from all processes
- Handles graceful shutdown
"""

import logging
import signal
import time
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING

from rich.console import Console

from llm_locust.core.models import TriggerShutdown
from llm_locust.race.state import RaceState
from llm_locust.race.summary import show_export_options, show_race_summary
from llm_locust.race.tui import RaceTUI, show_race_header

if TYPE_CHECKING:
    from llm_locust.race.config import RaceConfig

logger = logging.getLogger(__name__)


class RaceOrchestrator:
    """
    Orchestrates multi-engine races.

    Manages process lifecycle, metrics collection, and coordination
    between multiple benchmark processes running in parallel.
    """

    def __init__(self, config: "RaceConfig", enable_tui: bool = True) -> None:
        """
        Initialize race orchestrator.

        Args:
            config: Validated race configuration
            enable_tui: Enable live TUI (default: True)
        """
        self.config = config
        self.enable_tui = enable_tui
        self.metrics_queue: Queue = Queue()
        self.control_queues: dict[str, Queue] = {}
        self.processes: list[Process] = []
        self.running = False
        self.console = Console()

        # Initialize state tracking
        self.state = RaceState(config, self.metrics_queue)

        # Initialize TUI if enabled
        self.tui = RaceTUI(config, self.state) if enable_tui else None

    def start_race(self) -> None:
        """
        Start the race with countdown and synchronized execution.

        This is the main entry point for running a race. It will:
        1. Show countdown (3...2...1...GO!)
        2. Spawn process for each engine
        3. Wait for completion or user interrupt
        4. Clean up resources
        """
        # Show race header
        if self.enable_tui:
            show_race_header(self.config, self.console)
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"🏁 RACE: {self.config.name}")
            logger.info("=" * 80)
            logger.info("")

        # Spawn processes for each engine
        if not self.enable_tui:
            logger.info("🚀 Spawning engine processes...")
        self._spawn_engine_processes()

        # Warm-up phase: Let engines load datasets
        self._warmup_phase()

        # Countdown after warm-up
        self._countdown()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        start_time = time.time()  # Start timer AFTER warm-up

        if self.enable_tui and self.tui:
            # Run with TUI
            self._run_with_tui(start_time)
        else:
            # Run without TUI (text mode)
            self._run_without_tui(start_time)

        # Stop all engines
        self.stop_race()

        # Show race summary
        if self.enable_tui:
            show_race_summary(self.config, self.state, self.console)
            show_export_options(self.console)
        else:
            logger.info("✅ Race complete!")

    def _run_with_tui(self, start_time: float) -> None:
        """Run race with live TUI display."""
        from rich.live import Live

        if not self.tui:
            return

        self.tui.start_time = start_time  # Set start time for elapsed calculation
        self.tui.start()

        try:
            with Live(
                self.tui.render(),
                console=self.console,
                refresh_per_second=20,  # 20 FPS for smooth animations
                screen=False,
                transient=False,  # Don't clear on exit
            ) as live:
                while self.running and (time.time() - start_time) < self.config.duration:
                    # Update state from metrics queue
                    self.state.update()

                    # Force TUI update
                    live.update(self.tui.render(), refresh=True)

                    time.sleep(0.05)  # Update every 50ms (20 FPS for smooth animations)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Race interrupted by user[/yellow]")
        finally:
            self.tui.stop()

    def _run_without_tui(self, start_time: float) -> None:
        """Run race without TUI (text mode with periodic updates)."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("🏁 RACE IN PROGRESS")
        logger.info("=" * 80)
        logger.info(f"Running for {self.config.duration} seconds...")
        logger.info("Press Ctrl+C to stop early")
        logger.info("")

        try:
            while self.running and (time.time() - start_time) < self.config.duration:
                elapsed = int(time.time() - start_time)
                remaining = self.config.duration - elapsed

                # Update state
                self.state.update()

                # Progress update every 30 seconds
                if elapsed % 30 == 0 and elapsed > 0:
                    progress_pct = (elapsed / self.config.duration) * 100
                    logger.info(
                        f"⏱️  Progress: {elapsed}s / {self.config.duration}s "
                        f"({progress_pct:.1f}% complete, {remaining}s remaining)"
                    )

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Race interrupted by user")

        logger.info("")
        logger.info("=" * 80)
        logger.info("⏹️  Race Duration Complete")
        logger.info("=" * 80)
        logger.info("")

    def stop_race(self) -> None:
        """Gracefully stop the race and clean up resources."""
        logger.info("🛑 Stopping race...")
        self.running = False

        # Send shutdown signal to all engines
        for engine_name, control_queue in self.control_queues.items():
            logger.info(f"   Stopping {engine_name}...")
            try:
                control_queue.put(TriggerShutdown())
            except Exception as e:
                logger.warning(f"   Failed to send shutdown to {engine_name}: {e}")

        # Wait for processes to finish gracefully
        logger.info("⏳ Waiting for engines to finish...")
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"   Process {process.name} did not stop gracefully, terminating...")
                process.terminate()
                process.join(timeout=5)

        logger.info("✅ All engines stopped")

    def _warmup_phase(self) -> None:
        """Wait for engines to load datasets and warm up."""
        if self.enable_tui:
            self.console.print()
            self.console.print("📦 [bold yellow]Warming up engines...[/bold yellow]")
            self.console.print("   Loading datasets and initializing...")
        else:
            logger.info("")
            logger.info("📦 Warming up engines...")
            logger.info("   Loading datasets and initializing...")

        # Wait for engines to load (give them time to initialize)
        # We'll wait until we see some activity or max 90 seconds
        warmup_start = time.time()
        max_warmup = 90  # Maximum 90 seconds for warmup

        while (time.time() - warmup_start) < max_warmup:
            # Check if any requests have started
            self.state.update()
            if self.state.total_requests > 0:
                # At least one engine has started making requests
                break

            elapsed = int(time.time() - warmup_start)
            if elapsed % 10 == 0 and elapsed > 0:
                if self.enable_tui:
                    self.console.print(f"   Still loading... ({elapsed}s)")
                else:
                    logger.info(f"   Still loading... ({elapsed}s)")

            time.sleep(1)

        warmup_time = int(time.time() - warmup_start)
        if self.enable_tui:
            self.console.print(f"✅ [green]Engines ready![/green] (warmup: {warmup_time}s)")
            self.console.print()
        else:
            logger.info(f"✅ Engines ready! (warmup: {warmup_time}s)")
            logger.info("")

    def _countdown(self) -> None:
        """Show countdown before starting race."""
        if self.enable_tui:
            self.console.print("🏁 [bold]Starting race in...[/bold]")
            for i in range(3, 0, -1):
                self.console.print(f"   [bold yellow]{i}...[/bold yellow]")
                time.sleep(1)
            self.console.print("   [bold green]GO! 🚀[/bold green]")
            self.console.print()
        else:
            logger.info("🏁 Starting race in...")
            for i in range(3, 0, -1):
                logger.info(f"   {i}...")
                time.sleep(1)
            logger.info("   GO! 🚀")
            logger.info("")

    def _spawn_engine_processes(self) -> None:
        """Spawn a process for each engine."""
        from llm_locust.race.runner import run_engine_benchmark

        for engine in self.config.engines:
            # Create control queue for this engine
            control_queue: Queue = Queue()
            self.control_queues[engine.name] = control_queue

            # Create process
            process = Process(
                target=run_engine_benchmark,
                args=(engine, self.config, self.metrics_queue, control_queue),
                name=f"engine-{engine.name}",
            )

            self.processes.append(process)
            process.start()
            logger.info(f"   ✅ Spawned process for {engine.emoji} {engine.name} (PID: {process.pid})")

        logger.info(f"✅ Spawned {len(self.processes)} engine processes")

    def _signal_handler(self, _sig: int, _frame: object) -> None:
        """Handle interrupt signals gracefully."""
        logger.info("")
        logger.info("🛑 Received interrupt signal, shutting down...")
        self.running = False
