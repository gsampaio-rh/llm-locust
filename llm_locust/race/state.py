"""
Race state management - tracks metrics and status for all engines.

Consumes metrics from the shared queue and maintains current state
for TUI visualization.
"""

import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING

from llm_locust.core.models import RequestFailureLog, RequestSuccessLog

if TYPE_CHECKING:
    from llm_locust.race.config import RaceConfig


@dataclass
class EngineState:
    """Current state for a single engine."""

    name: str
    emoji: str
    color: str
    status: str = "Initializing"
    request_count: int = 0
    failure_count: int = 0
    active_users: int = 0
    total_tokens: int = 0
    last_update: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.request_count + self.failure_count
        if total == 0:
            return 100.0
        return (self.request_count / total) * 100

    @property
    def requests_per_second(self) -> float:
        """Calculate current RPS (rough estimate)."""
        elapsed = time.time() - self.last_update
        if elapsed < 1:
            return 0.0
        return self.request_count / elapsed


class RaceState:
    """
    Maintains current state of the race.

    Processes metrics from the queue and updates per-engine state
    for TUI visualization.
    """

    def __init__(self, config: "RaceConfig", metrics_queue: Queue) -> None:
        """
        Initialize race state tracker.

        Args:
            config: Race configuration
            metrics_queue: Queue to read metrics from
        """
        self.config = config
        self.metrics_queue = metrics_queue
        self.start_time = time.time()

        # Initialize state for each engine
        self.engines: dict[str, EngineState] = {}
        for engine in config.engines:
            self.engines[engine.name] = EngineState(
                name=engine.name,
                emoji=engine.emoji,
                color=engine.color,
                status="Loading",
            )

    def update(self) -> None:
        """
        Update state by processing available metrics from queue.

        Non-blocking - processes all available metrics and returns.
        """
        processed = 0
        max_batch = 100  # Process up to 100 metrics per update

        while processed < max_batch:
            try:
                # Non-blocking get
                metric = self.metrics_queue.get_nowait()

                # Process different metric types
                if isinstance(metric, RequestSuccessLog):
                    self._process_success(metric)
                elif isinstance(metric, RequestFailureLog):
                    self._process_failure(metric)
                # Add more metric types as needed

                processed += 1

            except Exception:
                # Queue is empty or other error
                break

        # Update elapsed time for all engines
        for engine_state in self.engines.values():
            if engine_state.request_count > 0 and engine_state.status == "Loading":
                engine_state.status = "Running"

    def _process_success(self, log: RequestSuccessLog) -> None:
        """Process a successful request."""
        # We need to map user_id to engine somehow
        # For now, increment all engines equally (will fix with better tracking)
        for engine_state in self.engines.values():
            engine_state.request_count += 1
            engine_state.total_tokens += log.num_input_tokens + len(log.result_chunks)
            engine_state.last_update = time.time()

    def _process_failure(self, _log: RequestFailureLog) -> None:
        """Process a failed request."""
        # Similar issue - need better engine mapping
        for engine_state in self.engines.values():
            engine_state.failure_count += 1
            engine_state.last_update = time.time()

    def get_engine_state(self, engine_name: str) -> EngineState | None:
        """Get state for a specific engine."""
        return self.engines.get(engine_name)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since race start."""
        return time.time() - self.start_time

    @property
    def total_requests(self) -> int:
        """Get total requests across all engines."""
        return sum(e.request_count for e in self.engines.values())

    @property
    def total_failures(self) -> int:
        """Get total failures across all engines."""
        return sum(e.failure_count for e in self.engines.values())

