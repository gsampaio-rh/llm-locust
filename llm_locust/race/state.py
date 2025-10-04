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
from llm_locust.race.animation import CounterAnimation

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

    # Metric history for sparklines (last 60 data points)
    ttft_history: list[float] = field(default_factory=list)
    tpot_history: list[float] = field(default_factory=list)
    throughput_history: list[float] = field(default_factory=list)
    request_rate_history: list[float] = field(default_factory=list)

    # Animated counters for smooth display
    _animated_requests: CounterAnimation | None = field(default=None, init=False)
    _animated_tokens: CounterAnimation | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize animated counters."""
        object.__setattr__(self, "_animated_requests", CounterAnimation(0, speed=20.0))
        object.__setattr__(self, "_animated_tokens", CounterAnimation(0, speed=50.0))

    def get_animated_requests(self) -> int:
        """Get smoothly animated request count."""
        if self._animated_requests:
            self._animated_requests.set(self.request_count)
            return self._animated_requests.get()
        return self.request_count

    def get_animated_tokens(self) -> int:
        """Get smoothly animated token count."""
        if self._animated_tokens:
            self._animated_tokens.set(self.total_tokens)
            return self._animated_tokens.get()
        return self.total_tokens

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

    @property
    def avg_ttft(self) -> float:
        """Get average TTFT from recent history."""
        if not self.ttft_history:
            return 0.0
        return sum(self.ttft_history) / len(self.ttft_history)

    @property
    def avg_tpot(self) -> float:
        """Get average TPOT from recent history."""
        if not self.tpot_history:
            return 0.0
        return sum(self.tpot_history) / len(self.tpot_history)

    @property
    def avg_throughput(self) -> float:
        """Get average throughput from recent history."""
        if not self.throughput_history:
            return 0.0
        return sum(self.throughput_history) / len(self.throughput_history)


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
        # Calculate metrics from the request
        num_output_tokens = len(log.result_chunks)
        duration = log.end_time - log.start_time

        # Calculate TTFT (time to first token)
        ttft_ms = (log.token_times[0] - log.start_time) * 1000 if log.token_times else 0

        # Calculate TPOT (time per output token)
        if num_output_tokens > 1 and len(log.token_times) > 1:
            generation_time = log.token_times[-1] - log.token_times[0]
            tpot_ms = (generation_time / (num_output_tokens - 1)) * 1000
        else:
            tpot_ms = 0

        # Calculate throughput (tokens per second)
        throughput = (num_output_tokens / duration) if duration > 0 else 0

        # We need to map user_id to engine somehow
        # For now, increment all engines equally (will fix with better tracking)
        for engine_state in self.engines.values():
            engine_state.request_count += 1
            engine_state.total_tokens += log.num_input_tokens + num_output_tokens
            engine_state.last_update = time.time()

            # Add to metric history (keep last 60 points)
            max_history = 60
            if ttft_ms > 0:
                engine_state.ttft_history.append(ttft_ms)
                if len(engine_state.ttft_history) > max_history:
                    engine_state.ttft_history.pop(0)

            if tpot_ms > 0:
                engine_state.tpot_history.append(tpot_ms)
                if len(engine_state.tpot_history) > max_history:
                    engine_state.tpot_history.pop(0)

            if throughput > 0:
                engine_state.throughput_history.append(throughput)
                if len(engine_state.throughput_history) > max_history:
                    engine_state.throughput_history.pop(0)

            # Track request rate
            current_rate = engine_state.requests_per_second
            engine_state.request_rate_history.append(current_rate)
            if len(engine_state.request_rate_history) > max_history:
                engine_state.request_rate_history.pop(0)

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

