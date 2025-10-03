"""
Type definitions and data models for LLM load testing.
"""

import dataclasses
import math
import time
from typing import Any


@dataclasses.dataclass(frozen=True)
class TriggerShutdown:
    """Signal to trigger shutdown of a process or thread."""


@dataclasses.dataclass(frozen=True)
class SetStartTime:
    """Tracks start time of UserSpawner loop."""

    start_time: int


@dataclasses.dataclass(frozen=True)
class SetLastProcessedTime:
    """Tracks current time of UserSpawner loop."""

    current_time: int


@dataclasses.dataclass(frozen=True)
class SetActiveUsers:
    """Tracks changes in active user count."""

    total_users: int


@dataclasses.dataclass(frozen=True)
class SetUserInfo:
    """Configuration changes for user spawning."""

    max_users: int
    user_addition_count: int
    user_addition_time: float


@dataclasses.dataclass(frozen=True)
class RequestFailureLog:
    """Records information about failed requests."""

    timestamp: int
    start_time: float
    end_time: float
    status_code: int
    user_id: int = 0  # Which user made this request


@dataclasses.dataclass(frozen=True)
class RequestSuccessLog:
    """Records information about successful requests."""

    result_chunks: tuple[bytes, ...]  # Use tuple for immutability
    num_input_tokens: int
    timestamp: int
    token_times: tuple[float, ...]  # Use tuple for immutability
    start_time: float
    end_time: float
    status_code: int
    user_id: int = 0  # Which user made this request
    input_prompt: str = ""  # The actual input prompt text


@dataclasses.dataclass(frozen=True)
class MetricsLog:
    """Records metrics based on time and active user."""

    timestamp: int
    data: int | float


@dataclasses.dataclass(frozen=True)
class ErrorLog:
    """Records error information."""

    error_message: str
    error_type: str = "unknown"
    context: dict[str, Any] = dataclasses.field(default_factory=dict)


def get_timestamp_seconds() -> int:
    """Get current time floor value in seconds."""
    return math.floor(time.time())
