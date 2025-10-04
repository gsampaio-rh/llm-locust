"""
Health monitoring and status indicators for race engines.

Tracks engine health based on error rates, latency, and performance metrics.
"""

from enum import Enum


class HealthStatus(Enum):
    """Health status levels for engines."""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    INITIALIZING = "initializing"


def get_health_badge(status: HealthStatus) -> str:
    """
    Get emoji badge for health status.

    Args:
        status: Health status

    Returns:
        Emoji badge string
    """
    badges = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.WARNING: "⚠️",
        HealthStatus.ERROR: "❌",
        HealthStatus.INITIALIZING: "⏳",
    }
    return badges.get(status, "❓")


def get_health_color(status: HealthStatus) -> str:
    """
    Get Rich color for health status.

    Args:
        status: Health status

    Returns:
        Rich color name
    """
    colors = {
        HealthStatus.HEALTHY: "green",
        HealthStatus.WARNING: "yellow",
        HealthStatus.ERROR: "red",
        HealthStatus.INITIALIZING: "blue",
    }
    return colors.get(status, "white")


def calculate_health_status(
    request_count: int,
    _failure_count: int,
    success_rate: float,
    avg_ttft: float,
    recent_errors: int = 0,
) -> HealthStatus:
    """
    Calculate overall health status based on metrics.

    Args:
        request_count: Total successful requests
        failure_count: Total failed requests
        success_rate: Success rate percentage
        avg_ttft: Average TTFT in milliseconds
        recent_errors: Errors in last window

    Returns:
        Health status
    """
    # Still initializing
    if request_count == 0:
        return HealthStatus.INITIALIZING

    # Error conditions
    if success_rate < 90.0:
        return HealthStatus.ERROR

    if recent_errors > 5:
        return HealthStatus.ERROR

    # Warning conditions
    if success_rate < 98.0:
        return HealthStatus.WARNING

    if avg_ttft > 2000:  # >2 seconds TTFT
        return HealthStatus.WARNING

    if recent_errors > 0:
        return HealthStatus.WARNING

    # All good
    return HealthStatus.HEALTHY


def get_status_message(status: HealthStatus, request_count: int = 0) -> str:
    """
    Get human-readable status message.

    Args:
        status: Health status
        request_count: Number of requests (for context)

    Returns:
        Status message
    """
    if status == HealthStatus.HEALTHY:
        return f"Healthy ({request_count} reqs)"
    elif status == HealthStatus.WARNING:
        return "Performance degraded"
    elif status == HealthStatus.ERROR:
        return "High error rate!"
    elif status == HealthStatus.INITIALIZING:
        return "Initializing..."
    else:
        return "Unknown"

