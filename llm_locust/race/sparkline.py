"""
Sparkline rendering for compact metric visualization.

Provides ASCII sparklines to show metric trends over time in minimal space.
"""

from collections.abc import Sequence

# Sparkline characters (8 levels)
SPARKLINE_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def render_sparkline(
    values: Sequence[float],
    width: int = 20,
    min_val: float | None = None,
    max_val: float | None = None,
) -> str:
    """
    Render a sparkline from a sequence of values.

    Args:
        values: Numeric values to visualize
        width: Maximum width in characters
        min_val: Minimum value for scaling (auto if None)
        max_val: Maximum value for scaling (auto if None)

    Returns:
        String containing sparkline characters
    """
    if not values:
        return "▁" * min(width, 5)  # Flat line if no data

    # Take last N values to fit width
    values = list(values)[-width:]

    # Calculate range
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)

    # Avoid division by zero
    value_range = max_val - min_val
    if value_range == 0:
        return SPARKLINE_CHARS[0] * len(values)

    # Map values to sparkline characters
    sparkline = ""
    for value in values:
        # Normalize to 0-1
        normalized = (value - min_val) / value_range
        # Map to character index (0-7)
        index = int(normalized * (len(SPARKLINE_CHARS) - 1))
        index = max(0, min(index, len(SPARKLINE_CHARS) - 1))
        sparkline += SPARKLINE_CHARS[index]

    return sparkline


def render_sparkline_with_color(
    values: Sequence[float],
    width: int = 20,
    threshold_good: float | None = None,
    threshold_bad: float | None = None,
) -> tuple[str, str]:
    """
    Render sparkline with color based on thresholds.

    Args:
        values: Numeric values to visualize
        width: Maximum width in characters
        threshold_good: Value below which is "good" (green)
        threshold_bad: Value above which is "bad" (red)

    Returns:
        Tuple of (sparkline_string, color_style)
    """
    sparkline = render_sparkline(values, width)

    if not values:
        return sparkline, "dim"

    # Determine color based on recent values (last 5)
    recent_values = list(values)[-5:]
    avg_recent = sum(recent_values) / len(recent_values)

    if threshold_good and avg_recent <= threshold_good:
        color = "green"
    elif threshold_bad and avg_recent >= threshold_bad:
        color = "red"
    else:
        color = "yellow"

    return sparkline, color


def get_trend_indicator(values: Sequence[float]) -> str:
    """
    Get trend indicator based on recent values.

    Args:
        values: Numeric values to analyze

    Returns:
        Trend indicator: "↗" (improving), "↘" (degrading), "→" (stable)
    """
    if len(values) < 2:
        return "→"

    # Compare last 5 values to previous 5
    recent = list(values)[-5:]
    previous = list(values)[-10:-5] if len(values) >= 10 else list(values)[:-5]

    if not previous:
        return "→"

    avg_recent = sum(recent) / len(recent)
    avg_previous = sum(previous) / len(previous)

    # 10% threshold for change
    threshold = abs(avg_previous) * 0.1

    if avg_recent > avg_previous + threshold:
        return "↗"
    elif avg_recent < avg_previous - threshold:
        return "↘"
    else:
        return "→"


def format_metric_with_sparkline(
    metric_name: str,
    current_value: float,
    values_history: Sequence[float],
    unit: str = "ms",
    width: int = 15,
    threshold_good: float | None = None,
    threshold_bad: float | None = None,
) -> str:
    """
    Format a metric line with sparkline.

    Args:
        metric_name: Name of metric (e.g., "TTFT")
        current_value: Current value to display
        values_history: Historical values for sparkline
        unit: Unit suffix (e.g., "ms", "tok/s")
        width: Sparkline width
        threshold_good: Good threshold for coloring
        threshold_bad: Bad threshold for coloring

    Returns:
        Formatted string with metric name, value, and sparkline
    """
    sparkline, color = render_sparkline_with_color(
        values_history, width, threshold_good, threshold_bad
    )
    trend = get_trend_indicator(values_history)

    # Format: "TTFT: 234ms  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  ↗"
    return f"{metric_name}: {current_value:6.1f}{unit}  {sparkline} {trend}"

