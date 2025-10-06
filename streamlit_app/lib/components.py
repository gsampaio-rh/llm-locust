"""
Reusable UI components for the dashboard.

Follows the Single Responsibility Principle - each function renders one type of component.
"""

from typing import Any

import streamlit as st

from lib.explanations import reliability_to_emoji_and_label, speed_to_emoji_and_label
from models.benchmark import BenchmarkData


def render_platform_header(platform_name: str, is_winner: bool = False) -> None:
    """
    Render platform header card with gradient background.

    Args:
        platform_name: Name of the platform
        is_winner: Whether this platform is the overall winner
    """
    header_text = f"🏆 {platform_name}" if is_winner else platform_name

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 12px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
        <h2 style="color: white; margin: 0; font-size: 16px; font-weight: 600;">{}</h2>
    </div>
    """.format(
            header_text
        ),
        unsafe_allow_html=True,
    )


def render_benchmark_metadata(
    total_requests: int,
    duration_seconds: float,
    concurrency: int,
    quality_score: float,
    benchmark_id: str,
) -> None:
    """
    Render benchmark metadata info card.

    Args:
        total_requests: Number of requests
        duration_seconds: Test duration
        concurrency: Number of concurrent users
        quality_score: Data quality score (0-100)
        benchmark_id: Benchmark identifier
    """
    st.markdown(
        """
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <div style="font-size: 11px; color: #6c757d;">
            📊 {total:,} requests • 🕐 {duration:.0f}s • 👥 {concurrency} users
        </div>
        <div style="font-size: 11px; color: #6c757d; margin-top: 3px;">
            Quality: {quality:.0f}% • {benchmark_id}
        </div>
    </div>
    """.format(
            total=total_requests,
            duration=duration_seconds,
            concurrency=concurrency,
            quality=quality_score,
            benchmark_id=benchmark_id,
        ),
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str,
    value: float,
    unit: str,
    subtitle: str = "",
    bg_color: str = "#f8f9fa",
    text_color: str = "#212529",
    border_color: str = "#dee2e6",
    is_winner: bool = False,
    icon: str = "📊",
) -> None:
    """
    Render a colored metric card.

    Args:
        title: Metric title (e.g., "TTFT")
        value: Main metric value
        unit: Unit string (e.g., "ms", "%", "tok/s")
        subtitle: Additional info below main value
        bg_color: Background color hex
        text_color: Text color hex
        border_color: Left border color hex
        is_winner: Show trophy badge
        icon: Emoji icon for the metric
    """
    winner_badge = " 🏆" if is_winner else ""

    st.markdown(
        """
    <div style="background-color: {}; border-left: 4px solid {}; 
                padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <div style="color: {}; font-size: 12px; font-weight: 600; margin-bottom: 5px;">{} {}</div>
        <div style="color: {}; font-size: 32px; font-weight: 700; line-height: 1;">
            {:.2f}<span style="font-size: 18px;">{}</span>{}
        </div>
        <div style="color: {}; font-size: 11px; margin-top: 8px;">
            {}
        </div>
    </div>
    """.format(
            bg_color,
            border_color,
            text_color,
            icon,
            title,
            text_color,
            value,
            unit,
            winner_badge,
            text_color,
            subtitle,
        ),
        unsafe_allow_html=True,
    )


def get_status_colors(value: float, thresholds: dict[str, float], lower_is_better: bool = True) -> tuple[str, str, str]:
    """
    Get background, text, and border colors based on value and thresholds.

    Args:
        value: The metric value
        thresholds: Dict with 'excellent' and 'good' threshold values
        lower_is_better: If True, lower values are better (for latency)

    Returns:
        (bg_color, text_color, border_color)
    """
    if lower_is_better:
        if value <= thresholds.get("excellent", 500):
            return "#d4edda", "#155724", "#28a745"  # Green
        elif value <= thresholds.get("good", 1000):
            return "#fff3cd", "#856404", "#ffc107"  # Yellow
        else:
            return "#f8d7da", "#721c24", "#dc3545"  # Red
    else:
        if value >= thresholds.get("excellent", 99.9):
            return "#d4edda", "#155724", "#28a745"  # Green
        elif value >= thresholds.get("good", 99):
            return "#fff3cd", "#856404", "#ffc107"  # Yellow
        else:
            return "#f8d7da", "#721c24", "#dc3545"  # Red


def render_help_expander(title: str, content: dict[str, str]) -> None:
    """
    Render expandable help section.

    Args:
        title: Expander title
        content: Dict of {metric_name: explanation}
    """
    with st.expander(title):
        cols = st.columns(2)

        items = list(content.items())
        mid_point = len(items) // 2

        with cols[0]:
            for key, value in items[:mid_point]:
                st.markdown(f"**{key}**")
                st.markdown(value)
                st.markdown("")

        with cols[1]:
            for key, value in items[mid_point:]:
                st.markdown(f"**{key}**")
                st.markdown(value)
                st.markdown("")

