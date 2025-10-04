"""
Interactive race module for multi-endpoint benchmarking.

This module provides the "Great Model Race" functionality - an interactive TUI
that allows racing multiple LLM endpoints head-to-head with live visualization.
"""

from llm_locust.race.animation import AnimatedValue, CounterAnimation, ProgressAnimation
from llm_locust.race.charts import render_metric_chart, show_charts_view
from llm_locust.race.config import EngineConfig, RaceConfig, load_race_config
from llm_locust.race.health import HealthStatus, calculate_health_status, get_health_badge
from llm_locust.race.orchestrator import RaceOrchestrator
from llm_locust.race.runner import run_engine_benchmark
from llm_locust.race.sparkline import render_sparkline, render_sparkline_with_color
from llm_locust.race.state import EngineState, RaceState
from llm_locust.race.summary import show_export_options, show_race_summary
from llm_locust.race.tui import RaceTUI, show_race_header

__all__ = [
    "AnimatedValue",
    "CounterAnimation",
    "EngineConfig",
    "EngineState",
    "HealthStatus",
    "ProgressAnimation",
    "RaceConfig",
    "RaceOrchestrator",
    "RaceState",
    "RaceTUI",
    "calculate_health_status",
    "get_health_badge",
    "load_race_config",
    "render_metric_chart",
    "render_sparkline",
    "render_sparkline_with_color",
    "run_engine_benchmark",
    "show_charts_view",
    "show_export_options",
    "show_race_header",
    "show_race_summary",
]
