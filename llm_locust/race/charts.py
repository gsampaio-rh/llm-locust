"""
Terminal-based time-series charts using plotext.

Provides detailed metric visualization for race analysis.
"""

from typing import TYPE_CHECKING

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False

if TYPE_CHECKING:
    from llm_locust.race.state import RaceState


def render_metric_chart(
    state: "RaceState",
    metric: str = "ttft",
    width: int = 80,
    height: int = 20,
) -> str:
    """
    Render a time-series chart for a specific metric.

    Args:
        state: Race state with metric history
        metric: Metric to chart ("ttft", "tpot", "throughput")
        width: Chart width in characters
        height: Chart height in characters

    Returns:
        Rendered chart as string
    """
    if not PLOTEXT_AVAILABLE:
        return "⚠️  plotext not installed. Run: pip install plotext>=5.2.8"

    plt.clf()  # Clear previous plot
    plt.plotsize(width, height)

    # Set theme
    plt.theme("dark")

    # Metric configurations
    metric_config = {
        "ttft": {
            "title": "Time to First Token (TTFT)",
            "ylabel": "ms",
            "attr": "ttft_history",
        },
        "tpot": {
            "title": "Time Per Output Token (TPOT)",
            "ylabel": "ms",
            "attr": "tpot_history",
        },
        "throughput": {
            "title": "Throughput",
            "ylabel": "tokens/s",
            "attr": "throughput_history",
        },
    }

    config = metric_config.get(metric, metric_config["ttft"])

    # Plot each engine's data
    has_data = False
    for engine_name, engine_state in state.engines.items():
        history = getattr(engine_state, config["attr"], [])
        if history:
            has_data = True
            x_values = list(range(len(history)))
            plt.plot(x_values, history, label=f"{engine_state.emoji} {engine_name}")

    if has_data:
        plt.title(config["title"])
        plt.xlabel("Time (data points)")
        plt.ylabel(config["ylabel"])
        plt.grid(True)

        # Build the plot
        return plt.build()
    else:
        return f"⚠️  No {metric.upper()} data available yet"


def show_charts_view(state: "RaceState") -> None:
    """
    Display interactive charts view (terminal-based).

    Args:
        state: Race state with metric data
    """
    if not PLOTEXT_AVAILABLE:
        print("⚠️  plotext not installed. Run: pip install plotext>=5.2.8")
        return

    print("\n" + "=" * 80)
    print("📊 TIME-SERIES CHARTS")
    print("=" * 80 + "\n")

    # TTFT Chart
    print("1️⃣  TTFT (Time to First Token)")
    print(render_metric_chart(state, "ttft", width=80, height=15))
    print()

    # TPOT Chart
    print("2️⃣  TPOT (Time Per Output Token)")
    print(render_metric_chart(state, "tpot", width=80, height=15))
    print()

    # Throughput Chart
    print("3️⃣  Throughput")
    print(render_metric_chart(state, "throughput", width=80, height=15))
    print()

    print("=" * 80)
    print("Press [c] to close charts view")
    print("=" * 80 + "\n")

