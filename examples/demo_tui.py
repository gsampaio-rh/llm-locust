"""
Demo script for testing the Race TUI.

This script demonstrates the TUI layout and live updates without
actually running benchmarks.
"""

from llm_locust.race import load_race_config
from llm_locust.race.tui import RaceTUI, show_race_header


def main() -> None:
    """Demo the race TUI."""
    # Load a test config
    config = load_race_config("configs/races/test-quick.yaml")

    # Show static header
    show_race_header(config)

    # Create and run TUI
    tui = RaceTUI(config)

    print("🎨 Starting TUI demo (will run for 10 seconds)...")
    print()

    # Run live for 10 seconds
    tui.run_live(duration=10)

    print()
    print("✅ TUI demo complete!")


if __name__ == "__main__":
    main()

