"""
Race CLI command - interactive multi-endpoint benchmarking.

Usage:
    llm-locust race --config examples/races/demo-race.yaml
"""

import argparse
import logging
import sys

from llm_locust.race import RaceOrchestrator, load_race_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for race CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Locust Race - Interactive Multi-Engine Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo race (2 minutes, 3 engines)
  llm-locust race --config configs/races/demo-race.yaml

  # Run production comparison
  llm-locust race --config configs/races/production-candidates.yaml

  # Run with custom theme
  llm-locust race --config demo-race.yaml --theme hacker

For more information, see: docs/RACE_CLI.md
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to race configuration YAML file",
    )

    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        help="Visual theme (default, dark, light, hacker, retro)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="expert",
        choices=["novice", "expert", "teacher"],
        help="Information density mode (novice, expert, teacher)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration without running race",
    )

    args = parser.parse_args()

    # Load and validate configuration
    logger.info("📋 Loading race configuration...")
    try:
        config = load_race_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"❌ Configuration file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    logger.info(f"✅ Configuration loaded: {config.name}")
    logger.info(f"   Engines: {len(config.engines)}")
    for engine in config.engines:
        logger.info(f"   - {engine.emoji} {engine.name} ({engine.url})")

    # Validate only mode
    if args.validate_only:
        logger.info("✅ Configuration is valid!")
        return

    # Run the race
    logger.info("")
    logger.info("🏁 Starting race...")

    orchestrator = RaceOrchestrator(config)
    try:
        orchestrator.start_race()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Race interrupted by user")
        orchestrator.stop_race()
    except Exception as e:
        logger.error(f"❌ Race failed: {e}", exc_info=True)
        orchestrator.stop_race()
        sys.exit(1)


if __name__ == "__main__":
    main()
