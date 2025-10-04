"""
Test script for race configuration loading and validation.

This script tests the race configuration system without running actual benchmarks.
"""

import sys
from pathlib import Path

from llm_locust.race import load_race_config


def main() -> None:
    """Test race configuration loading."""
    print("🧪 Testing Race Configuration System\n")

    # Test files
    test_configs = [
        "examples/races/demo-race.yaml",
        "examples/races/production-candidates.yaml",
        "examples/races/rag-benchmark.yaml",
        "examples/races/stress-test.yaml",
    ]

    success_count = 0
    fail_count = 0

    for config_file in test_configs:
        config_path = Path(config_file)
        print(f"Testing: {config_path.name}")

        try:
            config = load_race_config(config_path)
            print(f"  ✅ Valid configuration")
            print(f"     Name: {config.name}")
            print(f"     Engines: {len(config.engines)}")
            print(f"     Duration: {config.duration}s")
            print(f"     Users: {config.users}")

            # Print engines
            for engine in config.engines:
                print(f"       {engine.emoji} {engine.name} - {engine.model}")

            success_count += 1

        except FileNotFoundError as e:
            print(f"  ❌ File not found: {e}")
            fail_count += 1

        except ValueError as e:
            print(f"  ❌ Validation error: {e}")
            fail_count += 1

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            fail_count += 1

        print()

    # Summary
    print("=" * 60)
    print(f"Results: {success_count} passed, {fail_count} failed")

    if fail_count > 0:
        print("❌ Some tests failed")
        sys.exit(1)
    else:
        print("✅ All tests passed!")


if __name__ == "__main__":
    main()

