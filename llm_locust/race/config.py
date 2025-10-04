"""
Race configuration models and validation.

Defines the schema for race.yaml configuration files and provides
validation logic for race setup.
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """Configuration for a single engine in the race."""

    name: str
    url: str
    model: str
    emoji: str = "🚀"
    color: str = "cyan"
    tokenizer: str | None = None

    def __post_init__(self) -> None:
        """Validate engine configuration."""
        if not self.name:
            raise ValueError("Engine name cannot be empty")
        if not self.url:
            raise ValueError(f"Engine '{self.name}' must have a URL")
        if not self.model:
            raise ValueError(f"Engine '{self.name}' must specify a model")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Engine '{self.name}' URL must start with http:// or https://")


@dataclasses.dataclass(frozen=True)
class RaceConfig:
    """Complete race configuration."""

    name: str
    engines: tuple[EngineConfig, ...]
    duration: int = 600  # seconds
    users: int = 50
    spawn_rate: float = 5.0  # users/second
    dataset: str = "sharegpt"
    target_input_tokens: int = 256
    target_output_tokens: int = 128
    output_dir: str = "results"

    def __post_init__(self) -> None:
        """Validate race configuration."""
        if not self.name:
            raise ValueError("Race name cannot be empty")

        if not self.engines:
            raise ValueError("Race must have at least one engine")

        if len(self.engines) > 10:
            raise ValueError("Race cannot have more than 10 engines")

        if self.duration <= 0:
            raise ValueError("Duration must be positive")

        if self.users <= 0:
            raise ValueError("User count must be positive")

        if self.spawn_rate <= 0:
            raise ValueError("Spawn rate must be positive")

        # Check for duplicate engine names
        engine_names = [e.name for e in self.engines]
        if len(engine_names) != len(set(engine_names)):
            raise ValueError("Engine names must be unique")


def load_race_config(config_path: Path | str) -> RaceConfig:
    """
    Load and validate race configuration from YAML file.

    Args:
        config_path: Path to race.yaml configuration file

    Returns:
        Validated RaceConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Race configuration file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Race configuration path is not a file: {path}")

    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    if not data:
        raise ValueError("Configuration file is empty")

    if "race" not in data:
        raise ValueError("Configuration must have a 'race' top-level key")

    race_data = data["race"]

    # Parse engines
    if "engines" not in race_data:
        raise ValueError("Race configuration must have 'engines' list")

    engines_data = race_data["engines"]
    if not isinstance(engines_data, list):
        raise ValueError("'engines' must be a list")

    engines = []
    for i, engine_data in enumerate(engines_data):
        try:
            # Handle optional tokenizer (defaults to None)
            tokenizer = engine_data.get("tokenizer")

            engine = EngineConfig(
                name=engine_data["name"],
                url=engine_data["url"],
                model=engine_data["model"],
                emoji=engine_data.get("emoji", "🚀"),
                color=engine_data.get("color", "cyan"),
                tokenizer=tokenizer,
            )
            engines.append(engine)
        except KeyError as e:
            raise ValueError(f"Engine {i} missing required field: {e}") from e
        except Exception as e:
            raise ValueError(f"Engine {i} configuration error: {e}") from e

    # Build race config with defaults
    return RaceConfig(
        name=race_data["name"],
        engines=tuple(engines),
        duration=race_data.get("duration", 600),
        users=race_data.get("users", 50),
        spawn_rate=race_data.get("spawn_rate", 5.0),
        dataset=race_data.get("dataset", "sharegpt"),
        target_input_tokens=race_data.get("target_input_tokens", 256),
        target_output_tokens=race_data.get("target_output_tokens", 128),
        output_dir=race_data.get("output_dir", "results"),
    )


def validate_race_config(config: dict[str, Any]) -> list[str]:
    """
    Validate race configuration and return list of errors.

    Args:
        config: Raw configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if "race" not in config:
        errors.append("Missing 'race' top-level key")
        return errors

    race = config["race"]

    # Check required fields
    if "name" not in race:
        errors.append("Missing 'name' field")

    if "engines" not in race:
        errors.append("Missing 'engines' field")
    elif not isinstance(race["engines"], list):
        errors.append("'engines' must be a list")
    elif len(race["engines"]) == 0:
        errors.append("'engines' list cannot be empty")
    elif len(race["engines"]) > 10:
        errors.append("'engines' list cannot have more than 10 entries")
    else:
        # Validate each engine
        for i, engine in enumerate(race["engines"]):
            if not isinstance(engine, dict):
                errors.append(f"Engine {i} must be a dictionary")
                continue

            for field in ["name", "url", "model"]:
                if field not in engine:
                    errors.append(f"Engine {i} missing required field: {field}")

            if "url" in engine and not engine["url"].startswith(("http://", "https://")):
                errors.append(f"Engine {i} URL must start with http:// or https://")

    # Validate optional numeric fields
    if "duration" in race and (not isinstance(race["duration"], int) or race["duration"] <= 0):
        errors.append("'duration' must be a positive integer")

    if "users" in race and (not isinstance(race["users"], int) or race["users"] <= 0):
        errors.append("'users' must be a positive integer")

    if "spawn_rate" in race and (
        not isinstance(race["spawn_rate"], (int, float)) or race["spawn_rate"] <= 0
    ):
        errors.append("'spawn_rate' must be a positive number")

    return errors
