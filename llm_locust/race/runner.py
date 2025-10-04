"""
Benchmark runner for individual engines in a race.

Each engine runs in its own process, executing benchmarks independently
while reporting metrics to a shared queue.
"""

import asyncio
import logging
from multiprocessing import Queue
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

from llm_locust.clients.openai import OpenAIChatStreamingClient
from llm_locust.core.spawner import start_user_loop
from llm_locust.utils.prompts import load_databricks_dolly, load_sharegpt

if TYPE_CHECKING:
    from llm_locust.race.config import EngineConfig, RaceConfig

logger = logging.getLogger(__name__)


def run_engine_benchmark(
    engine: "EngineConfig",
    race_config: "RaceConfig",
    metrics_queue: Queue,
    control_queue: Queue,
) -> None:
    """
    Run benchmark for a single engine (executed in separate process).

    This function is the entry point for each engine's process. It:
    1. Sets up the tokenizer and prompts
    2. Creates the OpenAI client
    3. Runs the benchmark with specified user count
    4. Reports metrics to shared queue

    Args:
        engine: Configuration for this specific engine
        race_config: Overall race configuration
        metrics_queue: Queue for sending metrics to orchestrator
        control_queue: Queue for receiving control signals
    """
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [{engine.name}] - %(levelname)s - %(message)s",
    )
    logger.info(f"🏁 Starting benchmark for {engine.emoji} {engine.name}")

    try:
        # Load tokenizer
        tokenizer_name = engine.tokenizer or engine.model
        logger.info(f"📦 Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not tokenizer.chat_template:
            tokenizer.chat_template = "{{prompt}}"

        # Load prompts based on dataset
        logger.info(f"📚 Loading {race_config.dataset} dataset...")
        if race_config.dataset == "sharegpt":
            prompts = load_sharegpt(
                tokenizer,
                min_input_length=race_config.target_input_tokens - 50,
                max_input_length=race_config.target_input_tokens + 50,
            )
        elif race_config.dataset == "dolly":
            prompts = load_databricks_dolly(
                tokenizer,
                min_input_length=race_config.target_input_tokens - 50,
                max_input_length=race_config.target_input_tokens + 50,
            )
        else:
            # Default to sharegpt
            prompts = load_sharegpt(
                tokenizer,
                min_input_length=race_config.target_input_tokens - 50,
                max_input_length=race_config.target_input_tokens + 50,
            )

        if len(prompts) < 10:
            logger.error(f"❌ Insufficient prompts: {len(prompts)} (need at least 10)")
            return

        logger.info(f"✅ Loaded {len(prompts)} prompts")

        # Create OpenAI client
        client = OpenAIChatStreamingClient(
            base_url=engine.url,
            prompts=prompts,
            system_prompt="You are a helpful AI assistant.",
            openai_model_name=engine.model,
            tokenizer=tokenizer,
            max_tokens=race_config.target_output_tokens,
        )

        # Calculate spawn parameters
        spawn_rate = race_config.spawn_rate
        user_addition_time = 1 / spawn_rate if spawn_rate > 0 else 0

        logger.info(f"🚀 Starting {race_config.users} users (spawn rate: {spawn_rate}/s)")

        # Run the benchmark using asyncio
        asyncio.run(
            start_user_loop(
                max_users=race_config.users,
                user_addition_count=1,
                user_addition_time=user_addition_time,
                model_client=client,
                metrics_queue=metrics_queue,
                user_control_queue=control_queue,
            )
        )

        logger.info(f"✅ Benchmark complete for {engine.name}")

    except Exception as e:
        logger.error(f"❌ Benchmark failed for {engine.name}: {e}", exc_info=True)
        raise

