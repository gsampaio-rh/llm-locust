"""
Simple load test example for LLM endpoints.

Usage:
    python examples/simple_test.py --host http://localhost:8000 --model llama-3.1-8b --users 10
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from multiprocessing import Process, Queue

from transformers import AutoTokenizer

from llm_locust.clients.openai import OpenAIChatStreamingClient
from llm_locust.core.spawner import start_user_loop
from llm_locust.metrics.collector import MetricsCollector
from llm_locust.metrics.per_request_logger import PerRequestLogger
from llm_locust.utils.prompts import (
    load_databricks_dolly,
    load_sharegpt,
    load_cnn_dailymail,
    load_billsum,
    load_infinity_instruct,
    SYSTEM_PROMPT,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_spawner(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    max_users: int,
    spawn_rate: float,
) -> None:
    """Run user spawner in a separate process."""
    user_addition_time = 1 / spawn_rate if spawn_rate > 0 else 0

    asyncio.run(
        start_user_loop(
            max_users=max_users,
            user_addition_count=1,
            user_addition_time=user_addition_time,
            model_client=client,
            metrics_queue=metrics_queue,
            user_control_queue=control_queue,
        )
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Load Testing Tool")
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="LLM endpoint URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=1.0,
        help="Users to spawn per second (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--prompt-min-tokens",
        type=int,
        default=100,
        help="Minimum prompt length in tokens (default: 100)",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=500,
        help="Maximum prompt length in tokens (default: 500)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dolly", "sharegpt", "cnn_dailymail", "billsum", "infinity_instruct"],
        default="dolly",
        help="Dataset to use for prompts (default: dolly)",
    )
    parser.add_argument(
        "--log-per-request",
        action="store_true",
        help="Enable per-request metrics logging",
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Print per-request metrics to console (WARNING: noisy with high concurrency)",
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=10,
        help="Show summary every N requests instead of each request (default: 10)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/per_request_metrics.csv",
        help="Output file for per-request metrics (default: results/per_request_metrics.csv)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format for per-request metrics (default: csv)",
    )

    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 LLM LOAD TEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Target Configuration:")
    logger.info(f"  🎯 Endpoint:    {args.host}")
    logger.info(f"  🤖 Model:       {args.model}")
    logger.info(f"  🔧 Tokenizer:   {args.tokenizer}")
    logger.info("")
    logger.info("Load Configuration:")
    logger.info(f"  👥 Users:       {args.users} concurrent users")
    logger.info(f"  ⚡ Spawn Rate:  {args.spawn_rate} users/second")
    logger.info(f"  ⏱️  Duration:    {args.duration} seconds")
    logger.info(f"  📊 Total Reqs:  ~{int(args.users * args.duration * 0.5)} (estimated)")
    logger.info("")
    logger.info("Request Configuration:")
    logger.info(f"  🔤 Max Tokens:  {args.max_tokens}")
    logger.info(f"  📝 Input Range: {args.prompt_min_tokens}-{args.prompt_max_tokens} tokens")
    logger.info(f"  📊 Dataset:     {args.dataset.upper()}")
    logger.info("")
    if args.log_per_request:
        logger.info("Per-Request Logging:")
        logger.info(f"  📁 Output File: {args.output_file}")
        logger.info(f"  📄 Format:      {args.output_format.upper()}")
        logger.info(f"  🖥️  Console:     {'Yes' if args.log_to_console else 'No'}")
        if args.log_to_console:
            logger.info(f"  📊 Display:     Every {args.summary_interval}th request")
        logger.info("")
    logger.info("Metrics Display:")
    logger.info("  [AGGREGATE] = Sliding window metrics (all users, last 30s)")
    logger.info("  [REQUEST]   = Individual request metrics")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts
    logger.info(f"📚 Loading {args.dataset.upper()} dataset...")
    if args.dataset == "dolly":
        prompts = load_databricks_dolly(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "sharegpt":
        prompts = load_sharegpt(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "cnn_dailymail":
        prompts = load_cnn_dailymail(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "billsum":
        prompts = load_billsum(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "infinity_instruct":
        prompts = load_infinity_instruct(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    logger.info(f"✅ Loaded {len(prompts)} prompts")

    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT,
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )

    # Setup queues
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()

    # Setup per-request logging if enabled
    per_request_logger = None
    if args.log_per_request:
        logger.info(f"📝 Enabling per-request logging to {args.output_file}")
        if args.log_to_console:
            if args.summary_interval > 0:
                logger.info(f"   Console: Showing every {args.summary_interval}th request")
            else:
                logger.info("   Console: Showing all requests (may be noisy!)")
        per_request_logger = PerRequestLogger(
            output_file=args.output_file,
            format=args.output_format,
            print_to_console=args.log_to_console,
            summary_interval=args.summary_interval,
        )

    # Start metrics collector
    logger.info("📊 Starting metrics collector...")
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=30,
        quantiles=[50, 90, 99],
        per_request_logger=per_request_logger,
    )
    collector.start_logging()

    # Start user spawner in separate process
    logger.info("🚀 Starting user spawner...")
    spawner_process = Process(
        target=run_spawner,
        args=(client, metrics_queue, control_queue, args.users, args.spawn_rate),
    )
    spawner_process.start()
    
    # Give it a moment to start
    time.sleep(1)

    # Setup graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("")
        logger.info("🛑 Shutting down gracefully...")
        from llm_locust.core.models import TriggerShutdown

        control_queue.put(TriggerShutdown())
        logger.info("⏳ Waiting for active requests to complete...")
        time.sleep(5)
        spawner_process.terminate()
        spawner_process.join()
        collector.stop_logging()
        
        # Close per-request logger if enabled
        if per_request_logger:
            per_request_logger.close()
        
        logger.info("✅ Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run for specified duration with progress updates
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 TEST STARTING")
    logger.info("=" * 80)
    logger.info(f"Running for {args.duration} seconds...")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 LIVE METRICS")
    logger.info("=" * 80)

    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = int(time.time() - start_time)
            remaining = args.duration - elapsed
            if elapsed % 30 == 0 and elapsed > 0:  # Progress update every 30 seconds
                logger.info(
                    f"⏱️  Progress: {elapsed}/{args.duration}s elapsed "
                    f"({remaining}s remaining)"
                )
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")

    logger.info("")
    logger.info("=" * 80)
    logger.info("⏹️  Test Duration Complete")
    logger.info("=" * 80)
    
    # Print per-request summary if enabled
    if per_request_logger:
        per_request_logger.print_summary()
    
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

