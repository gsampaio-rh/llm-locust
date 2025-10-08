"""
Cost Estimation Benchmark Test

Objective:
    Run a controlled test to measure token throughput and calculate cost per 1M tokens.
    This test is optimized for getting accurate cost estimates rather than stress testing.

What This Test Does:
    1. Runs a sustained workload for accurate throughput measurement
    2. Generates a CSV file with per-request token metrics
    3. Calculates actual tokens/second throughput
    4. Helps you answer: "How much does it cost to process 1M tokens?"

How to Calculate Cost per 1M Tokens:
    1. Run this benchmark
    2. Upload the CSV to the dashboard (streamlit_app)
    3. Configure your instance type and hourly cost
    4. Dashboard automatically calculates $/1M tokens based on:
       - Your measured throughput (tokens/sec)
       - Your infrastructure cost ($/hour)
       - Formula: Cost per 1M tokens = ($/hour ÷ tokens/hour) × 1,000,000

Test Parameters:
    - Duration: 5-10 minutes (default: 5 min for quick estimate)
    - Concurrency: Moderate load (default: 20 users)
    - Token Profile: Configurable (default: ~512 input / 256 output)
    - Dataset: Flexible (default: dolly)

Usage Examples:

    # Basic cost estimation test (5 minutes, 20 users)
    python examples/benchmark_cost_estimation.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --tokenizer meta-llama/Llama-3.2-3B-Instruct

    # Longer test for more accurate cost estimates (10 minutes)
    python examples/benchmark_cost_estimation.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --duration 600

    # Test with different token sizes (simulate your actual workload)
    python examples/benchmark_cost_estimation.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --prompt-min-tokens 100 \\
        --prompt-max-tokens 200 \\
        --max-tokens 512

    # Test with RAG workload (long context)
    python examples/benchmark_cost_estimation.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --dataset billsum \\
        --prompt-min-tokens 1500 \\
        --prompt-max-tokens 2000

After Running:
    1. Find your results CSV in results/ directory
    2. Open the Streamlit dashboard: streamlit run streamlit_app/app.py
    3. Upload your CSV file
    4. Go to "Cost Analysis" page
    5. Select your instance type (or enter custom pricing)
    6. See your actual $/1M tokens cost!
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

from transformers import AutoTokenizer

from llm_locust.clients.openai import OpenAIChatStreamingClient
from llm_locust.core.spawner import start_user_loop
from llm_locust.metrics.collector import MetricsCollector
from llm_locust.metrics.per_request_logger import PerRequestLogger
from llm_locust.utils.prompts import (
    load_databricks_dolly,
    load_sharegpt,
    load_billsum,
    SYSTEM_PROMPT,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Constants
BENCHMARK_NAME = "Cost Estimation Test"
BENCHMARK_ID = "cost-estimation"
DEFAULT_DURATION = 300  # 5 minutes - enough for stable throughput measurement
DEFAULT_USERS = 150  # HEAVY load to maximize throughput (lowest cost per token!)
DEFAULT_SPAWN_RATE = 10.0  # Fast ramp-up to reach max capacity quickly
DEFAULT_INPUT_MIN = 400  # ~512 average
DEFAULT_INPUT_MAX = 600
DEFAULT_OUTPUT_TOKENS = None  # None = unlimited (natural stopping)
DEFAULT_DATASET = "dolly"
MAX_OUTPUT_TOKENS_SAFETY = 2048  # Safety limit (conservative for most models with 4096 context)

# Logging configuration
LOG_TO_CONSOLE = False  # Don't spam console during cost tests
SUMMARY_INTERVAL = 50  # Show summary every 50 requests


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
    """Main entry point for cost estimation benchmark."""
    parser = argparse.ArgumentParser(
        description=f"LLM Benchmark - {BENCHMARK_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
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
        "--engine",
        type=str,
        required=True,
        help="Engine/platform name (e.g., vllm, tgi, ollama) - used in output filename",
    )

    # Optional arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer to use (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_USERS,
        help=f"Number of concurrent users (default: {DEFAULT_USERS})",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=DEFAULT_SPAWN_RATE,
        help=f"Users to spawn per second (default: {DEFAULT_SPAWN_RATE})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION}s / {DEFAULT_DURATION // 60} minutes)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Maximum output tokens to generate (default: unlimited/natural stopping). Set a number to limit output.",
    )
    parser.add_argument(
        "--no-unlimited-output",
        action="store_true",
        help="Use a fixed max_tokens limit instead of natural stopping (requires --max-tokens).",
    )
    parser.add_argument(
        "--prompt-min-tokens",
        type=int,
        default=DEFAULT_INPUT_MIN,
        help=f"Minimum input tokens (default: {DEFAULT_INPUT_MIN})",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=DEFAULT_INPUT_MAX,
        help=f"Maximum input tokens (default: {DEFAULT_INPUT_MAX})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dolly", "sharegpt", "billsum"],
        default=DEFAULT_DATASET,
        help=f"Dataset to use (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for benchmark results (default: results)",
    )

    args = parser.parse_args()

    # Handle unlimited output (default behavior)
    if args.max_tokens is None or args.max_tokens == 0:
        # Unlimited mode (default)
        actual_max_tokens = MAX_OUTPUT_TOKENS_SAFETY  # Safety limit
        unlimited_mode = True
    else:
        # Limited mode (user specified max_tokens)
        actual_max_tokens = args.max_tokens
        unlimited_mode = False

    # Calculate average token sizes for display
    avg_input_tokens = (args.prompt_min_tokens + args.prompt_max_tokens) // 2
    avg_output_tokens = "unlimited (natural stopping)" if unlimited_mode else str(args.max_tokens)

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}.csv"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    # Print benchmark configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"💰 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🎯 Purpose:")
    logger.info("   Measure token throughput to calculate cost per 1M tokens")
    logger.info("")
    logger.info("📋 Test Profile:")
    logger.info(f"   • Target Input Tokens:  ~{avg_input_tokens} tokens ({args.prompt_min_tokens}-{args.prompt_max_tokens})")
    if unlimited_mode:
        logger.info(f"   • Target Output Tokens: UNLIMITED (natural stopping, safety limit: {MAX_OUTPUT_TOKENS_SAFETY})")
    else:
        logger.info(f"   • Target Output Tokens: ~{avg_output_tokens} tokens")
    logger.info(f"   • Dataset:              {args.dataset.upper()}")
    logger.info(f"   • Workload Type:        Sustained high load (max throughput)")
    logger.info("")
    logger.info("🎯 Target Configuration:")
    logger.info(f"   • Engine:       {args.engine}")
    logger.info(f"   • Endpoint:     {args.host}")
    logger.info(f"   • Model:        {args.model}")
    logger.info(f"   • Tokenizer:    {args.tokenizer}")
    logger.info("")
    logger.info("⚙️  Test Configuration:")
    logger.info(f"   • Users:        {args.users} concurrent users {'🔥 HEAVY LOAD!' if args.users >= 100 else ''}")
    logger.info(f"   • Spawn Rate:   {args.spawn_rate} users/second")
    logger.info(f"   • Duration:     {args.duration}s ({args.duration // 60} minutes)")
    logger.info(f"   • Total Reqs:   ~{int(args.users * (args.duration / 60) * 1.5)} (estimated)")
    logger.info("")
    logger.info("💡 Strategy:")
    logger.info(f"   Pushing to MAX throughput = spreading fixed costs over MORE tokens")
    logger.info(f"   = LOWER cost per 1M tokens! 🚀")
    logger.info("")
    logger.info("📊 Output:")
    logger.info(f"   • File:         {output_filename}")
    logger.info(f"   • Location:     {output_dir}/")
    logger.info(f"   • Format:       CSV with full per-request metrics")
    logger.info("")
    logger.info("📈 What You'll Get:")
    logger.info("   ✓ Total tokens processed (input + output)")
    logger.info("   ✓ Actual test duration")
    logger.info("   ✓ Measured throughput (tokens/second)")
    logger.info("   ✓ Per-request latency metrics (TTFT, TPOT)")
    logger.info("   ✓ Success rate and error tracking")
    logger.info("")
    if args.users >= 100:
        logger.info("⚠️  HEAVY LOAD WARNING:")
        logger.info("   Watch for: High failure rate, OOM errors, extreme latency")
        logger.info("   If failures > 5%, reduce --users and retest")
        logger.info("   Goal: Find max sustainable throughput (not max chaos!)")
    logger.info("")
    logger.info("💡 Next Steps After Test:")
    logger.info("   1. Open Streamlit dashboard:")
    logger.info("      streamlit run streamlit_app/app.py")
    logger.info("")
    logger.info("   2. Upload this CSV file")
    logger.info("")
    logger.info("   3. Go to 'Cost Analysis' page")
    logger.info("")
    logger.info("   4. Select your instance type or enter custom pricing")
    logger.info("")
    logger.info("   5. Dashboard shows your actual $/1M tokens cost!")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts based on dataset
    logger.info(f"📚 Loading {args.dataset.upper()} dataset...")
    logger.info(f"   Filtering for {args.prompt_min_tokens}-{args.prompt_max_tokens} input tokens")

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
    elif args.dataset == "billsum":
        prompts = load_billsum(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts found ({len(prompts)}). Need at least 10.")
        logger.error("   Try adjusting token range or use a different dataset.")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(prompts)} prompts")

    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT,
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=actual_max_tokens,
    )
    
    if unlimited_mode:
        logger.info(f"⚡ Running in UNLIMITED OUTPUT mode (default)")
        logger.info(f"   Model will decide when to stop (safety limit: {MAX_OUTPUT_TOKENS_SAFETY} tokens)")
        logger.info(f"   This captures NATURAL output token distribution!")
        logger.info(f"   Tip: Use --max-tokens 256 to limit output if needed")
        logger.info("")
    else:
        logger.info(f"⚠️  Running with FIXED OUTPUT LIMIT: {args.max_tokens} tokens")
        logger.info(f"   Tip: Remove --max-tokens for natural/unlimited output (more realistic)")
        logger.info("")

    # Setup queues
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()

    # Setup per-request logging
    logger.info(f"📝 Enabling per-request logging...")
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=LOG_TO_CONSOLE,
        summary_interval=SUMMARY_INTERVAL,
        include_text=True,  # Include text for analysis
        max_text_length=200,
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

    # Start user spawner
    logger.info("🚀 Starting user spawner...")
    spawner_process = Process(
        target=run_spawner,
        args=(client, metrics_queue, control_queue, args.users, args.spawn_rate),
    )
    spawner_process.start()
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
        per_request_logger.close()

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ COST ESTIMATION TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"📊 Results saved to: {output_file}")
        logger.info("")
        logger.info("📈 Quick Summary:")
        per_request_logger.print_summary()
        logger.info("")
        logger.info("💰 Calculate Your Cost Per 1M Tokens:")
        logger.info("")
        logger.info("   Step 1: Open the dashboard")
        logger.info("      $ streamlit run streamlit_app/app.py")
        logger.info("")
        logger.info("   Step 2: Upload your results")
        logger.info(f"      File: {output_file}")
        logger.info("")
        logger.info("   Step 3: Go to 'Cost Analysis' page")
        logger.info("")
        logger.info("   Step 4: Configure your infrastructure cost")
        logger.info("      - Select instance type (AWS, GCP, Azure, On-prem)")
        logger.info("      - Or enter custom hourly cost")
        logger.info("")
        logger.info("   Step 5: See your results!")
        logger.info("      ✓ Cost per 1M input tokens")
        logger.info("      ✓ Cost per 1M output tokens")
        logger.info("      ✓ Monthly cost projections")
        logger.info("      ✓ Comparison with API providers")
        logger.info("      ✓ Break-even analysis")
        logger.info("")
        logger.info("💡 Pro Tip:")
        logger.info("   Run this test multiple times at different loads to see")
        logger.info("   how throughput (and cost efficiency) changes with scale!")
        logger.info("")
        logger.info("=" * 80)

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run benchmark
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 TEST STARTING")
    logger.info("=" * 80)
    logger.info(f"Running for {args.duration} seconds ({args.duration // 60} minutes)...")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")
    logger.info("⏳ Ramping up users...")
    logger.info(f"   ({args.users} users at {args.spawn_rate} users/sec = {args.users / args.spawn_rate:.0f}s ramp-up)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 LIVE METRICS (30-second sliding window)")
    logger.info("=" * 80)

    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = int(time.time() - start_time)
            remaining = args.duration - elapsed

            # Progress update every 60 seconds
            if elapsed % 60 == 0 and elapsed > 0:
                progress_pct = (elapsed / args.duration) * 100
                logger.info("")
                logger.info(
                    f"⏱️  Progress: {elapsed}s / {args.duration}s "
                    f"({progress_pct:.1f}% complete, {remaining}s remaining)"
                )

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")

    logger.info("")
    logger.info("=" * 80)
    logger.info("⏹️  Test Duration Complete")
    logger.info("=" * 80)

    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
