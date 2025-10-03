"""
Benchmark Test 2a: Constant Rate (Sustained Load) (512 input / 256 output tokens)

Objective:
    Validate system reliability and performance under continuous, predictable workloads.

Workload Profile:
    - Input tokens: ~512 per request
    - Output tokens: ~256 per request
    - Interaction type: Steady production-like traffic flow

Test Parameters:
    - Duration: 15-20 minutes (default: 1200s / 20 minutes)
    - Concurrency: ~40 concurrent streams
    - Rate: Fixed at ~2 requests/second across all users
    - Total requests: ~2400 over 20 minutes at 2 req/s

Benchmark Focus:
    - Sustained Performance: Identify whether latency degrades over time
    - Stability: Measure throughput consistency and error rates
    - SLA Readiness: Ensures performance guarantees can be met under steady load

Business Context:
    Enterprise deployments with predictable usage patterns, such as internal 
    productivity copilots or workflow automation tools.

Usage:
    python examples/benchmark_constant_rate.py \\
        --host https://your-llm-endpoint.com \\
        --model your-model-name \\
        --engine vllm \\
        --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
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
from llm_locust.utils.prompts import load_sharegpt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Test 2a Constants
BENCHMARK_NAME = "Constant Rate / Sustained Load (Test 2a)"
BENCHMARK_ID = "2a-constant-rate"
DATASET_NAME = "sharegpt"  # Mixed conversational prompts
TARGET_INPUT_TOKENS = 512
TARGET_OUTPUT_TOKENS = 256
DEFAULT_DURATION = 1200  # 20 minutes
DEFAULT_USERS = 40  # Concurrent streams
DEFAULT_REQUEST_RATE = 2.0  # Requests per second (total across all users)
INPUT_TOKEN_MIN = 400
INPUT_TOKEN_MAX = 600

# Per-request logging defaults
LOG_TO_CONSOLE = True
SUMMARY_INTERVAL = 10  # Show every 10th request to reduce console spam

# Calculate spawn rate to achieve target request rate
# With 40 users each making ~1 request per 20 seconds = 2 req/s total
# spawn_rate: get all 40 users active quickly (within 10 seconds)
DEFAULT_SPAWN_RATE = 4.0  # Users per second


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
    """Main entry point for benchmark test."""
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
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer to use (default: Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_USERS,
        help=f"Number of concurrent users (default: {DEFAULT_USERS})",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=DEFAULT_REQUEST_RATE,
        help=f"Target request rate (req/s) across all users (default: {DEFAULT_REQUEST_RATE})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION}s / 20 minutes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for benchmark results (default: results)",
    )

    args = parser.parse_args()

    # Calculate expected total requests
    expected_requests = int(args.request_rate * args.duration)

    # Generate output filename: {engine}-{datetime}-{benchmark-name}.csv
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}.csv"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file for this benchmark run
    output_file = output_dir / output_filename

    # Print benchmark configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🎯 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Benchmark Specification:")
    logger.info(f"  • Target Input Tokens:  ~{TARGET_INPUT_TOKENS} tokens")
    logger.info(f"  • Target Output Tokens: ~{TARGET_OUTPUT_TOKENS} tokens")
    logger.info(f"  • Workload Type:        Constant Rate (sustained production load)")
    logger.info(f"  • Success Criteria:     No latency degradation, consistent throughput")
    logger.info("")
    logger.info("🎯 Target Configuration:")
    logger.info(f"  • Engine:       {args.engine}")
    logger.info(f"  • Endpoint:     {args.host}")
    logger.info(f"  • Model:        {args.model}")
    logger.info(f"  • Tokenizer:    {args.tokenizer}")
    logger.info("")
    logger.info("⚙️  Test Configuration:")
    logger.info(f"  • Users:        {args.users} concurrent streams")
    logger.info(f"  • Request Rate: {args.request_rate} req/s (constant across all users)")
    logger.info(f"  • Duration:     {args.duration}s ({args.duration // 60} minutes)")
    logger.info(f"  • Total Reqs:   ~{expected_requests} (at constant rate)")
    logger.info(f"  • Per User:     ~{expected_requests / args.users:.1f} requests per user")
    logger.info(f"  • Dataset:      {DATASET_NAME.upper()}")
    logger.info("")
    logger.info("📊 Metrics Configuration:")
    logger.info(f"  • Output File:  {output_filename}")
    logger.info(f"  • Format:       CSV with full metrics")
    logger.info(f"  • Console Log:  {'Enabled' if LOG_TO_CONSOLE else 'Disabled'}")
    logger.info(f"  • Log Interval: Every {SUMMARY_INTERVAL}th request")
    logger.info("")
    logger.info("🎯 Focus Areas:")
    logger.info("  • Sustained Performance Over Time")
    logger.info("  • Latency Stability (no degradation)")
    logger.info("  • Throughput Consistency")
    logger.info("  • Error Rate Under Steady Load")
    logger.info("  • SLA Compliance")
    logger.info("")
    logger.info("⚙️  Constant Rate Strategy:")
    logger.info(f"  • Users spawn quickly (within {args.users / DEFAULT_SPAWN_RATE:.1f}s)")
    logger.info(f"  • Each user paces requests to maintain {args.request_rate} req/s total")
    logger.info(f"  • Wait time between requests: ~{args.users / args.request_rate:.1f}s per user")
    logger.info("  • This simulates steady enterprise workload patterns")
    logger.info("")
    logger.info("📈 What to Watch For:")
    logger.info("  • Gradual latency increase (indicates resource saturation)")
    logger.info("  • Throughput drop-off (system struggling to keep up)")
    logger.info("  • Error rate spikes (capacity limits reached)")
    logger.info("  • Memory leaks (degradation over time)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts
    logger.info(f"📚 Loading {DATASET_NAME.upper()} dataset (mixed conversational)...")
    logger.info(f"   Filtering for ~{TARGET_INPUT_TOKENS} input tokens")
    logger.info(f"   ({INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX} token range)")
    
    prompts = load_sharegpt(
        tokenizer,
        min_input_length=INPUT_TOKEN_MIN,
        max_input_length=INPUT_TOKEN_MAX,
    )
    
    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts found ({len(prompts)}). Need at least 10.")
        logger.error("   Try adjusting token range or use a different dataset.")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(prompts)} prompts")
    avg_input_tokens = sum(p["num_input_tokens"] for p in prompts) / len(prompts)
    logger.info(f"   Average input tokens: {avg_input_tokens:.0f}")

    # Create client with request pacing for constant rate
    # Each user will wait between requests to achieve target rate
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt="You are a helpful AI assistant. Provide clear, concise responses.",
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=TARGET_OUTPUT_TOKENS,
        # Note: The User class in core/user.py will handle pacing
        # We pass wait_time via client for users to respect
    )

    # Setup queues
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()

    # Setup per-request logging
    logger.info(f"📝 Enabling detailed per-request logging...")
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=LOG_TO_CONSOLE,
        summary_interval=SUMMARY_INTERVAL,
        include_text=True,
        max_text_length=200,
    )

    # Start metrics collector
    logger.info("📊 Starting metrics collector...")
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=60,  # 60-second sliding window for better long-term visibility
        quantiles=[50, 90, 99],  # P50, P90, P99
        per_request_logger=per_request_logger,
    )
    collector.start_logging()

    # Start user spawner in separate process
    logger.info("🚀 Starting user spawner...")
    spawner_process = Process(
        target=run_spawner,
        args=(client, metrics_queue, control_queue, args.users, DEFAULT_SPAWN_RATE),
    )
    spawner_process.start()
    
    # Give it time to spawn all users
    spawn_time = args.users / DEFAULT_SPAWN_RATE
    logger.info(f"⏳ Spawning {args.users} users (will take ~{spawn_time:.1f}s)...")
    time.sleep(min(spawn_time + 2, 15))  # Cap at 15 seconds max

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
        
        # Close per-request logger
        per_request_logger.close()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Results saved to: {output_file}")
        logger.info("")
        logger.info("📈 Analysis Guidelines:")
        logger.info("   1. Plot latency over time (look for upward trends)")
        logger.info("   2. Check if throughput remained constant at ~{:.1f} req/s".format(args.request_rate))
        logger.info("   3. Analyze error rate (should be near 0% for production)")
        logger.info("   4. Compare P50/P90/P99 across time buckets")
        logger.info("   5. Verify SLA compliance throughout the test")
        logger.info("")
        logger.info("🎯 SLA Evaluation:")
        logger.info("   • Latency: Check if P99 stayed within acceptable bounds")
        logger.info("   • Throughput: Sustained {:.1f} req/s = PASS".format(args.request_rate))
        logger.info("   • Error Rate: <0.1% = PASS, <1% = WARN, >1% = FAIL")
        logger.info("   • Degradation: No upward latency trend = PASS")
        logger.info("")
        per_request_logger.print_summary()
        logger.info("=" * 80)
        
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run benchmark
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 BENCHMARK STARTING")
    logger.info("=" * 80)
    logger.info(f"Running for {args.duration} seconds ({args.duration // 60} minutes)...")
    logger.info(f"Maintaining constant rate of {args.request_rate} req/s")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")
    logger.info("⚠️  This is a long-running test designed to reveal degradation over time")
    logger.info("=" * 80)
    logger.info("📊 LIVE METRICS (60-second sliding window)")
    logger.info("=" * 80)

    start_time = time.time()
    last_report_time = start_time
    
    try:
        while time.time() - start_time < args.duration:
            elapsed = int(time.time() - start_time)
            remaining = args.duration - elapsed
            
            # Progress update every 120 seconds (2 minutes)
            if elapsed % 120 == 0 and elapsed > 0:
                progress_pct = (elapsed / args.duration) * 100
                logger.info("")
                logger.info("=" * 80)
                logger.info(
                    f"⏱️  Progress: {elapsed}s / {args.duration}s "
                    f"({progress_pct:.1f}% complete, {remaining // 60}m {remaining % 60}s remaining)"
                )
                logger.info(f"📊 Check metrics window above for performance trends")
                logger.info("=" * 80)
            
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Benchmark interrupted by user")

    logger.info("")
    logger.info("=" * 80)
    logger.info("⏹️  Benchmark Duration Complete")
    logger.info("=" * 80)
    
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

