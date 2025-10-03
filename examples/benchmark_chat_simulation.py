"""
Benchmark Test 1a: Chat Simulation (256 input / 128 output tokens)

Objective:
    Evaluate system performance under short, interactive workloads representative 
    of conversational AI.

Workload Profile:
    - Input tokens: ~256 per request
    - Output tokens: ~128 per request
    - Interaction type: Compact prompts and concise responses

Test Parameters:
    - Duration: 5-10 minutes (default: 600s)
    - Concurrency: ~50 parallel chat sessions
    - Rate: Steady conversational pace (1-2 requests per user per minute)

Benchmark Focus:
    - Latency Sensitivity: TTFT and p99 latency as indicators of responsiveness
    - Throughput: Ability to sustain dozens of interactive conversations
    - User Experience: Ensures responses remain conversational (<1s median, <2s p99)

Business Context:
    Customer-facing assistants, support bots, or copilots where responsiveness 
    is critical for usability and adoption.

Usage:
    python examples/benchmark_chat_simulation.py \\
        --host https://your-llm-endpoint.com \\
        --model your-model-name \\
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

# Benchmark Test 1a Constants
BENCHMARK_NAME = "Chat Simulation (Test 1a)"
BENCHMARK_ID = "1a-chat-simulation"
DATASET_NAME = "sharegpt"  # Fixed dataset for this benchmark
TARGET_INPUT_TOKENS = 256
TARGET_OUTPUT_TOKENS = 128
DEFAULT_DURATION = 600  # 10 minutes
DEFAULT_USERS = 50
DEFAULT_SPAWN_RATE = 5.0  # Users per second
INPUT_TOKEN_MIN = 200  # ~256 target with some variance
INPUT_TOKEN_MAX = 300

# Per-request logging defaults (always enabled for benchmarks)
LOG_TO_CONSOLE = True  # Show individual requests in console
SUMMARY_INTERVAL = 0  # 0 = show all requests, >0 = show every Nth request


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
        "--spawn-rate",
        type=float,
        default=DEFAULT_SPAWN_RATE,
        help=f"Users to spawn per second (default: {DEFAULT_SPAWN_RATE})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION}s / 10 minutes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for benchmark results (default: results)",
    )

    args = parser.parse_args()

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
    logger.info(f"  • Workload Type:        Conversational AI (short interactive)")
    logger.info(f"  • Success Criteria:     TTFT <1s (median), <2s (p99)")
    logger.info("")
    logger.info("🎯 Target Configuration:")
    logger.info(f"  • Engine:       {args.engine}")
    logger.info(f"  • Endpoint:     {args.host}")
    logger.info(f"  • Model:        {args.model}")
    logger.info(f"  • Tokenizer:    {args.tokenizer}")
    logger.info("")
    logger.info("⚙️  Test Configuration:")
    logger.info(f"  • Users:        {args.users} concurrent sessions")
    logger.info(f"  • Spawn Rate:   {args.spawn_rate} users/second")
    logger.info(f"  • Duration:     {args.duration}s ({args.duration // 60} minutes)")
    logger.info(f"  • Total Reqs:   ~{int(args.users * (args.duration / 60) * 1.5)} (estimated)")
    logger.info(f"  • Dataset:      {DATASET_NAME.upper()}")
    logger.info("")
    logger.info("📊 Metrics Configuration:")
    logger.info(f"  • Output File:  {output_filename}")
    logger.info(f"  • Format:       CSV with full metrics")
    logger.info(f"  • Console Log:  {'Enabled' if LOG_TO_CONSOLE else 'Disabled'}")
    logger.info(f"  • Log Interval: {'All requests' if SUMMARY_INTERVAL == 0 else f'Every {SUMMARY_INTERVAL}th request'}")
    logger.info("")
    logger.info("🎯 Focus Areas:")
    logger.info("  • Time to First Token (TTFT)")
    logger.info("  • P99 Latency Distribution")
    logger.info("  • Sustained Throughput")
    logger.info("  • Conversational Responsiveness")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts - dataset is fixed at benchmark level
    logger.info(f"📚 Loading {DATASET_NAME.upper()} dataset (conversational prompts)...")
    logger.info(f"   Filtering for ~{TARGET_INPUT_TOKENS} input tokens")
    logger.info(f"   ({INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX} token range)")
    
    # Dataset selection is fixed for this benchmark (ShareGPT = conversational)
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

    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt="You are a helpful AI assistant. Provide concise, clear responses.",
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=TARGET_OUTPUT_TOKENS,
    )

    # Setup queues
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()

    # Setup per-request logging (always enabled for benchmarks)
    logger.info(f"📝 Enabling detailed per-request logging...")
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=LOG_TO_CONSOLE,
        summary_interval=SUMMARY_INTERVAL,
        include_text=True,  # Include prompts/responses for analysis
        max_text_length=200,  # Truncate long text
    )

    # Start metrics collector
    logger.info("📊 Starting metrics collector...")
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=30,  # 30-second sliding window
        quantiles=[50, 90, 99],  # P50, P90, P99
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
        
        # Close per-request logger
        per_request_logger.close()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Results saved to: {output_file}")
        logger.info("")
        logger.info("📈 Next Steps:")
        logger.info("   1. Analyze TTFT distribution (target: <1s median, <2s p99)")
        logger.info("   2. Review throughput sustainability")
        logger.info("   3. Check for latency degradation over time")
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
    logger.info("Press Ctrl+C to stop early")
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
        logger.info("\n🛑 Benchmark interrupted by user")

    logger.info("")
    logger.info("=" * 80)
    logger.info("⏹️  Benchmark Duration Complete")
    logger.info("=" * 80)
    
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

