"""
Benchmark Test 1b: RAG Simulation (4096 input / 512 output tokens)

Objective:
    Assess performance when handling large input contexts and longer responses 
    typical of retrieval-augmented generation (RAG) systems.

Workload Profile:
    - Input tokens: ~4096 per request
    - Output tokens: ~512 per request
    - Interaction type: Long-form context ingestion with detailed answers

Test Parameters:
    - Duration: 10-15 minutes (default: 900s)
    - Concurrency: ~20 parallel sessions
    - Rate: Moderate, with bursts representing multiple users querying documents

Benchmark Focus:
    - Memory Load: Stress-test KV cache growth and GPU memory usage
    - Latency Distribution: Observe how latency scales with large token counts
    - Throughput Impact: Identify drop-offs as request size increases

Business Context:
    Knowledge-base assistants, research copilots, or enterprise search systems 
    requiring context-heavy queries.

Usage:
    python examples/benchmark_rag_simulation.py \\
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
from llm_locust.utils.prompts import load_billsum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Test 1b Constants
BENCHMARK_NAME = "RAG Simulation (Test 1b)"
BENCHMARK_ID = "1b-rag-simulation"
DATASET_NAME = "billsum"  # Long legislative documents ideal for RAG testing
TARGET_INPUT_TOKENS = 4096
TARGET_OUTPUT_TOKENS = 512
DEFAULT_DURATION = 900  # 15 minutes
DEFAULT_USERS = 20
DEFAULT_SPAWN_RATE = 2.0  # Users per second (slower than Test 1a due to larger context)
INPUT_TOKEN_MIN = 800   # Adjusted for servers with 1024 max input tokens
INPUT_TOKEN_MAX = 950   # Safely below 1024 limit, leaves room for 512 output tokens

# Per-request logging defaults
LOG_TO_CONSOLE = True
SUMMARY_INTERVAL = 0  # 0 = show all requests


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
        help=f"Test duration in seconds (default: {DEFAULT_DURATION}s / 15 minutes)",
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
    logger.info(f"  • Workload Type:        RAG/Long Context (document processing)")
    logger.info(f"  • Success Criteria:     TTFT <3s (median), throughput stability")
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
    logger.info(f"  • Total Reqs:   ~{int(args.users * (args.duration / 60) * 0.5)} (estimated)")
    logger.info(f"  • Dataset:      {DATASET_NAME.upper()}")
    logger.info("")
    logger.info("📊 Metrics Configuration:")
    logger.info(f"  • Output File:  {output_filename}")
    logger.info(f"  • Format:       CSV with full metrics")
    logger.info(f"  • Console Log:  {'Enabled' if LOG_TO_CONSOLE else 'Disabled'}")
    logger.info(f"  • Log Interval: {'All requests' if SUMMARY_INTERVAL == 0 else f'Every {SUMMARY_INTERVAL}th request'}")
    logger.info("")
    logger.info("🎯 Focus Areas:")
    logger.info("  • KV Cache Growth & GPU Memory Usage")
    logger.info("  • Latency Scaling with Large Contexts")
    logger.info("  • Throughput Impact vs Short Contexts")
    logger.info("  • Time to First Token (TTFT) at Scale")
    logger.info("")
    logger.info("⚠️  RAG-Specific Notes:")
    logger.info("  • Large contexts stress memory management")
    logger.info("  • TTFT expected to be higher than Test 1a")
    logger.info("  • Watch for OOM errors and throughput degradation")
    logger.info("  • Measure tokens/second efficiency")
    logger.info("")
    logger.info("⚙️  Server Requirements:")
    logger.info(f"  • Minimum max_model_len: {INPUT_TOKEN_MAX + TARGET_OUTPUT_TOKENS} tokens")
    logger.info(f"    ({INPUT_TOKEN_MAX} input + {TARGET_OUTPUT_TOKENS} output)")
    logger.info(f"  • For full 4K context: {TARGET_INPUT_TOKENS + TARGET_OUTPUT_TOKENS}+ tokens")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts - BillSum dataset for long-form documents
    logger.info(f"📚 Loading {DATASET_NAME.upper()} dataset (long legislative documents)...")
    logger.info(f"   Source: https://huggingface.co/datasets/FiscalNote/billsum")
    logger.info(f"   Filtering for ~{TARGET_INPUT_TOKENS} input tokens")
    logger.info(f"   ({INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX} token range)")
    logger.info("   This may take a moment due to large document sizes...")
    
    try:
        prompts = load_billsum(
            tokenizer,
            min_input_length=INPUT_TOKEN_MIN,
            max_input_length=INPUT_TOKEN_MAX,
            num_samples=500,  # Get 500 bills to have enough variety
        )
    except Exception as e:
        logger.error(f"❌ Failed to load {DATASET_NAME} dataset: {e}")
        logger.error("   This dataset requires pandas or pyarrow: pip install pandas pyarrow")
        sys.exit(1)
    
    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts found ({len(prompts)}). Need at least 10.")
        logger.error(f"   Token range: {INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX}")
        logger.error("")
        logger.error("   💡 BillSum bills distribution:")
        logger.error("      • Most bills: 1500-3500 tokens")
        logger.error("      • Large bills: 3500-5000+ tokens")
        logger.error("")
        logger.error("   📋 To fix:")
        logger.error("      • Delete datasets/billsum.jsonl cache and retry")
        logger.error("      • Script will re-download with correct token range")
        logger.error("")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(prompts)} long-context prompts")
    avg_input_tokens = sum(p["num_input_tokens"] for p in prompts) / len(prompts)
    logger.info(f"   Average input tokens: {avg_input_tokens:.0f}")

    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt="You are a helpful AI assistant specialized in document analysis. Provide detailed, comprehensive responses based on the given context.",
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=TARGET_OUTPUT_TOKENS,
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
        logger.info("   1. Analyze KV cache memory usage patterns")
        logger.info("   2. Compare TTFT vs Test 1a (expect 2-3x higher)")
        logger.info("   3. Review throughput degradation (tokens/sec)")
        logger.info("   4. Check for OOM errors or request failures")
        logger.info("   5. Measure latency distribution (P50, P90, P99)")
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
    logger.info("⚠️  Note: Large contexts may cause slower initial responses")
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

