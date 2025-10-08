"""
Million Token Race Benchmark

Objective:
    Measure how fast your infrastructure can produce 1 MILLION OUTPUT TOKENS.
    This is a throughput-focused test optimized for maximum token generation speed.

What This Test Does:
    1. Spawns high concurrency users (default: 80 for L4 GPU)
    2. Each user continuously generates tokens
    3. Tracks total output tokens produced
    4. Stops when 1M output tokens are generated
    5. Reports final throughput and time to completion

Key Metrics:
    - Time to 1M tokens (minutes/seconds)
    - Average throughput (tokens/second)
    - Peak throughput (tokens/second)
    - Total requests completed
    - Success rate

Hardware Optimization:
    - L4 GPU (24GB): 60-80 users recommended
    - A10G (24GB): 80-100 users recommended
    - A100 (80GB): 150-200 users recommended

Usage Examples:

    # Default: Race to 1M tokens with 80 users (optimized for L4)
    python examples/benchmark_million_tokens.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm

    # Conservative load (safer for smaller GPUs)
    python examples/benchmark_million_tokens.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --users 50

    # Aggressive load (for larger GPUs like A100)
    python examples/benchmark_million_tokens.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --users 150

    # Race to 10M tokens (stress test)
    python examples/benchmark_million_tokens.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --target-tokens 10000000

    # Short prompts for maximum output token generation
    python examples/benchmark_million_tokens.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --prompt-min-tokens 50 \\
        --prompt-max-tokens 100

Expected Results (L4 GPU, Llama-3.2-3B):
    - Time to 1M tokens: 5-10 minutes
    - Throughput: ~2000-3000 tokens/sec
    - Success rate: >99%

After Running:
    - CSV file saved with all request details
    - Final report with throughput statistics
    - Use Cost Analysis dashboard to calculate $/1M tokens
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
from llm_locust.core.models import RequestSuccessLog, RequestFailureLog, TriggerShutdown
from llm_locust.core.user import User
from llm_locust.metrics.per_request_logger import PerRequestLogger
from llm_locust.utils.prompts import load_databricks_dolly, SYSTEM_PROMPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Constants
BENCHMARK_NAME = "Million Token Race"
BENCHMARK_ID = "million-tokens"
DEFAULT_TARGET_TOKENS = 1_000_000  # 1 million output tokens
DEFAULT_USERS = 80  # Optimized for L4 GPU with 3B model
DEFAULT_SPAWN_RATE = 20.0  # Fast ramp-up
DEFAULT_INPUT_MIN = 50  # Short prompts = more output tokens
DEFAULT_INPUT_MAX = 150
DEFAULT_OUTPUT_TOKENS = 2048  # Max output per request
DEFAULT_DATASET = "dolly"
MAX_DURATION = 3600  # Safety timeout: 1 hour

# Logging configuration
LOG_TO_CONSOLE = False  # Don't spam console
SUMMARY_INTERVAL = 100  # Show summary every 100 requests


class TokenRaceTracker:
    """Tracks progress toward token target"""
    
    def __init__(self, target_tokens: int):
        self.target_tokens = target_tokens
        self.total_output_tokens = 0
        self.total_input_tokens = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Throughput tracking
        self.token_timestamps = []  # (timestamp, cumulative_tokens)
        self.peak_throughput = 0
        
    def add_request(self, progress: dict):
        """Add request to tracker"""
        if progress['is_success']:
            self.total_output_tokens += progress['output_tokens']
            self.total_input_tokens += progress['input_tokens']
            self.successful_requests += 1
            self.token_timestamps.append((time.time(), self.total_output_tokens))
        else:
            self.failed_requests += 1
        
        self.total_requests += 1
    
    def get_progress_pct(self) -> float:
        """Get progress percentage"""
        return (self.total_output_tokens / self.target_tokens) * 100
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def get_current_throughput(self) -> float:
        """Get current throughput (tokens/sec)"""
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            return self.total_output_tokens / elapsed
        return 0
    
    def get_eta_seconds(self) -> float:
        """Estimate time remaining"""
        throughput = self.get_current_throughput()
        if throughput > 0:
            remaining_tokens = self.target_tokens - self.total_output_tokens
            return remaining_tokens / throughput
        return 0
    
    def calculate_peak_throughput(self, window_seconds: int = 10) -> float:
        """Calculate peak throughput over a sliding window"""
        if len(self.token_timestamps) < 2:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Find tokens at window start
        recent_timestamps = [t for t in self.token_timestamps if t[0] >= cutoff_time]
        
        if len(recent_timestamps) < 2:
            return 0
        
        start_time, start_tokens = recent_timestamps[0]
        end_time, end_tokens = recent_timestamps[-1]
        
        time_diff = end_time - start_time
        token_diff = end_tokens - start_tokens
        
        if time_diff > 0:
            return token_diff / time_diff
        return 0
    
    def is_complete(self) -> bool:
        """Check if target reached"""
        return self.total_output_tokens >= self.target_tokens
    
    def should_log_progress(self, interval_seconds: int = 5) -> bool:
        """Check if should log progress update"""
        current_time = time.time()
        if current_time - self.last_log_time >= interval_seconds:
            self.last_log_time = current_time
            return True
        return False


def run_million_token_race(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    progress_queue: Queue,
    target_tokens: int,
    max_users: int,
    spawn_rate: float,
) -> None:
    """Run million token race in separate process"""
    asyncio.run(
        _async_million_token_race(
            client, metrics_queue, control_queue, progress_queue, target_tokens, max_users, spawn_rate
        )
    )


async def _async_million_token_race(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    progress_queue: Queue,
    target_tokens: int,
    max_users: int,
    spawn_rate: float,
) -> None:
    """Async implementation of million token race"""
    tracker = TokenRaceTracker(target_tokens)
    users: list[User] = []
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 MILLION TOKEN RACE STARTING")
    logger.info("=" * 80)
    logger.info(f"Target: {target_tokens:,} output tokens")
    logger.info(f"Users: {max_users} concurrent")
    logger.info(f"Spawn rate: {spawn_rate} users/second")
    logger.info("")
    logger.info("⏱️  Spawning users...")
    
    # Spawn users rapidly - User.__init__ automatically starts the task
    spawn_delay = 1.0 / spawn_rate if spawn_rate > 0 else 0
    for i in range(max_users):
        user = User(
            model_client=client,
            metrics_queue=metrics_queue,
            user_id=i,
        )
        users.append(user)
        
        if i % 10 == 0:
            logger.info(f"   Spawned {i}/{max_users} users...")
        
        if spawn_delay > 0:
            await asyncio.sleep(spawn_delay)
    
    logger.info(f"✅ All {max_users} users active!")
    logger.info("")
    
    # Give users a moment to initialize their request loops
    await asyncio.sleep(1)
    
    logger.info("=" * 80)
    logger.info("🏃 RACING TO 1 MILLION TOKENS")
    logger.info("=" * 80)
    logger.info("")
    
    # Main race loop
    while not tracker.is_complete():
        # Check for shutdown signal
        if not control_queue.empty():
            msg = control_queue.get()
            if isinstance(msg, TriggerShutdown):
                break
        
        # Check safety timeout
        if tracker.get_elapsed_time() > MAX_DURATION:
            logger.warning(f"⚠️  Safety timeout reached ({MAX_DURATION}s)")
            break
        
        # Collect progress updates from queue
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                tracker.add_request(msg)
            except:
                break
        
        # Log progress
        if tracker.should_log_progress(interval_seconds=5):
            progress_pct = tracker.get_progress_pct()
            elapsed = tracker.get_elapsed_time()
            throughput = tracker.get_current_throughput()
            eta = tracker.get_eta_seconds()
            peak = tracker.calculate_peak_throughput(window_seconds=10)
            
            if peak > tracker.peak_throughput:
                tracker.peak_throughput = peak
            
            logger.info(
                f"📊 Progress: {tracker.total_output_tokens:>9,}/{target_tokens:,} tokens "
                f"({progress_pct:>5.1f}%) | "
                f"⏱️  {elapsed:>5.0f}s | "
                f"🚀 {throughput:>6.0f} tok/s | "
                f"⏰ ETA: {eta:>4.0f}s | "
                f"👥 {len(users)} users | "
                f"📋 {tracker.successful_requests} reqs"
            )
        
        await asyncio.sleep(0.1)
    
    # Race complete!
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 RACE COMPLETE!")
    logger.info("=" * 80)
    
    # Stop all users (they handle cleanup internally)
    logger.info("🛑 Stopping users...")
    stop_tasks = [user.stop() for user in users]
    await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    # Final stats
    elapsed = tracker.get_elapsed_time()
    avg_throughput = tracker.get_current_throughput()
    success_rate = (tracker.successful_requests / tracker.total_requests * 100) if tracker.total_requests > 0 else 0
    
    logger.info("")
    logger.info("🏆 FINAL RESULTS:")
    logger.info(f"   • Output Tokens:      {tracker.total_output_tokens:,}")
    logger.info(f"   • Input Tokens:       {tracker.total_input_tokens:,}")
    logger.info(f"   • Total Tokens:       {tracker.total_output_tokens + tracker.total_input_tokens:,}")
    logger.info(f"   • Time Elapsed:       {elapsed:.1f}s ({elapsed/60:.2f} minutes)")
    logger.info(f"   • Avg Throughput:     {avg_throughput:.0f} tokens/sec")
    logger.info(f"   • Peak Throughput:    {tracker.peak_throughput:.0f} tokens/sec")
    logger.info(f"   • Total Requests:     {tracker.total_requests:,}")
    logger.info(f"   • Successful:         {tracker.successful_requests:,}")
    logger.info(f"   • Failed:             {tracker.failed_requests:,}")
    logger.info(f"   • Success Rate:       {success_rate:.2f}%")
    logger.info("")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"LLM Benchmark - {BENCHMARK_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument("--host", type=str, required=True, help="LLM endpoint URL")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--engine", type=str, required=True, help="Engine name (vllm, tgi, etc.)")
    
    # Optional arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=f"Target output tokens (default: {DEFAULT_TARGET_TOKENS:,})",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_USERS,
        help=f"Concurrent users (default: {DEFAULT_USERS})",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=DEFAULT_SPAWN_RATE,
        help=f"Users to spawn per second (default: {DEFAULT_SPAWN_RATE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help=f"Max output tokens per request (default: {DEFAULT_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--prompt-min-tokens",
        type=int,
        default=DEFAULT_INPUT_MIN,
        help=f"Min input tokens (default: {DEFAULT_INPUT_MIN})",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=DEFAULT_INPUT_MAX,
        help=f"Max input tokens (default: {DEFAULT_INPUT_MAX})",
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
        help="Output directory (default: results)",
    )
    
    args = parser.parse_args()
    
    # Generate output directory and filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    test_folder = f"{BENCHMARK_ID}-{timestamp}"
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}.csv"
    
    # Create nested directory: results/million-tokens-20251008-083653/
    output_dir = Path(args.output_dir) / test_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    
    # Print configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🏁 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🎯 Goal:")
    logger.info(f"   Generate {args.target_tokens:,} OUTPUT tokens as fast as possible!")
    logger.info("")
    logger.info("⚙️  Configuration:")
    logger.info(f"   • Engine:         {args.engine}")
    logger.info(f"   • Host:           {args.host}")
    logger.info(f"   • Model:          {args.model}")
    logger.info(f"   • Users:          {args.users} concurrent")
    logger.info(f"   • Spawn Rate:     {args.spawn_rate} users/sec")
    logger.info(f"   • Max Tokens:     {args.max_tokens} per request")
    logger.info(f"   • Input Tokens:   {args.prompt_min_tokens}-{args.prompt_max_tokens}")
    logger.info(f"   • Dataset:        {args.dataset.upper()}")
    logger.info("")
    logger.info("💡 Strategy:")
    logger.info("   • Short prompts = more tokens for output generation")
    logger.info("   • High concurrency = maximize throughput")
    logger.info("   • Unlimited output = natural token generation")
    logger.info("")
    logger.info("📊 Output:")
    logger.info(f"   • Folder: {test_folder}/")
    logger.info(f"   • File:   {output_filename}")
    logger.info(f"   • Path:   {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"
    
    # Load prompts
    logger.info(f"📚 Loading {args.dataset.upper()} dataset...")
    if args.dataset == "dolly":
        from llm_locust.utils.prompts import load_databricks_dolly
        prompts = load_databricks_dolly(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "sharegpt":
        from llm_locust.utils.prompts import load_sharegpt
        prompts = load_sharegpt(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    elif args.dataset == "billsum":
        from llm_locust.utils.prompts import load_billsum
        prompts = load_billsum(
            tokenizer,
            min_input_length=args.prompt_min_tokens,
            max_input_length=args.prompt_max_tokens,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts ({len(prompts)}). Need at least 10.")
        sys.exit(1)
    
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
    progress_queue: Queue = Queue()  # NEW: Separate queue for progress tracking
    
    # Setup per-request logging with progress tracking
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=LOG_TO_CONSOLE,
        summary_interval=SUMMARY_INTERVAL,
        include_text=True,
        max_text_length=200,
        progress_queue=progress_queue,  # Send progress updates here
    )
    
    # Start metrics collection (lightweight - just for CSV logging)
    from llm_locust.metrics.collector import MetricsCollector
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=30,
        quantiles=[50, 90, 99],
        per_request_logger=per_request_logger,
    )
    collector.start_logging()
    
    # Start race process
    race_process = Process(
        target=run_million_token_race,
        args=(client, metrics_queue, control_queue, progress_queue, args.target_tokens, args.users, args.spawn_rate),
    )
    race_process.start()
    
    # Setup graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("\n🛑 Shutting down...")
        control_queue.put(TriggerShutdown())
        time.sleep(5)
        race_process.terminate()
        race_process.join()
        collector.stop_logging()
        per_request_logger.close()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 Results saved to:")
        logger.info(f"   Folder: {test_folder}/")
        logger.info(f"   File:   {output_file}")
        logger.info("")
        per_request_logger.print_summary()
        logger.info("")
        logger.info("💡 Next Steps:")
        logger.info("   • Upload CSV to dashboard: streamlit run streamlit_app/app.py")
        logger.info("   • Analyze throughput and cost efficiency")
        logger.info("   • Compare with other configurations")
        logger.info("=" * 80)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for race to complete
    race_process.join()
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

