"""
Million Token Race - STRESS TEST MODE

This variant progressively increases load until the system reaches maximum capacity.
Automatically finds the optimal user count for peak throughput.

Strategy:
1. Start with baseline load (e.g., 20 users)
2. Increase users every 30 seconds
3. Monitor error rate and latency degradation
4. Stop when system saturates (errors > threshold or latency degrades)
5. Report optimal configuration

What it finds:
- Maximum sustainable throughput (tokens/sec)
- Optimal concurrent user count
- Failure threshold
- Performance degradation curve

Usage:
    # Auto-scale stress test
    python examples/benchmark_million_tokens_stress.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm-performance \\
        --start-users 20 \\
        --max-users 200 \\
        --step-users 10 \\
        --step-duration 30

    # Conservative approach (slower ramp)
    python examples/benchmark_million_tokens_stress.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm-performance \\
        --start-users 10 \\
        --max-users 150 \\
        --step-users 5 \\
        --step-duration 60

Safety Features:
- Automatic stop when error rate exceeds threshold (default: 10%)
- Latency degradation detection (P99 TTFT > 10s)
- Memory pressure monitoring
- Graceful shutdown on saturation
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
from typing import Optional

from transformers import AutoTokenizer

from llm_locust.clients.openai import OpenAIChatStreamingClient
from llm_locust.core.models import TriggerShutdown
from llm_locust.core.user import User
from llm_locust.metrics.collector import MetricsCollector
from llm_locust.metrics.per_request_logger import PerRequestLogger
from llm_locust.utils.prompts import load_databricks_dolly, SYSTEM_PROMPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Constants
BENCHMARK_NAME = "Million Token Race - STRESS TEST"
BENCHMARK_ID = "million-tokens-stress"
DEFAULT_START_USERS = 20
DEFAULT_MAX_USERS = 200
DEFAULT_STEP_USERS = 10
DEFAULT_STEP_DURATION = 30  # seconds between scaling steps
DEFAULT_INPUT_MIN = 50
DEFAULT_INPUT_MAX = 150
DEFAULT_OUTPUT_TOKENS = 2048
DEFAULT_DATASET = "dolly"
MAX_DURATION = 7200  # Safety: 2 hours max
DEFAULT_SPAWN_RATE = 10.0

# Failure thresholds
DEFAULT_ERROR_THRESHOLD = 0.10  # Stop if >10% errors
DEFAULT_LATENCY_THRESHOLD = 10000  # Stop if P99 TTFT >10s
DEFAULT_MIN_THROUGHPUT_RATIO = 0.5  # Stop if throughput drops below 50% of peak

# Logging
LOG_TO_CONSOLE = False
SUMMARY_INTERVAL = 50


class StressTestMetrics:
    """Tracks metrics for each load level"""
    
    def __init__(self, user_count: int):
        self.user_count = user_count
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_output_tokens = 0
        self.ttft_samples: list[float] = []
        
    def add_progress(self, progress: dict):
        """Add progress update"""
        self.total_requests += 1
        if progress['is_success']:
            self.successful_requests += 1
            self.total_output_tokens += progress['output_tokens']
        else:
            self.failed_requests += 1
    
    def add_ttft(self, ttft_ms: float):
        """Add TTFT sample"""
        self.ttft_samples.append(ttft_ms)
    
    def finalize(self):
        """Mark this load level as complete"""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Get duration of this load level"""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (tokens/sec)"""
        if self.duration == 0:
            return 0.0
        return self.total_output_tokens / self.duration
    
    @property
    def p99_ttft(self) -> float:
        """Calculate P99 TTFT"""
        if not self.ttft_samples:
            return 0.0
        sorted_samples = sorted(self.ttft_samples)
        p99_idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[p99_idx] if p99_idx < len(sorted_samples) else sorted_samples[-1]
    
    def should_stop(self, error_threshold: float, latency_threshold: float, peak_throughput: float, min_ratio: float) -> tuple[bool, str]:
        """Check if we should stop scaling"""
        # Need minimum data
        if self.total_requests < 10:
            return False, ""
        
        # Check error rate
        if self.error_rate > error_threshold:
            return True, f"Error rate {self.error_rate*100:.1f}% exceeds threshold {error_threshold*100:.1f}%"
        
        # Check latency degradation
        if self.ttft_samples and self.p99_ttft > latency_threshold:
            return True, f"P99 TTFT {self.p99_ttft:.0f}ms exceeds threshold {latency_threshold:.0f}ms"
        
        # Check throughput degradation (only if we have a peak to compare to)
        if peak_throughput > 0 and self.throughput < peak_throughput * min_ratio:
            return True, f"Throughput {self.throughput:.0f} tok/s dropped below {min_ratio*100:.0f}% of peak {peak_throughput:.0f} tok/s"
        
        return False, ""


async def stress_test_race(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    progress_queue: Queue,
    ttft_queue: Queue,
    start_users: int,
    max_users: int,
    step_users: int,
    step_duration: int,
    spawn_rate: float,
    error_threshold: float,
    latency_threshold: float,
    min_throughput_ratio: float,
) -> None:
    """Run stress test with progressive load increase"""
    
    users: list[User] = []
    current_level = StressTestMetrics(0)
    all_levels: list[StressTestMetrics] = []
    peak_throughput = 0.0
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🔥 STRESS TEST STARTING")
    logger.info("=" * 80)
    logger.info(f"Strategy: Progressively scale from {start_users} to {max_users} users")
    logger.info(f"Step size: +{step_users} users every {step_duration}s")
    logger.info(f"")
    logger.info(f"Stop conditions:")
    logger.info(f"  • Error rate > {error_threshold*100:.0f}%")
    logger.info(f"  • P99 TTFT > {latency_threshold:.0f}ms")
    logger.info(f"  • Throughput drops > {(1-min_throughput_ratio)*100:.0f}% from peak")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Start with initial users
    current_user_count = start_users
    spawn_delay = 1.0 / spawn_rate if spawn_rate > 0 else 0
    
    logger.info(f"📊 LOAD LEVEL 1: {current_user_count} users")
    current_level = StressTestMetrics(current_user_count)
    
    for i in range(current_user_count):
        user = User(
            model_client=client,
            metrics_queue=metrics_queue,
            user_id=i,
        )
        users.append(user)
        if spawn_delay > 0:
            await asyncio.sleep(spawn_delay)
    
    logger.info(f"✅ Spawned {current_user_count} users")
    await asyncio.sleep(2)  # Let them initialize
    
    level_start_time = time.time()
    last_log_time = time.time()
    
    while True:
        # Check for shutdown signal
        if not control_queue.empty():
            msg = control_queue.get()
            if isinstance(msg, TriggerShutdown):
                break
        
        # Check safety timeout
        if time.time() - current_level.start_time > MAX_DURATION:
            logger.warning("⚠️  Safety timeout reached")
            break
        
        # Collect progress updates
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                current_level.add_progress(progress)
            except:
                break
        
        # Collect TTFT samples
        while not ttft_queue.empty():
            try:
                ttft_ms = ttft_queue.get_nowait()
                current_level.add_ttft(ttft_ms)
            except:
                break
        
        # Log progress every 5 seconds
        if time.time() - last_log_time >= 5:
            elapsed = time.time() - level_start_time
            remaining = step_duration - elapsed
            logger.info(
                f"   Users: {current_user_count:>3} | "
                f"Reqs: {current_level.total_requests:>4} | "
                f"Throughput: {current_level.throughput:>6.0f} tok/s | "
                f"Errors: {current_level.error_rate*100:>4.1f}% | "
                f"Time: {elapsed:>4.0f}s / {step_duration}s"
            )
            last_log_time = time.time()
        
        # Check if it's time to scale up
        level_elapsed = time.time() - level_start_time
        if level_elapsed >= step_duration:
            # Finalize current level
            current_level.finalize()
            
            # Update peak throughput
            if current_level.throughput > peak_throughput:
                peak_throughput = current_level.throughput
            
            # Check if we should stop
            should_stop, reason = current_level.should_stop(
                error_threshold, latency_threshold, peak_throughput, min_throughput_ratio
            )
            
            if should_stop:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"🛑 STOPPING: {reason}")
                logger.info("=" * 80)
                all_levels.append(current_level)
                break
            
            # Save level and check if we should continue
            all_levels.append(current_level)
            
            # Check if we've reached max users
            if current_user_count >= max_users:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"✅ REACHED MAX USERS: {max_users}")
                logger.info("=" * 80)
                break
            
            # Scale up
            next_user_count = min(current_user_count + step_users, max_users)
            users_to_add = next_user_count - current_user_count
            
            logger.info("")
            logger.info(f"📈 SCALING UP: {current_user_count} → {next_user_count} users (+{users_to_add})")
            logger.info("")
            
            # Add new users
            for i in range(users_to_add):
                user = User(
                    model_client=client,
                    metrics_queue=metrics_queue,
                    user_id=len(users),
                )
                users.append(user)
                if spawn_delay > 0:
                    await asyncio.sleep(spawn_delay)
            
            current_user_count = next_user_count
            current_level = StressTestMetrics(current_user_count)
            level_start_time = time.time()
            
            logger.info(f"📊 LOAD LEVEL {len(all_levels) + 1}: {current_user_count} users")
        
        await asyncio.sleep(0.5)
    
    # Stop all users
    logger.info("")
    logger.info("🛑 Stopping all users...")
    stop_tasks = [user.stop() for user in users]
    await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 STRESS TEST RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Users':>6} | {'Duration':>8} | {'Reqs':>5} | {'Errors':>7} | {'Throughput':>11} | {'P99 TTFT':>9}")
    logger.info("-" * 80)
    
    best_level = None
    best_throughput = 0.0
    
    for level in all_levels:
        logger.info(
            f"{level.user_count:>6} | "
            f"{level.duration:>7.0f}s | "
            f"{level.total_requests:>5} | "
            f"{level.error_rate*100:>6.1f}% | "
            f"{level.throughput:>9.0f} tok/s | "
            f"{level.p99_ttft:>8.0f}ms"
        )
        if level.throughput > best_throughput and level.error_rate < error_threshold:
            best_throughput = level.throughput
            best_level = level
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏆 OPTIMAL CONFIGURATION")
    logger.info("=" * 80)
    
    if best_level:
        logger.info(f"  • Users:        {best_level.user_count}")
        logger.info(f"  • Throughput:   {best_level.throughput:.0f} tokens/sec")
        logger.info(f"  • Error Rate:   {best_level.error_rate*100:.2f}%")
        logger.info(f"  • P99 TTFT:     {best_level.p99_ttft:.0f}ms")
        logger.info(f"  • Requests:     {best_level.successful_requests}")
        logger.info("")
        logger.info(f"💡 Recommendation: Use {best_level.user_count} concurrent users for production")
    else:
        logger.info("  No stable configuration found")
    
    logger.info("=" * 80)


def run_stress_test(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    progress_queue: Queue,
    ttft_queue: Queue,
    start_users: int,
    max_users: int,
    step_users: int,
    step_duration: int,
    spawn_rate: float,
    error_threshold: float,
    latency_threshold: float,
    min_throughput_ratio: float,
) -> None:
    """Run stress test in separate process"""
    asyncio.run(
        stress_test_race(
            client, metrics_queue, control_queue, progress_queue, ttft_queue,
            start_users, max_users, step_users, step_duration, spawn_rate,
            error_threshold, latency_threshold, min_throughput_ratio
        )
    )


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
    parser.add_argument("--engine", type=str, required=True, help="Engine name")
    
    # Optional arguments
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Tokenizer")
    parser.add_argument("--start-users", type=int, default=DEFAULT_START_USERS, help=f"Starting users (default: {DEFAULT_START_USERS})")
    parser.add_argument("--max-users", type=int, default=DEFAULT_MAX_USERS, help=f"Maximum users (default: {DEFAULT_MAX_USERS})")
    parser.add_argument("--step-users", type=int, default=DEFAULT_STEP_USERS, help=f"Users to add per step (default: {DEFAULT_STEP_USERS})")
    parser.add_argument("--step-duration", type=int, default=DEFAULT_STEP_DURATION, help=f"Duration per step in seconds (default: {DEFAULT_STEP_DURATION})")
    parser.add_argument("--spawn-rate", type=float, default=DEFAULT_SPAWN_RATE, help=f"Spawn rate (default: {DEFAULT_SPAWN_RATE})")
    parser.add_argument("--error-threshold", type=float, default=DEFAULT_ERROR_THRESHOLD, help=f"Error threshold (default: {DEFAULT_ERROR_THRESHOLD})")
    parser.add_argument("--latency-threshold", type=float, default=DEFAULT_LATENCY_THRESHOLD, help=f"TTFT threshold in ms (default: {DEFAULT_LATENCY_THRESHOLD})")
    parser.add_argument("--min-throughput-ratio", type=float, default=DEFAULT_MIN_THROUGHPUT_RATIO, help=f"Min throughput ratio (default: {DEFAULT_MIN_THROUGHPUT_RATIO})")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate output
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    test_folder = f"{BENCHMARK_ID}-{timestamp}"
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}.csv"
    output_dir = Path(args.output_dir) / test_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    
    # Print config
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🔥 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🎯 Goal: Find maximum sustainable throughput")
    logger.info("")
    logger.info("⚙️  Configuration:")
    logger.info(f"   • Engine:       {args.engine}")
    logger.info(f"   • Host:         {args.host}")
    logger.info(f"   • Model:        {args.model}")
    logger.info(f"   • Start Users:  {args.start_users}")
    logger.info(f"   • Max Users:    {args.max_users}")
    logger.info(f"   • Step Size:    +{args.step_users} users")
    logger.info(f"   • Step Duration: {args.step_duration}s")
    logger.info("")
    logger.info("🛑 Stop Conditions:")
    logger.info(f"   • Error rate > {args.error_threshold*100:.0f}%")
    logger.info(f"   • P99 TTFT > {args.latency_threshold:.0f}ms")
    logger.info(f"   • Throughput drops > {(1-args.min_throughput_ratio)*100:.0f}%")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"
    
    # Load prompts
    logger.info("📚 Loading DOLLY dataset...")
    prompts = load_databricks_dolly(tokenizer, min_input_length=DEFAULT_INPUT_MIN, max_input_length=DEFAULT_INPUT_MAX)
    logger.info(f"✅ Loaded {len(prompts)} prompts")
    
    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT,
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=DEFAULT_OUTPUT_TOKENS,
    )
    
    # Setup queues
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()
    progress_queue: Queue = Queue()
    ttft_queue: Queue = Queue()
    
    # Setup logging
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=LOG_TO_CONSOLE,
        summary_interval=SUMMARY_INTERVAL,
        include_text=False,
        progress_queue=progress_queue,
        ttft_queue=ttft_queue,
    )
    
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=30,
        quantiles=[50, 90, 99],
        per_request_logger=per_request_logger,
    )
    collector.start_logging()
    
    # Start stress test
    test_process = Process(
        target=run_stress_test,
        args=(
            client, metrics_queue, control_queue, progress_queue, ttft_queue,
            args.start_users, args.max_users, args.step_users, args.step_duration,
            args.spawn_rate, args.error_threshold, args.latency_threshold, args.min_throughput_ratio
        ),
    )
    test_process.start()
    
    # Graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("\n🛑 Shutting down...")
        control_queue.put(TriggerShutdown())
        time.sleep(5)
        test_process.terminate()
        test_process.join()
        collector.stop_logging()
        per_request_logger.close()
        logger.info(f"\n📊 Results: {output_file}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for completion
    test_process.join()
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

