"""
Variable Load Cost Estimation Test

This test varies the request rate during the test to simulate realistic traffic patterns
and help you understand how your infrastructure performs under different load levels.

Traffic Patterns:
    - Ramp-up: Gradual increase from light to heavy load
    - Steady: Constant request rate
    - Wave: Oscillating between high and low load
    - Business Hours: Simulates daily traffic pattern
    - Burst: Random spikes in traffic

Usage:
    python examples/benchmark_variable_load.py \\
        --host http://localhost:8000 \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --engine vllm \\
        --pattern ramp-up
"""

import argparse
import asyncio
import logging
import math
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
from llm_locust.utils.prompts import load_databricks_dolly, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BENCHMARK_NAME = "Variable Load Cost Estimation"
BENCHMARK_ID = "variable-load-cost"


class LoadPattern:
    """Defines how request rate varies over time"""
    
    def __init__(self, pattern_name: str, duration: int):
        self.pattern_name = pattern_name
        self.duration = duration
    
    def get_target_users(self, elapsed_seconds: float) -> int:
        """Return target number of active users at given time"""
        progress = elapsed_seconds / self.duration
        
        if self.pattern_name == "ramp-up":
            # Gradual increase: 10 → 100 users
            return int(10 + (90 * progress))
        
        elif self.pattern_name == "ramp-down":
            # Gradual decrease: 100 → 10 users
            return int(100 - (90 * progress))
        
        elif self.pattern_name == "wave":
            # Sine wave: oscillate between 20-80 users
            return int(50 + 30 * math.sin(progress * 2 * math.pi))
        
        elif self.pattern_name == "business-hours":
            # Simulate daily pattern: low → high → low
            # Morning ramp-up, peak, evening ramp-down
            if progress < 0.25:  # Morning: 10 → 80
                return int(10 + 70 * (progress / 0.25))
            elif progress < 0.75:  # Day: steady at 80
                return 80
            else:  # Evening: 80 → 20
                return int(80 - 60 * ((progress - 0.75) / 0.25))
        
        elif self.pattern_name == "burst":
            # Random bursts every ~60 seconds
            burst_cycle = (elapsed_seconds % 60) / 60
            if burst_cycle < 0.2:  # 20% of time: HIGH
                return 100
            else:  # 80% of time: LOW
                return 20
        
        elif self.pattern_name == "steady-light":
            return 20
        
        elif self.pattern_name == "steady-medium":
            return 50
        
        elif self.pattern_name == "steady-heavy":
            return 100
        
        else:
            return 50  # Default
    
    def describe(self) -> str:
        """Return human-readable description"""
        descriptions = {
            "ramp-up": "Gradual increase from 10 to 100 users",
            "ramp-down": "Gradual decrease from 100 to 10 users",
            "wave": "Sine wave oscillating between 20-80 users",
            "business-hours": "Daily pattern: morning ramp (10→80), peak (80), evening drop (80→20)",
            "burst": "Random bursts: 20 users baseline, 100 users during bursts",
            "steady-light": "Constant 20 users",
            "steady-medium": "Constant 50 users",
            "steady-heavy": "Constant 100 users",
        }
        return descriptions.get(self.pattern_name, "Unknown pattern")


def run_variable_load_test(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    pattern: LoadPattern,
) -> None:
    """Run test with variable load pattern"""
    asyncio.run(_async_variable_load_test(client, metrics_queue, control_queue, pattern))


async def _async_variable_load_test(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    pattern: LoadPattern,
) -> None:
    """Async implementation of variable load test"""
    from llm_locust.core.models import SetUserInfo
    from llm_locust.core.user import User
    
    users: list[User] = []
    start_time = time.time()
    last_log_time = start_time
    
    # Track metrics for each 30-second window
    window_start_time = start_time
    window_requests = 0
    window_tokens = 0
    window_successes = 0
    window_failures = 0
    
    logger.info("🚀 Variable load test starting...")
    logger.info(f"📊 Pattern: {pattern.describe()}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 LIVE METRICS (updated every 30 seconds)")
    logger.info("=" * 80)
    
    while True:
        # Check for control messages
        if not control_queue.empty():
            msg = control_queue.get()
            if msg.__class__.__name__ == "TriggerShutdown":
                break
        
        elapsed = time.time() - start_time
        if elapsed >= pattern.duration:
            break
        
        # Calculate target users for current time
        target_users = pattern.get_target_users(elapsed)
        current_users = len(users)
        
        # Adjust user count
        if current_users < target_users:
            # Add users
            for _ in range(target_users - current_users):
                user = User(
                    model_client=client,
                    metrics_queue=metrics_queue,
                    user_id=len(users),
                )
                users.append(user)
        elif current_users > target_users:
            # Remove users
            for _ in range(current_users - target_users):
                if users:
                    user = users.pop()
                    await user.stop()
        
        # Collect metrics from queue (non-blocking)
        while not metrics_queue.empty():
            try:
                msg = metrics_queue.get_nowait()
                msg_type = msg.__class__.__name__
                
                if msg_type == "RequestSuccessLog":
                    window_requests += 1
                    window_successes += 1
                    window_tokens += msg.input_tokens + msg.output_tokens
                elif msg_type == "RequestFailureLog":
                    window_requests += 1
                    window_failures += 1
            except:
                break
        
        # Log metrics every 30 seconds
        current_time = time.time()
        window_duration = current_time - window_start_time
        
        if window_duration >= 30:
            # Calculate metrics for this window
            if window_duration > 0:
                req_per_sec = window_requests / window_duration
                tokens_per_sec = window_tokens / window_duration
                success_rate = (window_successes / window_requests * 100) if window_requests > 0 else 0
            else:
                req_per_sec = 0
                tokens_per_sec = 0
                success_rate = 0
            
            # Log metrics
            logger.info("")
            logger.info(f"⏱️  Time: {int(elapsed)}s | Users: {len(users)}")
            logger.info(f"   Requests:      {window_requests:>6} total  ({req_per_sec:>5.1f} req/s)")
            logger.info(f"   Tokens:        {window_tokens:>6} total  ({tokens_per_sec:>5.1f} tok/s)")
            logger.info(f"   Success Rate:  {success_rate:>5.1f}% ({window_successes} success, {window_failures} failed)")
            
            # Reset window counters
            window_start_time = current_time
            window_requests = 0
            window_tokens = 0
            window_successes = 0
            window_failures = 0
        
        await asyncio.sleep(1)
    
    # Cleanup
    logger.info("")
    logger.info("=" * 80)
    logger.info("🛑 Stopping all users...")
    for user in users:
        await user.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"LLM Benchmark - {BENCHMARK_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument("--host", type=str, required=True, help="LLM endpoint URL")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--engine", type=str, required=True, help="Engine/platform name")
    
    # Optional arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["ramp-up", "ramp-down", "wave", "business-hours", "burst", 
                 "steady-light", "steady-medium", "steady-heavy"],
        default="ramp-up",
        help="Load pattern to test",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300s / 5 minutes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create output file
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}-{args.pattern}.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    
    # Create load pattern
    pattern = LoadPattern(args.pattern, args.duration)
    
    # Print configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🎯 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Load Pattern:")
    logger.info(f"   • Pattern:  {args.pattern}")
    logger.info(f"   • Description: {pattern.describe()}")
    logger.info(f"   • Duration: {args.duration}s ({args.duration // 60} minutes)")
    logger.info("")
    logger.info("🎯 Target:")
    logger.info(f"   • Engine:   {args.engine}")
    logger.info(f"   • Host:     {args.host}")
    logger.info(f"   • Model:    {args.model}")
    logger.info("")
    logger.info("💡 Purpose:")
    logger.info("   See how your infrastructure handles varying load levels")
    logger.info("   This helps you understand:")
    logger.info("     - How throughput scales with load")
    logger.info("     - Where performance degrades")
    logger.info("     - Optimal cost efficiency point")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Load tokenizer and prompts
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"
    
    logger.info("📚 Loading prompts...")
    prompts = load_databricks_dolly(tokenizer, min_input_length=400, max_input_length=600)
    logger.info(f"✅ Loaded {len(prompts)} prompts")
    
    # Create client with unlimited output
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT,
        openai_model_name=args.model,
        tokenizer=tokenizer,
        max_tokens=2048,  # Safety limit
    )
    
    # Setup queues and metrics
    metrics_queue: Queue = Queue()
    control_queue: Queue = Queue()
    
    per_request_logger = PerRequestLogger(
        output_file=str(output_file),
        format="csv",
        print_to_console=False,
        summary_interval=50,
        include_text=True,
        max_text_length=200,
    )
    
    collector = MetricsCollector(
        metrics_queue=metrics_queue,
        model_client=client,
        metrics_window_size=30,
        quantiles=[50, 90, 99],
        per_request_logger=per_request_logger,
    )
    collector.start_logging()
    
    # Start test
    test_process = Process(
        target=run_variable_load_test,
        args=(client, metrics_queue, control_queue, pattern),
    )
    test_process.start()
    
    # Graceful shutdown handler
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("\n🛑 Shutting down...")
        from llm_locust.core.models import TriggerShutdown
        control_queue.put(TriggerShutdown())
        time.sleep(5)
        test_process.terminate()
        test_process.join()
        collector.stop_logging()
        per_request_logger.close()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Results: {output_file}")
        logger.info("")
        
        # Print detailed summary
        per_request_logger.print_summary()
        
        # Print pattern summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 LOAD PATTERN SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Pattern: {args.pattern}")
        logger.info(f"Description: {pattern.describe()}")
        logger.info(f"Duration: {args.duration}s ({args.duration // 60} minutes)")
        logger.info("")
        logger.info("💡 What to look for in the CSV:")
        logger.info("   • Group by timestamp to see metrics at different load levels")
        logger.info("   • Compare throughput (tokens/sec) vs number of concurrent users")
        logger.info("   • Look for latency degradation (TTFT, TPOT) as load increases")
        logger.info("   • Find the sweet spot: max throughput with acceptable latency")
        logger.info("")
        logger.info("📈 Next Steps:")
        logger.info("   1. Upload CSV to dashboard: streamlit run streamlit_app/app.py")
        logger.info("   2. Filter by timestamp ranges to compare load levels")
        logger.info("   3. Calculate $/1M tokens at different load points")
        logger.info("   4. Find your optimal cost efficiency!")
        logger.info("")
        logger.info("=" * 80)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for test to complete
    logger.info("")
    logger.info("🏁 TEST RUNNING")
    logger.info("=" * 80)
    
    test_process.join()
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
