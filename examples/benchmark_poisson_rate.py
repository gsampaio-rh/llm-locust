"""
Benchmark Test 2b: Poisson Rate (Bursty Traffic) (512 input / 256 output tokens)

Objective:
    Evaluate system robustness under irregular, unpredictable bursts of traffic.

Workload Profile:
    - Input tokens: ~512 per request
    - Output tokens: ~256 per request
    - Interaction type: Requests arrive in sudden spikes, modeled with Poisson distribution

Test Parameters:
    - Duration: 10-15 minutes (default: 900s / 15 minutes)
    - Concurrency: Varies dynamically with traffic spikes
    - Rate: Average ~2 requests/second, with unpredictable peaks above baseline
    - Burst Factor: Peak can reach 5-10x average rate

Benchmark Focus:
    - Autoscaling: Tests system's ability to allocate resources dynamically
    - Queueing & Batching: Reveals how the system manages traffic spikes
    - Tail Latency: Identifies user experience risks under peak load

Business Context:
    Real-world enterprise apps with spiky traffic, such as e-commerce assistants 
    during flash sales, or knowledge tools during peak work hours.

Usage:
    python examples/benchmark_poisson_rate.py \\
        --host https://your-llm-endpoint.com \\
        --model your-model-name \\
        --engine vllm \\
        --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import logging
import random
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

# Benchmark Test 2b Constants
BENCHMARK_NAME = "Poisson Rate / Bursty Traffic (Test 2b)"
BENCHMARK_ID = "2b-poisson-rate"
DATASET_NAME = "sharegpt"  # Mixed conversational prompts
TARGET_INPUT_TOKENS = 512
TARGET_OUTPUT_TOKENS = 256
DEFAULT_DURATION = 900  # 15 minutes
DEFAULT_AVERAGE_RATE = 2.0  # Average requests per second
DEFAULT_BURST_FACTOR = 5.0  # Peak can be 5x average
DEFAULT_MAX_USERS = 100  # Maximum concurrent users during peak
INPUT_TOKEN_MIN = 400
INPUT_TOKEN_MAX = 600

# Per-request logging defaults
LOG_TO_CONSOLE = True
SUMMARY_INTERVAL = 10  # Show every 10th request

# Poisson distribution parameters
# Lambda (λ) = average rate (requests per second)
# Inter-arrival time follows exponential distribution: -ln(U) / λ


def generate_poisson_intervals(
    duration: int,
    average_rate: float,
    burst_factor: float = 5.0,
    burst_frequency: int = 120,  # Burst every 2 minutes
) -> list[float]:
    """
    Generate request timestamps following Poisson distribution with periodic bursts.
    
    Args:
        duration: Test duration in seconds
        average_rate: Average requests per second
        burst_factor: Multiplier for burst periods (peak rate = avg * burst_factor)
        burst_frequency: How often bursts occur (in seconds)
        
    Returns:
        List of timestamps when requests should be made
    """
    timestamps = []
    current_time = 0.0
    burst_duration = 30  # Each burst lasts 30 seconds
    
    while current_time < duration:
        # Determine if we're in a burst period
        time_in_cycle = current_time % burst_frequency
        in_burst = time_in_cycle < burst_duration
        
        # Adjust rate based on burst status
        current_rate = average_rate * burst_factor if in_burst else average_rate
        
        # Generate inter-arrival time using exponential distribution
        # This creates Poisson-distributed arrivals
        inter_arrival = random.expovariate(current_rate)
        current_time += inter_arrival
        
        if current_time < duration:
            timestamps.append(current_time)
    
    logger.info(f"Generated {len(timestamps)} request timestamps with Poisson distribution")
    logger.info(f"  • Average rate: {average_rate:.2f} req/s")
    logger.info(f"  • Burst rate: {average_rate * burst_factor:.2f} req/s")
    logger.info(f"  • Burst every: {burst_frequency}s for {burst_duration}s")
    logger.info(f"  • Expected total: ~{int(duration * average_rate)} requests")
    
    return timestamps


class PoissonLoadGenerator:
    """Generates load following Poisson distribution with bursts."""
    
    def __init__(
        self,
        client: OpenAIChatStreamingClient,
        metrics_queue: Queue,
        control_queue: Queue,
        timestamps: list[float],
        max_users: int,
    ):
        self.client = client
        self.metrics_queue = metrics_queue
        self.control_queue = control_queue
        self.timestamps = sorted(timestamps)
        self.max_users = max_users
        self.active_tasks = []
        self.user_counter = 0
    
    async def make_single_request(self, user_id: int) -> None:
        """Make a single request and record metrics."""
        import aiohttp
        from llm_locust.core.models import (
            ErrorLog,
            RequestFailureLog,
            RequestSuccessLog,
            get_timestamp_seconds,
        )
        
        url, headers, data, input_data = self.client.get_request_params()
        start_time = time.perf_counter()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        time_key = get_timestamp_seconds()
                        end_time = time.perf_counter()
                        self.metrics_queue.put(
                            RequestFailureLog(
                                timestamp=time_key,
                                start_time=start_time,
                                end_time=end_time,
                                status_code=response.status,
                                user_id=user_id,
                            )
                        )
                        return
                    
                    result_chunks: list[bytes] = []
                    token_times: list[float] = []
                    
                    async for chunk_data, _ in response.content.iter_chunks():
                        token_times.append(time.perf_counter())
                        result_chunks.append(chunk_data)
                    
                    time_key = get_timestamp_seconds()
                    end_time = time.perf_counter()
                    self.metrics_queue.put(
                        RequestSuccessLog(
                            result_chunks=tuple(result_chunks),
                            num_input_tokens=input_data["num_input_tokens"],
                            timestamp=time_key,
                            token_times=tuple(token_times),
                            start_time=start_time,
                            end_time=end_time,
                            status_code=response.status,
                            user_id=user_id,
                            input_prompt=input_data.get("prompt", ""),
                        )
                    )
        except Exception as e:
            logger.warning(f"Request failed for user {user_id}: {e}")
            self.metrics_queue.put(
                ErrorLog(
                    error_message=str(e),
                    error_type=type(e).__name__,
                    context={"user_id": user_id},
                )
            )
    
    async def run(self) -> None:
        """Run the Poisson load generation."""
        from llm_locust.core.models import TriggerShutdown
        
        start_time = time.time()
        request_index = 0
        
        logger.info(f"🌊 Starting Poisson load generation with {len(self.timestamps)} requests")
        
        while request_index < len(self.timestamps):
            # Check for shutdown signal
            if not self.control_queue.empty():
                msg = self.control_queue.get()
                if isinstance(msg, TriggerShutdown):
                    logger.info("Received shutdown signal")
                    break
            
            current_time = time.time() - start_time
            target_time = self.timestamps[request_index]
            
            # Wait until it's time for the next request
            if current_time < target_time:
                await asyncio.sleep(min(0.1, target_time - current_time))
                continue
            
            # Clean up completed tasks
            self.active_tasks = [t for t in self.active_tasks if not t.done()]
            
            # Create a new request task
            if len(self.active_tasks) < self.max_users:
                self.user_counter += 1
                task = asyncio.create_task(self.make_single_request(self.user_counter))
                self.active_tasks.append(task)
                
                request_index += 1
                
                # Progress logging
                if request_index % 100 == 0:
                    progress = (request_index / len(self.timestamps)) * 100
                    logger.info(
                        f"📊 Progress: {request_index}/{len(self.timestamps)} requests "
                        f"({progress:.1f}%) | Active requests: {len(self.active_tasks)}"
                    )
            else:
                # Too many concurrent requests, system is saturated
                # This reveals queueing behavior
                await asyncio.sleep(0.1)
        
        # Wait for remaining tasks to complete
        if self.active_tasks:
            logger.info(f"⏳ Waiting for {len(self.active_tasks)} active requests to complete...")
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        logger.info("✅ Poisson load generation complete")


def run_poisson_spawner(
    client: OpenAIChatStreamingClient,
    metrics_queue: Queue,
    control_queue: Queue,
    timestamps: list[float],
    max_users: int,
) -> None:
    """Run Poisson load generator in a separate process."""
    generator = PoissonLoadGenerator(
        client=client,
        metrics_queue=metrics_queue,
        control_queue=control_queue,
        timestamps=timestamps,
        max_users=max_users,
    )
    asyncio.run(generator.run())


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
        "--average-rate",
        type=float,
        default=DEFAULT_AVERAGE_RATE,
        help=f"Average request rate (req/s) (default: {DEFAULT_AVERAGE_RATE})",
    )
    parser.add_argument(
        "--burst-factor",
        type=float,
        default=DEFAULT_BURST_FACTOR,
        help=f"Burst multiplier (peak = avg * factor) (default: {DEFAULT_BURST_FACTOR})",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=DEFAULT_MAX_USERS,
        help=f"Maximum concurrent users during bursts (default: {DEFAULT_MAX_USERS})",
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

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"{args.engine}-{timestamp}-{BENCHMARK_ID}.csv"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    # Calculate expected requests
    expected_requests = int(args.average_rate * args.duration)

    # Print benchmark configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🎯 BENCHMARK: {BENCHMARK_NAME}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Benchmark Specification:")
    logger.info(f"  • Target Input Tokens:  ~{TARGET_INPUT_TOKENS} tokens")
    logger.info(f"  • Target Output Tokens: ~{TARGET_OUTPUT_TOKENS} tokens")
    logger.info(f"  • Workload Type:        Poisson/Bursty (unpredictable spikes)")
    logger.info(f"  • Success Criteria:     System handles bursts gracefully, low tail latency")
    logger.info("")
    logger.info("🎯 Target Configuration:")
    logger.info(f"  • Engine:       {args.engine}")
    logger.info(f"  • Endpoint:     {args.host}")
    logger.info(f"  • Model:        {args.model}")
    logger.info(f"  • Tokenizer:    {args.tokenizer}")
    logger.info("")
    logger.info("⚙️  Test Configuration:")
    logger.info(f"  • Average Rate:  {args.average_rate} req/s")
    logger.info(f"  • Peak Rate:     {args.average_rate * args.burst_factor} req/s (burst)")
    logger.info(f"  • Burst Factor:  {args.burst_factor}x")
    logger.info(f"  • Max Users:     {args.max_users} concurrent (during bursts)")
    logger.info(f"  • Duration:      {args.duration}s ({args.duration // 60} minutes)")
    logger.info(f"  • Total Reqs:    ~{expected_requests} (average)")
    logger.info(f"  • Dataset:       {DATASET_NAME.upper()}")
    logger.info("")
    logger.info("📊 Metrics Configuration:")
    logger.info(f"  • Output File:   {output_filename}")
    logger.info(f"  • Format:        CSV with full metrics")
    logger.info(f"  • Console Log:   {'Enabled' if LOG_TO_CONSOLE else 'Disabled'}")
    logger.info(f"  • Log Interval:  Every {SUMMARY_INTERVAL}th request")
    logger.info("")
    logger.info("🎯 Focus Areas:")
    logger.info("  • Burst Handling Capability")
    logger.info("  • Tail Latency During Peaks (P99)")
    logger.info("  • Queue Management & Batching")
    logger.info("  • Resource Allocation Dynamics")
    logger.info("  • Error Rate During Overload")
    logger.info("")
    logger.info("🌊 Poisson Distribution Details:")
    logger.info("  • Arrivals follow exponential inter-arrival times")
    logger.info("  • Bursts occur periodically every ~2 minutes")
    logger.info("  • Each burst lasts ~30 seconds")
    logger.info("  • Models real-world unpredictable traffic")
    logger.info("")
    logger.info("📈 What to Watch For:")
    logger.info("  • P99 latency spikes during bursts")
    logger.info("  • Error rate increase under peak load")
    logger.info("  • Queue depth and request queueing time")
    logger.info("  • System recovery after burst periods")
    logger.info("  • Concurrent user count variations")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Generate Poisson timestamps
    logger.info("🎲 Generating Poisson-distributed request timestamps...")
    random.seed(42)  # For reproducibility
    timestamps = generate_poisson_intervals(
        duration=args.duration,
        average_rate=args.average_rate,
        burst_factor=args.burst_factor,
    )
    logger.info(f"✅ Generated {len(timestamps)} timestamps")
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts
    logger.info(f"📚 Loading {DATASET_NAME.upper()} dataset...")
    prompts = load_sharegpt(
        tokenizer,
        min_input_length=INPUT_TOKEN_MIN,
        max_input_length=INPUT_TOKEN_MAX,
    )
    
    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts found ({len(prompts)}). Need at least 10.")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(prompts)} prompts")

    # Create client
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt="You are a helpful AI assistant. Provide clear, concise responses.",
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
        metrics_window_size=30,
        quantiles=[50, 90, 99],
        per_request_logger=per_request_logger,
    )
    collector.start_logging()

    # Start Poisson load generator
    logger.info("🌊 Starting Poisson load generator...")
    spawner_process = Process(
        target=run_poisson_spawner,
        args=(client, metrics_queue, control_queue, timestamps, args.max_users),
    )
    spawner_process.start()
    
    time.sleep(2)

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
        logger.info("✅ BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Results saved to: {output_file}")
        logger.info("")
        logger.info("📈 Analysis Guidelines:")
        logger.info("   1. Plot latency over time - look for spikes during bursts")
        logger.info("   2. Analyze P99 latency during burst vs normal periods")
        logger.info("   3. Check error rate correlation with burst periods")
        logger.info("   4. Measure system recovery time after bursts")
        logger.info("   5. Compare concurrent user count variations")
        logger.info("")
        logger.info("🎯 Burst Performance Evaluation:")
        logger.info("   • Low P99 during bursts = Good queueing/batching")
        logger.info("   • Fast recovery after bursts = Good resource management")
        logger.info("   • Low error rate during peaks = Adequate capacity")
        logger.info("   • Graceful degradation = Production-ready")
        logger.info("")
        per_request_logger.print_summary()
        logger.info("=" * 80)
        
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Monitor the test
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏁 BENCHMARK STARTING")
    logger.info("=" * 80)
    logger.info(f"Running for {args.duration} seconds ({args.duration // 60} minutes)...")
    logger.info("Generating bursty traffic following Poisson distribution")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 LIVE METRICS (30-second sliding window)")
    logger.info("=" * 80)

    start_time = time.time()
    try:
        while time.time() - start_time < args.duration + 10:  # Extra time for completion
            if not spawner_process.is_alive():
                break
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

