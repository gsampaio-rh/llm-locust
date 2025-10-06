"""
Benchmark Test 1c: Code Generation Simulation (80-200 input / 512 output tokens)

Objective:
    Benchmark realistic code generation scenarios common in development assistance.
    Real-world code generation prompts are typically concise (80-200 tokens),
    with longer generated outputs.

Workload Profile:
    - Input tokens: 80-200 per request (realistic code prompts)
    - Output tokens: ~512 per request
    - Interaction type: Concise prompts with substantial code completions

Test Parameters:
    - Duration: 5-10 minutes (default: 600s)
    - Concurrency: ~30 developer sessions
    - Rate: Constant flow of requests, reflecting active programming cycles

Benchmark Focus:
    - Generation Performance: Measures efficiency with short prompts but substantial output
    - Latency: Focus on median and tail latencies for developer workflow smoothness
    - Throughput: Can the system sustain multiple code completions in parallel?
    - Real-world Simulation: Mimics actual code assistant usage patterns

Business Context:
    AI-powered coding copilots, auto-completion engines, or dev tool integrations 
    where balanced input/output is typical.

Usage:
    python examples/benchmark_code_generation.py \\
        --host https://your-llm-endpoint.com \\
        --model your-model-name \\
        --engine vllm \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct
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
from llm_locust.utils.prompts import create_single_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark Test 1c Constants
BENCHMARK_NAME = "Code Generation Simulation (Test 1c)"
BENCHMARK_ID = "1c-code-generation"
DATASET_NAME = "synthetic_code"  # Synthetic code generation prompts
TARGET_INPUT_TOKENS = 512
TARGET_OUTPUT_TOKENS = 512
DEFAULT_DURATION = 600  # 10 minutes
DEFAULT_USERS = 30
DEFAULT_SPAWN_RATE = 3.0  # Users per second
INPUT_TOKEN_MIN = 80   # Adjusted for realistic code prompts
INPUT_TOKEN_MAX = 200  # Code prompts are typically concise

# Per-request logging defaults
LOG_TO_CONSOLE = True
SUMMARY_INTERVAL = 0  # 0 = show all requests

# Code generation prompt templates
CODE_GENERATION_PROMPTS = [
    # Python tasks
    "Write a Python function that implements a binary search algorithm on a sorted list. Include proper error handling, type hints, and docstrings. The function should return the index of the target element or -1 if not found. Also include unit tests using pytest.",
    
    "Create a Python class for a RESTful API client that handles authentication, rate limiting, and retry logic. Include methods for GET, POST, PUT, and DELETE requests. Use the requests library and implement exponential backoff for retries.",
    
    "Implement a Python decorator that measures function execution time and logs it. The decorator should handle both synchronous and asynchronous functions, and include options for custom logging levels and formatting.",
    
    "Write a Python script that parses a large CSV file in chunks, performs data validation and transformation, and writes the results to a PostgreSQL database using SQLAlchemy. Include proper error handling and progress tracking.",
    
    # JavaScript/TypeScript tasks
    "Create a React component for a data table with sorting, filtering, and pagination. Use TypeScript, include proper prop types, and implement virtualization for large datasets. Add unit tests using Jest and React Testing Library.",
    
    "Write a Node.js Express middleware for JWT authentication. Include token validation, refresh token logic, and role-based access control. Handle edge cases and provide proper error responses with HTTP status codes.",
    
    "Implement a Redux slice for managing user authentication state, including login, logout, and token refresh actions. Use Redux Toolkit and TypeScript. Include selectors and proper action creators with error handling.",
    
    # Java tasks
    "Create a Java Spring Boot service for managing user accounts with CRUD operations. Include validation, exception handling, and integration with a PostgreSQL database using JPA. Add unit tests with JUnit 5 and Mockito.",
    
    "Write a Java class that implements the Observer design pattern for a real-time notification system. Include proper thread safety, documentation, and example usage showing multiple observers subscribing to events.",
    
    # Go tasks
    "Implement a Go HTTP server with middleware for logging, CORS, and request timeout. Include graceful shutdown, context propagation, and structured logging using zap. Add integration tests using httptest.",
    
    "Write a Go function that implements a connection pool for database connections. Include proper resource management, health checks, and configurable pool size. Use channels for synchronization and context for cancellation.",
    
    # System design tasks
    "Design and implement a rate limiter in Python using the token bucket algorithm. Include Redis for distributed rate limiting across multiple servers. Provide both sync and async implementations with proper testing.",
    
    "Create a caching layer in Python that supports multiple backends (Redis, Memcached, in-memory). Implement cache invalidation strategies, TTL support, and cache-aside pattern. Include benchmarks comparing performance.",
    
    # Data structures & algorithms
    "Implement a Trie data structure in Python for efficient string prefix searches. Include methods for insertion, search, deletion, and auto-completion. Optimize for both time and space complexity with detailed comments.",
    
    "Write a Python implementation of a LRU (Least Recently Used) cache using a combination of a doubly-linked list and hash map. Include thread-safety, proper capacity management, and comprehensive unit tests.",
    
    # API & Integration tasks
    "Create a Python client for interacting with the Stripe API. Include methods for creating customers, charges, and subscriptions. Handle webhooks, implement idempotency, and add retry logic for failed requests.",
    
    "Write a Python script that integrates with the GitHub API to analyze repository metrics. Fetch commit history, pull requests, and issues. Generate a comprehensive report with statistics and visualizations using matplotlib.",
    
    # Database tasks
    "Implement a Python script for database migration using Alembic. Include schema changes, data migrations, and rollback capabilities. Add validation to ensure data integrity before and after migration.",
    
    "Create a Python class for building complex SQL queries dynamically. Support WHERE clauses with multiple conditions, JOINs, GROUP BY, and HAVING. Include SQL injection prevention and query optimization hints.",
    
    # Testing & Quality
    "Write a pytest fixture that sets up a test database with sample data. Include factories for creating test objects, cleanup logic, and utilities for asserting database state. Support both PostgreSQL and SQLite.",
    
    "Implement a Python script for load testing an API endpoint. Use asyncio for concurrent requests, measure latency percentiles (P50, P90, P99), and generate a detailed performance report with graphs.",
]


def generate_code_prompts(tokenizer: AutoTokenizer, target_tokens: int = 512) -> list[dict]:
    """
    Generate code generation prompts targeting specific token count.
    
    Args:
        tokenizer: Tokenizer for counting tokens
        target_tokens: Target token count for prompts
        
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    system_prompt = (
        "You are an expert software engineer and coding assistant. "
        "Provide complete, production-ready code with proper error handling, "
        "documentation, and tests. Follow best practices and write clean, "
        "maintainable code."
    )
    
    # Test tokenization method first
    test_chat = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
    ]
    
    use_chat_template = True
    try:
        _ = tokenizer.apply_chat_template(
            test_chat,
            tokenize=True,
            add_generation_prompt=True,
        )
        logger.info("✅ Using chat template for tokenization")
    except Exception as e:
        logger.warning(f"⚠️  Chat template not working: {e}")
        logger.warning("   Falling back to direct tokenization")
        use_chat_template = False
    
    # Track token counts for debugging
    all_token_counts = []
    
    for prompt_text in CODE_GENERATION_PROMPTS:
        # Create multiple variations by adding context
        variations = [
            prompt_text,
            f"Context: Building a production system.\n\n{prompt_text}",
            f"Requirements:\n- Follow PEP 8 / best practices\n- Include comprehensive error handling\n- Add type hints and documentation\n\nTask: {prompt_text}",
        ]
        
        for variation in variations:
            try:
                if use_chat_template:
                    chat = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": variation},
                    ]
                    num_tokens = len(
                        tokenizer.apply_chat_template(
                            chat,
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                    )
                else:
                    # Fallback: tokenize combined text
                    combined_text = f"{system_prompt}\n\n{variation}"
                    num_tokens = len(tokenizer.encode(combined_text))
                
                # Track all token counts
                all_token_counts.append(num_tokens)
                
                # Accept prompts within range
                if INPUT_TOKEN_MIN <= num_tokens <= INPUT_TOKEN_MAX:
                    prompts.append({
                        "prompt": variation,
                        "num_input_tokens": num_tokens,
                        "system_prompt": system_prompt,
                    })
            except Exception as e:
                logger.error(f"Failed to tokenize prompt: {e}")
                logger.error(f"Prompt preview: {variation[:100]}...")
                continue
    
    # Report token count statistics
    if all_token_counts:
        min_tokens = min(all_token_counts)
        max_tokens = max(all_token_counts)
        avg_tokens = sum(all_token_counts) / len(all_token_counts)
        logger.info(f"   Token count range: {min_tokens} - {max_tokens} (avg: {avg_tokens:.0f})")
        logger.info(f"   Target range: {INPUT_TOKEN_MIN} - {INPUT_TOKEN_MAX}")
        
        if len(prompts) == 0:
            logger.error(f"   ⚠️  No prompts match target range!")
            logger.error(f"   Consider adjusting INPUT_TOKEN_MIN/MAX or using shorter/longer prompts")
    
    logger.info(f"Generated {len(prompts)} code generation prompts")
    return prompts


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
        default="Qwen/Qwen2.5-7B-Instruct",
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
    logger.info(f"  • Input Token Range:    {INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX} tokens (realistic code prompts)")
    logger.info(f"  • Target Output Tokens: ~{TARGET_OUTPUT_TOKENS} tokens")
    logger.info(f"  • Workload Type:        Code Generation (short prompt, long output)")
    logger.info(f"  • Success Criteria:     Median latency <2s, P99 <5s")
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
    logger.info("  • Code Generation Performance (short input, long output)")
    logger.info("  • Median & Tail Latency (P50, P90, P99)")
    logger.info("  • Sustained Throughput")
    logger.info("  • Developer Workflow Smoothness")
    logger.info("  • Real-world Code Assistant Simulation")
    logger.info("")
    logger.info("💡 Code Generation Context:")
    logger.info("  • Simulates real-world coding assistant usage")
    logger.info("  • Mix of Python, JavaScript, Java, Go tasks")
    logger.info("  • Concise prompts with substantial code generation")
    logger.info("  • Tests model performance on technical content")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Load tokenizer
    logger.info(f"📦 Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Debug tokenizer info
    logger.info(f"   Tokenizer type: {type(tokenizer).__name__}")
    logger.info(f"   Has chat template: {bool(tokenizer.chat_template)}")
    if tokenizer.chat_template:
        logger.info(f"   Chat template preview: {str(tokenizer.chat_template)[:100]}...")

    # Generate code prompts
    logger.info(f"🔧 Generating {DATASET_NAME.upper()} prompts...")
    logger.info(f"   Token range: {INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX} tokens")
    logger.info(f"   (Realistic code generation prompts are typically concise)")
    
    prompts = generate_code_prompts(tokenizer, TARGET_INPUT_TOKENS)
    
    if len(prompts) < 10:
        logger.error(f"❌ Insufficient prompts generated ({len(prompts)}). Need at least 10.")
        logger.error("   This should not happen with synthetic prompts. Check tokenizer.")
        sys.exit(1)
    
    logger.info(f"✅ Generated {len(prompts)} code generation prompts")
    avg_input_tokens = sum(p["num_input_tokens"] for p in prompts) / len(prompts)
    logger.info(f"   Average input tokens: {avg_input_tokens:.0f}")

    # Create client
    system_prompt = prompts[0].get("system_prompt") if prompts else None
    client = OpenAIChatStreamingClient(
        base_url=args.host,
        prompts=prompts,
        system_prompt=system_prompt,
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
        logger.info("   1. Analyze code generation performance (short input, long output)")
        logger.info("   2. Review median latency (target: <2s for smooth workflow)")
        logger.info("   3. Check tail latency (P99 target: <5s)")
        logger.info("   4. Compare throughput vs Test 1a (chat) and Test 1b (RAG)")
        logger.info("   5. Assess suitability for real-time coding assistance")
        logger.info("   6. Review token efficiency: input vs output ratio")
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

