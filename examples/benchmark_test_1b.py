"""
Benchmark Test 1b: RAG Simulation (4096 input / 512 output tokens)

This test evaluates system performance under RAG workloads with large input contexts
and longer responses typical of retrieval-augmented generation systems.

Test Specifications:
- Input tokens: ~4096 per request
- Output tokens: ~512 per request
- Duration: 10-15 minutes
- Concurrency: ~20 parallel sessions
- Rate: Moderate with bursts representing multiple users querying documents
- Focus: Memory load, latency distribution, throughput impact

Usage:
    python examples/benchmark_test_1b.py --host http://localhost:8000 --model llama-3.1-8b
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from multiprocessing import Process, Queue
from typing import Any

from transformers import AutoTokenizer

from llm_locust.clients.openai import OpenAIChatStreamingClient
from llm_locust.core.spawner import start_user_loop
from llm_locust.metrics.collector import MetricsCollector
from llm_locust.metrics.per_request_logger import PerRequestLogger
from llm_locust.utils.prompts import (
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


def create_rag_prompts(
    tokenizer: AutoTokenizer,
    num_prompts: int = 100,
) -> list[dict[str, Any]]:
    """
    Create RAG-style prompts with large input contexts (~4096 tokens).
    
    Combines long document context with specific questions to simulate
    retrieval-augmented generation scenarios.
    """
    prompts: list[dict[str, Any]] = []
    
    # RAG-style prompts combining long context with specific questions
    rag_templates = [
        {
            "context": "Based on the following comprehensive technical documentation about machine learning model deployment and optimization strategies, including detailed explanations of model serving architectures, performance monitoring, and scaling considerations for production environments...",
            "questions": [
                "What are the key performance bottlenecks in this system?",
                "How should we optimize memory usage for large models?",
                "What monitoring strategies are recommended?",
                "Explain the scaling architecture in detail.",
                "What are the security considerations mentioned?",
                "How does the system handle concurrent requests?",
                "What are the recommended deployment strategies?",
                "Explain the caching mechanisms described.",
                "What performance metrics should we track?",
                "How does the system handle model updates?"
            ]
        },
        {
            "context": "Given this extensive research paper on natural language processing advances, covering transformer architectures, attention mechanisms, pre-training strategies, fine-tuning approaches, and evaluation methodologies for various NLP tasks including text classification, question answering, summarization, and generation...",
            "questions": [
                "What are the main architectural innovations described?",
                "How do attention mechanisms work in this context?",
                "What pre-training strategies are most effective?",
                "Explain the fine-tuning methodologies.",
                "What evaluation metrics are recommended?",
                "How do these approaches compare to previous methods?",
                "What are the computational requirements?",
                "What datasets were used in the experiments?",
                "What are the limitations mentioned?",
                "What future research directions are suggested?"
            ]
        },
        {
            "context": "Consider this detailed business analysis report covering market trends, competitive landscape, customer segmentation, pricing strategies, operational efficiency metrics, financial projections, risk assessments, and strategic recommendations for a technology company operating in the AI/ML space...",
            "questions": [
                "What are the key market opportunities identified?",
                "How does our competitive positioning compare?",
                "What customer segments should we focus on?",
                "What pricing strategies are recommended?",
                "How can we improve operational efficiency?",
                "What are the financial projections?",
                "What risks should we be aware of?",
                "What strategic initiatives are suggested?",
                "How should we allocate resources?",
                "What partnerships are recommended?"
            ]
        }
    ]
    
    # Generate prompts by combining contexts with questions
    for i in range(num_prompts):
        template = rag_templates[i % len(rag_templates)]
        question = template["questions"][i % len(template["questions"])]
        
        # Create a long context by repeating and expanding the base context
        base_context = template["context"]
        expanded_context = base_context * 3  # Repeat to increase length
        
        # Add more detailed content to reach ~4096 tokens
        additional_content = f"""
        
        The document continues with extensive technical details, implementation guidelines, 
        code examples, configuration parameters, troubleshooting guides, best practices, 
        case studies, performance benchmarks, scalability considerations, security protocols, 
        monitoring dashboards, alerting systems, backup strategies, disaster recovery plans, 
        compliance requirements, audit trails, user management, access controls, API documentation, 
        integration examples, testing procedures, quality assurance processes, deployment pipelines, 
        version control strategies, documentation standards, training materials, support procedures, 
        maintenance schedules, upgrade procedures, migration guides, optimization techniques, 
        resource allocation strategies, cost analysis, ROI calculations, and strategic recommendations.
        
        Additional sections include detailed technical specifications, architectural diagrams, 
        system requirements, hardware recommendations, software dependencies, configuration files, 
        environment variables, database schemas, API endpoints, authentication mechanisms, 
        authorization policies, data encryption methods, network configurations, load balancing 
        strategies, caching mechanisms, CDN integration, performance tuning guidelines, 
        monitoring tools, logging systems, error handling procedures, exception management, 
        debugging techniques, profiling tools, optimization strategies, and continuous 
        improvement processes.
        """
        
        full_context = expanded_context + additional_content
        prompt_text = f"{full_context}\n\nQuestion: {question}"
        
        # Build chat for tokenization
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        
        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize RAG prompt: {e}")
            continue
        
        # Target ~4096 input tokens (allow some variance)
        if 3500 <= num_tokens <= 4500:
            prompts.append({
                "prompt": prompt_text,
                "num_input_tokens": num_tokens,
                "context_type": template["context"][:50] + "...",
                "question": question,
            })
    
    logger.info(f"Created {len(prompts)} RAG-style prompts with ~4096 input tokens")
    return prompts


def main() -> None:
    """Main entry point for Benchmark Test 1b."""
    parser = argparse.ArgumentParser(
        description="Benchmark Test 1b: RAG Simulation (4096 input / 512 output tokens)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://ollama-test-vllm-benchmark.apps.cluster-njnqr.njnqr.sandbox1049.opentlc.com",
        help="LLM endpoint URL (default: Ollama endpoint)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b-instruct-fp16",
        help="Model name (default: qwen2.5:7b-instruct-fp16)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=20,
        help="Number of concurrent users (default: 20 for RAG simulation)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=0.5,
        help="Users to spawn per second (default: 0.5 for moderate rate)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512 for RAG responses)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=900,  # 15 minutes
        help="Test duration in seconds (default: 900 = 15 minutes)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer to use (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--use-billsum",
        action="store_true",
        help="Use BillSum dataset for long context (alternative to custom RAG prompts)",
    )
    parser.add_argument(
        "--use-infinity-instruct",
        action="store_true",
        help="Use Infinity Instruct dataset for long context (alternative to custom RAG prompts)",
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
        default=5,
        help="Show summary every N requests instead of each request (default: 5)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        default=True,
        help="Include input prompt and output text in CSV (default: True)",
    )
    parser.add_argument(
        "--no-include-text",
        action="store_false",
        dest="include_text",
        help="Disable including text in CSV (metrics only)",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=1000,
        help="Maximum text length to save in CSV (default: 1000 chars for long contexts)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/benchmark_1b_rag_metrics.csv",
        help="Output file for per-request metrics (default: results/benchmark_1b_rag_metrics.csv)",
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
    logger.info("🚀 BENCHMARK TEST 1B: RAG SIMULATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Test Configuration:")
    logger.info(f"  🎯 Endpoint:    {args.host}")
    logger.info(f"  🤖 Model:       {args.model}")
    logger.info(f"  🔧 Tokenizer:   {args.tokenizer}")
    logger.info("")
    logger.info("RAG Workload Profile:")
    logger.info(f"  📊 Input Tokens:  ~4096 per request")
    logger.info(f"  📊 Output Tokens: ~{args.max_tokens} per request")
    logger.info(f"  👥 Users:        {args.users} concurrent sessions")
    logger.info(f"  ⚡ Spawn Rate:   {args.spawn_rate} users/second")
    logger.info(f"  ⏱️  Duration:     {args.duration} seconds ({args.duration//60} minutes)")
    logger.info(f"  📊 Total Reqs:   ~{int(args.users * args.duration * 0.1)} (estimated)")
    logger.info("")
    logger.info("Benchmark Focus:")
    logger.info("  🧠 Memory Load:     Stress-test KV cache growth and GPU memory usage")
    logger.info("  📈 Latency Dist:    Observe how latency scales with large token counts")
    logger.info("  🔄 Throughput:      Identify drop-offs as request size increases")
    logger.info("  🏢 Business Context: Knowledge-base assistants, research copilots")
    logger.info("")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{{prompt}}"

    # Load prompts based on dataset choice
    if args.use_billsum:
        logger.info("📚 Loading BillSum dataset for long context...")
        prompts = load_billsum(
            tokenizer,
            min_input_length=3500,
            max_input_length=4500,
        )
    elif args.use_infinity_instruct:
        logger.info("📚 Loading Infinity Instruct dataset for long context...")
        prompts = load_infinity_instruct(
            tokenizer,
            min_input_length=3500,
            max_input_length=4500,
        )
    else:
        logger.info("📚 Creating custom RAG prompts with ~4096 input tokens...")
        prompts = create_rag_prompts(tokenizer, num_prompts=200)
    
    logger.info(f"✅ Loaded {len(prompts)} prompts for RAG simulation")

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
            include_text=args.include_text,
            max_text_length=args.max_text_length,
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
    logger.info("🏁 BENCHMARK TEST 1B STARTING")
    logger.info("=" * 80)
    logger.info(f"Running RAG simulation for {args.duration} seconds...")
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
            if elapsed % 60 == 0 and elapsed > 0:  # Progress update every 60 seconds
                logger.info(
                    f"⏱️  Progress: {elapsed}/{args.duration}s elapsed "
                    f"({remaining}s remaining)"
                )
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")

    logger.info("")
    logger.info("=" * 80)
    logger.info("⏹️  BENCHMARK TEST 1B COMPLETE")
    logger.info("=" * 80)
    
    # Print per-request summary if enabled
    if per_request_logger:
        per_request_logger.print_summary()
    
    signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
