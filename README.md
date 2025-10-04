# LLM Locust

A specialized load testing tool for Large Language Model inference endpoints with streaming support.

## ✨ Features

- **Streaming-First**: Built for SSE/streaming LLM endpoints
- **Rich Metrics**: TTFT (Time to First Token), TPOT (Time Per Output Token), throughput
- **Multi-Process**: Scales to high concurrency without GIL bottlenecks
- **OpenAI Compatible**: Works with any OpenAI-compatible API
- **Quantile Analysis**: P50, P90, P99 latency percentiles
- **Async/Await**: High-performance async request handling

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Basic usage
python examples/simple_test.py --host https://vllm-test-vllm-benchmark.apps.cluster-njnqr.njnqr.sandbox1049.opentlc.com --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct --users 10
```

## 📦 Installation

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"
```

## 🎯 Benchmarks

Standardized benchmark tests for evaluating LLM inference endpoint performance:

```bash
# Run Chat Simulation benchmark (Test 1a)
# Target: 256 input tokens, 128 output tokens, 50 concurrent users
python examples/benchmark_chat_simulation.py \
    --host http://localhost:8000 \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

**Available Benchmarks:**
- ✅ **Test 1a: Chat Simulation** - Conversational AI workload (256/128 tokens)
- ✅ **Test 1b: RAG Simulation** - Large context processing (4096/512 tokens)
- ✅ **Test 1c: Code Generation** - Balanced workload (512/512 tokens)
- ✅ **Test 2a: Constant Rate** - Sustained load testing (512/256 tokens)
- ✅ **Test 2b: Poisson Rate** - Bursty traffic patterns (512/256 tokens)

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for detailed benchmark documentation.

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Benchmark Guide](docs/BENCHMARKS.md)** - Running standardized benchmarks and detailed specifications
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design, datasets, and internals
- **[Metrics Coverage](docs/METRICS_COVERAGE.md)** - Complete metrics reference and KPI analysis
- **[Changelog](docs/CHANGELOG.md)** - Version history, refactoring summary, and feature additions
- **[Interactive CLI Plan](docs/AGILE_PLAN_INTERACTIVE_CLI.md)** - Future roadmap: "The Great Model Race"

See [`docs/README.md`](docs/README.md) for a complete documentation index.

## 🏗️ Architecture

```
llm_locust/
├── core/       # Load testing engine (User, Spawner, Models)
├── clients/    # LLM client implementations (OpenAI-compatible)
├── metrics/    # Metrics collection and aggregation
└── utils/      # Utilities (prompts, helpers)
```

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **RPS** | Requests completed per second |
| **TTFT** | Time to First Token (P50, P90, P99) |
| **TPOT** | Time Per Output Token (average generation time per token) |
| **Throughput** | Output tokens generated per second |
| **Latency** | Total request completion time |

## 🎯 Example Usage

```python
from llm_locust import OpenAIChatStreamingClient, UserSpawner, MetricsCollector
from transformers import AutoTokenizer
from llm_locust.utils import load_databricks_dolly
from multiprocessing import Process, Queue
import asyncio

# Load tokenizer and prompts
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
prompts = load_databricks_dolly(tokenizer, min_input_length=100, max_input_length=500)

# Create client
client = OpenAIChatStreamingClient(
    base_url="http://localhost:8000",
    prompts=prompts,
    system_prompt=None,
    openai_model_name="llama-3.1-8b",
    tokenizer=tokenizer,
    max_tokens=128,
)

# Setup metrics collection
metrics_queue = Queue()
collector = MetricsCollector(
    metrics_queue=metrics_queue,
    model_client=client,
    metrics_window_size=30,
    quantiles=[50, 90, 99],
)
collector.start_logging()

# Run load test
control_queue = Queue()
spawner = UserSpawner(
    model_client=client,
    metrics_queue=metrics_queue,
    max_user_count=10,
    user_addition_count=1,
    user_addition_time=1.0,
)

# ... run your test ...

collector.stop_logging()
```

## 🔧 Configuration

All components support configuration via:
- Constructor arguments
- Environment variables (for API keys, hosts, etc.)
- Config files (coming soon)

## 🔧 Recent Updates

**✅ Critical Fix (Oct 2025):** Failed requests now logged to CSV files
- Success rate calculations now accurate
- Complete visibility into error patterns
- See [CHANGELOG.md](docs/CHANGELOG.md) for details

## 📝 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check .

# Run type checker
mypy llm_locust/

# Format code
ruff format .
```

## 🤝 Contributing

Contributions welcome! Please:
1. Follow the existing code structure
2. Add type annotations
3. Update tests
4. Run linters before submitting

## 📄 License

Apache 2.0 - See LICENSE file for details

## 🙏 Credits

Inspired by [Locust](https://locust.io/) but specialized for LLM workloads.
