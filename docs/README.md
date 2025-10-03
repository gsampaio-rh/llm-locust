# LLM Locust Documentation

Welcome to the LLM Locust documentation! This directory contains comprehensive guides for using and extending the load testing framework.

## 📚 Documentation Index

### Getting Started
- **[Main README](../README.md)** - Quick start and basic usage
- **[Examples](../examples/)** - Working code examples

### Core Documentation

#### [Datasets Guide](DATASETS.md)
Guide to supported prompt datasets and how to use them.

**Topics covered:**
- Databricks Dolly 15k
- ShareGPT conversational dataset
- Custom dataset loading
- Dataset selection for different use cases
- Caching and performance

**Read this if you want to:**
- Choose the right dataset for your test
- Load ShareGPT or custom datasets
- Mix multiple datasets
- Understand dataset characteristics

#### [Architecture Guide](ARCHITECTURE.md)
Detailed explanation of the system design, components, and communication patterns.

**Topics covered:**
- Multi-process architecture
- AsyncIO concurrency model
- Metrics pipeline
- Extension points
- Performance characteristics
- Best practices

**Read this if you want to:**
- Understand how the system works internally
- Extend the framework with custom components
- Debug performance issues
- Contribute to the codebase

#### [Metrics Coverage Analysis](METRICS_COVERAGE.md)
Detailed analysis of which industry-standard KPIs are available.

**Topics covered:**
- Per-request runtime metrics
- Latency metrics (TTFT, TPOT, E2E)
- Throughput metrics (TPS)
- Metric mapping to standard KPIs
- Data analysis examples

**Read this if you want to:**
- Know exactly what metrics you can collect
- Compare against benchmark requirements
- Perform post-test analysis
- Calculate derived metrics

### Reference Documentation

#### [Benchmark Guide](BENCHMARKS.md)
Complete guide to running standardized benchmark tests with examples and analysis.

**Topics covered:**
- Available benchmarks and how to run them
- Output format and file naming
- Analyzing results with example code
- Best practices and troubleshooting
- Customizing benchmarks

**Read this if you want to:**
- Run standardized performance tests
- Understand benchmark output files
- Analyze benchmark results
- Customize benchmarks for your needs

#### [Benchmark Specifications](TESTS.md)
Detailed technical specifications for each benchmark test.

**Topics covered:**
- Chat Simulation (Test 1a) - 256/128 tokens
- RAG Simulation (Test 1b) - 4096/512 tokens
- Code Generation (Test 1c) - 512/512 tokens
- Constant Rate Testing (Test 2a)
- Poisson Rate Testing (Test 2b)

**Read this if you want to:**
- Understand benchmark requirements and objectives
- Design custom benchmarks based on specs
- Compare against industry standards

#### [Refactoring Summary](REFACTORING_SUMMARY.md)
Complete history of changes from the original repository.

**Topics covered:**
- What was removed (WebUI)
- What was reorganized (package structure)
- What was improved (type safety, error handling)
- Migration guide for existing users

**Read this if you:**
- Are familiar with the original llm-locust
- Want to understand the changes
- Need to migrate existing code

---

## 📖 Quick Navigation

### By Use Case

**I want to run a basic load test:**
→ Start with [Main README](../README.md) and [examples/simple_test.py](../examples/simple_test.py)

**I want to run standardized benchmarks:**
→ See [Benchmark Specifications](TESTS.md) and [Benchmark Guide](BENCHMARKS.md)

**I want to understand the metrics:**
→ Read [Metrics Guide](METRICS_GUIDE.md)

**I want to add custom metrics:**
→ See [Architecture Guide](ARCHITECTURE.md) → Extension Points

**I want to analyze results:**
→ See [Metrics Coverage](METRICS_COVERAGE.md) → Post-Processing Analysis

**I want to understand the codebase:**
→ Read [Architecture Guide](ARCHITECTURE.md)

**I'm migrating from the original:**
→ Read [Refactoring Summary](REFACTORING_SUMMARY.md)

---

## 🔍 Quick Reference

### Key Metrics Definitions

| Metric | Definition | Unit |
|--------|------------|------|
| **TTFT** | Time to First Token | milliseconds |
| **TPOT** | Time Per Output Token | milliseconds/token |
| **E2E** | End-to-End Latency | seconds |
| **RPS** | Requests per Second | requests/second |
| **TPS** | Tokens per Second | tokens/second |

### Package Structure

```
llm_locust/
├── core/          # User simulation and spawning
├── clients/       # LLM client implementations (OpenAI-compatible)
├── metrics/       # Metrics collection, aggregation, and logging
└── utils/         # Utilities (prompts, helpers)
```

### Common Commands

```bash
# Run basic test
python examples/simple_test.py --host http://localhost:8000 --model llama-3.1-8b --users 10

# Enable per-request logging
python examples/simple_test.py --host http://localhost:8000 --model llama-3.1-8b --log-per-request

# Show per-request metrics in console
python examples/simple_test.py --host http://localhost:8000 --model llama-3.1-8b --log-per-request --log-to-console

# Run type checking
mypy llm_locust/

# Run linter
ruff check .
```

---

## 🤝 Contributing

See the [Architecture Guide](ARCHITECTURE.md) for information on extending the framework and contributing code.

## 📄 License

Apache 2.0 - See [LICENSE](../LICENSE) file for details.

