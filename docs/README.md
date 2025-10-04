# LLM Locust Documentation

Welcome to the LLM Locust documentation! This directory contains comprehensive guides for using and extending the load testing framework.

## 📚 Documentation Index

### Getting Started
- **[Main README](../README.md)** - Quick start and basic usage
- **[Examples](../examples/)** - Working code examples

### Core Documentation

#### [Architecture Guide](ARCHITECTURE.md)
Detailed explanation of the system design, components, and communication patterns.

**Topics covered:**
- Multi-process architecture
- AsyncIO concurrency model
- Metrics pipeline
- **Dataset architecture** (Dolly, ShareGPT, BillSum, Custom)
- Extension points
- Performance characteristics
- Best practices

**Read this if you want to:**
- Understand how the system works internally
- Choose the right dataset for your test
- Load and mix datasets
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

#### [Changelog](CHANGELOG.md)
Complete version history, feature additions, and refactoring details.

**Topics covered:**
- Recent changes and features (Unreleased, v0.2.0)
- YAML cost integration feature
- Failed request tracking fix
- Historical refactoring (v0.1.0)
- Migration guide for existing users

**Read this if you want to:**
- See what's changed in each version
- Understand new features like YAML cost integration
- Migrate from older versions
- Track project evolution

#### [Interactive CLI Plan](AGILE_PLAN_INTERACTIVE_CLI.md)
Future roadmap for interactive "racing" CLI with live TUI.

**Topics covered:**
- "The Great Model Race" concept
- 4-sprint agile plan (8 weeks)
- 27 user stories with acceptance criteria
- Live request/response inspector
- Achievements, replays, and sharing

**Read this if you want to:**
- See future direction of the project
- Contribute to interactive CLI development
- Understand the product vision

---

## 📖 Quick Navigation

### By Use Case

**I want to run a basic load test:**
→ Start with [Main README](../README.md) and [examples/simple_test.py](../examples/simple_test.py)

**I want to run standardized benchmarks:**
→ See [Benchmark Guide](BENCHMARKS.md) (includes all specifications)

**I want to understand the metrics:**
→ Read [Metrics Coverage](METRICS_COVERAGE.md) and [Architecture](ARCHITECTURE.md)

**I want to add custom metrics:**
→ See [Architecture Guide](ARCHITECTURE.md) → Extension Points

**I want to analyze results:**
→ See [Metrics Coverage](METRICS_COVERAGE.md) → Post-Processing Analysis

**I want to understand the codebase:**
→ Read [Architecture Guide](ARCHITECTURE.md)

**I'm migrating from the original:**
→ Read [Changelog](CHANGELOG.md) → Historical Context section

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

