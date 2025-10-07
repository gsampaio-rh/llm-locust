# LLM Locust Benchmarks

This document contains standardized benchmark tests for evaluating LLM inference endpoint performance under various workload conditions.

## Overview

LLM Locust provides 5 standardized benchmark tests that cover different real-world workloads:

| Test ID | Name | Input/Output | Concurrency | Duration | Focus |
|---------|------|--------------|-------------|----------|-------|
| **1a** | Chat Simulation | 256/128 tokens | 50 users | 10 min | Interactive responsiveness |
| **1b** | RAG Simulation | 4096/512 tokens | 20 users | 15 min | Large context processing |
| **1c** | Code Generation | 512/512 tokens | 30 users | 10 min | Balanced workload |
| **2a** | Constant Rate | 512/256 tokens | 40 users | 20 min | Sustained reliability |
| **2b** | Poisson Rate | 512/256 tokens | Variable | 15 min | Burst handling |

---

## Benchmark Specifications

### ✅ Test 1a: Chat Simulation (256 input / 128 output tokens)

**File:** `examples/benchmark_chat_simulation.py`

#### Objective
Evaluate system performance under short, interactive workloads representative of conversational AI.

#### Workload Profile
- **Input tokens:** ~256 per request
- **Output tokens:** ~128 per request
- **Interaction type:** Compact prompts and concise responses, mimicking natural back-and-forth dialogue
- **Dataset:** ShareGPT (conversational, fixed at benchmark level)

#### Test Parameters
- **Duration:** 5–10 minutes (default: 10 minutes)
  - Long enough to measure stability and response consistency
- **Concurrency:** ~50 parallel chat sessions (default: 50 users)
- **Rate:** Steady conversational pace (1–2 requests per user per minute)
- **Number of Users Simulated:** Dozens of customer or assistant interactions in parallel

#### Benchmark Focus
- **Latency Sensitivity:** Time-to-first-token (TTFT) and p99 latency as indicators of responsiveness
- **Throughput:** Ability to sustain dozens of interactive conversations simultaneously
- **User Experience Impact:** Ensures responses remain conversational

#### Success Criteria
- ✅ TTFT median < 1 second
- ✅ TTFT p99 < 2 seconds
- ✅ Sustained throughput across all sessions
- ✅ No latency degradation over time

#### Business Context
Customer-facing assistants, support bots, or copilots where responsiveness is critical for usability and adoption.

#### Run Example
```bash
python examples/benchmark_chat_simulation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

---

### ✅ Test 1b: RAG Simulation (4096 input / 512 output tokens)

**File:** `examples/benchmark_rag_simulation.py`

#### Objective
Assess performance when handling large input contexts and longer responses typical of retrieval-augmented generation (RAG) systems.

#### Workload Profile
- **Input tokens:** ~4096 per request
- **Output tokens:** ~512 per request
- **Interaction type:** Long-form context ingestion with detailed answers
- **Dataset:** BillSum (long legislative documents from [FiscalNote/billsum](https://huggingface.co/datasets/FiscalNote/billsum))

#### Test Parameters
- **Duration:** 10–15 minutes (default: 15 minutes)
  - Longer runs needed for large context processing
- **Concurrency:** ~20 parallel sessions (default: 20 users)
- **Rate:** Moderate, with bursts representing multiple users querying documents simultaneously
- **Number of Users Simulated:** Enterprise-scale workloads, such as analysts querying knowledge bases

#### Benchmark Focus
- **Memory Load:** Stress-test KV cache growth and GPU memory usage
- **Latency Distribution:** Observe how latency scales with large token counts
- **Throughput Impact:** Identify drop-offs as request size increases

#### Success Criteria
- ✅ TTFT median < 3 seconds (higher due to long context)
- ✅ Throughput stability (no significant degradation)
- ✅ No OOM errors or memory-related failures
- ✅ Consistent tokens/second efficiency

#### Business Context
Knowledge-base assistants, research copilots, or enterprise search systems requiring context-heavy queries.

#### Run Example
```bash
python examples/benchmark_rag_simulation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

#### Focus Areas
- KV cache growth and GPU memory usage
- Latency scaling with large token counts
- Throughput impact vs short contexts
- Memory management efficiency

#### RAG-Specific Notes
- Large contexts stress memory management
- TTFT expected to be higher than Test 1a
- Watch for OOM errors and throughput degradation
- Measure tokens/second efficiency

---

### ✅ Test 1c: Code Generation Simulation (512 input / 512 output tokens)

**File:** `examples/benchmark_code_generation.py`

#### Objective
Benchmark balanced input-output scenarios common in development assistance and code generation.

#### Workload Profile
- **Input tokens:** ~512 per request
- **Output tokens:** ~512 per request
- **Interaction type:** Medium-sized prompts with equally long completions
- **Dataset:** Synthetic code generation prompts (Python, JavaScript, Java, Go tasks)

#### Test Parameters
- **Duration:** 5–10 minutes (default: 10 minutes)
- **Concurrency:** ~30 developer sessions (default: 30 users)
- **Rate:** Constant flow of requests, reflecting active programming cycles
- **Number of Users Simulated:** Teams of developers using AI assistants concurrently

#### Benchmark Focus
- **Balanced Load:** Measures efficiency when both prompt parsing and response generation are significant
- **Latency:** Focus on median and tail latencies for developer workflow smoothness
- **Throughput:** Can the system sustain multiple code completions in parallel?

#### Success Criteria
- ✅ Median latency < 2 seconds
- ✅ P99 latency < 5 seconds
- ✅ Sustained throughput across all sessions
- ✅ No degradation during continuous coding assistance

#### Business Context
AI-powered coding copilots, auto-completion engines, or dev tool integrations where balanced input/output is typical.

#### Run Example
```bash
python examples/benchmark_code_generation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

#### Focus Areas
- Balanced load performance (equal prompt/response sizes)
- Developer workflow smoothness (median/tail latencies)
- Multi-language code generation capability
- Real-time coding assistance feasibility

---

### ✅ Test 2a: Constant Rate (Sustained Load)

**File:** `examples/benchmark_constant_rate.py`

#### Objective
Validate system reliability and performance under continuous, predictable workloads.

#### Workload Profile
- **Input tokens:** ~512 per request
- **Output tokens:** ~256 per request
- **Interaction type:** Steady production-like traffic flow
- **Dataset:** ShareGPT (mixed conversational, fixed at benchmark level)

#### Test Parameters
- **Duration:** 15–20 minutes (default: 20 minutes)
  - Longer duration to reveal long-term degradation trends
- **Concurrency:** ~40 concurrent streams (default: 40 users)
- **Rate:** Fixed at ~2 requests/second across all users
- **Number of Users Simulated:** Dozens of sustained user sessions

#### Benchmark Focus
- **Sustained Performance:** Identify whether latency degrades over time
- **Stability:** Measure throughput consistency and error rates
- **SLA Readiness:** Ensures performance guarantees can be met under steady load

#### Success Criteria
- ✅ No latency degradation over time
- ✅ Throughput consistency (maintains target req/s)
- ✅ Error rate < 0.1% for production readiness
- ✅ Stable P99 latency throughout test

#### Business Context
Enterprise deployments with predictable usage patterns, such as internal productivity copilots or workflow automation tools.

#### Run Example
```bash
python examples/benchmark_constant_rate.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

#### Custom Rate Example
```bash
# Test with different constant rate
python examples/benchmark_constant_rate.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --request-rate 5.0 \
    --users 50 \
    --duration 1800
```

#### Focus Areas
- Long-term performance stability (no degradation)
- Sustained throughput consistency
- SLA compliance under steady load
- Resource utilization patterns
- Memory leak detection

#### Analysis Tips
- Plot latency over time to detect gradual degradation
- Check if actual req/s matches target throughout test
- Compare P50/P90/P99 across different time buckets
- Look for correlation between errors and specific time periods

---

### ✅ Test 2b: Poisson Rate (Bursty Traffic)

**File:** `examples/benchmark_poisson_rate.py`

#### Objective
Evaluate system robustness under irregular, unpredictable bursts of traffic.

#### Workload Profile
- **Input tokens:** ~512 per request
- **Output tokens:** ~256 per request
- **Interaction type:** Requests arrive in sudden spikes, modeled with Poisson distribution
- **Dataset:** ShareGPT (mixed conversational, fixed at benchmark level)

#### Test Parameters
- **Duration:** 10–15 minutes (default: 15 minutes)
  - Long enough to capture multiple burst cycles
- **Concurrency:** Varies dynamically (up to 100 concurrent during bursts)
- **Rate:** Average 2 req/s, peak 10 req/s during bursts (5x multiplier)
- **Number of Users Simulated:** Dozens to hundreds depending on burst profile

#### Benchmark Focus
- **Autoscaling:** Tests system's ability to allocate resources dynamically
- **Queueing & Batching:** Reveals how the system manages traffic spikes
- **Tail Latency:** Identifies user experience risks under peak load

#### Success Criteria
- ✅ Low P99 latency during bursts
- ✅ Fast recovery after burst periods
- ✅ Error rate < 1% even during peaks
- ✅ Graceful degradation under overload

#### Business Context
Real-world enterprise apps with spiky traffic, such as e-commerce assistants during flash sales, or knowledge tools during peak work hours.

#### Run Example
```bash
python examples/benchmark_poisson_rate.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

#### Custom Burst Configuration
```bash
# Test with different burst parameters
python examples/benchmark_poisson_rate.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --average-rate 3.0 \
    --burst-factor 8.0 \
    --max-users 150 \
    --duration 900
```

#### Focus Areas
- Burst handling and queueing mechanisms
- Tail latency during traffic spikes (P99)
- Dynamic resource allocation
- System recovery time post-burst
- Error rates under overload conditions

#### Poisson Distribution Details
- Requests follow exponential inter-arrival times
- Bursts occur periodically (~every 2 minutes)
- Each burst lasts ~30 seconds
- Models real-world unpredictable traffic patterns

#### Analysis Tips
- Plot latency over time to visualize burst impact
- Compare P99 during burst vs normal periods
- Measure system recovery time after bursts
- Correlate error rates with traffic spikes
- Analyze concurrent request counts during peaks

---

## Quick Start

### Basic Usage

```bash
# Run Chat Simulation benchmark (Test 1a)
python examples/benchmark_chat_simulation.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

### Custom Configuration

```bash
# Adjust concurrency and duration
python examples/benchmark_chat_simulation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine tgi \
    --users 100 \
    --duration 1800 \
    --output-dir results
```

### Engine Parameter

The `--engine` parameter is required and used in the output filename:
- `vllm` - vLLM serving engine
- `tgi` - Text Generation Inference
- `ollama` - Ollama
- `openai` - OpenAI API
- Or any custom name for your serving platform

**Output filename format:** `{engine}-{datetime}-{benchmark-id}.csv`

Example: `vllm-20250103-120530-1a-chat-simulation.csv`

---

## Output Format

Each benchmark generates detailed CSV output with per-request metrics:

**File Location:** `results/{engine}-{datetime}-{benchmark-id}.csv`

Example: `results/vllm-20250103-120530-1a-chat-simulation.csv`

### Per-Request Logging
- ✅ **Always enabled** - All benchmarks log every request to CSV by default
- ✅ **Console output enabled** - Individual requests shown in console
- ✅ **Full detail** - Shows all requests (summary_interval = 0)

These settings ensure maximum visibility during benchmark runs.

### CSV Columns
- `request_id` - Unique request identifier
- `timestamp` - Unix timestamp
- `user_id` - User/session identifier
- `user_request_num` - Request number for this user
- `input_tokens` - Number of input tokens
- `output_tokens` - Number of output tokens generated
- `ttft_ms` - Time to First Token (milliseconds)
- `tpot_ms` - Time Per Output Token (milliseconds)
- `end_to_end_s` - Total request time (seconds)
- `total_tokens_per_sec` - Overall token throughput
- `output_tokens_per_sec` - Generation speed
- `status_code` - HTTP status code
- `input_prompt` - Truncated input text (for analysis)
- `output_text` - Truncated output text (for analysis)

---

## Analyzing Results

### Key Metrics to Check

#### 1. Time to First Token (TTFT)
- Median should be < 1s for chat workloads
- P99 should be < 2s for good UX
- Watch for degradation over time

#### 2. Throughput
- Should remain stable throughout test
- No significant drop-offs
- Consistent tokens/second

#### 3. Latency Distribution
- Check P50, P90, P99 percentiles
- Look for long tail issues
- Identify outliers

#### 4. Error Rate
- Should be < 0.1% for production systems
- Check status codes for failure patterns

### Example Analysis (Python)

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/vllm-20250103-120530-1a-chat-simulation.csv')

# Calculate TTFT percentiles
ttft_p50 = df['ttft_ms'].quantile(0.50)
ttft_p90 = df['ttft_ms'].quantile(0.90)
ttft_p99 = df['ttft_ms'].quantile(0.99)

print(f"TTFT P50: {ttft_p50:.1f}ms")
print(f"TTFT P90: {ttft_p90:.1f}ms")
print(f"TTFT P99: {ttft_p99:.1f}ms")

# Check for degradation over time
df['time_bucket'] = pd.cut(df['timestamp'], bins=10)
degradation = df.groupby('time_bucket')['ttft_ms'].mean()
print("\nLatency over time:")
print(degradation)

# Success rate
total = len(df)
success = len(df[df['status_code'] == 200])
print(f"\nSuccess Rate: {(success/total)*100:.2f}%")
```

---

## Best Practices

### Before Running Benchmarks

1. **Warm up the endpoint** - Run a few test requests first
2. **Check resource availability** - Ensure sufficient CPU/GPU/memory
3. **Baseline measurements** - Know your expected performance
4. **Network conditions** - Test from similar network conditions as production

### During Benchmarks

1. **Monitor server metrics** - Watch GPU utilization, memory, etc.
2. **Check for errors** - Watch logs for failures
3. **Avoid other load** - Don't run multiple benchmarks simultaneously
4. **Let it run** - Don't interrupt unless necessary

### After Benchmarks

1. **Save raw results** - Keep CSV files for future analysis
2. **Calculate percentiles** - P50, P90, P99 are critical
3. **Compare to baseline** - Track performance over time
4. **Document findings** - Note any anomalies or issues

---

## Customizing Benchmarks

### Adjusting Token Counts

Edit the benchmark script constants:

```python
TARGET_INPUT_TOKENS = 256    # Adjust input size
TARGET_OUTPUT_TOKENS = 128   # Adjust output size
INPUT_TOKEN_MIN = 200        # Min input range
INPUT_TOKEN_MAX = 300        # Max input range
```

### Changing Datasets

Replace the dataset loader:

```python
# Use Dolly instead of ShareGPT
from llm_locust.utils.prompts import load_databricks_dolly

prompts = load_databricks_dolly(
    tokenizer,
    min_input_length=INPUT_TOKEN_MIN,
    max_input_length=INPUT_TOKEN_MAX,
)
```

### Custom Success Criteria

Modify the validation logic in your analysis scripts to match your SLAs.

---

## Troubleshooting

### "Insufficient prompts found"
- The dataset doesn't have enough prompts in your token range
- Adjust `INPUT_TOKEN_MIN` and `INPUT_TOKEN_MAX`
- Try a different dataset (ShareGPT vs Dolly)

### "Connection errors"
- Check endpoint URL is correct
- Verify endpoint is running and accessible
- Check for firewall/network issues

### "Memory errors"
- Reduce number of concurrent users
- Check server has sufficient memory
- Monitor GPU memory usage

### "Results file missing"
- Ensure output directory exists and is writable
- Check file permissions
- Verify benchmark completed successfully

---

## Contributing

To add new benchmarks:

1. Copy `examples/benchmark_chat_simulation.py` as a template
2. Adjust constants and configuration for your test
3. Update this document with test details
4. Add test specification details
5. Submit a PR with documentation

---

## References

- [Metrics Guide](METRICS_GUIDE.md) - Understanding metrics
- [Architecture](ARCHITECTURE.md) - How the system works
- [Datasets Guide](DATASETS.md) - Supported datasets
- [Results README](../results/README.md) - Understanding results output

---

**Last Updated:** 2025-10-04  
**Version:** 1.0  
**Maintained By:** LLM Locust Team
