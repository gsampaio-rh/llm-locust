# LLM Locust Benchmarks

This directory contains standardized benchmark tests for evaluating LLM inference endpoint performance under various workload conditions.

## Available Benchmarks

### ✅ Test 1a: Chat Simulation
**File:** `examples/benchmark_chat_simulation.py`

**Objective:** Evaluate system performance under short, interactive workloads representative of conversational AI.

**Workload Profile:**
- **Input tokens:** ~256 per request
- **Output tokens:** ~128 per request
- **Interaction type:** Compact prompts and concise responses
- **Duration:** 10 minutes (default)
- **Concurrency:** 50 parallel chat sessions (default)
- **Rate:** Steady conversational pace (1-2 requests per user per minute)

**Success Criteria:**
- TTFT median < 1 second
- TTFT p99 < 2 seconds
- Sustained throughput across all sessions
- No latency degradation over time

**Use Cases:**
Customer-facing assistants, support bots, copilots where responsiveness is critical.

**Dataset:** ShareGPT (conversational, fixed at benchmark level)

**Run:**
```bash
python examples/benchmark_chat_simulation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

---

### ✅ Test 1b: RAG Simulation
**File:** `examples/benchmark_rag_simulation.py`

**Objective:** Assess performance when handling large input contexts and longer responses typical of retrieval-augmented generation (RAG) systems.

**Workload Profile:**
- **Input tokens:** ~4096 per request
- **Output tokens:** ~512 per request
- **Interaction type:** Long-form context ingestion with detailed answers
- **Duration:** 15 minutes (default)
- **Concurrency:** 20 parallel sessions (default)
- **Rate:** Moderate pace representing document querying workloads

**Success Criteria:**
- TTFT median < 3 seconds (higher due to long context)
- Throughput stability (no significant degradation)
- No OOM errors or memory-related failures
- Consistent tokens/second efficiency

**Use Cases:**
Knowledge-base assistants, research copilots, enterprise search systems requiring context-heavy queries.

**Dataset:** BillSum (long legislative documents from [FiscalNote/billsum](https://huggingface.co/datasets/FiscalNote/billsum), fixed at benchmark level)

**Run:**
```bash
python examples/benchmark_rag_simulation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

**Focus Areas:**
- KV cache growth and GPU memory usage
- Latency scaling with large token counts
- Throughput impact vs short contexts
- Memory management efficiency

---

### ✅ Test 1c: Code Generation Simulation
**File:** `examples/benchmark_code_generation.py`

**Objective:** Benchmark balanced input-output scenarios common in development assistance and code generation.

**Workload Profile:**
- **Input tokens:** ~512 per request
- **Output tokens:** ~512 per request
- **Interaction type:** Medium-sized prompts with equally long completions
- **Duration:** 10 minutes (default)
- **Concurrency:** 30 developer sessions (default)
- **Rate:** Constant flow reflecting active programming cycles

**Success Criteria:**
- Median latency < 2 seconds
- P99 latency < 5 seconds
- Sustained throughput across all sessions
- No degradation during continuous coding assistance

**Use Cases:**
AI-powered coding copilots, auto-completion engines, dev tool integrations where balanced input/output is typical.

**Dataset:** Synthetic code generation prompts (Python, JavaScript, Java, Go tasks)

**Run:**
```bash
python examples/benchmark_code_generation.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

**Focus Areas:**
- Balanced load performance (equal prompt/response sizes)
- Developer workflow smoothness (median/tail latencies)
- Multi-language code generation capability
- Real-time coding assistance feasibility

---

### ✅ Test 2a: Constant Rate (Sustained Load)
**File:** `examples/benchmark_constant_rate.py`

**Objective:** Validate system reliability and performance under continuous, predictable workloads.

**Workload Profile:**
- **Input tokens:** ~512 per request
- **Output tokens:** ~256 per request
- **Interaction type:** Steady production-like traffic flow
- **Duration:** 20 minutes (default)
- **Concurrency:** 40 concurrent streams (default)
- **Rate:** Fixed at 2 requests/second across all users

**Success Criteria:**
- No latency degradation over time
- Throughput consistency (maintains target req/s)
- Error rate < 0.1% for production readiness
- Stable P99 latency throughout test

**Use Cases:**
Enterprise deployments with predictable usage patterns, such as internal productivity copilots or workflow automation tools.

**Dataset:** ShareGPT (mixed conversational, fixed at benchmark level)

**Run:**
```bash
python examples/benchmark_constant_rate.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

**Custom Rate:**
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

**Focus Areas:**
- Long-term performance stability (no degradation)
- Sustained throughput consistency
- SLA compliance under steady load
- Resource utilization patterns
- Memory leak detection

**Analysis Tips:**
- Plot latency over time to detect gradual degradation
- Check if actual req/s matches target throughout test
- Compare P50/P90/P99 across different time buckets
- Look for correlation between errors and specific time periods

---

### ✅ Test 2b: Poisson Rate (Bursty Traffic)
**File:** `examples/benchmark_poisson_rate.py`

**Objective:** Evaluate system robustness under irregular, unpredictable bursts of traffic.

**Workload Profile:**
- **Input tokens:** ~512 per request
- **Output tokens:** ~256 per request
- **Interaction type:** Requests arrive in sudden spikes, modeled with Poisson distribution
- **Duration:** 15 minutes (default)
- **Concurrency:** Varies dynamically (up to 100 concurrent during bursts)
- **Rate:** Average 2 req/s, peak 10 req/s during bursts (5x multiplier)

**Success Criteria:**
- Low P99 latency during bursts
- Fast recovery after burst periods
- Error rate < 1% even during peaks
- Graceful degradation under overload

**Use Cases:**
Real-world enterprise apps with spiky traffic, such as e-commerce assistants during flash sales, or knowledge tools during peak work hours.

**Dataset:** ShareGPT (mixed conversational, fixed at benchmark level)

**Run:**
```bash
python examples/benchmark_poisson_rate.py \
    --host https://your-llm-endpoint.com \
    --model your-model-name \
    --engine vllm \
    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
```

**Custom Burst Configuration:**
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

**Focus Areas:**
- Burst handling and queueing mechanisms
- Tail latency during traffic spikes (P99)
- Dynamic resource allocation
- System recovery time post-burst
- Error rates under overload conditions

**Poisson Distribution Details:**
- Requests follow exponential inter-arrival times
- Bursts occur periodically (~every 2 minutes)
- Each burst lasts ~30 seconds
- Models real-world unpredictable traffic patterns

**Analysis Tips:**
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
    --model Qwen/Qwen2.5-7B-Instruct \
    --engine vllm \
    --tokenizer Qwen/Qwen2.5-7B-Instruct
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

**Per-Request Logging:**
- ✅ **Always enabled** - All benchmarks log every request to CSV by default
- ✅ **Console output enabled** - Individual requests shown in console
- ✅ **Full detail** - Shows all requests (summary_interval = 0)

These settings ensure maximum visibility during benchmark runs.

**Columns:**
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

1. **Time to First Token (TTFT)**
   - Median should be < 1s for chat workloads
   - P99 should be < 2s for good UX
   - Watch for degradation over time

2. **Throughput**
   - Should remain stable throughout test
   - No significant drop-offs
   - Consistent tokens/second

3. **Latency Distribution**
   - Check P50, P90, P99 percentiles
   - Look for long tail issues
   - Identify outliers

4. **Error Rate**
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
3. Update this README with test details
4. Submit a PR with test specification from `docs/TESTS.md`

---

## References

- [Test Specifications](../docs/TESTS.md) - Detailed benchmark requirements
- [Metrics Guide](../docs/METRICS_GUIDE.md) - Understanding metrics
- [Architecture](../docs/ARCHITECTURE.md) - How the system works
- [Results README](README.md) - Understanding results output

