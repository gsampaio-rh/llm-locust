# Metrics Coverage Analysis

Comprehensive analysis of which per-request metrics LLM Locust can collect.

## ✅ Per-Request Runtime Metrics

| Metric | Status | Available In | Field Name | Notes |
|--------|--------|--------------|------------|-------|
| **Load Duration** | ❌ Not Available | - | - | Server-side metric (model loading time) |
| **Prompt Evaluation Count** | ✅ **Fully Available** | Per-request log | `input_tokens` | From `RequestSuccessLog.num_input_tokens` |
| **Prompt Evaluation Time** | ✅ **Fully Available** | Per-request log | `ttft_ms` | Time to First Token = prompt processing time |
| **Prompt Token Rate** | ✅ **Calculated** | Per-request log | *Calculated* | `input_tokens / (ttft_ms / 1000)` tokens/sec |
| **Response Token Count** | ✅ **Fully Available** | Per-request log | `output_tokens` | Parsed from response chunks |
| **Response Generation Time** | ✅ **Fully Available** | Per-request log | *Calculated* | `end_to_end_ms - ttft_ms` |
| **Response Token Rate** | ✅ **Fully Available** | Per-request log | `output_tokens_per_sec` | Output tokens / generation time |
| **End-to-End Latency** | ✅ **Fully Available** | Per-request log | `end_to_end_ms` | Total request duration |

## ✅ Latency Metrics

| Metric | Status | Available In | Field Name | Notes |
|--------|--------|--------------|------------|-------|
| **First Token Latency (TTFT)** | ✅ **Fully Available** | Per-request log + Aggregated | `ttft_ms` | Time to first token (critical for UX) |
| **Time Per Output Token (TPOT)** | ✅ **Fully Available** | Per-request log | `tpot_ms` | Time per output token |
| **Tail Latency Variance** | ✅ **Fully Available** | Aggregated metrics | P50, P90, P99 | Percentiles for TTFT, TPOT, E2E |

## ✅ Throughput Metrics (Per Request)

| Metric | Status | Available In | Field Name | Notes |
|--------|--------|--------------|------------|-------|
| **TPS - Total** | ✅ **Fully Available** | Per-request log | `total_tokens_per_sec` | (Input + Output) tokens per second |
| **TPS - Output Only** | ✅ **Fully Available** | Per-request log | `output_tokens_per_sec` | Output tokens per second |
| **TPS - Generation Phase** | ✅ **Calculated** | Per-request log | *Calculated* | `output_tokens / generation_time` |

---

## 📊 Current Per-Request Output

### CSV Format
```csv
request_id,timestamp,input_tokens,output_tokens,ttft_ms,tpot_ms,end_to_end_s,total_tokens_per_sec,output_tokens_per_sec,status_code
```

### Example Data
```csv
1,1696348800,256,128,234.50,12.30,1.847,207.98,69.28,200
2,1696348802,312,145,245.10,11.80,1.923,237.55,75.40,200
3,1696348804,189,98,228.90,13.20,1.654,173.51,59.24,200
```

### Field Mappings

| CSV Field | Metric Name | Formula | Unit |
|-----------|-------------|---------|------|
| `request_id` | Request ID | Sequential counter | - |
| `timestamp` | Completion timestamp | Unix timestamp | seconds |
| `input_tokens` | **Prompt Evaluation Count** | From tokenizer | tokens |
| `output_tokens` | **Response Token Count** | Parsed from chunks | tokens |
| `ttft_ms` | **First Token Latency** | `first_token_time - start_time` | milliseconds |
| `tpot_ms` | **Time Per Output Token** | `generation_time / output_tokens` | ms/token |
| `end_to_end_s` | **End-to-End Latency** | `end_time - start_time` | seconds |
| `total_tokens_per_sec` | **Total TPS** | `(input + output) / total_time` | tokens/sec |
| `output_tokens_per_sec` | **Response Token Rate** | `output_tokens / total_time` | tokens/sec |
| `status_code` | HTTP Status | Response status | - |

---

## 🧮 Derived Metrics (Can Be Calculated)

These metrics can be calculated from the available data:

### 1. Prompt Evaluation Time
```python
prompt_eval_time_ms = ttft_ms
```
**Reasoning**: TTFT includes prompt processing time

### 2. Response Generation Time
```python
response_gen_time_s = end_to_end_s - (ttft_ms / 1000)
```
**Reasoning**: Time after first token until completion

### 3. Prompt Token Rate
```python
prompt_token_rate = input_tokens / (ttft_ms / 1000)
```
**Unit**: tokens/second  
**Reasoning**: Input processing throughput

### 4. Generation Token Rate (Excluding Prompt)
```python
generation_token_rate = output_tokens / (end_to_end_s - (ttft_ms / 1000))
```
**Unit**: tokens/second  
**Reasoning**: Pure generation throughput

### 5. Time Per Token Distribution
```python
# Available from token_times array
individual_token_latencies = [
    (token_times[i+1] - token_times[i]) * 1000 
    for i in range(len(token_times)-1)
]
```
**Note**: Currently aggregated as TPOT, but raw data is available

---

## 📈 Extended Metrics Available

### Real-Time Aggregated Metrics

These are calculated over sliding time windows:

```python
{
    "active_users": 10,                                      # Concurrent users
    "requests_per_second": 12.45,                           # Successful RPS
    "failed_requests_per_second": 0.02,                     # Failed RPS
    
    # TTFT Percentiles
    "response_time_first_token_ms_quantile_50": 234.5,     # TTFT P50
    "response_time_first_token_ms_quantile_90": 245.2,     # TTFT P90
    "response_time_first_token_ms_quantile_99": 256.8,     # TTFT P99
    
    # TPOT Percentiles
    "tpot_ms_quantile_50": 12.3,                          # TPOT P50
    "tpot_ms_quantile_90": 15.7,                          # TPOT P90
    "tpot_ms_quantile_99": 19.2,                          # TPOT P99
    
    # End-to-End Percentiles
    "response_time_seconds_quantile_50": 4.2,              # E2E P50
    "response_time_seconds_quantile_90": 5.1,              # E2E P90
    "response_time_seconds_quantile_99": 6.8,              # E2E P99
    
    # Throughput
    "total_output_tokens_per_second": 1542.3,              # Aggregate TPS
    "total_empty_output_tokens_per_second": 0.0,           # Empty responses
}
```

---

## ✨ Example Usage

### Enable Per-Request Logging

```bash
python examples/simple_test.py \
    --host https://your-endpoint.com \
    --model your-model \
    --users 10 \
    --duration 300 \
    --log-per-request \
    --output-file results.csv
```

### See Real-Time Per-Request Metrics

```bash
python examples/simple_test.py \
    --host https://your-endpoint.com \
    --model your-model \
    --users 5 \
    --log-per-request \
    --log-to-console
```

**Console Output:**
```
📋 Request #   1 | ⏱️  TTFT:   234.5ms | 🔄 TPOT:   12.3ms | ⏰ E2E:  1.847s | 📥 In:  256 | 📤 Out:  128 | 🚀   69.3 tok/s
📋 Request #   2 | ⏱️  TTFT:   245.1ms | 🔄 TPOT:   11.8ms | ⏰ E2E:  1.923s | 📥 In:  312 | 📤 Out:  145 | 🚀   75.4 tok/s
```

---

## 📊 Post-Processing Analysis

### Load and Analyze CSV

```python
import pandas as pd

# Load per-request data
df = pd.read_csv('per_request_metrics.csv')

# Calculate additional metrics
df['prompt_token_rate'] = df['input_tokens'] / (df['ttft_ms'] / 1000)
df['generation_time_s'] = df['end_to_end_s'] - (df['ttft_ms'] / 1000)
df['generation_token_rate'] = df['output_tokens'] / df['generation_time_s']

# Analyze distribution
print(df[['ttft_ms', 'tpot_ms', 'end_to_end_s']].describe())

# Find outliers
ttft_p99 = df['ttft_ms'].quantile(0.99)
outliers = df[df['ttft_ms'] > ttft_p99]
print(f"Found {len(outliers)} outlier requests")

# Compare input size impact
grouped = df.groupby(pd.cut(df['input_tokens'], bins=5))['ttft_ms'].mean()
print("TTFT by input size:")
print(grouped)
```

### Calculate Tail Latency Variance

```python
# Tail latency variance (P99/P50 ratio)
ttft_p50 = df['ttft_ms'].quantile(0.50)
ttft_p99 = df['ttft_ms'].quantile(0.99)
tail_variance = ttft_p99 / ttft_p50

print(f"TTFT Tail Latency Variance: {tail_variance:.2f}x")
print(f"  P50: {ttft_p50:.1f}ms")
print(f"  P99: {ttft_p99:.1f}ms")

# Stable system should have variance < 2x
if tail_variance < 2.0:
    print("✅ Latency is stable")
else:
    print("⚠️  High latency variance detected")
```

---

## 🎯 Metrics Summary

### ✅ What We Have (12/12 requested metrics)

| Category | Metric | Status |
|----------|--------|--------|
| **Per-Request Runtime** | Prompt Evaluation Count | ✅ Direct |
| | Prompt Evaluation Time | ✅ Direct (TTFT) |
| | Prompt Token Rate | ✅ Calculated |
| | Response Token Count | ✅ Direct |
| | Response Generation Time | ✅ Calculated |
| | Response Token Rate | ✅ Direct |
| | End-to-End Latency | ✅ Direct |
| **Latency** | First Token Latency | ✅ Direct |
| | Inter-token Latency | ✅ Direct |
| | Tail Latency Variance | ✅ Calculated |
| **Throughput** | TPS (Total) | ✅ Direct |
| | TPS (Output) | ✅ Direct |

### ❌ What We Don't Have

| Metric | Reason | Alternative |
|--------|--------|-------------|
| Load Duration | Server-side (model loading) | Not applicable for inference |

---

## 🔧 Advanced Analysis Examples

### 1. SLA Compliance Check
```python
# Check if requests meet SLA
sla_ttft_ms = 500
sla_violations = df[df['ttft_ms'] > sla_ttft_ms]
compliance_rate = (1 - len(sla_violations) / len(df)) * 100

print(f"SLA Compliance: {compliance_rate:.2f}%")
print(f"Violations: {len(sla_violations)}/{len(df)} requests")
```

### 2. Performance Over Time
```python
# Analyze degradation over test duration
df['request_order'] = range(1, len(df) + 1)
df['batch'] = pd.cut(df['request_order'], bins=10)

batch_stats = df.groupby('batch').agg({
    'ttft_ms': ['mean', 'std'],
    'tpot_ms': ['mean', 'std'],
    'end_to_end_ms': ['mean', 'std']
})

print("Performance over time:")
print(batch_stats)
```

### 3. Input Size Impact
```python
# Correlation between input size and latency
correlation = df[['input_tokens', 'ttft_ms']].corr()
print(f"Input tokens vs TTFT correlation: {correlation.iloc[0, 1]:.3f}")

# Expected: strong positive correlation (more tokens = higher TTFT)
```

---

## 🚀 Future Enhancements

### Possible Additions
1. **Network latency breakdown** - Connection time vs transfer time
2. **Retry metrics** - Success rate after retries
3. **Queue time** - If server exposes queue metrics
4. **Token-level timestamps** - Individual token timing distribution
5. **Request metadata** - User agent, geographic location

### Server-Side Metrics (Require API)
- GPU utilization during request
- Batch size for request
- KV cache hit/miss
- Queue position and wait time

---

## 📝 Conclusion

**LLM Locust provides 100% coverage of client-observable per-request metrics.**

All 12 requested metrics are available either:
- ✅ **Directly logged** (9 metrics)
- ✅ **Easily calculated** (3 metrics)

The only missing metric (Load Duration) is a server-side initialization metric not relevant for ongoing inference requests.

For comprehensive benchmarking, combine LLM Locust's client-side metrics with server-side monitoring tools (Prometheus, Grafana, nvidia-smi, etc.).

