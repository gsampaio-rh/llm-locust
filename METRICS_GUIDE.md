# LLM Locust Metrics Collection Guide

This document defines all metrics collected by LLM Locust and maps them to industry-standard LLM benchmarking KPIs.

## Current Metrics Collected

### ✅ Fully Implemented

| **Category** | **Metric** | **Definition** | **Implementation** | **Output** |
|--------------|-----------|----------------|-------------------|-----------|
| **Per-Request Runtime** | Prompt Token Count | Number of input tokens processed | `RequestSuccessLog.num_input_tokens` | Per request |
| | Response Token Count | Number of output tokens generated | Parsed from `result_chunks` | Per request |
| | End-to-End Latency | Total request runtime | `end_time - start_time` | Per request |
| | Response Token Rate | Output tokens per second | `tokens / generation_time` | Per request |
| **Latency** | Time to First Token (TTFT) | Time until first output token | `token_times[0] - start_time` | P50, P90, P99 |
| | Time Per Output Token (TPOT) | Average time per generated token | `generation_time / output_tokens` | P50, P90, P99 |
| | Response Latency | Total response time | `response_time_seconds` | P50, P90, P99 |
| | Tail Latency Variance | Latency stability | P99/P50 ratio | Calculated |
| **Throughput** | Requests per Second (RPS) | Successful requests per second | `requests_per_second` | Aggregate |
| | Failed Requests per Second | Failed requests per second | `failed_requests_per_second` | Aggregate |
| | Aggregate Token Throughput | All tokens/sec across users | `total_output_tokens_per_second` | Aggregate |
| | Empty Response Rate | Responses with no tokens | `total_empty_output_tokens_per_second` | Aggregate |
| **Concurrency** | Active Users | Concurrent simulated users | `active_users` | Real-time |
| | Spawn Rate | User ramp-up rate | Configurable | Per test |
| **Reliability** | Success Rate | Completed requests % | `RPS / (RPS + Failed)` | Calculated |
| | Status Codes | HTTP response codes | `status_code` | Per request |

### 🟡 Partially Implemented (Can be Extended)

| **Category** | **Metric** | **Definition** | **How to Add** | **Priority** |
|--------------|-----------|----------------|----------------|--------------|
| **Per-Request Runtime** | Prompt Evaluation Time | Time processing input tokens | Measure before first token | Medium |
| | Response Generation Time | Time generating output only | `end_time - first_token_time` | Medium |
| | Prompt Token Rate | Input tokens/sec | `input_tokens / prompt_eval_time` | Low |
| **Latency** | Per-Token Latency Distribution | Individual token timings | Store all inter-token deltas | High |
| | Network Latency | Time to establish connection | Measure connection time | Medium |
| **Throughput** | Streaming vs Non-Streaming TPS | Compare streaming modes | Add non-streaming mode | Low |
| | Peak Throughput | Maximum sustained RPS | Track rolling max | Medium |
| **Resource Tracking** | Client CPU/Memory | Load test client resources | Add `psutil` monitoring | Medium |
| **Reliability** | Error Breakdown | Failures by type | Enhanced error categorization | High |
| | Retry Success Rate | Retries that succeeded | Add retry logic | Medium |
| | Timeout Rate | Requests exceeding SLA | Add timeout tracking | High |

### ❌ Not Implemented (Requires Server-Side Access)

| **Category** | **Metric** | **Definition** | **Why Not Available** | **Alternative** |
|--------------|-----------|----------------|----------------------|-----------------|
| **Server Resources** | GPU Utilization | GPU compute % used | Server-side metric | Monitor separately |
| | GPU Memory | VRAM usage | Server-side metric | Monitor separately |
| | CPU Utilization | Server CPU load | Server-side metric | Monitor separately |
| | KV Cache Hit Rate | Cache reuse effectiveness | Engine internal | Not available |
| **Serving Features** | Batch Size | Requests batched together | Engine internal | Not available |
| | Queue Length | Pending requests | Engine internal | Monitor via API if exposed |
| | Continuous Batching | Dynamic batching metrics | Engine-specific | Not available |
| | Speculative Decoding | Speculative decode speedup | Engine-specific | Not available |
| **Multi-System** | Multi-GPU Scaling | Speedup across GPUs | Requires cluster | Run separate tests |
| | Multi-node Scalability | Cross-node performance | Requires cluster | Run separate tests |
| | Load Distribution | Fairness across nodes | Requires instrumentation | Not available |

---

## Detailed Metric Descriptions

### Core Metrics (Always Collected)

#### 1. Time to First Token (TTFT)
**Definition**: Time from request start to receiving the first token  
**Importance**: Critical for user experience - perceived responsiveness  
**Collection**: 
```python
ttft = token_times[0] - start_time  # milliseconds
```
**Output**: P50, P90, P99 percentiles over sliding window

#### 2. Time Per Output Token (TPOT)
**Definition**: Average time per generated token  
**Importance**: Generation speed, streaming smoothness  
**Collection**:
```python
tpot = generation_time / total_output_tokens
```
**Output**: P50, P90, P99 percentiles

#### 3. Requests per Second (RPS)
**Definition**: Successful requests completed per second  
**Importance**: System throughput capacity  
**Collection**: Count successful requests in time window  
**Output**: Aggregate rate

#### 4. Token Throughput
**Definition**: Total output tokens generated per second  
**Importance**: Generation capacity under load  
**Collection**: Sum all tokens across users  
**Output**: Aggregate tokens/sec

#### 5. End-to-End Latency
**Definition**: Total time from request start to completion  
**Importance**: Overall request duration  
**Collection**: `end_time - start_time`  
**Output**: P50, P90, P99 percentiles

### Per-Request Data Available

Every successful request logs:
- `request_id`: Unique identifier
- `timestamp`: Request completion time
- `start_time`: Request start (perf_counter)
- `end_time`: Request end (perf_counter)
- `num_input_tokens`: Input token count
- `result_chunks`: Raw response chunks (bytes)
- `token_times`: Timestamp for each chunk
- `status_code`: HTTP status

Every failed request logs:
- `timestamp`: Failure time
- `start_time`: Request start
- `end_time`: Request end
- `status_code`: HTTP error code

---

## Metrics Output Format

### Real-Time Console Output
```
👥 Users:  10 | 📊 RPS:  12.45 | ❌ Failed:  0.02 | 
⚡ TTFT P50: 234.5ms | P90: 245.2ms | P99: 256.8ms | 
🔤 TPOT:  12.3ms | 🚀 Tokens/s: 1,542.3
```

### Available for Custom Export
All metrics are available via the `MetricsCollector.logging_function` callback:
```python
{
    "active_users": 10,
    "requests_per_second": 12.45,
    "failed_requests_per_second": 0.02,
    "response_time_first_token_ms_quantile_50": 234.5,
    "response_time_first_token_ms_quantile_90": 245.2,
    "response_time_first_token_ms_quantile_99": 256.8,
    "tpot_ms_quantile_50": 12.3,
    "tpot_ms_quantile_90": 15.7,
    "tpot_ms_quantile_99": 19.2,
    "response_time_seconds_quantile_50": 4.2,
    "response_time_seconds_quantile_90": 5.1,
    "response_time_seconds_quantile_99": 6.8,
    "total_output_tokens_per_second": 1542.3,
    "total_empty_output_tokens_per_second": 0.0,
}
```

---

## Extending Metrics Collection

### Adding Custom Metrics

1. **Create a new metric class**:
```python
# In llm_locust/metrics/metrics.py
class CustomMetric(QuantileMetric):
    @property
    def name(self) -> str:
        return "my_custom_metric"
    
    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        # Your metric calculation
        value = calculate_value(request_log)
        self.collect(MetricsLog(
            timestamp=request_log.timestamp,
            data=value,
        ))
```

2. **Add to LLMMetricsList**:
```python
class LLMMetricsList(MetricsList):
    def __init__(self, quantiles: list[int]) -> None:
        super().__init__()
        self.metrics.extend([
            # ... existing metrics ...
            CustomMetric(quantiles),  # Add here
        ])
```

### Adding Per-Request Export

To export individual request data:

```python
# Custom logging function
def save_per_request(log_dict: dict[str, Any]) -> None:
    with open("requests.jsonl", "a") as f:
        json.dump(log_dict, f)
        f.write("\n")

# Use in MetricsCollector
collector = MetricsCollector(
    metrics_queue=metrics_queue,
    model_client=client,
    logging_function=save_per_request,  # Custom function
)
```

---

## Comparison to Industry Standard KPIs

| **Standard KPI** | **LLM Locust Equivalent** | **Coverage** |
|------------------|---------------------------|--------------|
| Load Duration | Not measured | ❌ |
| Prompt Evaluation Time | Can be derived from TTFT | 🟡 |
| Prompt Token Rate | Can be calculated | 🟡 |
| Response Generation Time | `end_time - first_token_time` | ✅ |
| Response Token Rate | Calculated per request | ✅ |
| First Token Latency | TTFT (P50/P90/P99) | ✅ |
| Time Per Output Token | TPOT (P50/P90/P99) | ✅ |
| Aggregate TPS | `total_output_tokens_per_second` | ✅ |
| Requests per Second | `requests_per_second` | ✅ |
| Success Rate | Calculated from RPS + Failed RPS | ✅ |
| Timeout Rate | Can be added | 🟡 |
| GPU Utilization | Requires server monitoring | ❌ |
| Batch Size Efficiency | Engine-specific | ❌ |
| Queue Time | Requires server API | ❌ |
| Streaming Smoothness | TPOT variance | ✅ |

**Legend:**
- ✅ Fully implemented
- 🟡 Partially implemented or can be derived
- ❌ Not available (requires server access)

---

## Recommended Additions

### High Priority
1. **Per-request CSV export** - Individual request logging
2. **Timeout tracking** - SLA compliance monitoring
3. **Error categorization** - Detailed failure analysis
4. **Token-level timing** - Full per-token latency distribution

### Medium Priority
5. **Custom aggregation windows** - Configurable time periods
6. **Percentile configuration** - User-defined percentiles (P95, P99.9, etc.)
7. **Client resource monitoring** - CPU/memory of load generator
8. **Historical tracking** - Time-series data export

### Low Priority
9. **Non-streaming mode** - Compare streaming vs non-streaming
10. **Warm-up period** - Exclude initial requests from metrics
11. **Custom prompt patterns** - Regex-based prompt filtering
12. **Cost estimation** - Tokens per dollar calculations

---

## Usage Examples

### Basic Metrics Collection
```python
from llm_locust import MetricsCollector

collector = MetricsCollector(
    metrics_queue=queue,
    model_client=client,
    metrics_window_size=30,  # 30-second windows
    quantiles=[50, 90, 95, 99],  # P50, P90, P95, P99
)
```

### Custom Metrics Export
```python
def export_to_prometheus(metrics: dict[str, Any]) -> None:
    # Export to Prometheus
    for key, value in metrics.items():
        prometheus_gauge.labels(metric=key).set(value)

collector = MetricsCollector(
    metrics_queue=queue,
    model_client=client,
    logging_function=export_to_prometheus,
)
```

### Analysis Workflow
1. Run load test with LLM Locust
2. Collect real-time metrics (client-side)
3. Monitor server metrics separately (GPU, memory, etc.)
4. Combine for comprehensive analysis
5. Compare across engines (Ollama, TGI, vLLM)

---

## Future Enhancements

See [ARCHITECTURE.md](ARCHITECTURE.md) for planned extensions and [GitHub Issues](../../issues) for tracking implementation progress.

