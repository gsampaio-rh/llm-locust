# Architecture

## Overview

LLM Locust is a specialized load testing tool designed for streaming LLM endpoints. It follows a **multi-process, async-first architecture** to maximize concurrency and minimize GIL bottlenecks.

## Design Principles

1. **Streaming-First**: Built specifically for SSE/streaming responses
2. **Async/Await**: High-performance request handling using `aiohttp`
3. **Multi-Process**: Separate processes for user simulation and metrics collection
4. **Type-Safe**: Comprehensive type annotations throughout
5. **Immutable Data**: Use of frozen dataclasses for message passing
6. **Extensible**: Easy to add new metrics, clients, and prompt sources

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Main Process                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           MetricsCollector (Thread)                  │  │
│  │  - Aggregates metrics from queue                     │  │
│  │  - Calculates quantiles (P50, P90, P99)             │  │
│  │  - Logs metrics every N seconds                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ▲                                  │
│                           │ Queue (IPC)                      │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────┐
│                  User Spawner Process                        │
│                           │                                  │
│  ┌───────────────────────▼────────────────────────────┐    │
│  │            UserSpawner (AsyncIO)                    │    │
│  │  - Dynamically spawns/stops users                   │    │
│  │  - Manages spawn rate and ramp-up                   │    │
│  │  - Sends control signals                            │    │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │                                       │
│         ┌────────────┼────────────┐                         │
│         │            │             │                         │
│    ┌────▼───┐   ┌───▼────┐   ┌───▼────┐                    │
│    │ User 1 │   │ User 2 │...│ User N │                     │
│    │(Task)  │   │ (Task) │   │ (Task) │                     │
│    └────┬───┘   └───┬────┘   └───┬────┘                    │
│         │           │             │                          │
│         └───────────┼─────────────┘                          │
│                     │                                        │
│                     │ HTTP/SSE Streaming                     │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   LLM Serving Endpoint   │
          │  (OpenAI-compatible API) │
          └─────────────────────────┘
```

## Component Details

### Core Components

#### 1. User (`llm_locust/core/user.py`)
- **Purpose**: Simulates a single concurrent client
- **Concurrency**: Each user is an `asyncio.Task`
- **Behavior**: Continuously sends requests until stopped
- **Metrics**: Records TTFT, TPOT, throughput for each request
- **Error Handling**: Retries and error logging with context

**Key Features:**
- Non-blocking async request loop
- Per-token timing collection
- Graceful shutdown via `stop()` method
- Structured error logging with context

#### 2. UserSpawner (`llm_locust/core/spawner.py`)
- **Purpose**: Manages user lifecycle
- **Process**: Runs in separate process to avoid GIL
- **Dynamic Scaling**: Can adjust user count during test
- **Spawn Rate**: Gradual ramp-up/ramp-down

**Key Features:**
- Responsive to control queue messages
- Graceful user addition/removal
- No abrupt termination of in-flight requests
- Real-time user count reporting

#### 3. MetricsCollector (`llm_locust/metrics/collector.py`)
- **Purpose**: Aggregates and reports metrics
- **Threading**: Two background threads:
  - Collection thread: Reads from metrics queue
  - Reporting thread: Calculates and logs periodically
- **Sliding Window**: Configurable time window for aggregation

**Key Features:**
- Non-blocking queue consumption
- Quantile calculation using numpy
- Customizable logging function
- Thread-safe metric collection

### Client Implementations

#### OpenAIChatStreamingClient (`llm_locust/clients/openai.py`)
- **API**: OpenAI Chat Completions (streaming)
- **Format**: Server-Sent Events (SSE)
- **Caching**: Response chunk parsing cache
- **Random Selection**: Randomly selects from prompt pool

**Key Features:**
- SSE streaming parser
- Token-level timing capture
- Chunk-to-token conversion
- Support for custom system prompts

### Data Models

All data models use `frozen=True` dataclasses for immutability and safe IPC:

- **RequestSuccessLog**: Complete request metrics
- **RequestFailureLog**: Failed request info
- **ErrorLog**: Error context with type and metadata
- **Control Messages**: `TriggerShutdown`, `SetUserInfo`, etc.

### Metrics Pipeline

```
Request → User → Queue → Collector → Aggregation → Logging
   │                                      │
   └─ TTFT, TPOT, tokens            └─ P50, P90, P99
```

**Metrics Collected:**
1. **Response Metrics**
   - Requests per second (RPS)
   - Failed requests per second
   
2. **Latency Metrics** (quantiles: P50, P90, P99)
   - Time to First Token (TTFT)
   - Time Per Output Token (TPOT)
   - Total response time

3. **Throughput Metrics**
   - Output tokens per second
   - Empty response rate

## Communication Patterns

### 1. Metrics Queue (Spawner → Collector)
```python
# User sends metrics
metrics_queue.put(RequestSuccessLog(...))

# Collector receives and aggregates
metrics_data = metrics_queue.get(timeout=1)
```

### 2. Control Queue (Main → Spawner)
```python
# Main process sends control
control_queue.put(SetUserInfo(max_users=50))

# Spawner adjusts dynamically
user_spawner.max_user_count = msg.max_users
```

### 3. Shutdown Sequence
1. Main sends `TriggerShutdown` to control queue
2. Spawner stops creating new users
3. Spawner gracefully stops all active users
4. Spawner sends `TriggerShutdown` to metrics queue
5. Collector finishes processing and exits
6. Main process joins all threads/processes

## Performance Characteristics

### Concurrency Model
- **Process-Level**: 1 spawner process + 1 main process
- **Task-Level**: N async tasks (users) per spawner process
- **Thread-Level**: 2 threads in metrics collector

### Scalability
- **Typical**: 100-500 concurrent users per spawner process
- **Bottleneck**: Network I/O (not CPU)
- **GIL Impact**: Minimal (I/O-bound workload in separate process)

### Memory Usage
- **Base**: ~50MB per process
- **Per User**: ~1-2MB (mostly for request/response buffers)
- **Metrics Queue**: ~100 bytes per request log

## Extension Points

### 1. Custom Clients
Implement `BaseModelClient` ABC:
```python
class CustomClient(BaseModelClient):
    def ping_url(self) -> str: ...
    def get_request_params(self) -> tuple: ...
    def parse_response(self, chunk: bytes) -> list[int]: ...
```

### 2. Custom Metrics
Implement `SimpleMetric` or `QuantileMetric`:
```python
class CustomMetric(QuantileMetric):
    @property
    def name(self) -> str: ...
    def collect_request(self, request_log, collector) -> None: ...
```

### 3. Custom Prompt Sources
Add functions to `llm_locust/utils/prompts.py`:
```python
def load_custom_source(tokenizer, ...) -> list[dict]: ...
```

---

## Dataset Architecture

### Overview

LLM Locust supports multiple prompt datasets for different testing scenarios. Datasets are a critical component of the architecture, providing realistic workload patterns.

### Supported Datasets

| Dataset | Type | Use Case | Input Range | Ideal For |
|---------|------|----------|-------------|-----------|
| **Dolly** | Q&A | General benchmarking | 100-500 | Baseline tests |
| **ShareGPT** | Chat | Conversational | 50-2048 | Chat models |
| **BillSum** | Summarization | Long context prefill | 1024-8192 | Prefill-heavy tests |
| **Custom** | Any | Domain-specific | Variable | Specific use cases |

### 1. Databricks Dolly 15k (Default)

**Type**: Instruction-following  
**Size**: ~15,000 prompts  
**Use Case**: General-purpose Q&A, instruction following  
**Format**: JSONL with context + instruction

**Source**: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Characteristics:**
- Diverse instruction types
- Variable prompt lengths
- Context + instruction format
- Good for general benchmarking

**Usage:**
```python
from llm_locust.utils import load_databricks_dolly

prompts = load_databricks_dolly(
    tokenizer,
    min_input_length=100,
    max_input_length=500,
)
```

### 2. ShareGPT

**Type**: Conversational  
**Size**: ~90,000 conversations  
**Use Case**: Chat applications, multi-turn conversations  
**Format**: Multi-turn conversations

**Source**: [ShareGPT Vicuna Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

**Characteristics:**
- Real user conversations
- Natural language patterns
- Multi-turn context (we use first user message)
- Good for chat model benchmarking

**Usage:**
```python
from llm_locust.utils import load_sharegpt

prompts = load_sharegpt(
    tokenizer,
    min_input_length=50,
    max_input_length=2048,
    num_samples=1000,
)
```

**Format Example:**
```json
{
  "conversations": [
    {"from": "human", "value": "How do I learn Python?"},
    {"from": "gpt", "value": "Here are some steps..."},
    {"from": "human", "value": "What about advanced topics?"}
  ]
}
```

### 3. BillSum

**Type**: Long Document Summarization  
**Size**: ~23,000 bills  
**Use Case**: Long context prefill testing (heavy input processing)  
**Format**: US legislative bills

**Source**: [BillSum](https://huggingface.co/datasets/FiscalNote/billsum)

**Characteristics:**
- Very long documents (2k-10k tokens)
- Complex legislative text
- **Prefill-heavy** workload
- Tests long context handling
- Ideal for testing prompt processing performance

**Usage:**
```python
from llm_locust.utils import load_billsum

prompts = load_billsum(
    tokenizer,
    min_input_length=1500,
    max_input_length=2000,
    num_samples=500,
)
```

**Prompt Format:**
```
Summarize this legislative bill:

[Very long bill text...]
```

### 4. Custom Datasets

**Type**: User-provided  
**Use Case**: Domain-specific testing

**Supported Formats:**

**JSONL (recommended):**
```json
{"prompt": "Your prompt here"}
{"prompt": "Another prompt"}
```

**JSON Array:**
```json
[
  {"prompt": "Your prompt here"},
  {"prompt": "Another prompt"}
]
```

**Usage:**
```python
from llm_locust.utils import load_custom_prompts
from pathlib import Path

prompts = load_custom_prompts(
    tokenizer=tokenizer,
    prompts_file=Path("my_prompts.jsonl"),
)
```

### Dataset Selection Guide

**For General Benchmarking:**
→ Use **Dolly** (default)
- Well-balanced prompt distribution
- Good mix of lengths
- Instruction-following focus

**For Chat Applications:**
→ Use **ShareGPT**
- Real conversational patterns
- Natural language flow
- Chat-style interactions

**For Long Context (RAG) Testing:**
→ Use **BillSum**
- Very long input documents (2k-8k tokens)
- Tests prompt processing performance
- Heavy prefill workload
- Reveals prefill bottlenecks

**For Domain-Specific Testing:**
→ Use **Custom dataset**
- Your own prompts
- Domain-specific vocabulary
- Controlled test scenarios

### Caching

Datasets are automatically cached on first download:

- **Dolly**: `datasets/databricks-dolly-15k.jsonl`
- **ShareGPT**: `datasets/sharegpt.jsonl`
- **BillSum**: `datasets/billsum.jsonl`

All cache files are gitignored. Delete to force re-download.

### Performance Characteristics

| Dataset | Load Time | Memory | Cache Size |
|---------|-----------|--------|------------|
| Dolly | ~5s | ~50MB | ~13MB |
| ShareGPT | ~30s | ~200MB | ~150MB |
| BillSum | ~60s | ~300MB | ~250MB |
| Custom | Varies | Varies | Varies |

**Note**: First load includes download time. Subsequent loads use cached files.

### Advanced Usage

**Mix Multiple Datasets:**
```python
from llm_locust.utils import load_databricks_dolly, load_sharegpt

# Load both datasets
dolly = load_databricks_dolly(tokenizer, 100, 500)
sharegpt = load_sharegpt(tokenizer, 100, 500)

# Combine
all_prompts = dolly + sharegpt

# Use in client
client = OpenAIChatStreamingClient(
    base_url=host,
    prompts=all_prompts,
    ...
)
```

**Filter by Length:**
```python
# Short prompts (chat-like)
short = load_sharegpt(tokenizer, min_input_length=10, max_input_length=100)

# Medium prompts (Q&A)
medium = load_dolly(tokenizer, min_input_length=100, max_input_length=500)

# Long prompts (document analysis)
long = load_billsum(tokenizer, min_input_length=1500, max_input_length=2000)
```

---

## Best Practices

1. **User Count**: Start with 10-20 users, scale up gradually
2. **Spawn Rate**: Use 1-5 users/second to avoid overwhelming endpoints
3. **Metrics Window**: 30-60 seconds for stable quantile calculations
4. **Timeouts**: Set reasonable timeouts (5-10 minutes for large contexts)
5. **Prompt Distribution**: Use diverse prompt lengths for realistic load
6. **Monitoring**: Watch both client and server metrics

## Future Enhancements

- [ ] Distributed mode (multiple spawner processes)
- [ ] Real-time WebUI dashboard
- [ ] CSV/JSON metrics export
- [ ] Scenario-based testing (ramp-up/ramp-down patterns)
- [ ] Token budget tracking
- [ ] Cost estimation per request
- [ ] Automatic rate limiting detection

