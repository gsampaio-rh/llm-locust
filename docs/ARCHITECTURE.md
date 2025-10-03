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

