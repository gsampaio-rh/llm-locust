# Benchmark Test 1b: RAG Simulation

This test evaluates system performance under RAG (Retrieval-Augmented Generation) workloads with large input contexts and longer responses.

## Test Specifications

- **Input tokens:** ~4096 per request
- **Output tokens:** ~512 per request  
- **Duration:** 10-15 minutes
- **Concurrency:** ~20 parallel sessions
- **Rate:** Moderate with bursts representing multiple users querying documents
- **Focus:** Memory load, latency distribution, throughput impact

## Default Configuration

The test is pre-configured for Ollama with:
- **Endpoint:** `http://ollama-test-vllm-benchmark.apps.cluster-njnqr.njnqr.sandbox1049.opentlc.com`
- **Model:** `qwen2.5:7b-instruct-fp16` (Ollama format)
- **Tokenizer:** `Qwen/Qwen2.5-7B-Instruct` (HuggingFace tokenizer)
- **Best for:** Local development, ease of use

## Usage Examples

### Basic RAG Simulation (Ollama)
```bash
python examples/benchmark_test_1b.py
```

### With Custom Endpoint
```bash
python examples/benchmark_test_1b.py \
    --host http://localhost:8000 \
    --model llama-3.1-8b
```

### With Custom Parameters
```bash
python examples/benchmark_test_1b.py \
    --users 25 \
    --duration 1200 \
    --max-tokens 512
```

### Using BillSum Dataset (Long Context)
```bash
python examples/benchmark_test_1b.py \
    --use-billsum
```

### Using Infinity Instruct Dataset
```bash
python examples/benchmark_test_1b.py \
    --use-infinity-instruct
```

### With Per-Request Logging
```bash
python examples/benchmark_test_1b.py \
    --log-per-request \
    --output-file results/rag_benchmark_$(date +%Y%m%d_%H%M%S).csv
```

## Key Features

1. **Custom RAG Prompts**: Creates realistic RAG scenarios with long contexts (~4096 tokens) and specific questions
2. **Multiple Dataset Options**: Supports custom RAG prompts, BillSum, or Infinity Instruct datasets
3. **Memory Stress Testing**: Designed to stress-test KV cache growth and GPU memory usage
4. **Burst Traffic Simulation**: Moderate rate with bursts representing enterprise-scale workloads
5. **Comprehensive Metrics**: Tracks latency distribution, throughput impact, and memory usage

## Business Context

This test is ideal for evaluating:
- Knowledge-base assistants
- Research copilots  
- Enterprise search systems
- Document analysis tools
- RAG-based applications

## Expected Results

The test will help identify:
- How latency scales with large token counts
- Throughput drop-offs as request size increases
- Memory usage patterns under RAG workloads
- System performance under enterprise-scale RAG scenarios
