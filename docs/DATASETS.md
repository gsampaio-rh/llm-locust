# Dataset Guide

LLM Locust supports multiple prompt datasets for different testing scenarios.

## Supported Datasets

### Overview

| Dataset | Type | Use Case | Input Range | Ideal For |
|---------|------|----------|-------------|-----------|
| **Dolly** | Q&A | General benchmarking | 100-500 | Baseline tests |
| **ShareGPT** | Chat | Conversational | 50-2048 | Chat models |
| **CNN/DailyMail** | Summarization | Document summarization | 500-2048 | Summarization workloads |
| **BillSum** | Summarization | Long context prefill | 1024-8192 | Prefill-heavy tests |
| **Infinity Instruct** | Instructions | Long context decode | 512-4096 | Decode-heavy tests |
| **Shared Prefix** | Custom | Prefix caching | Variable | Cache efficiency |

### 1. Databricks Dolly 15k (Default)

**Type**: Instruction-following  
**Size**: ~15,000 prompts  
**Use Case**: General-purpose Q&A, instruction following  
**Format**: JSONL with context + instruction

**Usage:**
```bash
python examples/simple_test.py \
    --dataset dolly \
    --prompt-min-tokens 100 \
    --prompt-max-tokens 500
```

**Source**: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Characteristics:**
- Diverse instruction types
- Variable prompt lengths
- Context + instruction format
- Good for general benchmarking

### 2. ShareGPT

**Type**: Conversational  
**Size**: ~90,000 conversations  
**Use Case**: Chat applications, multi-turn conversations  
**Format**: Multi-turn conversations

**Usage:**
```bash
python examples/simple_test.py \
    --dataset sharegpt \
    --prompt-min-tokens 50 \
    --prompt-max-tokens 2048
```

**Source**: [ShareGPT Vicuna Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

**Characteristics:**
- Real user conversations
- Natural language patterns
- Multi-turn context (we use first user message)
- Good for chat model benchmarking

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

### 3. CNN/DailyMail

**Type**: Summarization  
**Size**: ~300,000 articles  
**Use Case**: Document summarization, article processing  
**Format**: News articles with summaries

**Usage:**
```bash
python examples/simple_test.py \
    --dataset cnn_dailymail \
    --prompt-min-tokens 500 \
    --prompt-max-tokens 2048 \
    --users 10
```

**Source**: [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)

**Characteristics:**
- News articles (varying lengths)
- Real-world summarization task
- 500-2000 token articles typical
- Good for summarization model testing

**Prompt Format:**
```
Summarize the following article:

[Article text...]
```

### 4. BillSum

**Type**: Long Document Summarization  
**Size**: ~23,000 bills  
**Use Case**: Long context prefill testing (heavy input processing)  
**Format**: US legislative bills

**Usage:**
```bash
python examples/simple_test.py \
    --dataset billsum \
    --prompt-min-tokens 1024 \
    --prompt-max-tokens 8192 \
    --users 5
```

**Source**: [BillSum](https://huggingface.co/datasets/billsum)

**Characteristics:**
- Very long documents (2k-10k tokens)
- Complex legislative text
- **Prefill-heavy** workload
- Tests long context handling
- Ideal for testing prompt processing performance

**Prompt Format:**
```
Summarize this legislative bill:

[Very long bill text...]
```

### 5. Infinity Instruct

**Type**: Long-form Instructions  
**Size**: 7M+ instructions  
**Use Case**: Long context decode testing (heavy output generation)  
**Format**: Instructions requiring detailed responses

**Usage:**
```bash
python examples/simple_test.py \
    --dataset infinity_instruct \
    --prompt-min-tokens 512 \
    --prompt-max-tokens 4096 \
    --max-tokens 1024 \
    --users 10
```

**Source**: [Infinity Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)

**Characteristics:**
- Instructions requiring long responses
- **Decode-heavy** workload
- Tests generation capacity
- Good for throughput benchmarking
- Variable complexity instructions

**Example Prompts:**
```
"Write a comprehensive guide to..."
"Explain in detail how..."
"Create a tutorial about..."
```

### 6. Shared Prefix (Prefix Caching)

**Type**: Custom-generated  
**Use Case**: Testing prefix/KV cache efficiency  
**Format**: Programmatically created

**Usage (Programmatic):**
```python
from llm_locust.utils import create_shared_prefix_dataset

# Long shared context
prefix = """
Given this 10,000 word research paper about quantum computing:
[... very long document ...]
"""

# Different questions about the same document
suffixes = [
    "What is the main conclusion?",
    "Who are the authors?",
    "What methodology was used?",
    "Summarize the results.",
]

prompts = create_shared_prefix_dataset(
    tokenizer=tokenizer,
    shared_prefix=prefix,
    variable_suffixes=suffixes,
)
```

**Characteristics:**
- All prompts share identical prefix
- Only suffix varies
- Tests KV cache effectiveness
- Ideal for prefix caching benchmarks
- Reveals cache hit/miss performance

**Why This Matters:**
Engines with good prefix caching (vLLM, TGI) should show:
- Fast TTFT after first request
- Lower compute for subsequent requests
- Better throughput with shared contexts

### 7. Custom Datasets

**Type**: User-provided  
**Use Case**: Domain-specific testing  

**Usage:**
```python
from llm_locust.utils import load_custom_prompts
from pathlib import Path

prompts = load_custom_prompts(
    tokenizer=tokenizer,
    prompts_file=Path("my_prompts.jsonl"),
)
```

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

**Alternative keys:** `prompt`, `text`, or `content`

---

## Dataset Comparison

| Dataset | Size | Type | Avg Input Length | Max Tokens | Use Case |
|---------|------|------|------------------|------------|----------|
| **Dolly** | 15k | Instruction | 200-800 | 128-512 | General Q&A |
| **ShareGPT** | 90k | Chat | 100-2000 | 256-1024 | Conversational |
| **CNN/DailyMail** | 300k | Summarization | 500-2000 | 128-256 | News summarization |
| **BillSum** | 23k | Long Summarization | 2000-8000 | 256-512 | **Long prefill** |
| **Infinity Instruct** | 7M | Long Instructions | 512-4096 | 512-2048 | **Long decode** |
| **Shared Prefix** | Custom | Caching Test | Variable | Variable | **Prefix caching** |
| **Custom** | Variable | Any | Variable | Variable | Domain-specific |

---

## Choosing a Dataset by Use Case

### For General Benchmarking
→ Use **Dolly** (default)
- Well-balanced prompt distribution
- Good mix of lengths
- Instruction-following focus

### For Chat Applications
→ Use **ShareGPT**
- Real conversational patterns
- Natural language flow
- Chat-style interactions

### For Summarization Workloads
→ Use **CNN/DailyMail**
- Real news articles
- Summarization tasks
- Moderate input length

### For Long Context Prefill Testing
→ Use **BillSum**
- Very long input documents (2k-8k tokens)
- **Tests prompt processing performance**
- **Heavy prefill workload**
- Reveals prefill bottlenecks

### For Long Context Decode Testing
→ Use **Infinity Instruct**
- Instructions requiring long responses
- **Tests generation capacity**
- **Heavy decode workload**
- Reveals generation bottlenecks

### For Prefix Caching Benchmarks
→ Use **Shared Prefix**
- Test KV cache efficiency
- Measure cache hit performance
- Compare engines' caching strategies
- Identify cache optimization opportunities

### For Domain-Specific Testing
→ Use **Custom dataset**
- Your own prompts
- Domain-specific vocabulary
- Controlled test scenarios

---

## Advanced Usage

### Mix Multiple Datasets

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

### Filter by Length

```python
# Short prompts (chat-like)
short = load_sharegpt(tokenizer, min_input_length=10, max_input_length=100)

# Medium prompts (Q&A)
medium = load_dolly(tokenizer, min_input_length=100, max_input_length=500)

# Long prompts (document analysis)
long = load_dolly(tokenizer, min_input_length=500, max_input_length=2000)
```

### Sample Size Control

```python
# Limit ShareGPT to first 1000 samples
prompts = load_sharegpt(
    tokenizer,
    min_input_length=100,
    max_input_length=500,
    num_samples=1000,  # Limit to 1000 prompts
)
```

---

## Caching

Datasets are automatically cached on first download:

- **Dolly**: `databricks-dolly-15k.jsonl`
- **ShareGPT**: `sharegpt.jsonl`

Both files are gitignored. Delete to re-download.

---

## Creating Custom Datasets

### From Your Data

```python
import json

# Convert your data to the expected format
prompts = []
for item in your_data:
    prompts.append({
        "prompt": item.text,
        # Optional metadata
        "source": item.source,
        "category": item.category,
    })

# Save as JSONL
with open("custom_prompts.jsonl", "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")
```

### Load Custom Dataset

```python
from llm_locust.utils import load_custom_prompts

prompts = load_custom_prompts(
    tokenizer=tokenizer,
    prompts_file=Path("custom_prompts.jsonl"),
    system_prompt="You are a helpful assistant",
)
```

---

## Performance Characteristics

| Dataset | Load Time | Memory | Cache Size |
|---------|-----------|--------|------------|
| Dolly | ~5s | ~50MB | ~13MB |
| ShareGPT | ~30s | ~200MB | ~150MB |
| Custom | Varies | Varies | Varies |

**Note**: First load includes download time. Subsequent loads use cached files.

---

## Troubleshooting

### "No prompts found within token range"
- Adjust `--prompt-min-tokens` and `--prompt-max-tokens`
- ShareGPT has longer prompts, try wider range
- Check if dataset downloaded correctly

### "Failed to download dataset"
- Check internet connection
- Try manual download and place in project root
- Verify HuggingFace is accessible

### "Failed to tokenize prompt"
- Ensure tokenizer is compatible with model
- Check tokenizer has chat_template configured
- Try different tokenizer

---

## Dataset Roadmap

Future dataset support:
- [ ] Alpaca
- [ ] OpenAssistant
- [ ] MMLU (for evaluation)
- [ ] Custom HuggingFace datasets
- [ ] Streaming dataset loading

See [GitHub Issues](https://github.com/gsampaio-rh/llm-locust/issues) for requests.

