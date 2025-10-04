# Race Configuration Examples

This directory contains example race configurations for the LLM Locust interactive racing CLI.

## 🚀 Quick Start

```bash
# Run a demo race (2 minutes, 3 engines)
llm-locust race --config configs/races/demo-race.yaml

# Run production comparison (10 minutes, 2 engines)
llm-locust race --config configs/races/production-candidates.yaml

# Run RAG benchmark (15 minutes, long context)
llm-locust race --config configs/races/rag-benchmark.yaml

# Run stress test (5 minutes, high load)
llm-locust race --config configs/races/stress-test.yaml

# Run cluster race (5 minutes, 4 real endpoints)
llm-locust race --config configs/races/cluster-race.yaml
```

## 📝 Configuration Format

All race configurations follow this schema:

```yaml
race:
  name: "Your Race Name"
  
  # Duration and concurrency
  duration: 600        # Race duration in seconds
  users: 50           # Number of concurrent users
  spawn_rate: 5.0     # Users spawned per second
  
  # Workload configuration
  dataset: "sharegpt"           # Dataset: sharegpt, dolly, billsum
  target_input_tokens: 256      # Target input length
  target_output_tokens: 128     # Target output length
  
  # Output
  output_dir: "results/races"   # Where to save results
  
  # Engines (2-10 supported)
  engines:
    - name: "engine-1"          # Unique engine name
      url: "http://..."         # Endpoint URL (required)
      model: "model-name"       # Model name (required)
      emoji: "🚀"              # Display emoji (optional)
      color: "cyan"            # Terminal color (optional)
      tokenizer: "..."         # Tokenizer name (optional, defaults to model)
```

## 🎨 Color Options

Valid color values for `color` field:
- Basic: `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`
- Bright: `bright_black`, `bright_red`, `bright_green`, `bright_yellow`, `bright_blue`, `bright_magenta`, `bright_cyan`, `bright_white`
- Named: `purple`, `pink`, `orange`, `gold`

## 📊 Dataset Options

### ShareGPT (Conversational)
- **Use case:** Chat applications, conversational AI
- **Characteristics:** Natural dialogue, variable lengths
- **Recommended for:** Chat simulation, general benchmarks

### Dolly (Instruction Following)
- **Use case:** Q&A, instruction-following tasks
- **Characteristics:** Structured prompts, diverse topics
- **Recommended for:** General-purpose benchmarks

### BillSum (Long Documents)
- **Use case:** RAG systems, document analysis
- **Characteristics:** Very long documents (2k-8k tokens)
- **Recommended for:** Long-context testing, prefill benchmarks

## 🎯 Example Use Cases

### Quick Demo (2 minutes)
```yaml
duration: 120
users: 10
spawn_rate: 2.0
```

### Production Comparison (10 minutes)
```yaml
duration: 600
users: 50
spawn_rate: 5.0
```

### Stress Test (5 minutes)
```yaml
duration: 300
users: 100
spawn_rate: 10.0
```

### Long Context / RAG (15 minutes)
```yaml
duration: 900
users: 20
spawn_rate: 2.0
dataset: "billsum"
target_input_tokens: 4096
```

## 🔧 Customization

To create your own race configuration:

1. Copy one of the example files
2. Modify the `name` to describe your test
3. Update `engines` list with your endpoints
4. Adjust `duration`, `users`, and `spawn_rate` as needed
5. Choose appropriate `dataset` for your workload
6. Save with a descriptive filename

## 📖 Learn More

- [Race CLI Documentation](../../docs/RACE_CLI.md)
- [Architecture Guide](../../docs/ARCHITECTURE.md)
- [Agile Plan](../../docs/AGILE_PLAN_INTERACTIVE_CLI.md)

## 🤝 Contributing

Have a useful race configuration? Submit a PR with:
- Descriptive filename
- Clear comments explaining the use case
- Tested against at least 2 endpoints

