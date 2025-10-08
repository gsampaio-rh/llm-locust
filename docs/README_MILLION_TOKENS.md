# Million Token Race Benchmark

## 🎯 Purpose

**Measure how fast your infrastructure can produce 1 MILLION output tokens.**

This is the ultimate **throughput test** - it races to generate as many tokens as possible, as fast as possible.

---

## 🏁 Quick Start

### Basic Race (Optimized for L4 GPU)

```bash
python examples/benchmark_million_tokens.py \
    --host http://vllm-performance:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm_performance
```

**What happens:**
- Spawns 80 concurrent users (optimized for L4)
- Races to 1M output tokens
- Shows live progress every 5 seconds
- Stops when target reached
- Reports final throughput and time
- Saves results to organized folder: `results/million-tokens-YYYYMMDD-HHMMSS/`

---

## 📁 Output Structure

Each test run creates a dedicated folder with timestamp:

```
results/
└── million-tokens-20251008-083653/
    └── vllm-performance-20251008-083653-million-tokens.csv
```

**Benefits:**
- ✅ Organized by test run
- ✅ Easy to find specific runs
- ✅ No CSV clutter in root results folder
- ✅ Can store additional metadata in folder later

**Example with multiple runs:**
```
results/
├── million-tokens-20251008-083653/
│   └── vllm-performance-20251008-083653-million-tokens.csv
├── million-tokens-20251008-091234/
│   └── vllm-quantized-20251008-091234-million-tokens.csv
└── million-tokens-20251008-105521/
    └── tgi-20251008-105521-million-tokens.csv
```

---

## ⚙️ Hardware-Specific Recommendations

### L4 GPU (24GB) - Your Setup
```bash
# Recommended: 80 users
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --users 80

# Conservative: 50 users (safer)
--users 50

# Aggressive: 100 users (may see failures)
--users 100
```

**Expected Results:**
- **Throughput:** 2,500-3,000 tokens/sec
- **Time to 1M:** 5-7 minutes
- **Success Rate:** >98%

### A10G / A10 (24GB)
```bash
--users 100  # Sweet spot for A10
```

**Expected:**
- Throughput: 3,000-4,000 tokens/sec
- Time to 1M: 4-6 minutes

### A100 (40GB/80GB)
```bash
--users 150  # Can handle higher concurrency
```

**Expected:**
- Throughput: 5,000-8,000 tokens/sec
- Time to 1M: 2-3 minutes

---

## 📊 Understanding the Output

### Live Progress
```
📊 Progress: 234,567/1,000,000 tokens (23.5%) | ⏱️  142s | 🚀 1,653 tok/s | ⏰ ETA: 463s | 👥 80 users | 📋 142 reqs
```

Breakdown:
- **234,567/1,000,000** - Output tokens generated so far
- **23.5%** - Progress percentage
- **142s** - Elapsed time
- **1,653 tok/s** - Current throughput (rolling average)
- **ETA: 463s** - Estimated time remaining (7.7 minutes)
- **80 users** - Active concurrent users
- **142 reqs** - Total requests completed

### Final Results
```
🏆 FINAL RESULTS:
   • Output Tokens:      1,000,543
   • Input Tokens:       15,234
   • Total Tokens:       1,015,777
   • Time Elapsed:       387.3s (6.5 minutes)
   • Avg Throughput:     2,584 tokens/sec
   • Peak Throughput:    2,891 tokens/sec
   • Total Requests:     523
   • Successful:         521
   • Failed:             2
   • Success Rate:       99.62%
```

---

## 🎮 Advanced Usage

### Race to 10 Million Tokens
```bash
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --target-tokens 10000000
```

**Use case:** Extreme stress test, multi-hour stability test

### Minimize Input Tokens (Max Output)
```bash
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --prompt-min-tokens 20 \
    --prompt-max-tokens 50
```

**Why:** Shorter prompts = more tokens available for output = faster to 1M

### Compare Different Configurations
```bash
# Test FP16 (performance config)
python examples/benchmark_million_tokens.py \
    --host http://vllm-performance:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm_performance

# Test INT8 (quantized config)
python examples/benchmark_million_tokens.py \
    --host http://vllm-quantized:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm_quantized

# Compare results:
# - Which is faster?
# - What's the throughput difference?
# - Cost tradeoff?
```

---

## 🔍 What to Look For

### ✅ Success Indicators

1. **High Success Rate (>98%)**
   - Your infrastructure can handle the load
   - No significant failures or timeouts

2. **Stable Throughput**
   - Should stay relatively consistent
   - Minor fluctuations are normal
   - Watch for major drops (indicates saturation)

3. **Fast Time to 1M**
   - L4: <10 minutes = good
   - L4: <7 minutes = excellent
   - L4: <5 minutes = exceptional

### ⚠️ Warning Signs

1. **Low Success Rate (<95%)**
   - **Problem:** Overloaded, too many users
   - **Solution:** Reduce `--users` by 20-30%

2. **Dropping Throughput**
   - **Problem:** GPU memory saturation, thermal throttling
   - **Solution:** Reduce users, check GPU stats

3. **Many Failed Requests**
   - **Problem:** Timeouts, OOM errors
   - **Solution:** Check logs, reduce load

---

## 📈 Comparing Runs

### Create Comparison Table

After multiple runs:

| Configuration | Users | Time | Throughput | Success % |
|--------------|-------|------|------------|-----------|
| vLLM FP16 | 80 | 6.5 min | 2,584 tok/s | 99.6% |
| vLLM INT8 | 80 | 7.2 min | 2,315 tok/s | 99.8% |
| TGI FP16 | 80 | 8.1 min | 2,058 tok/s | 98.9% |

**Analysis:**
- FP16 is 11% faster than INT8
- INT8 has slightly better stability
- TGI is 19% slower than vLLM

---

## 💰 Cost Analysis

Use the CSV output to calculate cost per 1M tokens:

```bash
# 1. Run the race
python examples/benchmark_million_tokens.py ...
# Results saved to: results/million-tokens-20251008-083653/

# 2. Open dashboard
streamlit run streamlit_app/app.py

# 3. Upload the CSV from the test folder
#    File: results/million-tokens-20251008-083653/vllm-performance-20251008-083653-million-tokens.csv

# 4. Go to Cost Analysis page

# 5. Select g6.4xlarge (L4) - $1.47/hr

# 6. Dashboard calculates:
#    - If throughput = 2,500 tok/s
#    - Time for 1M = 400 seconds (6.7 minutes)
#    - Cost = $1.47 × (400/3600) = $0.16 per 1M tokens
```

---

## 🎯 Optimization Strategy

### 1. Find Your Baseline (Current Script)
```bash
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --users 80
```

### 2. Test Higher Load
```bash
# Increase by 25%
python examples/benchmark_million_tokens.py ... --users 100

# If success rate >95%, try more
python examples/benchmark_million_tokens.py ... --users 120
```

### 3. Find the Sweet Spot
- Maximum throughput
- Success rate >95%
- Stable performance
- No GPU memory issues

### 4. Document Your Optimal Config
```yaml
# optimal_config.yaml
infrastructure:
  gpu: L4
  memory: 24GB
  
model:
  name: Llama-3.2-3B-Instruct
  precision: bfloat16
  
optimal_settings:
  concurrent_users: 80
  throughput: 2,584 tok/s
  time_to_1M: 387s
  success_rate: 99.6%
```

---

## 🚀 Pro Tips

### 1. Warm-Up First
```bash
# Run a quick warm-up (100K tokens)
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --target-tokens 100000 \
    --users 20

# Then run the full race
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm
```

### 2. Monitor GPU While Running
```bash
# In another terminal
watch -n 1 nvidia-smi
```

Watch for:
- GPU utilization (should be >90%)
- Memory usage (should be <95% to avoid OOM)
- Temperature (should be <85°C)

### 3. Run Multiple Times
```bash
# Run 3 times and average
for i in {1..3}; do
    python examples/benchmark_million_tokens.py \
        --host http://localhost:8000 \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --engine vllm_run$i
    sleep 60  # Cool-down between runs
done
```

### 4. Short Prompts = Faster Race
```bash
# Minimum prompts = maximum output
python examples/benchmark_million_tokens.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --prompt-min-tokens 20 \
    --prompt-max-tokens 50
```

---

## 📊 When to Use This Benchmark

✅ **Use this benchmark when:**
- You want to know **maximum throughput capacity**
- You're comparing different hardware/configurations
- You're stress-testing your infrastructure
- You want to calculate **cost per 1M tokens** at max capacity

❌ **Don't use this benchmark for:**
- Testing realistic user workloads (use 1a, 1b, 1c instead)
- Testing latency/responsiveness (use chat simulation)
- Testing burst handling (use Poisson rate)

---

## 🤝 Related Benchmarks

- **Cost Estimation** (`benchmark_cost_estimation.py`) - Time-based, calculates $/1M tokens
- **Variable Load** (`benchmark_variable_load.py`) - Tests different load patterns
- **Chat Simulation** (`benchmark_chat_simulation.py`) - Realistic user workload

---

## ❓ FAQ

### Q: Why 80 users by default?
**A:** Optimized for L4 GPU with 3B model. Leaves headroom for batching (max-num-seqs=96).

### Q: What if I see failures?
**A:** Reduce `--users` by 20-30%. Your infrastructure is saturated.

### Q: Can I test larger models?
**A:** Yes, but reduce users proportionally:
- 7B model: ~50 users
- 13B model: ~30 users
- 70B model: ~10 users (may need multiple GPUs)

### Q: How is this different from cost estimation?
**A:**
- **Million Token Race:** Stops at 1M tokens (variable time)
- **Cost Estimation:** Stops at 5 minutes (variable tokens)

### Q: Should I use this for production planning?
**A:** Use this to find **maximum capacity**. For production, run more realistic benchmarks (1a, 1b, 1c) at 60-70% of max capacity.

---

## 📚 Documentation

- [Main Benchmarks Guide](../../docs/BENCHMARKS.md)
- [Cost Estimation Guide](COST_ESTIMATION_GUIDE.md)
- [Architecture Guide](../../docs/ARCHITECTURE.md)

---

**Good luck with your million token race! 🏁🚀**

