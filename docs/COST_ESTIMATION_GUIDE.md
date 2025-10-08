# Cost Estimation Guide

## 🎯 Goal
Calculate how much it costs to process **1 million tokens** on your LLM infrastructure.

## 💡 Key Concept: Throughput = Cost Efficiency

**Critical insight**: Your infrastructure has a **fixed hourly cost** but **variable throughput**.

```
Infrastructure: $1.47/hour (fixed - whether idle or busy)

At Low Load (10 users):
  → 50 tokens/sec
  → Time for 1M tokens: 5.56 hours
  → Cost: $1.47 × 5.56 = $8.17 per 1M tokens

At High Load (50 users):
  → 250 tokens/sec  (5× more!)
  → Time for 1M tokens: 1.11 hours
  → Cost: $1.47 × 1.11 = $1.63 per 1M tokens  (5× cheaper!)
```

**The test measures: "How fast can my infrastructure produce tokens?"**

Faster production (higher throughput) = spreading fixed costs over more tokens = **lower cost per token**.

**Goal**: Find your **maximum sustainable throughput** for the **lowest cost per token**!

## 📋 Quick Start (3 Steps)

### Step 1: Run the Cost Estimation Benchmark

```bash
# Basic test (5 minutes, HEAVY load, unlimited output)
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

**What this does:**
- Runs a 5-minute sustained load test with **150 concurrent users** (HEAVY load! 🔥)
- Uses **unlimited output** (model decides when to stop - natural distribution)
- Pushes your infrastructure to **maximum throughput**
- **Why heavy?** → Higher throughput = lower $/token (spreading fixed costs!)
- Measures actual token throughput (tokens/second)
- Generates a CSV file in `results/` directory
- Example: `results/vllm-20251008-143022-cost-estimation.csv`

**💡 Tip:** This is aggressive! If you see failures/timeouts, reduce users:
```bash
# More conservative if 150 is too much
python examples/benchmark_cost_estimation.py ... --users 75
```

### Step 2: Open the Dashboard

```bash
streamlit run streamlit_app/app.py
```

### Step 3: Calculate Costs

1. **Upload your CSV** on the main dashboard page
2. **Go to "Cost Analysis"** page (left sidebar)
3. **Select your instance type** (or enter custom pricing)
   - AWS g6.4xlarge (L4) - $1.47/hr
   - AWS p4d.24xlarge (A100) - $32.77/hr
   - GCP, Azure, On-prem options available
4. **See your results!**
   - ✅ Cost per 1M input tokens
   - ✅ Cost per 1M output tokens
   - ✅ Monthly cost projections
   - ✅ Break-even analysis vs APIs

---

## 🔧 Customization Options

### Test Different Workloads

```bash
# Natural/Unlimited Output (let model decide when to stop)
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --unlimited-output

# Short prompts, short responses (chat-like)
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --prompt-min-tokens 100 \
    --prompt-max-tokens 300 \
    --max-tokens 128 \
    --dataset sharegpt

# Long context (RAG-like)
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --prompt-min-tokens 1500 \
    --prompt-max-tokens 2000 \
    --max-tokens 512 \
    --dataset billsum

# Code generation (balanced)
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --prompt-min-tokens 400 \
    --prompt-max-tokens 600 \
    --max-tokens 512 \
    --dataset dolly
```

### Adjust Test Duration

```bash
# Quick test (3 minutes) - for rapid iteration
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --duration 180

# Longer test (10 minutes) - for more accurate results
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --duration 600
```

### Adjust Load Level

```bash
# Light load (10 users) - testing lower utilization
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --users 10

# Heavy load (50 users) - testing higher utilization
python examples/benchmark_cost_estimation.py \
    --host http://localhost:8000 \
    --model your-model \
    --engine vllm \
    --users 50 \
    --spawn-rate 5.0
```

---

## 📊 Understanding the Results

### What the Dashboard Shows

#### 1. **Benchmark Performance Table**
- **Total tokens processed** during your test
- **Test duration** (actual time)
- **Throughput** (tokens/second) ← **KEY METRIC**
- **Time to process 1M tokens**

#### 2. **Cost Efficiency Table**
- **$/1M Input Tokens** - Cost to process 1M input tokens (prompts)
- **$/1M Output Tokens** - Cost to generate 1M output tokens (responses)
- **Price per Token** - For API comparison
- **Tokens per Dollar** - Efficiency metric

#### 3. **Monthly Projections**
- Token capacity per month (24/7 operation)
- Monthly cost at different scales
- Instance scaling calculator

#### 4. **API Comparison**
- Compare with OpenAI, Anthropic, Google, etc.
- See break-even point
- Economy of scale visualization

---

## 💰 How Cost Calculation Works

### The Math

```
Cost per 1M tokens = (Infrastructure Cost per Hour ÷ Tokens per Hour) × 1,000,000

Where:
- Infrastructure Cost = Your instance hourly cost ($/hr)
- Tokens per Hour = Measured throughput (tokens/sec) × 3600
```

### Example

```
Benchmark Results:
- Throughput: 50 tokens/sec
- Tokens per hour: 50 × 3600 = 180,000 tokens/hr

Infrastructure Cost:
- AWS g6.4xlarge (L4 GPU): $1.47/hr

Cost Calculation:
- Cost per 1M tokens = ($1.47 ÷ 180,000) × 1,000,000
- Cost per 1M tokens = $8.17

Therefore:
- Processing 1M tokens costs $8.17
- Processing 10M tokens/month costs $81.70
- Processing 100M tokens/month costs $817.00
```

---

## 🎯 Pro Tips

### 1. **Higher Load = Lower Cost Per Token! 💡**

**KEY INSIGHT**: Your infrastructure costs $X/hour whether it's processing 10 tokens/sec or 100 tokens/sec. Higher throughput = same cost spread over more tokens = cheaper per token!

```
Example with $1.47/hour infrastructure:

Light Load (10 users, 50 tok/s):
  Cost per 1M tokens = ($1.47/hr ÷ 180K tok/hr) × 1M = $8.17

Heavy Load (50 users, 250 tok/s):
  Cost per 1M tokens = ($1.47/hr ÷ 900K tok/hr) × 1M = $1.63

→ 5× more throughput = 5× cheaper per token! 🚀
```

**Strategy**: Test at increasing load to find your **maximum sustainable throughput**:

```bash
# Start low
python examples/benchmark_cost_estimation.py ... --users 10

# Increase load
python examples/benchmark_cost_estimation.py ... --users 30

# Push higher
python examples/benchmark_cost_estimation.py ... --users 50

# Find the limit
python examples/benchmark_cost_estimation.py ... --users 100
```

**Find the sweet spot where:**
- ✅ Throughput is maximized (most tokens/sec)
- ✅ Latency is acceptable (TTFT < 1-2s)
- ✅ Success rate > 99%
- ✅ No GPU/memory saturation

**This gives you the LOWEST cost per token!**

### 2. **⚠️ Don't Over-Optimize for Cost at the Expense of Quality**

While higher throughput = lower cost per token, watch out for:
- 🚨 Latency degradation (slow responses hurt user experience)
- 🚨 Increased failures (retries cost money too!)
- 🚨 GPU memory exhaustion (crashes are expensive)

**Best practice**: Find max throughput where latency/quality are still acceptable.

### 3. **Test Your Actual Workload**
Match the benchmark parameters to your real use case:
- Chat: Short prompts (100-300 tokens), short responses (100-200 tokens)
- RAG: Long prompts (1000-2000 tokens), medium responses (200-500 tokens)
- Code: Medium prompts (400-600 tokens), long responses (500-1000 tokens)

### 4. **Compare Configurations**
Run tests with different:
- Model sizes (7B vs 13B vs 70B)
- Quantization levels (FP16 vs INT8 vs INT4)
- GPU types (L4 vs A10 vs A100)
- Memory utilization settings

### 5. **Use YAML Auto-Load**
Place your deployment YAMLs in `configs/engines/` and the dashboard will auto-extract:
- GPU count and type
- Memory utilization
- Replicas
- Model name

This ensures accurate cost calculations!

---

## 🚀 Complete Example Workflow

```bash
# 1. Test vLLM deployment
python examples/benchmark_cost_estimation.py \
    --host http://vllm-endpoint:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm \
    --duration 300

# 2. Test TGI deployment
python examples/benchmark_cost_estimation.py \
    --host http://tgi-endpoint:8080 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --engine tgi \
    --duration 300

# 3. Open dashboard
streamlit run streamlit_app/app.py

# 4. Upload both CSVs
# 5. Go to Cost Analysis
# 6. Compare side-by-side!
```

---

## ❓ FAQ

### Q: How long should I run the test?
**A:** 5 minutes is usually enough for a good estimate. For production planning, run 10 minutes.

### Q: What if my throughput varies a lot?
**A:** Run multiple tests and average the results. Or run longer tests (10-15 minutes).

### Q: Should I test at peak load?
**A:** Test at your expected average load. The dashboard can help you scale up from there.

### Q: What about multi-GPU setups?
**A:** The cost calculator handles multi-GPU instances. Just select the right instance type (e.g., 8x A100).

### Q: How do I test different models?
**A:** Run the benchmark for each model, then compare in the dashboard!

### Q: Can I use this for on-prem?
**A:** Yes! Select "Custom" instance type and enter your amortized costs (hardware + power + maintenance).

---

## 📚 Related Documentation

- [Benchmark Guide](../docs/BENCHMARKS.md) - All benchmark tests
- [Architecture Guide](../docs/ARCHITECTURE.md) - System design
- [Cost Analysis Dashboard](../streamlit_app/pages/6_Cost_Analysis.py) - Dashboard code
- [README](../README.md) - Main documentation

---

## 🤝 Need Help?

If you have questions or issues:
1. Check the dashboard tooltips (hover over ℹ️ icons)
2. Review the example results in `results/` directory
3. Open an issue on GitHub
