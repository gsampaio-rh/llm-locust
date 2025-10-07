# API Cost Comparison Guide

## 🎯 Overview

The Cost Analysis page now includes comprehensive API provider comparison, allowing you to compare self-hosted LLM platforms against 1,376+ API models from OpenAI, Anthropic, Google, Meta, Mistral, and more.

## 🔍 Filtering Models

### 1. Filter by Provider

Select one or more providers to narrow down options:
- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
- **Anthropic**: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
- **Google**: gemini-1.5-pro, gemini-1.5-flash
- **Meta**: llama-3.3-70b, llama-3.1-8b, llama-3.1-70b
- **Mistral**: mistral-large, mistral-medium, mistral-small
- **And many more...**

### 2. Filter by Size

Select model sizes to compare similar-sized models:

**Numeric Sizes:**
- 7B, 8B, 13B, 34B, 70B, 405B (parameter count)

**Keyword Sizes:**
- mini, small, medium, large, xl

**Claude Tiers:**
- haiku (fastest, cheapest)
- sonnet (balanced)
- opus (most capable)

**GPT Versions:**
- 3.5, 4, 4-turbo, 4o

### 3. Search by Name

Type in the search box to find specific models:
- "llama" → All Llama models
- "gpt" → All GPT models
- "claude" → All Claude models
- "mistral" → All Mistral models

### 4. Combined Filters

Combine multiple filters for precise results:
- **Provider**: Anthropic + **Size**: sonnet → Only Claude Sonnet models
- **Provider**: Meta + **Size**: 70B → Only Llama 70B models
- **Search**: "instruct" + **Size**: 7B → All 7B instruct models

## 📊 Comparison Features

### Cost Metrics

For each platform, see:
- **$/1M Tokens**: Cost per million output tokens (normalized for comparison)
- **Workload Cost**: Actual cost to process your benchmark requests
- **$/Hour**: Infrastructure cost (self-hosted) or "Pay-per-use" (APIs)
- **TTFT P50**: Latency for self-hosted (N/A for APIs)

### Key Insights

The dashboard automatically identifies:
- 💰 **Most Cost-Efficient**: Cheapest option overall
- ⚡ **Fastest Self-Hosted**: Best latency among self-hosted options
- 📊 **Break-Even Analysis**: QPS level where self-hosting becomes cheaper

### Trade-Off Recommendations

Based on your data, the dashboard provides guidance on:
- **When to self-host**: High volume, low latency, data privacy, customization
- **When to use APIs**: Low volume, variable load, no infrastructure management

## 💡 Example Use Cases

### Use Case 1: Find Cheapest API for Low Volume

**Filters:**
- Provider: (leave empty)
- Size: mini, small, haiku
- Search: (empty)

**Result**: Compare cheapest API options for low-volume workloads

---

### Use Case 2: Compare 70B Models

**Filters:**
- Provider: Meta, Mistral
- Size: 70B
- Search: (empty)

**Result**: See all 70B parameter models and their costs

---

### Use Case 3: OpenAI vs Anthropic

**Filters:**
- Provider: openai, anthropic
- Size: (leave empty)
- Search: (empty)

**Result**: Side-by-side comparison of two major providers

---

### Use Case 4: Find Specific Model

**Filters:**
- Provider: (empty)
- Size: (empty)
- Search: "llama-3.3-70b"

**Result**: Exact model match

## 📈 Understanding the Analysis

### Break-Even Table

Shows monthly costs at different QPS levels:

| QPS | Monthly Requests | Self-Hosted | API | Winner |
|-----|------------------|-------------|-----|--------|
| 1 | 2.6M | $1,310 | $156 | 🏆 API |
| 10 | 25.9M | $1,310 | $1,560 | 🏆 Self-Hosted |
| 100 | 259.2M | $1,310 | $15,600 | 🏆 Self-Hosted |

**Interpretation:**
- At low QPS (1-10): APIs are cheaper (pay-per-use)
- At high QPS (100+): Self-hosting is cheaper (fixed infrastructure cost)

### Cost Calculation Formula

**Self-Hosted:**
```
Cost = ($/hour × hours) × instances_needed
instances_needed = ⌈(QPS × avg_output_tokens) ÷ throughput⌉
```

**API:**
```
Cost = (input_tokens × $/input_token) + (output_tokens × $/output_token)
```

## 🚀 Advanced Features

### Proportional Cost Allocation

When YAML configs are loaded with GPU memory utilization:
```
Cost = ($/hour ÷ GPUs) × GPU_util × deployment_GPUs × replicas
```

Example:
- Instance: AWS p4d.24xlarge (8x A100) @ $32.77/hr
- GPU Memory Utilization: 60%
- Deployment: 1 GPU, 1 replica
- Cost: ($32.77 ÷ 8) × 0.60 × 1 × 1 = $2.46/hr

### Full Instance Cost

When proportional allocation is disabled:
```
Cost = $/hour × replicas
```

## 📝 Tips & Best Practices

### 1. Start with Filters
Don't browse 1,376 models manually. Use filters to narrow down to relevant options.

### 2. Compare Similar Sizes
When comparing self-hosted 70B models, also look at API 70B models for context.

### 3. Consider Latency
APIs have network overhead. Self-hosted platforms show actual TTFT latency.

### 4. Factor in Scale
Break-even analysis helps determine when self-hosting makes economic sense.

### 5. Don't Forget Hidden Costs
- Self-hosted: Engineering time, maintenance, monitoring
- APIs: Network egress, rate limits, vendor lock-in

## 🔧 Troubleshooting

### "Showing 0 models"
- Remove all filters and start over
- Check if selected provider/size combination exists

### "No models in dropdown"
- Check the "Show all filtered models" checkbox
- Adjust filters to include more models

### "API pricing data not loaded"
- Ensure `model_prices.json` exists in the project root
- Check file permissions

## 📚 Model Naming Patterns

The size extraction recognizes these patterns:
- `llama-3.1-70b-instruct` → 70B
- `gpt-4o-mini` → mini
- `claude-3-5-sonnet-20241022` → sonnet
- `mistral-large-latest` → large
- `gemini-1.5-flash` → (version 1.5, but "flash" not extracted as size)

## 🆕 What's New

**October 7, 2025:**
- ✅ Added filtering by provider (50+ providers)
- ✅ Added filtering by model size (extracted from names)
- ✅ Added search box for model names
- ✅ Smart size sorting (7B → 8B → 13B → 70B → mini → small → opus)
- ✅ Auto-enable "show all" when filtered to <100 models
- ✅ Combined filters for precise model selection

## 📞 Support

For issues or feature requests, see the main [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) document.

