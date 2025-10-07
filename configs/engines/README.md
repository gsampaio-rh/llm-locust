# Engine Deployment Configurations

This directory contains **clean, standardized** Kubernetes deployment manifests for LLM inference engines used in benchmarking.

## 📁 Files

- `vllm.yaml` - vLLM deployment (baseline, 92% GPU utilization)
- `vllm_cost.yaml` - vLLM cost-optimized (60% GPU utilization)
- `tgi.yaml` - HuggingFace Text Generation Inference
- `ollama.yaml` - Ollama deployment

## 🎯 Purpose

These YAML files serve multiple purposes:

1. **Reference Deployments**: Real configurations used in production benchmarks
2. **Cost Analysis**: Auto-parsed by dashboard for GPU utilization and resource specs
3. **Reproducibility**: Anyone can deploy the same setup
4. **Documentation**: Shows optimal configurations for each engine

## 📋 Standardized Format

All deployment YAMLs follow this structure:

```yaml
---
# Deployment Configuration for [Engine Name]
# Purpose: [Brief description]
# Last Updated: [Date]
# Benchmark ID: [Short identifier used in CSV filenames]

apiVersion: apps/v1
kind: Deployment
metadata:
  name: [engine]-test           # Must match benchmark CSV platform name
  namespace: vllm-benchmark      # Optional, can be removed for other namespaces
  labels:
    app: [engine]
    component: inference-server
    benchmark-group: llm-performance
  annotations:
    description: "[Brief description of this config]"
    
spec:
  replicas: 1
  
  selector:
    matchLabels:
      app: [engine]
  
  template:
    metadata:
      labels:
        app: [engine]
        component: inference-server
    
    spec:
      # Resource allocation
      containers:
      - name: [engine]
        image: [container-image]
        
        # ⚠️ IMPORTANT: These are parsed by cost analysis dashboard
        resources:
          requests:
            cpu: "2"              # Minimum CPU cores
            memory: 16Gi          # Minimum memory
            nvidia.com/gpu: "1"   # Number of GPUs
          limits:
            cpu: "4"              # Maximum CPU cores
            memory: 24Gi          # Maximum memory  
            nvidia.com/gpu: "1"   # Number of GPUs (must match request)
        
        # Model and performance configuration
        args:
          - "--model=[model-name]"
          - "--gpu-memory-utilization=0.92"  # ⚠️ Used for cost calculation!
          - "--max-model-len=8192"
          # ... other engine-specific flags
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      
      # Optional: Node affinity for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

## 🔑 Key Fields for Cost Analysis

The dashboard's YAML parser extracts these fields:

### Required for Cost Calculation
- `metadata.name` → Platform name (must match CSV filename)
- `spec.replicas` → Number of pod replicas
- `resources.limits.nvidia.com/gpu` → GPU count per pod
- `resources.limits.cpu` → CPU allocation
- `resources.limits.memory` → Memory allocation
- `args` with `--gpu-memory-utilization` → GPU memory usage (vLLM specific)
- `args` with `--model` → Model name for reference

### Naming Convention
- **Deployment name format**: `{engine}-test` or `{engine}_{variant}`
- **Must normalize to CSV platform names**:
  - YAML: `vllm-test` → Parser: `vllm` → CSV: `vllm`
  - YAML: `vllm-cost` → Parser: `vllm_cost` → CSV: `vllm_cost`

## 🧹 What We Removed (Cleaning)

To keep files clean and maintainable, we removed:

❌ **Auto-generated metadata**:
- `resourceVersion`
- `uid`
- `creationTimestamp`
- `generation`
- `managedFields` (all of it!)
- `status` section

❌ **Unnecessary details**:
- Helm annotations (unless essential)
- Duplicate labels
- Verbose field specifications
- Empty/default values

✅ **What we kept**:
- Essential metadata (name, namespace, labels)
- Resource specifications
- Container configuration
- Health checks
- Node affinity/tolerations

## 📝 How to Use

### For Benchmarking
```bash
# Deploy an engine
kubectl apply -f configs/engines/vllm.yaml

# Verify deployment
kubectl get pods -n vllm-benchmark

# Run benchmark
python examples/benchmark_chat_simulation.py \
    --engine vllm \
    --host https://vllm-endpoint \
    --model meta-llama/Llama-3.2-3B-Instruct
```

### For Cost Analysis
```bash
# The dashboard automatically parses these YAMLs
cd streamlit_app
streamlit run app.py

# 1. Upload benchmark CSVs
# 2. Go to Cost Analysis page
# 3. Click "Load Instance Specs from Deployment YAMLs"
# 4. Select instance type from dropdown
# 5. View cost comparison!
```

### For Custom Configurations
```bash
# Copy a template
cp configs/engines/vllm.yaml configs/engines/my-vllm.yaml

# Edit configuration
vim configs/engines/my-vllm.yaml

# Change:
# - metadata.name to "my-vllm"
# - GPU memory utilization
# - Model name
# - Resource limits

# Deploy
kubectl apply -f configs/engines/my-vllm.yaml

# Benchmark
python examples/benchmark_chat_simulation.py \
    --engine my_vllm \
    --host https://my-vllm-endpoint \
    --model your-model
```

## 🔧 Configuration Guidelines

### GPU Memory Utilization

**High Performance (Default)**:
```yaml
args:
  - "--gpu-memory-utilization=0.92"  # Max performance
```
- Use 92% of GPU memory
- Best throughput
- Higher risk of OOM

**Cost Optimized**:
```yaml
args:
  - "--gpu-memory-utilization=0.60"  # Cost optimized
```
- Use 60% of GPU memory
- Lower cost per pod (can pack more models)
- Better for multi-tenant clusters

### Resource Limits

**Standard Setup** (7B model):
```yaml
resources:
  limits:
    cpu: "4"
    memory: 24Gi
    nvidia.com/gpu: "1"
```

**Large Model** (70B+ model):
```yaml
resources:
  limits:
    cpu: "16"
    memory: 96Gi
    nvidia.com/gpu: "4"  # Multi-GPU
```

### Replicas

**Development/Testing**:
```yaml
spec:
  replicas: 1
```

**Production**:
```yaml
spec:
  replicas: 3  # High availability
```

## 🐛 Troubleshooting

### YAML Parser Issues

**Problem**: Dashboard doesn't detect deployment
- **Solution**: Check `metadata.name` matches CSV platform name
- Example: If CSV is `vllm_cost-*.csv`, name should be `vllm-cost` (parser converts to `vllm_cost`)

**Problem**: GPU utilization not detected
- **Solution**: Only vLLM supports `--gpu-memory-utilization` flag
- For TGI/Ollama, utilization defaults to full instance cost

**Problem**: Wrong resource specs shown
- **Solution**: Ensure `resources.limits` are set (parser uses limits, not requests)

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Kubernetes Resources](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

## 🤝 Contributing

When adding new engine configs:

1. Copy an existing YAML as template
2. Follow the standardized format above
3. Remove all auto-generated metadata
4. Test deployment: `kubectl apply -f your-config.yaml`
5. Test parsing: Run cost analysis dashboard
6. Document any engine-specific flags
7. Add to this README

---

**Last Updated**: 2025-10-04  
**Maintained By**: LLM Locust Team
