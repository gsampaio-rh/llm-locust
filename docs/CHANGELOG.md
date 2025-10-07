# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **YAML Cost Integration** - Automatic extraction of deployment specs from Kubernetes YAMLs
  - Added `streamlit_app/lib/yaml_parser.py` for parsing K8s deployment manifests
  - Extracts: GPU count, CPU, memory, replicas, model name, GPU memory utilization
  - Auto-populates cost analysis dashboard with deployment specs
  - Supports GPU memory utilization-based cost allocation
  - Added proportional cost mode: allocates cost based on `--gpu-memory-utilization` parameter
  - Platform name normalization (handles `vllm-test` → `vllm`, `vllm-cost` → `vllm_cost`)
  
- **Cost Analysis Enhancements**
  - Two cost allocation modes:
    - **Proportional**: Cost = (Instance cost / GPUs) × GPU utilization × Replicas
    - **Full Instance**: Cost = Instance cost × Replicas
  - Extracts `--gpu-memory-utilization` from vLLM deployments for accurate cost modeling
  - Shows packing factor (e.g., 60% util → 1.67x models per GPU theoretical capacity)
  - Debug panel showing platform name matching for troubleshooting
  - Cost note column showing calculation method per platform

- **YAML Standardization**
  - Cleaned all deployment YAMLs (removed auto-generated K8s metadata)
  - Reduced file sizes: 400+ lines → 150-240 lines (60% reduction)
  - Added header comments with purpose, last updated date, benchmark ID
  - Standardized format across all engines (vLLM, TGI, Ollama)
  - Comprehensive `configs/engines/README.md` with usage guidelines

- **Documentation Consolidation**
  - Merged TESTS.md into BENCHMARKS.md (single source of truth)
  - Enhanced BENCHMARKS.md with detailed specifications from TESTS.md
  - Added overview table for quick reference
  - Merged REFACTORING_SUMMARY.md into CHANGELOG.md (this file)
  - Merged YAML_COST_INTEGRATION.md into CHANGELOG.md (this file)

- **Interactive CLI Planning**
  - Created comprehensive agile plan: `docs/AGILE_PLAN_INTERACTIVE_CLI.md`
  - 4 sprints (8 weeks) with 27 user stories
  - "The Great Model Race" concept - multi-endpoint racing with live TUI
  - Live request/response inspector for educational value
  - Achievements, replays, and sharing features planned
  - Story points, priorities, and dependencies mapped

### Changed
- Cost Analysis dashboard now handles dynamic platform names (no KeyError)
- YAML parser uses deployment name instead of app label for better matching
- All benchmark documentation now in single BENCHMARKS.md file

### Fixed
- Fixed KeyError when uploading benchmarks with new platform names
- Fixed variable shadowing in cost calculation loop (`config` vs `yaml_cfg`)
- Fixed platform name matching between YAMLs and CSV benchmarks

---

## [0.2.0] - 2025-10-03

### Added
- **Per-Request Logging** - Complete CSV output with all request metrics
  - Added `PerRequestLogger` class for structured logging
  - CSV format with columns: request_id, timestamp, user_id, input/output tokens, TTFT, TPOT, status
  - Console output with color-coded status (success/failure)
  - Configurable summary intervals
  - Optional text inclusion (truncated prompts/responses)

- **Failed Request Tracking** - [CRITICAL FIX]
  - Failed requests (HTTP errors, timeouts, etc.) now logged to CSV
  - Enables accurate success rate calculations
  - Added `user_id` field to `RequestFailureLog` model
  - Updated `PerRequestLogger.log_request()` to handle both success and failure
  - Failed requests logged with `status_code` ≠ 200 and zero metrics
  - Console shows failures with `[FAILURE] ❌` prefix and error details
  - Proper failure tracking in Streamlit dashboard

- **Standardized Benchmarks**
  - Test 1a: Chat Simulation (256/128 tokens)
  - Test 1b: RAG Simulation (4096/512 tokens)
  - Test 1c: Code Generation (512/512 tokens)
  - Test 2a: Constant Rate (sustained load)
  - Test 2b: Poisson Rate (bursty traffic)

- **Dataset Support**
  - ShareGPT dataset loader
  - Databricks Dolly dataset loader
  - BillSum dataset loader (long documents for RAG)
  - Token filtering and sampling

- **Streamlit Dashboard**
  - Multi-page dashboard for benchmark comparison
  - Overview with performance scorecards
  - Comparison page with side-by-side metrics
  - Latency Analysis with distributions, CDFs, statistical tests
  - Throughput Analysis with time series and stability metrics
  - Reliability Analysis with error breakdowns
  - Cost Analysis calculator with instance type selection
  - Token Analysis (planned)

- **Documentation**
  - Comprehensive ARCHITECTURE.md
  - BENCHMARKS.md with all test specifications
  - DATASETS.md with dataset guides
  - METRICS_COVERAGE.md with KPI mapping
  - Dashboard PRD and implementation status

### Changed
- **Files Changed** (Failed Request Fix):
  - `llm_locust/core/models.py` - Added `user_id` to `RequestFailureLog`
  - `llm_locust/core/user.py` - Include `user_id` when logging failures
  - `llm_locust/metrics/per_request_logger.py` - Handle both success/failure
  - `llm_locust/metrics/collector.py` - Pass all logs to per_request_logger

- Refactored metrics collection architecture
- Improved error handling and logging
- Enhanced benchmark output formatting

### Impact
- Benchmark CSV files now contain complete request history
- Success rate accurately calculated: `count(status_code=200) / count(*)`
- Failure patterns visible in temporal analysis
- Dashboard can show error rate over time and by user

---

## [0.1.0] - Initial Architecture Refactoring

### Major Refactoring

Successfully refactored LLM Locust from a monolithic structure with WebUI to a clean, well-organized Python package focused on programmatic load testing.

### Added

#### Package Structure
Created logical organization by functionality:

```
llm-locust/
├── llm_locust/
│   ├── __init__.py
│   ├── core/              # Core load testing engine
│   │   ├── models.py      # Data models (frozen dataclasses)
│   │   ├── user.py        # User simulator
│   │   └── spawner.py     # User spawner (multiprocess)
│   ├── clients/           # LLM client implementations
│   │   └── openai.py      # OpenAI-compatible client
│   ├── metrics/           # Metrics collection & aggregation
│   │   ├── collector.py   # Metrics collector (threaded)
│   │   └── metrics.py     # Metric calculations
│   └── utils/             # Utilities
│       └── prompts.py     # Prompt management
├── examples/
│   └── simple_test.py     # Basic usage example
└── tests/                 # Test directory
```

#### Type Safety
- Comprehensive type annotations throughout codebase
- Used `TYPE_CHECKING` imports to avoid circular dependencies
- Made all dataclasses `frozen=True` for immutability
- Changed mutable collections to immutable (list → tuple in dataclasses)
- Added return type annotations to all functions
- Full `mypy --strict` compliance

**Example:**
```python
# Before
def __init__(self, model_client, metrics_queue, user_id=0):

# After
def __init__(
    self,
    model_client: "BaseModelClient",
    metrics_queue: Queue,
    user_id: int = 0,
) -> None:
```

#### Code Quality
- Added docstrings to all modules
- Improved error handling with structured logging
- Used `logger.error()` instead of `print()`
- Added error context (user_id, error_type, etc.)
- Made error messages actionable

**Before:**
```python
except Exception as e:
    logger.exception(f"Error in user loop: {e}")
    self.metrics_queue.put(ErrorLog(error_message=str(e)))
```

**After:**
```python
except (aiohttp.ClientError, asyncio.TimeoutError) as e:
    logger.warning(
        "Network error in user loop",
        exc_info=e,
        extra={
            "user_id": self.user_id,
            "error_type": type(e).__name__,
            "url": url,
        },
    )
    self.metrics_queue.put(
        ErrorLog(
            error_message=str(e),
            error_type=type(e).__name__,
            context={"user_id": self.user_id, "phase": "request"},
        )
    )
```

#### Configuration
- Modernized `pyproject.toml` with proper metadata
- Added `ruff` for linting and formatting
- Added `mypy` configuration for type checking
- Added development dependencies section
- Proper package discovery configuration

#### Documentation
- Created comprehensive `README.md`
- Added detailed `ARCHITECTURE.md`
- Added inline code documentation
- Created working example in `examples/simple_test.py`

### Removed

- **WebUI** (entire React frontend)
  - Deleted entire `webui/` directory
  - Removed UI-related dependencies (Node.js, Yarn, etc.)
  - Eliminated WebUI endpoints from API
  - Removed misleading `EXTENSIONS_OVERVIEW.md` documentation

- **Unused Components**
  - FastAPI endpoints (kept as reference)
  - UI dependencies (Jinja2, staticfiles)
  - Unused files (inputs.json, Dockerfile, image.png)

### Changed

**Before Structure:**
```
llm-locust/
├── api.py
├── clients.py
├── user.py
├── user_spawner.py
├── metrics_collector.py
├── metrics.py
├── utils.py
├── prompt.py
└── webui/
```

**After Structure:**
```
llm-locust/
├── llm_locust/
│   ├── core/      # Load testing engine
│   ├── clients/   # LLM clients
│   ├── metrics/   # Metrics collection
│   └── utils/     # Utilities
├── examples/
└── tests/
```

### Key Improvements

#### 1. Better Organization
- Logical grouping by functionality (core, clients, metrics, utils)
- Clear separation of concerns
- Easy to navigate and understand

#### 2. Type Safety
- Full type coverage enables IDE autocomplete
- Catches errors at development time
- Self-documenting code
- ~95% type coverage (up from 0%)

#### 3. Immutability
- Frozen dataclasses prevent accidental mutations
- Safer for multiprocess communication
- Clearer data flow

#### 4. Extensibility
- Easy to add new clients (implement `BaseModelClient`)
- Easy to add new metrics (implement `SimpleMetric`)
- Clear extension points documented

#### 5. Production Ready
- Proper logging with context
- Graceful error handling
- Resource cleanup
- Signal handling for shutdown

### Metrics

- **Files Removed**: ~100+ (entire webui/ directory)
- **Lines of Code Reduced**: ~10,000+ (WebUI removal)
- **Type Coverage**: ~95% (up from 0%)
- **Package Structure**: Clean subdirectories
- **Documentation**: Multiple comprehensive docs

### Migration Guide

#### For Existing Users

If you were using the WebUI:
1. The old API is available as `examples/old_api_reference.py` for reference
2. Use the programmatic API instead:

```python
from llm_locust import (
    OpenAIChatStreamingClient,
    UserSpawner,
    MetricsCollector,
)

# Your load test code here
```

#### For New Users

See `examples/simple_test.py` for a complete working example:

```bash
python examples/simple_test.py \
    --host http://localhost:8000 \
    --model llama-3.1-8b \
    --users 10 \
    --duration 300
```

---

## Feature History: YAML Cost Integration (Oct 2025)

### Problem Solved

**Before:** Manually entering GPU count, CPU, memory for each platform in the cost dashboard was tedious and error-prone.

**After:** Click one button to load specs directly from actual deployment manifests!

### How It Works

```
configs/engines/*.yaml
         ↓
    YAML Parser
    (lib/yaml_parser.py)
         ↓
    Extract specs:
    - GPU count
    - CPU cores  
    - Memory
    - Replicas
    - Model name
    - GPU memory utilization
         ↓
    Cost Analysis Page
         ↓
    Calculate costs using:
    - Proportional mode (GPU util-based)
    - Full instance mode (dedicated nodes)
```

### Features

#### Automatic Extraction
The parser extracts from deployment YAMLs:

| Field | Source | Example |
|-------|--------|---------|
| **Platform Name** | `metadata.name` | `vllm-test` → `vllm` |
| **GPU Count** | `resources.limits["nvidia.com/gpu"]` | `1` |
| **CPU Cores** | `resources.limits.cpu` | `4` |
| **Memory** | `resources.limits.memory` | `24Gi` |
| **Replicas** | `spec.replicas` | `1` |
| **Model Name** | `container.args` (--model flag) | `meta-llama/Llama-3.2-3B-Instruct` |
| **GPU Utilization** | `container.args` (--gpu-memory-utilization) | `0.92` |

#### Cost Allocation Modes

**Proportional Mode** (for shared K8s clusters):
```
Cost = (Instance cost / Instance GPUs) × GPU utilization × Deployment GPUs × Replicas
```

Example:
- Instance: AWS p5.48xlarge = $98.32/hour for 8 GPUs
- Your pod: 1 GPU × 60% utilization × 1 replica
- **Cost = ($98.32 / 8) × 0.60 × 1 = $7.37/hour**

**Full Instance Mode** (for dedicated nodes):
```
Cost = Instance cost × Replicas
```

#### Dashboard Integration

1. Place deployment YAMLs in `configs/engines/`
2. Open Cost Analysis page in Streamlit dashboard
3. Click **"🔄 Load Instance Specs from Deployment YAMLs"**
4. Review extracted specs (GPU count, memory, replicas, GPU util)
5. Select instance type from dropdown (GPU type and pricing)
6. Choose cost allocation mode (Proportional vs Full Instance)
7. View accurate cost comparison!

### Technical Implementation

**Parser Class:**
```python
class DeploymentConfig:
    platform_name: str              # "vllm", "vllm_cost", "tgi", "ollama"
    gpu_count: int                  # 1, 2, 4, 8
    cpu_cores: str                  # "4", "8", "16"
    memory_gi: str                  # "24Gi", "48Gi"
    replicas: int                   # 1, 2, 3
    model_name: Optional[str]       # "meta-llama/Llama-3.2-3B-Instruct"
    gpu_memory_utilization: Optional[float]  # 0.60, 0.92
```

**Key Functions:**
- `parse_deployment_yaml(path)` - Parse single YAML file
- `parse_all_yamls_in_directory(dir)` - Parse all YAMLs in directory
- `extract_gpu_memory_utilization_from_args(args)` - Extract GPU util from vLLM args
- `parse_memory(memory_str)` - Convert K8s memory format to GB

### Design Philosophy

**What We Extract (Facts Only):**
✅ Resource requests/limits are reliable  
✅ Replica count is accurate  
✅ Model name is useful for reference  
✅ GPU memory utilization from actual deployment args

**What Users Select (Deployment-Specific):**
🎯 GPU type (can't be reliably detected)  
🎯 Cloud provider (AWS, GCP, Azure, on-prem)  
🎯 Instance type (from pre-configured list or custom)  
🎯 Cost allocation mode (proportional vs full instance)

### Security Considerations

**Safe to Extract:**
- Resource limits (public information)
- Model names (often public)
- Platform names (public)
- GPU utilization settings

**Never Extracted:**
- Hostnames or URLs
- API keys or secrets
- Internal annotations
- Private label values

### Files Added
- `streamlit_app/lib/yaml_parser.py` (243 lines)
- `streamlit_app/requirements.txt` (added `pyyaml>=6.0.0`)
- Enhanced `streamlit_app/pages/6_Cost_Analysis.py`

### Example Use Case

**Your Deployment:**
```yaml
# configs/engines/vllm_cost.yaml
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: 20Gi
    cpu: "4"
args:
  - "--gpu-memory-utilization=0.60"  # Cost-optimized!
replicas: 1
```

**Dashboard Shows:**
- Proportional Mode: $1.09/hour (60% of single GPU cost)
- Full Instance Mode: $1.82/hour
- **Savings: 40% with proportional allocation!**

---

## Historical Context: Major Refactoring (v0.1.0)

### Overview

Transformed LLM Locust from a UI-heavy application to a clean, type-safe, well-organized Python package.

### What Changed

#### Removed UI Components
- Deleted entire `webui/` directory (React/TypeScript frontend)
- Removed UI-related dependencies (Node.js, Yarn, etc.)
- Eliminated WebUI endpoints from API
- Removed misleading `EXTENSIONS_OVERVIEW.md` documentation

#### Reorganized Package Structure

**Old Structure (Monolithic):**
```
llm-locust/
├── api.py
├── clients.py
├── user.py
├── user_spawner.py
├── metrics_collector.py
├── metrics.py
├── utils.py
├── prompt.py
└── webui/
```

**New Structure (Modular):**
```
llm-locust/
├── llm_locust/
│   ├── __init__.py
│   ├── core/              # Core load testing engine
│   │   ├── models.py      # Data models (formerly utils.py)
│   │   ├── user.py        # User simulator
│   │   └── spawner.py     # User spawner (formerly user_spawner.py)
│   ├── clients/           # LLM client implementations
│   │   └── openai.py      # OpenAI-compatible client (formerly clients.py)
│   ├── metrics/           # Metrics collection & aggregation
│   │   ├── collector.py   # Metrics collector (formerly metrics_collector.py)
│   │   └── metrics.py     # Metric calculations
│   └── utils/             # Utilities
│       └── prompts.py     # Prompt management (formerly prompt.py)
├── examples/
│   ├── simple_test.py     # NEW: Basic usage example
│   └── old_api_reference.py  # OLD: Reference implementation
└── tests/                 # Test directory
```

### Key Improvements

#### 1. Type Safety
- Added comprehensive type hints throughout codebase
- Full `mypy --strict` compliance
- Self-documenting code
- IDE autocomplete support

#### 2. Immutability
- Frozen dataclasses prevent accidental mutations
- Safer for multiprocess communication (IPC)
- Clearer data flow

#### 3. Better Organization
- Logical grouping by functionality
- Clear separation of concerns
- Easy to navigate and understand

#### 4. Extensibility
- Easy to add new clients (implement `BaseModelClient`)
- Easy to add new metrics (implement `SimpleMetric`)
- Clear extension points documented

#### 5. Production Ready
- Proper logging with context
- Graceful error handling
- Resource cleanup
- Signal handling for graceful shutdown

### What We Kept

**Core Functionality:**
- Multi-process architecture
- Async user simulation
- Metrics collection and aggregation
- OpenAI-compatible client
- Prompt loading utilities

**Key Features:**
- Streaming support
- TTFT/TPOT metrics
- Quantile calculations
- Dynamic user scaling
- Graceful shutdown

### Conclusion

The refactoring successfully transformed LLM Locust from a UI-heavy application to a clean, type-safe, well-organized Python package. The code is now:

- ✅ Easier to understand
- ✅ Easier to maintain
- ✅ Easier to extend
- ✅ Better documented
- ✅ Type-safe
- ✅ Production-ready

All core functionality has been preserved while significantly improving code quality and organization.

---

## Version History Summary

| Version | Date | Key Features |
|---------|------|--------------|
| **Unreleased** | 2025-10-04 | YAML cost integration, doc consolidation, cleaned configs |
| **0.2.0** | 2025-10-03 | Per-request logging, failed request tracking, benchmarks, dashboard |
| **0.1.0** | 2025-09-XX | Major refactoring, type safety, modular structure |

---

**Maintained By:** LLM Locust Team  
**Last Updated:** 2025-10-04  
**Format:** [Keep a Changelog](https://keepachangelog.com/)
