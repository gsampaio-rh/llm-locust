"""
Configuration and constants for the LLM Benchmark Dashboard.
"""

from typing import Final

# App Configuration
APP_TITLE: Final[str] = "🎯 LLM Benchmark Comparison Dashboard"
APP_ICON: Final[str] = "🎯"
LAYOUT: Final[str] = "wide"
INITIAL_SIDEBAR_STATE: Final[str] = "expanded"

# File Upload Configuration
MAX_FILE_SIZE_MB: Final[int] = 500
MAX_FILES: Final[int] = 20
SUPPORTED_FORMATS: Final[list[str]] = ["csv"]

# CSV Schema - Expected Columns
REQUIRED_COLUMNS: Final[list[str]] = [
    "request_id",
    "timestamp",
    "user_id",
    "user_request_num",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
    "end_to_end_s",
    "total_tokens_per_sec",
    "output_tokens_per_sec",
    "status_code",
]

OPTIONAL_COLUMNS: Final[list[str]] = [
    "input_prompt",
    "output_text",
]

# Platform Colors (for charts)
PLATFORM_COLORS: Final[dict[str, str]] = {
    # vLLM variants (VERY distinct colors)
    "vllm": "#8B5CF6",              # Purple (base vLLM)
    "vllm_cost": "#F97316",         # Orange (cost-optimized)
    "vllm-performance": "#EAB308",  # Bright Yellow (max performance)
    "vllm-quantized": "#22C55E",    # Bright Green (quantized/efficient)
    "vllm-perf": "#EAB308",         # Bright Yellow (alias for performance)
    "vllm-quant": "#22C55E",        # Bright Green (alias for quantized)
    
    # Other platforms
    "tgi": "#EC4899",               # Pink/Magenta
    "ollama": "#06B6D4",            # Cyan
    "openai": "#3B82F6",            # Blue
    "default": "#EF4444",           # Red (unknown)
}

# Status Colors
STATUS_COLORS: Final[dict[str, str]] = {
    "success": "#10B981",   # Green
    "warning": "#F59E0B",   # Amber
    "error": "#EF4444",     # Red
    "info": "#3B82F6",      # Blue
}

# Metrics Configuration
DEFAULT_PERCENTILES: Final[list[int]] = [50, 90, 95, 99]
DEFAULT_ROLLING_WINDOW: Final[int] = 100  # requests
DEFAULT_TIME_BUCKET_SECONDS: Final[int] = 60  # 1 minute

# Statistical Significance
ALPHA: Final[float] = 0.05  # 95% confidence
MIN_EFFECT_SIZE: Final[float] = 0.05  # 5% practical significance

# Data Quality Thresholds
MIN_REQUESTS_WARNING: Final[int] = 100
MIN_DURATION_WARNING: Final[int] = 60  # seconds
MAX_ERROR_RATE_WARNING: Final[float] = 0.05  # 5%
MAX_OUTLIER_RATE_WARNING: Final[float] = 0.10  # 10%

# Visualization Configuration
CHART_HEIGHT: Final[int] = 400
CHART_HEIGHT_SMALL: Final[int] = 300
CHART_HEIGHT_LARGE: Final[int] = 600

# Export Configuration
EXPORT_DPI: Final[int] = 300
EXPORT_FORMAT: Final[str] = "png"

# Success Criteria (for SLA checks)
DEFAULT_TTFT_P50_THRESHOLD_MS: Final[float] = 1000.0  # 1 second
DEFAULT_TTFT_P99_THRESHOLD_MS: Final[float] = 2000.0  # 2 seconds
DEFAULT_SUCCESS_RATE_THRESHOLD: Final[float] = 0.999  # 99.9%

# GPU and Cloud Pricing (approximate, as of Oct 2025)
# Pre-configured instance types with pricing
INSTANCE_CONFIGS: Final[dict[str, dict]] = {
    # AWS
    "AWS p5.48xlarge (8x H100)": {"provider": "AWS", "gpu": "H100", "gpu_count": 8, "cost_per_hour": 98.32},
    "AWS p4d.24xlarge (8x A100)": {"provider": "AWS", "gpu": "A100", "gpu_count": 8, "cost_per_hour": 32.77},
    "AWS p4de.24xlarge (8x A100)": {"provider": "AWS", "gpu": "A100", "gpu_count": 8, "cost_per_hour": 40.97},
    "AWS g6.xlarge (1x L4)": {"provider": "AWS", "gpu": "L4", "gpu_count": 1, "cost_per_hour": 0.857},
    "AWS g6.2xlarge (1x L4)": {"provider": "AWS", "gpu": "L4", "gpu_count": 1, "cost_per_hour": 1.212},
    "AWS g6.4xlarge (1x L4)": {"provider": "AWS", "gpu": "L4", "gpu_count": 1, "cost_per_hour": 1.82},
    "AWS g6.12xlarge (4x L4)": {"provider": "AWS", "gpu": "L4", "gpu_count": 4, "cost_per_hour": 4.848},
    "AWS g6.48xlarge (8x L4)": {"provider": "AWS", "gpu": "L4", "gpu_count": 8, "cost_per_hour": 9.696},
    "AWS g5.xlarge (1x A10G)": {"provider": "AWS", "gpu": "A10G", "gpu_count": 1, "cost_per_hour": 1.006},
    "AWS g5.12xlarge (4x A10G)": {"provider": "AWS", "gpu": "A10G", "gpu_count": 4, "cost_per_hour": 5.672},
    "AWS g5.48xlarge (8x A10G)": {"provider": "AWS", "gpu": "A10G", "gpu_count": 8, "cost_per_hour": 16.288},
    "AWS p3.2xlarge (1x V100)": {"provider": "AWS", "gpu": "V100", "gpu_count": 1, "cost_per_hour": 3.06},
    "AWS p3.8xlarge (4x V100)": {"provider": "AWS", "gpu": "V100", "gpu_count": 4, "cost_per_hour": 12.24},
    
    # GCP
    "GCP a3-highgpu-8g (8x H100)": {"provider": "GCP", "gpu": "H100", "gpu_count": 8, "cost_per_hour": 29.39},
    "GCP a2-highgpu-1g (1x A100)": {"provider": "GCP", "gpu": "A100", "gpu_count": 1, "cost_per_hour": 3.67},
    "GCP a2-highgpu-8g (8x A100)": {"provider": "GCP", "gpu": "A100", "gpu_count": 8, "cost_per_hour": 29.39},
    "GCP g2-standard-4 (1x L4)": {"provider": "GCP", "gpu": "L4", "gpu_count": 1, "cost_per_hour": 0.79},
    "GCP g2-standard-48 (4x L4)": {"provider": "GCP", "gpu": "L4", "gpu_count": 4, "cost_per_hour": 3.16},
    
    # Azure
    "Azure ND_H100_v5 (8x H100)": {"provider": "Azure", "gpu": "H100", "gpu_count": 8, "cost_per_hour": 27.20},
    "Azure Standard_ND96amsr_A100_v4 (8x A100)": {"provider": "Azure", "gpu": "A100", "gpu_count": 8, "cost_per_hour": 27.20},
    "Azure Standard_NC6s_v3 (1x V100)": {"provider": "Azure", "gpu": "V100", "gpu_count": 1, "cost_per_hour": 3.06},
    "Azure Standard_NC24s_v3 (4x V100)": {"provider": "Azure", "gpu": "V100", "gpu_count": 4, "cost_per_hour": 12.24},
    
    # On-prem (amortized estimates)
    "On-prem H100 Server (8x H100)": {"provider": "On-prem", "gpu": "H100", "gpu_count": 8, "cost_per_hour": 15.00},
    "On-prem A100 Server (8x A100)": {"provider": "On-prem", "gpu": "A100", "gpu_count": 8, "cost_per_hour": 10.00},
    "On-prem L40S Server (8x L40S)": {"provider": "On-prem", "gpu": "L40S", "gpu_count": 8, "cost_per_hour": 8.00},
    
    # Custom option
    "Custom (Enter Manually)": {"provider": "Custom", "gpu": "Custom", "gpu_count": 1, "cost_per_hour": 0.0},
}

GPU_TYPES: Final[list[str]] = ["H100", "A100", "L40S", "L4", "A10G", "V100"]
CLOUD_PROVIDERS: Final[list[str]] = ["AWS", "GCP", "Azure", "On-prem"]

