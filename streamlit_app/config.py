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
    "vllm": "#8B5CF6",      # Purple
    "tgi": "#EC4899",       # Pink
    "ollama": "#06B6D4",    # Cyan
    "openai": "#10B981",    # Green
    "default": "#F59E0B",   # Amber
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

