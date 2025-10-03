"""
Data models for benchmark data.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class BenchmarkMetadata(BaseModel):
    """Metadata extracted from benchmark CSV."""

    platform: str  # vllm, tgi, ollama, etc.
    benchmark_id: str = "unknown"  # 1a-chat-simulation, etc.
    timestamp: datetime = Field(default_factory=datetime.now)
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    concurrency: int = 0  # Estimated from unique user_ids
    filename: str = ""


class BenchmarkData(BaseModel):
    """Complete benchmark dataset with calculated metrics."""

    class Config:
        arbitrary_types_allowed = True

    metadata: BenchmarkMetadata
    df: pd.DataFrame  # Raw data
    quality_score: float = Field(ge=0, le=100, default=100.0)

    # Pre-calculated metrics (successful requests only)
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p90: float = 0.0
    tpot_p99: float = 0.0
    throughput_avg: float = 0.0  # tokens/sec
    success_rate: float = 1.0  # 0.0 to 1.0
    rps: float = 0.0  # requests per second


class ComparisonResult(BaseModel):
    """Result of comparing two benchmarks."""

    platform_a: str
    platform_b: str
    winner: Literal["a", "b", "tie"]

    # Metric differences (percentage)
    ttft_p50_diff_pct: float
    ttft_p99_diff_pct: float
    tpot_p50_diff_pct: float
    throughput_diff_pct: float
    success_rate_diff_pct: float

    # Statistical significance
    ttft_significant: bool = False
    tpot_significant: bool = False
    p_value_ttft: float = 1.0
    p_value_tpot: float = 1.0

    # Winner determination
    better_latency: Literal["a", "b", "tie"] = "tie"
    better_throughput: Literal["a", "b", "tie"] = "tie"
    better_reliability: Literal["a", "b", "tie"] = "tie"

