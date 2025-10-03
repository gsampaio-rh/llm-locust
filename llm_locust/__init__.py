"""
LLM Locust - Load testing tool for Large Language Model inference endpoints.

A specialized load testing framework for streaming LLM workloads with real-time
metrics collection including TTFT (Time to First Token), TPOT (Time Per Output Token),
and throughput metrics.
"""

__version__ = "0.2.0"

from llm_locust.clients.openai import BaseModelClient, OpenAIChatStreamingClient
from llm_locust.core.models import (
    ErrorLog,
    RequestFailureLog,
    RequestSuccessLog,
)
from llm_locust.core.spawner import UserSpawner, start_user_loop
from llm_locust.core.user import User
from llm_locust.metrics.collector import MetricsCollector
from llm_locust.metrics.metrics import LLMMetricsList

__all__ = [
    "BaseModelClient",
    "OpenAIChatStreamingClient",
    "RequestFailureLog",
    "RequestSuccessLog",
    "ErrorLog",
    "UserSpawner",
    "start_user_loop",
    "User",
    "MetricsCollector",
    "LLMMetricsList",
]
