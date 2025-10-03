"""
Metrics collection from multiprocess queues with aggregation and logging.
"""

import logging
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable

from llm_locust.clients.openai import BaseModelClient
from llm_locust.core.models import (
    ErrorLog,
    RequestFailureLog,
    RequestSuccessLog,
    SetActiveUsers,
    SetLastProcessedTime,
    SetStartTime,
    TriggerShutdown,
    get_timestamp_seconds,
)
from llm_locust.metrics.metrics import LLMMetricsList

logger = logging.getLogger(__name__)

# TYPE_CHECKING import for PerRequestLogger to avoid circular dependency
if TYPE_CHECKING:
    from llm_locust.metrics.per_request_logger import PerRequestLogger


class MetricsCollector:
    """Handles collection of metrics via metrics queue, and their aggregation and logging"""

    def __init__(
        self,
        metrics_queue: Queue,
        model_client: BaseModelClient,
        metrics_window_size: int = 30,
        quantiles: list[int] | None = None,
        logging_function: Callable[[dict[str, Any]], None] | None = None,
        per_request_logger: "PerRequestLogger | None" = None,
    ) -> None:
        self.start_time = 0
        self.metrics_list = LLMMetricsList(quantiles or [50, 90, 99])
        self.on_going_users: int = 0
        self.quantiles: list[int] = quantiles or [50]
        self.metrics_window_size: int = metrics_window_size
        self.computed_metrics_task: Thread | None = None
        self.collection_task: Thread | None = None
        self.metrics_queue = metrics_queue
        self.model_client = model_client
        self.last_processed_request_time: int = 0
        self.running = False
        self._logging_function = logging_function or self._default_logging_function
        self.per_request_logger = per_request_logger

    def _default_logging_function(self, log_dict: dict[str, Any]) -> None:
        """Default logging function that prints to console."""
        # Format metrics for better readability
        active_users = log_dict.get("active_users", 0)
        rps = log_dict.get("requests_per_second", 0)
        failed_rps = log_dict.get("failed_requests_per_second", 0)
        ttft_p50 = log_dict.get("response_time_first_token_ms_quantile_50", 0)
        ttft_p90 = log_dict.get("response_time_first_token_ms_quantile_90", 0)
        ttft_p99 = log_dict.get("response_time_first_token_ms_quantile_99", 0)
        tpot_p50 = log_dict.get("tpot_ms_quantile_50", 0)
        tokens_per_sec = log_dict.get("total_output_tokens_per_second", 0)
        
        logger.info(
            f"👥 Users: {active_users:2d} | "
            f"📊 RPS: {rps:6.2f} | "
            f"❌ Failed: {failed_rps:5.2f} | "
            f"⚡ TTFT P50: {ttft_p50:6.1f}ms | "
            f"P90: {ttft_p90:6.1f}ms | "
            f"P99: {ttft_p99:6.1f}ms | "
            f"🔤 TPOT: {tpot_p50:5.1f}ms | "
            f"🚀 Tokens/s: {tokens_per_sec:6.1f}"
        )

    def logging_function(self, log_dict: dict[str, Any]) -> None:
        """Write log entries via configured logging function."""
        try:
            self._logging_function(log_dict)
        except Exception as e:
            logger.error(
                "Failed to log metrics",
                exc_info=e,
                extra={"metrics_count": len(log_dict)},
            )

    def start_logging(self) -> None:
        """Start worker threads for collection and reporting."""
        self.running = True

        self.collection_task = Thread(target=self.collect_metrics, daemon=True)
        self.collection_task.start()

        self.computed_metrics_task = Thread(
            target=self.report_metrics,
            kwargs={
                "sliding_window_size": self.metrics_window_size,
                "sliding_window_stride": 2,
                "metric_function": self.log_metrics,
            },
            daemon=True,
        )
        self.computed_metrics_task.start()

    def stop_logging(self) -> None:
        """Stop worker threads gracefully."""
        self.running = False
        if self.collection_task:
            self.collection_task.join(timeout=5)
        if self.computed_metrics_task:
            self.computed_metrics_task.join(timeout=5)

    def collect_metrics(self) -> None:
        """
        Collect all metrics from the queue.

        Runs in a background thread, processing messages until shutdown.
        """
        while self.running:
            try:
                metrics_data = self.metrics_queue.get(timeout=1)

                if isinstance(metrics_data, TriggerShutdown):
                    logger.info("Received shutdown signal in collector")
                    return

                if isinstance(metrics_data, RequestSuccessLog | RequestFailureLog):
                    self.metrics_list.collect_request(metrics_data, self)
                    
                    # Log per-request metrics if logger is enabled
                    if self.per_request_logger and isinstance(metrics_data, RequestSuccessLog):
                        self.per_request_logger.log_request(metrics_data, self.model_client)

                elif isinstance(metrics_data, ErrorLog):
                    logger.warning(
                        "Error from user loop",
                        extra={
                            "error": metrics_data.error_message,
                            "type": metrics_data.error_type,
                            "context": metrics_data.context,
                        },
                    )

                elif isinstance(metrics_data, SetActiveUsers):
                    self.on_going_users = metrics_data.total_users

                elif isinstance(metrics_data, SetStartTime):
                    self.start_time = metrics_data.start_time
                    logger.info("Test started", extra={"start_time": self.start_time})

                elif isinstance(metrics_data, SetLastProcessedTime):
                    self.last_processed_request_time = metrics_data.current_time

            except Empty:
                continue
            except Exception:
                logger.exception("Unexpected error in metrics collection")

    def log_metrics(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> None:
        """
        Calculate and log metrics for the specified time window.

        Args:
            start_timestamp: Start of time window
            end_timestamp: End of time window
        """
        if self.last_processed_request_time > 0:
            # Wait for processing to catch up to ensure complete metrics
            while self.running and self.last_processed_request_time < end_timestamp:
                time.sleep(max(0, end_timestamp - self.last_processed_request_time))

        metrics = {
            "active_users": self.on_going_users,
            **self.metrics_list.calculate(start_timestamp, end_timestamp),
        }

        self.logging_function(metrics)

    def report_metrics(
        self,
        sliding_window_size: int,
        sliding_window_stride: int,
        metric_function: Callable[[int, int], None],
    ) -> None:
        """
        Report metrics on a sliding window.

        Args:
            sliding_window_size: Window size in seconds
            sliding_window_stride: How often to report (seconds)
            metric_function: Function to call with (start_time, end_time)
        """
        # Wait for test to start
        while self.start_time == 0 and self.running:
            time.sleep(0.1)

        while self.running:
            end_timestamp = get_timestamp_seconds()
            start_timestamp = max(end_timestamp - sliding_window_size, self.start_time)

            metric_function(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

            time.sleep(sliding_window_stride)
