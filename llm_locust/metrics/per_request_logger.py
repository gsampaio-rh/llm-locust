"""
Per-request metrics logging for detailed analysis.

Logs individual request metrics including TTFT, TPOT, and end-to-end latency.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from llm_locust.core.models import RequestSuccessLog

logger = logging.getLogger(__name__)


class PerRequestLogger:
    """
    Logs detailed metrics for each individual request.
    
    Captures:
    - TTFT (Time to First Token)
    - TPOT (Time Per Output Token)
    - End-to-End Latency
    - Token counts
    - Throughput
    """

    def __init__(
        self,
        output_file: Path | str,
        format: str = "csv",
        print_to_console: bool = False,
    ) -> None:
        """
        Initialize per-request logger.

        Args:
            output_file: Path to output file
            format: Output format ('csv' or 'jsonl')
            print_to_console: Also print to console
        """
        self.output_file = Path(output_file)
        self.format = format.lower()
        self.print_to_console = print_to_console
        self.request_count = 0

        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        if self.format == "csv" and not self.output_file.exists():
            self._write_csv_header()

        logger.info(f"Per-request logging enabled: {self.output_file}")

    def _write_csv_header(self) -> None:
        """Write CSV header row."""
        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id",
                "timestamp",
                "input_tokens",
                "output_tokens",
                "ttft_ms",
                "tpot_ms",
                "end_to_end_s",
                "total_tokens_per_sec",
                "output_tokens_per_sec",
                "status_code",
            ])

    def log_request(
        self,
        request_log: RequestSuccessLog,
        model_client: Any,
    ) -> None:
        """
        Log metrics for a single request.

        Args:
            request_log: Request success log
            model_client: Client for parsing response
        """
        self.request_count += 1

        # Calculate output token count
        output_tokens = 0
        for chunk in request_log.result_chunks:
            tokens = model_client.parse_response(chunk)
            output_tokens += len(tokens)

        # Calculate timing metrics
        total_duration_s = request_log.end_time - request_log.start_time

        # TTFT - Time to First Token
        ttft_ms = 0.0
        if request_log.token_times:
            ttft_ms = (request_log.token_times[0] - request_log.start_time) * 1000

        # TPOT - Time Per Output Token
        tpot_ms = 0.0
        if output_tokens > 0:
            generation_time = request_log.end_time - request_log.start_time
            if request_log.token_times:
                # Time from first token to end
                generation_time = request_log.end_time - request_log.token_times[0]
            tpot_ms = (generation_time * 1000) / output_tokens

        # Throughput
        total_tokens = request_log.num_input_tokens + output_tokens
        total_tokens_per_sec = total_tokens / total_duration_s if total_duration_s > 0 else 0
        output_tokens_per_sec = output_tokens / total_duration_s if total_duration_s > 0 else 0

        metrics = {
            "request_id": self.request_count,
            "timestamp": request_log.timestamp,
            "input_tokens": request_log.num_input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "end_to_end_s": round(total_duration_s, 3),
            "total_tokens_per_sec": round(total_tokens_per_sec, 2),
            "output_tokens_per_sec": round(output_tokens_per_sec, 2),
            "status_code": request_log.status_code,
        }

        # Write to file
        if self.format == "csv":
            self._write_csv_row(metrics)
        else:
            self._write_jsonl_row(metrics)

        # Print to console if enabled
        if self.print_to_console:
            self._print_metrics(metrics)

    def _write_csv_row(self, metrics: dict[str, Any]) -> None:
        """Write metrics row to CSV."""
        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics["request_id"],
                metrics["timestamp"],
                metrics["input_tokens"],
                metrics["output_tokens"],
                metrics["ttft_ms"],
                metrics["tpot_ms"],
                metrics["end_to_end_s"],
                metrics["total_tokens_per_sec"],
                metrics["output_tokens_per_sec"],
                metrics["status_code"],
            ])

    def _write_jsonl_row(self, metrics: dict[str, Any]) -> None:
        """Write metrics row to JSONL."""
        with open(self.output_file, "a") as f:
            json.dump(metrics, f)
            f.write("\n")

    def _print_metrics(self, metrics: dict[str, Any]) -> None:
        """Print metrics to console."""
        logger.info(
            f"📋 Request #{metrics['request_id']:4d} | "
            f"⏱️  TTFT: {metrics['ttft_ms']:7.1f}ms | "
            f"🔄 TPOT: {metrics['tpot_ms']:6.1f}ms | "
            f"⏰ E2E: {metrics['end_to_end_s']:6.3f}s | "
            f"📥 In: {metrics['input_tokens']:4d} | "
            f"📤 Out: {metrics['output_tokens']:4d} | "
            f"🚀 {metrics['output_tokens_per_sec']:6.1f} tok/s"
        )

    def close(self) -> None:
        """Close logger and print summary."""
        logger.info(
            f"✅ Logged {self.request_count} requests to {self.output_file}"
        )

