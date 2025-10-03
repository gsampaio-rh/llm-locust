"""
Per-request metrics logging for detailed analysis.

Logs individual request metrics including TTFT, TPOT, and end-to-end latency.
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_locust.core.models import RequestFailureLog, RequestSuccessLog

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
        summary_interval: int = 10,
        include_text: bool = True,
        max_text_length: int = 500,
    ) -> None:
        """
        Initialize per-request logger.

        Args:
            output_file: Path to output file
            format: Output format ('csv' or 'jsonl')
            print_to_console: Also print to console
            summary_interval: Print summary every N requests (0 = print all)
            include_text: Include input prompt and output text in logs
            max_text_length: Maximum text length to log (prevents huge CSVs)
        """
        self.output_file = Path(output_file)
        self.format = format.lower()
        self.print_to_console = print_to_console
        self.summary_interval = summary_interval
        self.include_text = include_text
        self.max_text_length = max_text_length
        self.request_count = 0
        
        # Per-user tracking
        self.user_requests: dict[int, int] = defaultdict(int)
        self.user_metrics: dict[int, list[float]] = defaultdict(list)

        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        if self.format == "csv":
            # Write headers if file doesn't exist or is empty
            if not self.output_file.exists() or self.output_file.stat().st_size == 0:
                self._write_csv_header()
            else:
                logger.warning(
                    f"Appending to existing CSV file: {self.output_file}. "
                    "Delete file if you want a fresh start."
                )

        logger.info(f"Per-request logging enabled: {self.output_file}")

    def _write_csv_header(self) -> None:
        """Write CSV header row."""
        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            headers = [
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
            if self.include_text:
                headers.extend(["input_prompt", "output_text"])
            writer.writerow(headers)

    def log_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        model_client: Any,
    ) -> None:
        """
        Log metrics for a single request (success or failure).

        Args:
            request_log: Request success or failure log
            model_client: Client for parsing response
        """
        self.request_count += 1
        
        # Handle failures separately
        if isinstance(request_log, RequestFailureLog):
            self._log_failure(request_log)
            return

        # Calculate output token count and decode text
        output_tokens = 0
        output_token_ids: list[int] = []
        for chunk in request_log.result_chunks:
            tokens = model_client.parse_response(chunk)
            output_tokens += len(tokens)
            output_token_ids.extend(tokens)

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

        # Decode output text if requested
        input_prompt = ""
        output_text = ""
        if self.include_text:
            # TODO: Input prompt not currently stored in RequestSuccessLog
            # Would need to modify core data flow to include it
            # For now, input_prompt will be empty
            input_prompt = f"[{request_log.num_input_tokens} tokens]"
            
            # Decode output tokens
            try:
                if output_token_ids:
                    output_text = model_client.tokenizer.decode(
                        output_token_ids,
                        skip_special_tokens=True,
                    )
                    # Truncate if too long
                    if len(output_text) > self.max_text_length:
                        output_text = output_text[:self.max_text_length] + "..."
                else:
                    output_text = "[empty output]"
            except Exception as e:
                logger.debug(f"Failed to decode output text: {e}")
                output_text = "[decode failed]"

        metrics = {
            "request_id": self.request_count,
            "timestamp": request_log.timestamp,
            "user_id": request_log.user_id,
            "input_tokens": request_log.num_input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "end_to_end_s": round(total_duration_s, 3),
            "total_tokens_per_sec": round(total_tokens_per_sec, 2),
            "output_tokens_per_sec": round(output_tokens_per_sec, 2),
            "status_code": request_log.status_code,
        }
        
        if self.include_text:
            metrics["input_prompt"] = input_prompt
            metrics["output_text"] = output_text

        # Track per-user stats
        user_id = request_log.user_id
        self.user_requests[user_id] += 1
        user_request_num = self.user_requests[user_id]
        self.user_metrics[user_id].append(ttft_ms)
        
        # Add user request number to metrics
        metrics["user_request_num"] = user_request_num

        # Write to file
        if self.format == "csv":
            self._write_csv_row(metrics)
        else:
            self._write_jsonl_row(metrics)

        # Print to console if enabled
        if self.print_to_console:
            # Print every request or use summary mode
            if self.summary_interval == 0 or self.request_count % self.summary_interval == 0:
                self._print_metrics(metrics)

    def _log_failure(self, request_log: RequestFailureLog) -> None:
        """Log a failed request with minimal metrics."""
        # Calculate duration
        total_duration_s = request_log.end_time - request_log.start_time
        
        # Build metrics dictionary with null values for unavailable data
        metrics = {
            "request_id": self.request_count,
            "timestamp": request_log.timestamp,
            "user_id": request_log.user_id,
            "input_tokens": 0,  # Not available for failures
            "output_tokens": 0,
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "end_to_end_s": round(total_duration_s, 3),
            "total_tokens_per_sec": 0.0,
            "output_tokens_per_sec": 0.0,
            "status_code": request_log.status_code,
        }
        
        if self.include_text:
            metrics["input_prompt"] = "[FAILED]"
            metrics["output_text"] = f"[ERROR {request_log.status_code}]"
        
        # Track per-user stats
        user_id = request_log.user_id
        self.user_requests[user_id] += 1
        user_request_num = self.user_requests[user_id]
        
        # Add user request number to metrics
        metrics["user_request_num"] = user_request_num
        
        # Write to file
        if self.format == "csv":
            self._write_csv_row(metrics)
        else:
            self._write_jsonl_row(metrics)
        
        # Print to console if enabled
        if self.print_to_console:
            if self.summary_interval == 0 or self.request_count % self.summary_interval == 0:
                self._print_failure_metrics(metrics)

    def _write_csv_row(self, metrics: dict[str, Any]) -> None:
        """Write metrics row to CSV."""
        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                metrics["request_id"],
                metrics["timestamp"],
                metrics["user_id"],
                metrics["user_request_num"],
                metrics["input_tokens"],
                metrics["output_tokens"],
                metrics["ttft_ms"],
                metrics["tpot_ms"],
                metrics["end_to_end_s"],
                metrics["total_tokens_per_sec"],
                metrics["output_tokens_per_sec"],
                metrics["status_code"],
            ]
            if self.include_text:
                row.extend([
                    metrics.get("input_prompt", ""),
                    metrics.get("output_text", ""),
                ])
            writer.writerow(row)

    def _write_jsonl_row(self, metrics: dict[str, Any]) -> None:
        """Write metrics row to JSONL."""
        with open(self.output_file, "a") as f:
            json.dump(metrics, f)
            f.write("\n")

    def _print_metrics(self, metrics: dict[str, Any]) -> None:
        """Print metrics to console."""
        logger.info(
            f"[REQUEST]   "
            f"#{metrics['request_id']:4d} | User {metrics['user_id']:2d} / Req {metrics['user_request_num']:4d} | "
            f"⏱️  TTFT: {metrics['ttft_ms']:7.1f}ms | "
            f"🔄 TPOT: {metrics['tpot_ms']:6.1f}ms | "
            f"⏰ E2E: {metrics['end_to_end_s']:6.3f}s | "
            f"📥 In: {metrics['input_tokens']:4d} | "
            f"📤 Out: {metrics['output_tokens']:4d} | "
            f"🚀 {metrics['output_tokens_per_sec']:6.1f} tok/s"
        )

    def _print_failure_metrics(self, metrics: dict[str, Any]) -> None:
        """Print failure metrics to console."""
        logger.error(
            f"[FAILURE] ❌ "
            f"#{metrics['request_id']:4d} | User {metrics['user_id']:2d} / Req {metrics['user_request_num']:4d} | "
            f"Status: {metrics['status_code']} | "
            f"⏰ E2E: {metrics['end_to_end_s']:6.3f}s"
        )

    def print_summary(self) -> None:
        """Print test summary statistics."""
        if self.request_count == 0:
            logger.info("No requests logged yet")
            return

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Requests: {self.request_count}")
        logger.info("")
        logger.info("Per-User Request Distribution:")
        logger.info("-" * 80)
        
        for user_id in sorted(self.user_requests.keys()):
            count = self.user_requests[user_id]
            ttft_values = self.user_metrics[user_id]
            avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0
            logger.info(
                f"  User {user_id:2d}: {count:4d} requests | "
                f"Avg TTFT: {avg_ttft:6.1f}ms"
            )
        
        logger.info("-" * 80)
        logger.info(f"📁 Full results saved to: {self.output_file}")
        logger.info("=" * 80)

    def close(self) -> None:
        """Close logger and print summary."""
        self.print_summary()
        logger.info(f"✅ Per-request logging complete")

