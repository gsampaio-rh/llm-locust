"""
Export functionality for benchmark summary reports.

Generates executive summaries in CSV and Markdown formats.
"""

import io
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from models.benchmark import BenchmarkData


def generate_summary_data(benchmarks: list["BenchmarkData"]) -> pd.DataFrame:
    """
    Generate summary data for all benchmarks.

    Args:
        benchmarks: List of benchmark data objects

    Returns:
        DataFrame with summary metrics
    """
    summary_data = []

    for benchmark in benchmarks:
        summary = {
            "Platform": benchmark.metadata.platform,
            "Total Requests": benchmark.metadata.total_requests,
            "Failed Requests": benchmark.metadata.failed_requests,
            "Success Rate (%)": round(benchmark.success_rate * 100, 2),
            "Duration (s)": int(benchmark.metadata.duration_seconds),
            "TTFT P50 (ms)": round(benchmark.ttft_p50, 1),
            "TTFT P90 (ms)": round(benchmark.ttft_p90, 1),
            "TTFT P99 (ms)": round(benchmark.ttft_p99, 1),
            "TPOT P50 (ms)": round(benchmark.tpot_p50, 1),
            "TPOT P90 (ms)": round(benchmark.tpot_p90, 1),
            "TPOT P99 (ms)": round(benchmark.tpot_p99, 1),
            "Throughput Avg (tok/s)": round(benchmark.throughput_avg, 1),
            "Throughput P90 (tok/s)": round(benchmark.throughput_p90, 1),
            "Total Tokens": benchmark.df["output_tokens"].sum(),
            "Avg Input Tokens": round(benchmark.df["input_tokens"].mean(), 1),
            "Avg Output Tokens": round(benchmark.df["output_tokens"].mean(), 1),
        }
        summary_data.append(summary)

    return pd.DataFrame(summary_data)


def export_summary_csv(benchmarks: list["BenchmarkData"]) -> str:
    """
    Export summary as CSV string.

    Args:
        benchmarks: List of benchmark data objects

    Returns:
        CSV formatted string
    """
    df = generate_summary_data(benchmarks)
    return df.to_csv(index=False)


def export_summary_markdown(benchmarks: list["BenchmarkData"]) -> str:
    """
    Export summary as Markdown report.

    Args:
        benchmarks: List of benchmark data objects

    Returns:
        Markdown formatted string
    """
    df = generate_summary_data(benchmarks)

    # Generate markdown report
    md = io.StringIO()

    # Header
    md.write("# LLM Benchmark Comparison Report\n\n")
    md.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.write(f"**Platforms Compared:** {len(benchmarks)}\n\n")

    md.write("---\n\n")

    # Executive Summary
    md.write("## 📊 Executive Summary\n\n")

    # Find winners
    winner_ttft_idx = df["TTFT P50 (ms)"].idxmin()
    winner_throughput_idx = df["Throughput Avg (tok/s)"].idxmax()
    winner_reliability_idx = df["Success Rate (%)"].idxmax()

    md.write(f"**🏆 Best Latency (TTFT):** {df.loc[winner_ttft_idx, 'Platform']} "
             f"({df.loc[winner_ttft_idx, 'TTFT P50 (ms)']}ms)\n\n")
    md.write(f"**🚀 Best Throughput:** {df.loc[winner_throughput_idx, 'Platform']} "
             f"({df.loc[winner_throughput_idx, 'Throughput Avg (tok/s)']} tok/s)\n\n")
    md.write(f"**✅ Best Reliability:** {df.loc[winner_reliability_idx, 'Platform']} "
             f"({df.loc[winner_reliability_idx, 'Success Rate (%)']}%)\n\n")

    md.write("---\n\n")

    # Full results table
    md.write("## 📈 Detailed Results\n\n")
    md.write(df.to_markdown(index=False))
    md.write("\n\n")

    md.write("---\n\n")

    # Key insights
    md.write("## 💡 Key Insights\n\n")

    # TTFT comparison
    ttft_best = df["TTFT P50 (ms)"].min()
    ttft_worst = df["TTFT P50 (ms)"].max()
    if ttft_worst > 0:
        ttft_diff = ((ttft_worst - ttft_best) / ttft_best) * 100
        md.write(f"- **Latency Range:** {ttft_diff:.1f}% difference between "
                 f"fastest ({ttft_best:.1f}ms) and slowest ({ttft_worst:.1f}ms)\n")

    # Throughput comparison
    tput_best = df["Throughput Avg (tok/s)"].max()
    tput_worst = df["Throughput Avg (tok/s)"].min()
    if tput_worst > 0:
        tput_diff = ((tput_best - tput_worst) / tput_worst) * 100
        md.write(f"- **Throughput Range:** {tput_diff:.1f}% difference between "
                 f"highest ({tput_best:.1f} tok/s) and lowest ({tput_worst:.1f} tok/s)\n")

    # Reliability
    avg_success = df["Success Rate (%)"].mean()
    md.write(f"- **Average Success Rate:** {avg_success:.2f}%\n")

    # Total processing
    total_requests = df["Total Requests"].sum()
    total_tokens = df["Total Tokens"].sum()
    md.write(f"- **Total Requests Processed:** {total_requests:,}\n")
    md.write(f"- **Total Tokens Generated:** {total_tokens:,}\n")

    md.write("\n---\n\n")

    # Recommendations
    md.write("## 🎯 Recommendations\n\n")

    best_overall = df.loc[winner_ttft_idx, "Platform"]
    md.write(f"**For Low-Latency Applications:** Use `{best_overall}` "
             f"(fastest TTFT: {df.loc[winner_ttft_idx, 'TTFT P50 (ms)']}ms)\n\n")

    best_throughput = df.loc[winner_throughput_idx, "Platform"]
    md.write(f"**For High-Throughput Workloads:** Use `{best_throughput}` "
             f"(best throughput: {df.loc[winner_throughput_idx, 'Throughput Avg (tok/s)']} tok/s)\n\n")

    best_reliable = df.loc[winner_reliability_idx, "Platform"]
    md.write(f"**For Mission-Critical Systems:** Use `{best_reliable}` "
             f"(highest reliability: {df.loc[winner_reliability_idx, 'Success Rate (%)']}%)\n\n")

    md.write("---\n\n")
    md.write("*Report generated by LLM Locust Benchmark Dashboard*\n")

    return md.getvalue()


def create_download_button(
    data: str,
    filename: str,
    file_format: str,
    label: str,
) -> bytes:
    """
    Prepare data for Streamlit download button.

    Args:
        data: Data string to download
        filename: Suggested filename
        file_format: Format type (csv, md, etc.)
        label: Button label

    Returns:
        Data as bytes for download
    """
    return data.encode("utf-8")

