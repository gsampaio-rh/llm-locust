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
            "Total Tokens": int(benchmark.df["output_tokens"].sum()),
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


def export_summary_markdown(
    benchmarks: list["BenchmarkData"],
    cost_configs: dict | None = None,
    yaml_configs: dict | None = None,
) -> str:
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
    md.write("# LLM Inference Performance Benchmark Report\n\n")
    
    md.write("## 📄 Report Overview\n\n")
    md.write("This report contains performance analysis of Large Language Model (LLM) serving platforms "
             "under production-like workloads. Each platform was tested with identical prompts and "
             "configurations to ensure fair comparison.\n\n")
    
    md.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.write(f"**Platforms Tested:** {len(benchmarks)}\n\n")
    
    # List platforms
    md.write("**Platforms:**\n")
    for benchmark in benchmarks:
        md.write(f"- {benchmark.metadata.platform}\n")
    md.write("\n")
    
    # Benchmark test information
    if benchmarks:
        # Get benchmark ID from first platform (they should all be the same test)
        benchmark_id = benchmarks[0].metadata.benchmark_id
        md.write(f"**Test Type:** `{benchmark_id}`\n\n")
        
        # Add test description based on benchmark ID
        test_descriptions = {
            "1a-chat-simulation": "Chat Simulation (256 input / 128 output tokens) - Conversational AI workload",
            "1b-rag-simulation": "RAG Simulation (4096 input / 512 output tokens) - Large context processing",
            "1c-code-generation": "Code Generation (512 input / 512 output tokens) - Balanced workload",
            "2a-constant-rate": "Constant Rate (512 input / 256 output tokens) - Sustained load",
            "2b-poisson-rate": "Poisson Rate (512 input / 256 output tokens) - Bursty traffic",
        }
        
        description = test_descriptions.get(benchmark_id, "Custom benchmark test")
        md.write(f"**Description:** {description}\n\n")
        
        md.write("---\n\n")
        
        # What was tested
        md.write("## 🎯 What Was Tested\n\n")
        
        # Explain the test scenario
        test_objectives = {
            "1a-chat-simulation": (
                "This test simulates conversational AI applications like chatbots and virtual assistants. "
                "It measures how quickly the system responds to short prompts and generates concise answers, "
                "which is critical for interactive user experiences."
            ),
            "1b-rag-simulation": (
                "This test simulates Retrieval-Augmented Generation (RAG) systems that process large context "
                "windows. It measures the system's ability to handle long documents and generate detailed responses, "
                "typical of knowledge base assistants and document analysis tools."
            ),
            "1c-code-generation": (
                "This test simulates AI coding assistants and development tools. It uses balanced input/output "
                "lengths typical of code completion and generation tasks, measuring both prompt processing "
                "and generation efficiency."
            ),
            "2a-constant-rate": (
                "This test validates system reliability under sustained, predictable load. It simulates "
                "production environments with steady traffic patterns, measuring stability and consistency "
                "over extended periods."
            ),
            "2b-poisson-rate": (
                "This test evaluates system robustness under unpredictable, bursty traffic patterns. "
                "It simulates real-world scenarios with traffic spikes, measuring how well the system "
                "handles sudden load increases and queue management."
            ),
        }
        
        objective = test_objectives.get(benchmark_id, 
            "This test evaluates LLM serving platform performance under controlled workload conditions.")
        md.write(f"**Test Objective:** {objective}\n\n")
        
        md.write("---\n\n")
        
        # Technical Test Configuration
        md.write("## ⚙️ Technical Test Configuration\n\n")
        
        # Calculate aggregated test stats
        total_requests = sum(b.metadata.total_requests for b in benchmarks)
        total_failures = sum(b.metadata.failed_requests for b in benchmarks)
        avg_duration = sum(b.metadata.duration_seconds for b in benchmarks) / len(benchmarks)
        
        # Input/Output token stats (from actual data)
        all_input_tokens = pd.concat([b.df["input_tokens"] for b in benchmarks])
        all_output_tokens = pd.concat([b.df["output_tokens"] for b in benchmarks])
        
        md.write(f"**Test Duration:** {avg_duration:.0f} seconds ({avg_duration/60:.1f} minutes)\n\n")
        md.write(f"**Total Load:**\n")
        md.write(f"- Total Requests: {total_requests:,}\n")
        md.write(f"- Total Failures: {total_failures:,}\n")
        md.write(f"- Success Rate: {((total_requests-total_failures)/total_requests*100):.2f}%\n\n")
        
        md.write(f"**Token Distribution:**\n")
        md.write(f"- Input Tokens: {all_input_tokens.mean():.0f} avg "
                 f"(min: {all_input_tokens.min()}, max: {all_input_tokens.max()})\n")
        md.write(f"- Output Tokens: {all_output_tokens.mean():.0f} avg "
                 f"(min: {all_output_tokens.min()}, max: {all_output_tokens.max()})\n")
        md.write(f"- Total Tokens Processed: {(all_input_tokens.sum() + all_output_tokens.sum()):,}\n\n")
        
        md.write(f"**Concurrency:**\n")
        # Estimate concurrency from request rate
        for benchmark in benchmarks:
            rps = benchmark.metadata.total_requests / benchmark.metadata.duration_seconds if benchmark.metadata.duration_seconds > 0 else 0
            md.write(f"- {benchmark.metadata.platform}: ~{rps:.1f} req/s\n")
        
        md.write("\n")
        
        md.write(f"**Dataset:**\n")
        dataset_info = {
            "1a-chat-simulation": "ShareGPT (real user conversations)",
            "1b-rag-simulation": "BillSum (US legislative documents)",
            "1c-code-generation": "Synthetic code generation prompts",
            "2a-constant-rate": "ShareGPT (mixed conversational)",
            "2b-poisson-rate": "ShareGPT (mixed conversational)",
        }
        md.write(f"- Source: {dataset_info.get(benchmark_id, 'Custom dataset')}\n")
        md.write(f"- Prompts: Randomly sampled from dataset pool\n")
        md.write(f"- Same prompts sent to all platforms (fair comparison)\n\n")
        
        md.write(f"**Load Testing Tool:** LLM Locust {benchmarks[0].metadata.llm_locust_version if hasattr(benchmarks[0].metadata, 'llm_locust_version') else 'v0.2.0+'}\n\n")
        md.write(f"**Methodology:** Multi-process async load generation with streaming response capture\n\n")

    md.write("---\n\n")
    
    # Platform configurations (if available)
    if yaml_configs:
        md.write("## 🔧 Platform Configurations\n\n")
        md.write("Technical specifications for each platform:\n\n")
        
        for benchmark in benchmarks:
            platform = benchmark.metadata.platform
            yaml_config = yaml_configs.get(platform)
            
            if yaml_config:
                md.write(f"### {platform}\n\n")
                md.write(f"**Resources:**\n")
                md.write(f"- GPUs: {yaml_config.gpu_count}\n")
                md.write(f"- CPU Cores: {yaml_config.cpu_cores}\n")
                md.write(f"- Memory: {yaml_config.memory_gi}\n")
                md.write(f"- Replicas: {yaml_config.replicas}\n")
                
                if yaml_config.gpu_memory_utilization:
                    md.write(f"- GPU Memory Utilization: {yaml_config.gpu_memory_utilization:.0%}\n")
                
                if yaml_config.model_name:
                    md.write(f"\n**Model:** {yaml_config.model_name}\n")
                
                md.write("\n")
        
        md.write("---\n\n")
    
    # Cost analysis (if available)
    if cost_configs:
        md.write("## 💰 Cost Analysis\n\n")
        
        for benchmark in benchmarks:
            platform = benchmark.metadata.platform
            cost_config = cost_configs.get(platform)
            
            if cost_config:
                # Calculate cost metrics
                throughput = benchmark.throughput_avg
                if throughput > 0:
                    tokens_per_hour = throughput * 3600
                    cost_per_hour = cost_config.get("cost_per_hour", 0)
                    cost_per_1m_tokens = (cost_per_hour / tokens_per_hour * 1_000_000) if tokens_per_hour > 0 else 0
                    
                    md.write(f"**{platform}:**\n")
                    md.write(f"- Instance: {cost_config.get('instance_name', 'N/A')}\n")
                    md.write(f"- GPU Type: {cost_config.get('gpu_type', 'N/A')}\n")
                    md.write(f"- Cost per Hour: ${cost_per_hour:.2f}\n")
                    md.write(f"- Cost per 1M Tokens: ${cost_per_1m_tokens:.2f}\n")
                    md.write(f"- Tokens per Hour: {tokens_per_hour:,.0f}\n")
                    md.write("\n")
        
        md.write("---\n\n")
    
    # Metrics explanation
    md.write("## 📖 Understanding the Metrics\n\n")
    md.write("**Key Performance Indicators (KPIs):**\n\n")
    md.write("- **TTFT (Time to First Token)**: How long until the first token arrives. "
             "Lower is better. Critical for perceived responsiveness.\n")
    md.write("  - Excellent: <300ms | Good: 300-1000ms | Poor: >1000ms\n\n")
    md.write("- **TPOT (Time Per Output Token)**: Average time between tokens during generation. "
             "Lower is better. Indicates generation efficiency.\n")
    md.write("  - Excellent: <20ms | Good: 20-100ms | Poor: >100ms\n\n")
    md.write("- **Throughput**: Output tokens generated per second. "
             "Higher is better. Measures overall generation speed.\n\n")
    md.write("- **Success Rate**: Percentage of requests completed successfully. "
             "Should be >99.9% for production.\n\n")
    md.write("- **Percentiles (P50/P90/P99)**: Statistical distribution of latencies. "
             "P99 represents worst-case user experience.\n\n")

    md.write("---\n\n")

    # Executive Summary
    md.write("## 📊 Executive Summary\n\n")

    # Find winners
    winner_ttft_idx = df["TTFT P50 (ms)"].idxmin()
    winner_throughput_idx = df["Throughput Avg (tok/s)"].idxmax()
    winner_reliability_idx = df["Success Rate (%)"].idxmax()

    md.write(f"**🏆 Best Latency (TTFT):** {df.loc[winner_ttft_idx, 'Platform']} "
             f"({df.loc[winner_ttft_idx, 'TTFT P50 (ms)']:.1f}ms)\n\n")
    md.write(f"**🚀 Best Throughput:** {df.loc[winner_throughput_idx, 'Platform']} "
             f"({df.loc[winner_throughput_idx, 'Throughput Avg (tok/s)']:.1f} tok/s)\n\n")
    md.write(f"**✅ Best Reliability:** {df.loc[winner_reliability_idx, 'Platform']} "
             f"({df.loc[winner_reliability_idx, 'Success Rate (%)']:.2f}%)\n\n")

    md.write("---\n\n")

    # Full results table
    md.write("## 📈 Detailed Results\n\n")
    
    # Create markdown table manually (avoid tabulate dependency)
    columns = df.columns.tolist()
    
    # Header
    md.write("| " + " | ".join(columns) + " |\n")
    md.write("|" + "|".join(["---" for _ in columns]) + "|\n")
    
    # Rows
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        md.write("| " + " | ".join(values) + " |\n")
    
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
             f"(fastest TTFT: {df.loc[winner_ttft_idx, 'TTFT P50 (ms)']:.1f}ms)\n\n")

    best_throughput = df.loc[winner_throughput_idx, "Platform"]
    md.write(f"**For High-Throughput Workloads:** Use `{best_throughput}` "
             f"(best throughput: {df.loc[winner_throughput_idx, 'Throughput Avg (tok/s)']:.1f} tok/s)\n\n")

    best_reliable = df.loc[winner_reliability_idx, "Platform"]
    md.write(f"**For Mission-Critical Systems:** Use `{best_reliable}` "
             f"(highest reliability: {df.loc[winner_reliability_idx, 'Success Rate (%)']:.2f}%)\n\n")

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

