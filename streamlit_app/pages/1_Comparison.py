"""
Platform Comparison Page

Visual side-by-side comparison of all platforms.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.components import get_status_colors, render_metric_card, render_platform_header

st.set_page_config(page_title="Platform Comparison", page_icon="📊", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

# Find winners
winners = {
    "ttft": min(range(len(benchmarks)), key=lambda i: benchmarks[i].ttft_p50),
    "tpot": min(range(len(benchmarks)), key=lambda i: benchmarks[i].tpot_p50),
    "throughput": max(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg),
    "reliability": max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate),
}

st.title("📊 Platform Comparison Dashboard")
st.caption(f"Visual comparison of {len(benchmarks)} platform(s)")

st.markdown("---")

# ============= VISUAL DASHBOARD =============
# Create columns for each benchmark
cols = st.columns(len(benchmarks))

for idx, benchmark in enumerate(benchmarks):
    with cols[idx]:
        # Platform header
        is_overall_winner = idx == winners["ttft"]
        render_platform_header(benchmark.metadata.platform, is_overall_winner)

        # Benchmark metadata
        st.markdown(
            f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <div style="font-size: 11px; color: #6c757d;">
                📊 {benchmark.metadata.total_requests:,} requests • 🕐 {benchmark.metadata.duration_seconds:.0f}s
            </div>
            <div style="font-size: 11px; color: #6c757d; margin-top: 3px;">
                {benchmark.metadata.benchmark_id}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get successful requests for calculations
        success_df = benchmark.df[benchmark.df["status_code"] == 200]

        # TTFT Card
        ttft_p90 = success_df["ttft_ms"].quantile(0.90) if len(success_df) > 0 else 0
        ttft_p95 = success_df["ttft_ms"].quantile(0.95) if len(success_df) > 0 else 0

        bg_color, text_color, border_color = get_status_colors(
            benchmark.ttft_p50, {"excellent": 500, "good": 1000}, lower_is_better=True
        )

        render_metric_card(
            title="TTFT (Time to First Token)",
            value=benchmark.ttft_p50,
            unit="ms",
            subtitle=f"P50 | P90: {ttft_p90:.0f}ms | P95: {ttft_p95:.0f}ms | P99: {benchmark.ttft_p99:.0f}ms",
            bg_color=bg_color,
            text_color=text_color,
            border_color=border_color,
            is_winner=(idx == winners["ttft"]),
            icon="⚡",
        )

        # TPOT Card
        tpot_p90 = success_df["tpot_ms"].quantile(0.90) if len(success_df) > 0 else 0
        tpot_p95 = success_df["tpot_ms"].quantile(0.95) if len(success_df) > 0 else 0

        render_metric_card(
            title="TPOT (Time Per Output Token)",
            value=benchmark.tpot_p50,
            unit="ms",
            subtitle=f"P50 | P90: {tpot_p90:.1f}ms | P95: {tpot_p95:.1f}ms | P99: {benchmark.tpot_p99:.1f}ms",
            bg_color="#e7f3ff",
            text_color="#0d47a1",
            border_color="#2196F3",
            is_winner=(idx == winners["tpot"]),
            icon="🔄",
        )

        # TPS Card
        avg_latency = (
            benchmark.df[benchmark.df["status_code"] == 200]["end_to_end_s"].mean()
            if len(success_df) > 0
            else 0
        )

        render_metric_card(
            title="TPS (Tokens Per Second)",
            value=benchmark.throughput_avg,
            unit="tok/s",
            subtitle=f"RPS: {benchmark.rps:.2f} req/s | Avg latency: {avg_latency:.2f}s",
            bg_color="#f3e5f5",
            text_color="#4a148c",
            border_color="#9c27b0",
            is_winner=(idx == winners["throughput"]),
            icon="🚀",
        )

        # Token Statistics Card
        avg_input = benchmark.df["input_tokens"].mean()
        avg_output = success_df["output_tokens"].mean() if len(success_df) > 0 else 0

        st.markdown(
            f"""
        <div style="background-color: #fff8e1; border-left: 4px solid #ffa726; 
                    padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <div style="color: #e65100; font-size: 12px; font-weight: 600; margin-bottom: 5px;">🔤 TOKEN STATS</div>
            <div style="color: #e65100; font-size: 20px; font-weight: 700; line-height: 1;">
                {avg_input:.0f} → {avg_output:.0f}
            </div>
            <div style="color: #f57c00; font-size: 11px; margin-top: 8px;">
                Avg input → output tokens
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Reliability Card
        bg_color, text_color, border_color = get_status_colors(
            benchmark.success_rate * 100, {"excellent": 99.9, "good": 99}, lower_is_better=False
        )

        failures_per_1k = (1 - benchmark.success_rate) * 1000
        render_metric_card(
            title="SUCCESS RATE",
            value=benchmark.success_rate * 100,
            unit="%",
            subtitle=f"{benchmark.metadata.failed_requests} failures | {failures_per_1k:.1f} per 1K requests",
            bg_color=bg_color,
            text_color=text_color,
            border_color=border_color,
            is_winner=(idx == winners["reliability"]),
            icon="✅",
        )

        # End-to-End Latency Card
        e2e_p50 = success_df["end_to_end_s"].quantile(0.50) if len(success_df) > 0 else 0
        e2e_p90 = success_df["end_to_end_s"].quantile(0.90) if len(success_df) > 0 else 0
        e2e_p95 = success_df["end_to_end_s"].quantile(0.95) if len(success_df) > 0 else 0
        e2e_p99 = success_df["end_to_end_s"].quantile(0.99) if len(success_df) > 0 else 0

        # Calculate e2e winner
        e2e_p50_values = []
        for b in benchmarks:
            success_df_temp = b.df[b.df["status_code"] == 200]
            e2e_val = (
                success_df_temp["end_to_end_s"].quantile(0.50)
                if len(success_df_temp) > 0
                else float("inf")
            )
            e2e_p50_values.append(e2e_val)
        
        e2e_winner_idx = min(range(len(benchmarks)), key=lambda i: e2e_p50_values[i])

        render_metric_card(
            title="END-TO-END LATENCY",
            value=e2e_p50,
            unit="s",
            subtitle=f"P50 | P90: {e2e_p90:.2f}s | P95: {e2e_p95:.2f}s | P99: {e2e_p99:.2f}s",
            bg_color="#fce4ec",
            text_color="#880e4f",
            border_color="#e91e63",
            is_winner=(idx == e2e_winner_idx),
            icon="⏱️",
        )

# Legend
st.markdown("")
st.caption("🏆 = Best performance in category • Green = Excellent • Yellow = Good • Red = Needs attention")

st.markdown("---")

# ============= COMPARISON TABLE =============
st.subheader("📋 Detailed Comparison Table")

comparison_data = []
for benchmark in benchmarks:
    comparison_data.append({
        "Platform": benchmark.metadata.platform,
        "Requests": f"{benchmark.metadata.total_requests:,}",
        "Success %": f"{benchmark.success_rate*100:.2f}",
        "TTFT P50": f"{benchmark.ttft_p50:.0f}ms",
        "TTFT P99": f"{benchmark.ttft_p99:.0f}ms",
        "TPOT P50": f"{benchmark.tpot_p50:.1f}ms",
        "Throughput": f"{benchmark.throughput_avg:.0f}",
        "RPS": f"{benchmark.rps:.2f}",
    })

st.table(comparison_data)

