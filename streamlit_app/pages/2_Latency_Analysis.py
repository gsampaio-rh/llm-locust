"""
Latency Analysis - Deep Dive

Comprehensive analysis of TTFT, TPOT, and end-to-end latency.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.statistics import compare_benchmarks
from lib.visualizations import (
    create_box_plot_chart,
    create_cdf_chart,
    create_latency_distribution_chart,
    create_multi_platform_timeline,
)

st.set_page_config(page_title="Latency Analysis", page_icon="⚡", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("⚡ Latency Analysis")
st.caption("Deep dive into TTFT, TPOT, and end-to-end latency metrics")

st.markdown("---")

# ============= METRIC SELECTOR =============
metric_type = st.radio(
    "Select Metric",
    [
        "TTFT (Time to First Token)",
        "TPOT (Time Per Output Token)",
        "End-to-End Latency",
    ],
    horizontal=True,
)

# Determine metric column and labels
if "TTFT" in metric_type:
    metric_col = "ttft_ms"
    metric_label = "TTFT"
    metric_unit = "ms"
elif "TPOT" in metric_type:
    metric_col = "tpot_ms"
    metric_label = "TPOT"
    metric_unit = "ms"
else:
    metric_col = "end_to_end_s"
    metric_label = "End-to-End"
    metric_unit = "s"

st.markdown("---")

# ============= DISTRIBUTIONS =============
st.subheader("📊 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribution**")
    fig = create_latency_distribution_chart(
        benchmarks,
        metric_col,
        f"{metric_label} Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows how latency values are spread across requests")

with col2:
    st.markdown("**Box Plot**")
    fig = create_box_plot_chart(benchmarks, metric_col, f"{metric_label} Box Plot")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Visualizes median, quartiles, and outliers")

st.markdown("---")

# ============= CDF =============
st.subheader("📉 Cumulative Distribution (CDF)")
fig = create_cdf_chart(benchmarks, metric_col, f"{metric_label} CDF")
st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ Reading the CDF"):
    st.markdown("""
    The CDF shows what percentage of requests are faster than a given value.
    
    **How to use it:**
    - Find your target latency on the X-axis
    - Read up to see what % of requests meet that target
    - Steeper curve = more consistent performance
    - The line that's furthest left is the fastest platform
    """)

st.markdown("---")

# ============= PERFORMANCE OVER TIME =============
st.subheader("📈 Performance Over Time (All Platforms)")

fig = create_multi_platform_timeline(
    benchmarks,
    metric_col,
    f"{metric_label} Over Time - Stability & Degradation Analysis",
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
**What to look for:**
- Flat lines = stable performance
- Upward trends = performance degradation over time
- Spikes = inconsistent behavior
- Compare solid lines (rolling averages) to see which platform is more stable
""")

st.markdown("---")

# ============= DETAILED STATISTICS =============
st.subheader("📋 Detailed Statistics")

stats_data = []
for benchmark in benchmarks:
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    if len(success_df) == 0:
        continue

    data = success_df[metric_col].dropna()

    stats_data.append(
        {
            "Platform": benchmark.metadata.platform,
            "Count": len(data),
            "Mean": f"{data.mean():.2f}",
            "Median (P50)": f"{data.quantile(0.50):.2f}",
            "P90": f"{data.quantile(0.90):.2f}",
            "P95": f"{data.quantile(0.95):.2f}",
            "P99": f"{data.quantile(0.99):.2f}",
            "P99.9": f"{data.quantile(0.999):.2f}",
            "Min": f"{data.min():.2f}",
            "Max": f"{data.max():.2f}",
            "Std Dev": f"{data.std():.2f}",
        }
    )

st.table(stats_data)

st.markdown("---")

# ============= TECHNICAL INSIGHTS =============
st.subheader("🔬 Technical Insights")

if len(benchmarks) >= 2:
    # Find best and worst performers
    if metric_label in ["TTFT", "TPOT"]:
        best_idx = min(
            range(len(benchmarks)),
            key=lambda i: benchmarks[i].ttft_p50
            if metric_label == "TTFT"
            else benchmarks[i].tpot_p50,
        )
        worst_idx = max(
            range(len(benchmarks)),
            key=lambda i: benchmarks[i].ttft_p50
            if metric_label == "TTFT"
            else benchmarks[i].tpot_p50,
        )

        best = benchmarks[best_idx]
        worst = benchmarks[worst_idx]

        if metric_label == "TTFT":
            best_val = best.ttft_p50
            worst_val = worst.ttft_p50
            best_p99 = best.ttft_p99
            worst_p99 = worst.ttft_p99
        else:
            best_val = best.tpot_p50
            worst_val = worst.tpot_p50
            best_p99 = best.tpot_p99
            worst_p99 = worst.tpot_p99

        improvement = ((worst_val - best_val) / worst_val) * 100
        improvement_p99 = ((worst_p99 - best_p99) / worst_p99) * 100

        st.success(
            f"🏆 **{best.metadata.platform}** has the best {metric_label} P50: "
            f"**{best_val:.1f}{metric_unit}** ({improvement:.1f}% faster than {worst.metadata.platform})"
        )

        st.info(
            f"📊 P99 comparison: **{best.metadata.platform}** at {best_p99:.1f}{metric_unit} vs "
            f"**{worst.metadata.platform}** at {worst_p99:.1f}{metric_unit} ({improvement_p99:.1f}% difference)"
        )

        # Statistical significance (if 2 platforms)
        if len(benchmarks) == 2:
            comparison = compare_benchmarks(benchmarks[0], benchmarks[1])
            
            if metric_label == "TTFT":
                is_significant = comparison.ttft_significant
                p_value = comparison.p_value_ttft
            else:
                is_significant = comparison.tpot_significant
                p_value = comparison.p_value_tpot
            
            if is_significant:
                st.success(
                    f"✅ **Statistically significant** difference (p={p_value:.4f})"
                )
            else:
                st.warning(
                    f"⚠️ Difference is **not statistically significant** (p={p_value:.4f}) - could be random variation"
                )

        # Check SLA compliance
        if metric_label == "TTFT":
            for benchmark in benchmarks:
                if benchmark.ttft_p50 > 1000:
                    st.warning(
                        f"⚠️ **{benchmark.metadata.platform}** exceeds 1s TTFT P50 target: "
                        f"{benchmark.ttft_p50:.1f}ms"
                    )
                if benchmark.ttft_p99 > 2000:
                    st.error(
                        f"❌ **{benchmark.metadata.platform}** exceeds 2s TTFT P99 SLA: "
                        f"{benchmark.ttft_p99:.1f}ms"
                    )

        # Variability analysis
        st.markdown("**Consistency Analysis:**")
        for benchmark in benchmarks:
            success_df = benchmark.df[benchmark.df["status_code"] == 200]
            if len(success_df) > 0:
                data = success_df[metric_col].dropna()
                cv = (data.std() / data.mean()) * 100 if data.mean() > 0 else 0

                if cv < 20:
                    st.markdown(
                        f"- ✅ **{benchmark.metadata.platform}**: Very consistent (CV={cv:.1f}%)"
                    )
                elif cv < 40:
                    st.markdown(
                        f"- 📊 **{benchmark.metadata.platform}**: Moderate variance (CV={cv:.1f}%)"
                    )
                else:
                    st.markdown(
                        f"- ⚠️ **{benchmark.metadata.platform}**: High variance (CV={cv:.1f}%) - investigate spikes"
                    )

else:
    st.info("Upload at least 2 benchmarks to see comparative insights")

    # Show single platform insights
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]
        success_df = benchmark.df[benchmark.df["status_code"] == 200]

        if metric_label == "TTFT":
            st.metric(
                "TTFT P50",
                f"{benchmark.ttft_p50:.1f}ms",
                delta=f"P99: {benchmark.ttft_p99:.1f}ms",
            )

            if benchmark.ttft_p50 < 500:
                st.success("✅ Excellent TTFT performance (< 500ms)")
            elif benchmark.ttft_p50 < 1000:
                st.info("✅ Good TTFT performance (< 1s)")
            else:
                st.warning("⚠️ TTFT exceeds 1s - consider optimization")

        elif metric_label == "TPOT":
            st.metric(
                "TPOT P50",
                f"{benchmark.tpot_p50:.1f}ms",
                delta=f"P99: {benchmark.tpot_p99:.1f}ms",
            )

            if benchmark.tpot_p50 < 15:
                st.success("✅ Excellent TPOT performance (< 15ms)")
            elif benchmark.tpot_p50 < 30:
                st.info("✅ Good TPOT performance (< 30ms)")
            else:
                st.warning("⚠️ TPOT exceeds 30ms - streaming may feel slow")

