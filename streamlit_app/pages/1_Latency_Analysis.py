"""
Latency Analysis Page

Understand response speed in plain English, with deep technical analysis on demand.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.explanations import GLOSSARY, get_simple_explanation, speed_to_emoji_and_label
from lib.visualizations import (
    create_box_plot_chart,
    create_cdf_chart,
    create_latency_distribution_chart,
    create_percentile_comparison_chart,
)

st.set_page_config(page_title="Latency Analysis", page_icon="⚡", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]
mode = st.session_state.get("mode", "simple")

# Title based on mode
if mode == "simple":
    st.title("⚡ Response Speed Analysis")
    st.markdown("### How fast do users see your responses?")
else:
    st.title("⚡ Latency Analysis")
    st.caption("Deep dive into TTFT, TPOT, and end-to-end latency metrics")

st.markdown("---")

# Metric selector
metric_type = st.radio(
    "What do you want to understand?" if mode == "simple" else "Select Metric",
    [
        "Response Speed (TTFT)" if mode == "simple" else "TTFT (Time to First Token)",
        "Streaming Smoothness (TPOT)" if mode == "simple" else "TPOT (Time Per Output Token)",
        "Total Response Time" if mode == "simple" else "End-to-End Latency",
    ],
    horizontal=True,
)

# Determine metric column and labels
if "TTFT" in metric_type:
    metric_col = "ttft_ms"
    metric_label = "TTFT"
    simple_name = "Response Speed"
    simple_desc = "How long users wait before seeing the first word appear"
elif "TPOT" in metric_type:
    metric_col = "tpot_ms"
    metric_label = "TPOT"
    simple_name = "Streaming Smoothness"
    simple_desc = "How smoothly text appears word-by-word"
else:
    metric_col = "end_to_end_s"
    metric_label = "End-to-End"
    simple_name = "Total Response Time"
    simple_desc = "How long from request to complete response"

st.markdown("---")

if mode == "simple":
    # ============= SIMPLE MODE =============
    
    # Show what this metric means
    st.markdown(f"## 💡 What is {simple_name}?")
    st.info(simple_desc)
    
    if metric_label in GLOSSARY:
        with st.expander("🤔 Tell me more"):
            info = GLOSSARY[metric_label]
            st.markdown(f"**{info['full_name']}**")
            st.markdown(info['detailed'])
            st.markdown(f"**Why it matters:** {info['why_matters']}")
            if info['analogy']:
                st.markdown(f"**Think of it like:** {info['analogy']}")
    
    st.markdown("---")
    
    # Simple comparison
    st.markdown(f"## 📊 Which Platform is Fastest?")
    
    # Get successful requests only
    platform_speeds = []
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 0:
            if metric_col == "end_to_end_s":
                value = success_df[metric_col].median() * 1000  # Convert to ms
            else:
                value = success_df[metric_col].median()
            platform_speeds.append((benchmark.metadata.platform, value, benchmark))
    
    # Sort by speed (fastest first)
    platform_speeds.sort(key=lambda x: x[1])
    
    # Show ranking
    for rank, (platform, speed_ms, benchmark) in enumerate(platform_speeds, 1):
        emoji, label, description = speed_to_emoji_and_label(speed_ms)
        
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            if rank == 1:
                st.markdown("### 🏆")
            else:
                st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"### {emoji} {platform}")
            st.caption(description)
        
        with col3:
            st.metric("Typical Speed", f"{speed_ms:.0f}ms", label=label)
        
        # Show comparison to winner
        if rank > 1:
            diff_ms = speed_ms - platform_speeds[0][1]
            diff_pct = (diff_ms / platform_speeds[0][1]) * 100
            st.caption(f"→ {diff_ms:.0f}ms ({diff_pct:.0f}%) slower than {platform_speeds[0][0]}")
        
        st.markdown("")
    
    st.markdown("---")
    
    # What this means for users
    st.markdown("## 💭 What This Means For Your Users")
    
    if len(platform_speeds) >= 2:
        fastest_platform, fastest_speed, _ = platform_speeds[0]
        slowest_platform, slowest_speed, _ = platform_speeds[-1]
        diff_ms = slowest_speed - fastest_speed
        diff_s = diff_ms / 1000
        
        if metric_label == "TTFT":
            if fastest_speed < 500:
                st.success(f"✅ **{fastest_platform}** feels instant - users barely notice any delay")
            elif fastest_speed < 1000:
                st.success(f"✅ **{fastest_platform}** feels responsive - good user experience")
            elif fastest_speed < 2000:
                st.warning(f"⚠️ **{fastest_platform}** is acceptable, but users notice the wait")
            else:
                st.error(f"❌ **{fastest_platform}** feels slow - users may get frustrated")
            
            if diff_ms > 100:
                st.markdown(f"""
                **Speed difference matters:**
                - {fastest_platform} responds **{diff_ms:.0f}ms faster** than {slowest_platform}
                - Users absolutely notice a **{diff_s:.2f} second** difference
                - At 10,000 requests/day, that's **{(diff_s * 10000 / 3600):.1f} hours** less waiting
                """)
        
        elif metric_label == "TPOT":
            if fastest_speed < 15:
                st.success(f"✅ **{fastest_platform}** - text streams smoothly and naturally")
            elif fastest_speed < 30:
                st.info(f"ℹ️ **{fastest_platform}** - acceptable streaming speed")
            else:
                st.warning(f"⚠️ **{fastest_platform}** - text may feel choppy or slow")
    
    st.markdown("---")
    
    # Progressive disclosure
    if st.button("🔬 Show Me The Technical Charts", use_container_width=True):
        st.session_state["mode"] = "advanced"
        st.rerun()

else:
    # ============= ADVANCED MODE =============
    
    # Percentile Comparison
    st.subheader("📊 Percentile Comparison")
    
    if metric_label in ["TTFT", "TPOT"]:
        fig = create_percentile_comparison_chart(benchmarks, metric_label.lower())
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Understanding Percentiles"):
            st.markdown("""
            - **P50 (Median):** Half of requests are faster, half are slower
            - **P90:** 90% of requests are faster than this
            - **P99:** Only 1% of requests are slower - the "tail latency"
            
            Lower percentiles = better performance
            """)
    else:
        st.info("Percentile comparison available for TTFT and TPOT metrics")
    
    # Distribution Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribution")
        fig = create_latency_distribution_chart(
            benchmarks,
            metric_col,
            f"{metric_label} Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shows how latency values are spread across requests")
    
    with col2:
        st.subheader("📦 Box Plot")
        fig = create_box_plot_chart(benchmarks, metric_col, f"{metric_label} Box Plot")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Visualizes median, quartiles, and outliers")
    
    # CDF
    st.markdown("---")
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
        """)
    
    # Detailed statistics table
    st.markdown("---")
    st.subheader("📋 Detailed Statistics")
    
    stats_data = []
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) == 0:
            continue
        
        data = success_df[metric_col].dropna()
        
        stats_data.append({
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
        })
    
    st.table(stats_data)
    
    # Technical insights
    st.markdown("---")
    st.subheader("🔬 Technical Insights")
    
    if len(benchmarks) >= 2:
        # Find best and worst performers
        if metric_label in ["TTFT", "TPOT"]:
            best_idx = min(
                range(len(benchmarks)),
                key=lambda i: benchmarks[i].ttft_p50 if metric_label == "TTFT" else benchmarks[i].tpot_p50,
            )
            worst_idx = max(
                range(len(benchmarks)),
                key=lambda i: benchmarks[i].ttft_p50 if metric_label == "TTFT" else benchmarks[i].tpot_p50,
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
                f"**{best_val:.1f}ms** ({improvement:.1f}% faster than {worst.metadata.platform})"
            )
            
            st.info(
                f"📊 P99 comparison: **{best.metadata.platform}** at {best_p99:.1f}ms vs "
                f"**{worst.metadata.platform}** at {worst_p99:.1f}ms ({improvement_p99:.1f}% difference)"
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
    else:
        st.info("Upload at least 2 benchmarks to see comparative insights")
