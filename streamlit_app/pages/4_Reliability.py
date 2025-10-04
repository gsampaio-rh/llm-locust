"""
Reliability & Error Analysis

Success rates, error patterns, and failure analysis.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.visualizations import create_status_code_pie_chart, create_success_rate_chart

st.set_page_config(page_title="Reliability", page_icon="✅", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("✅ Reliability & Error Analysis")
st.caption("Success rates, failures, and error patterns")

st.markdown("---")

# ============= SUCCESS RATE COMPARISON =============
st.subheader("📊 Success Rate Comparison")

fig = create_success_rate_chart(benchmarks)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============= RELIABILITY TABLE =============
st.subheader("📋 Reliability Metrics")

reliability_data = []
for benchmark in benchmarks:
    failed_requests = benchmark.metadata.failed_requests
    total_requests = benchmark.metadata.total_requests
    success_rate = benchmark.success_rate

    # Calculate failures per 1K, 10K requests
    failures_per_1k = (1 - success_rate) * 1000
    failures_per_10k = (1 - success_rate) * 10000

    reliability_data.append(
        {
            "Platform": benchmark.metadata.platform,
            "Success Rate": f"{success_rate*100:.3f}%",
            "Total Failures": failed_requests,
            "Per 1K Requests": f"{failures_per_1k:.1f}",
            "Per 10K Requests": f"{failures_per_10k:.0f}",
            "SLA Compliant (99.9%)": "✅ Yes" if success_rate >= 0.999 else "❌ No",
        }
    )

st.table(reliability_data)

st.caption("""
**SLA Target:** 99.9% success rate = only 1 failure per 1,000 requests  
Lower failure rates = better user experience and fewer support tickets
""")

st.markdown("---")

# ============= ERROR BREAKDOWN BY PLATFORM =============
st.subheader("🔴 Error Breakdown by Platform")

# Create tabs for each platform
tabs = st.tabs([b.metadata.platform for b in benchmarks])

for idx, benchmark in enumerate(benchmarks):
    with tabs[idx]:
        failed_df = benchmark.df[benchmark.df["status_code"] != 200]

        if len(failed_df) == 0:
            st.success(f"🎉 **{benchmark.metadata.platform}** has no failures!")
            st.metric("Perfect Success Rate", "100%")
            continue

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            error_rate = (len(failed_df) / len(benchmark.df)) * 100
            st.metric("Error Rate", f"{error_rate:.3f}%")

        with col2:
            st.metric("Total Failures", len(failed_df))

        with col3:
            st.metric("Successful Requests", len(benchmark.df) - len(failed_df))

        st.markdown("---")

        # Status code pie chart
        st.markdown("**Status Code Distribution:**")
        fig = create_status_code_pie_chart(benchmark)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Error timeline
        st.markdown("**Failures Over Time:**")

        # Group by time buckets
        failed_df_copy = failed_df.copy()
        failed_df_copy["time_bucket"] = (failed_df_copy["request_id"] // 100) * 100
        errors_over_time = failed_df_copy.groupby("time_bucket").size()

        if len(errors_over_time) > 0:
            st.line_chart(errors_over_time)

            # Check for patterns
            if errors_over_time.max() > errors_over_time.mean() * 3:
                st.warning(
                    "⚠️ Error spikes detected - failures are not evenly distributed"
                )
            else:
                st.info("✅ Errors are evenly distributed over time")

st.markdown("---")

# ============= TECHNICAL INSIGHTS =============
st.subheader("🔬 Technical Insights")

if len(benchmarks) >= 2:
    # Find most reliable
    best_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)
    worst_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)

    best = benchmarks[best_idx]
    worst = benchmarks[worst_idx]

    st.success(
        f"🏆 **{best.metadata.platform}** is most reliable: "
        f"**{best.success_rate*100:.3f}%** "
        f"({best.metadata.failed_requests} failures out of {best.metadata.total_requests:,} requests)"
    )

    # SLA compliance
    for benchmark in benchmarks:
        if benchmark.success_rate < 0.999:
            shortfall = (0.999 - benchmark.success_rate) * 100
            st.warning(
                f"⚠️ **{benchmark.metadata.platform}** below 99.9% SLA: "
                f"{benchmark.success_rate*100:.3f}% "
                f"({shortfall:.3f}% below target)"
            )

    # Failure rate comparison
    if len(benchmarks) == 2:
        a, b = benchmarks[0], benchmarks[1]
        diff = abs(a.success_rate - b.success_rate) * 100

        if diff > 0.1:
            better = a if a.success_rate > b.success_rate else b
            worse = b if a.success_rate > b.success_rate else a

            st.info(
                f"📊 **{better.metadata.platform}** has a **{diff:.2f}% higher** "
                f"success rate than **{worse.metadata.platform}**"
            )

            # Impact calculation
            daily_requests = 10000  # Assume 10K requests/day
            extra_failures_per_day = daily_requests * (
                1 - worse.success_rate - (1 - better.success_rate)
            )

            if extra_failures_per_day > 10:
                st.error(
                    f"⚠️ **Impact:** {worse.metadata.platform} would generate "
                    f"**~{extra_failures_per_day:.0f} extra failures per day** "
                    f"at 10K requests/day"
                )

    # Error pattern analysis
    st.markdown("**Error Pattern Analysis:**")
    for benchmark in benchmarks:
        failed_df = benchmark.df[benchmark.df["status_code"] != 200]

        if len(failed_df) > 0:
            # Check if errors increasing over time
            mid_point = len(benchmark.df) // 2
            first_half_errors = len(
                benchmark.df[:mid_point][benchmark.df[:mid_point]["status_code"] != 200]
            )
            second_half_errors = len(
                benchmark.df[mid_point:][benchmark.df[mid_point:]["status_code"] != 200]
            )

            if second_half_errors > first_half_errors * 1.5:
                st.error(
                    f"🔴 **{benchmark.metadata.platform}**: Error rate increasing over time! "
                    f"(First half: {first_half_errors} errors, Second half: {second_half_errors} errors)"
                )
            elif second_half_errors < first_half_errors * 0.5:
                st.success(
                    f"✅ **{benchmark.metadata.platform}**: Error rate decreasing over time "
                    f"(First half: {first_half_errors} errors, Second half: {second_half_errors} errors)"
                )
            else:
                st.info(
                    f"📊 **{benchmark.metadata.platform}**: Consistent error rate "
                    f"(First half: {first_half_errors} errors, Second half: {second_half_errors} errors)"
                )

else:
    st.info("Upload at least 2 benchmarks to see comparative insights")

    # Show single platform insights
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]

        if benchmark.success_rate >= 0.999:
            st.success(
                f"✅ **{benchmark.metadata.platform}** meets 99.9% SLA: "
                f"{benchmark.success_rate*100:.3f}%"
            )
        else:
            failures_per_1k = (1 - benchmark.success_rate) * 1000
            st.warning(
                f"⚠️ **{benchmark.metadata.platform}** below 99.9% SLA: "
                f"{benchmark.success_rate*100:.3f}% ({failures_per_1k:.0f} failures per 1K)"
            )

st.markdown("---")

# ============= RECOMMENDATIONS =============
with st.expander("💡 Recommendations"):
    st.markdown("""
    **If error rates are high:**
    1. Check server logs for root causes
    2. Monitor resource utilization (CPU, memory, GPU)
    3. Review timeout settings
    4. Consider rate limiting or load balancing
    5. Test with lower concurrency to isolate issues
    
    **Target SLAs:**
    - Production: 99.9%+ (three nines)
    - Critical systems: 99.99%+ (four nines)
    - Internal tools: 99%+ acceptable
    
    **Common causes of failures:**
    - Timeout errors: Increase timeout thresholds
    - Rate limiting (429): Reduce concurrency or add delays
    - Server errors (5xx): Check backend logs and resource limits
    - Resource exhaustion: Scale up infrastructure or optimize workload
    """)

