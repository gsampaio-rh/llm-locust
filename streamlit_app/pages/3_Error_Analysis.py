"""
Error Analysis Page

Understand reliability in plain English, with technical debugging on demand.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.explanations import reliability_to_emoji_and_label
from lib.visualizations import create_success_rate_chart

st.set_page_config(page_title="Error Analysis", page_icon="✅", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]
mode = st.session_state.get("mode", "simple")

# Title based on mode
if mode == "simple":
    st.title("✅ Reliability Analysis")
    st.markdown("### How often does it actually work?")
else:
    st.title("❌ Error Analysis")
    st.caption("Failures, error rates, and reliability metrics")

st.markdown("---")

if mode == "simple":
    # ============= SIMPLE MODE =============
    
    st.markdown("## 💡 What is Reliability?")
    st.info("""
    **Reliability** is the percentage of requests that work without errors.
    
    Think of it like a car:
    - **99.9% reliability** = Your car starts 999 out of 1000 times (excellent!)
    - **99% reliability** = Fails to start 10 times out of 1000 (frustrating!)
    - **95% reliability** = Fails 50 times out of 1000 (unacceptable!)
    """)
    
    with st.expander("🤔 Why does 0.1% matter?"):
        st.markdown("""
        **At scale, small differences are huge:**
        
        With 10,000 requests per day:
        - **99.9% = 10 failures/day** → Barely noticeable
        - **99.0% = 100 failures/day** → Lots of support tickets
        - **95.0% = 500 failures/day** → Business impact
        
        **Industry standards:**
        - Consumer apps: aim for 99.9%+ ("three nines")
        - Mission-critical: aim for 99.99%+ ("four nines")
        - Experimental/internal: 99%+ may be acceptable
        """)
    
    st.markdown("---")
    
    # Simple comparison
    st.markdown("## 📊 Which Platform is Most Reliable?")
    
    # Sort by success rate
    sorted_benchmarks = sorted(benchmarks, key=lambda b: b.success_rate, reverse=True)
    
    for rank, benchmark in enumerate(sorted_benchmarks, 1):
        emoji, label, description = reliability_to_emoji_and_label(benchmark.success_rate)
        
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            if rank == 1:
                st.markdown("### 🏆")
            else:
                st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"### {emoji} {benchmark.metadata.platform}")
            st.caption(description)
        
        with col3:
            st.metric(
                "Success Rate",
                f"{benchmark.success_rate*100:.2f}%",
                label=label,
            )
            st.caption(f"{benchmark.metadata.failed_requests} failures")
        
        # Show comparison to leader
        if rank > 1:
            leader = sorted_benchmarks[0]
            diff_pct = (leader.success_rate - benchmark.success_rate) * 100
            extra_failures = benchmark.metadata.failed_requests - leader.metadata.failed_requests
            st.caption(f"→ {diff_pct:.2f}% more failures ({extra_failures} extra failed requests)")
        
        st.markdown("")
    
    st.markdown("---")
    
    # What this means
    st.markdown("## 💭 What This Means For Your Users")
    
    if len(benchmarks) >= 2:
        best = sorted_benchmarks[0]
        worst = sorted_benchmarks[-1]
        
        # Calculate failure rates
        best_failures_per_1k = (1 - best.success_rate) * 1000
        worst_failures_per_1k = (1 - worst.success_rate) * 1000
        
        st.markdown(f"""
        **With {best.metadata.platform}:**
        - Only **~{best_failures_per_1k:.0f} failures per 1,000 requests**
        - {best.success_rate*100:.2f}% of users have a perfect experience
        """)
        
        if worst.success_rate < 0.999:
            st.warning(f"""
            **With {worst.metadata.platform}:**
            - **~{worst_failures_per_1k:.0f} failures per 1,000 requests**
            - That's **{worst_failures_per_1k - best_failures_per_1k:.0f}x more failures** than {best.metadata.platform}
            """)
        
        # User impact
        st.markdown("### 📈 Real-World Impact")
        
        daily_requests = 10000  # Assume 10k requests/day
        best_daily_failures = daily_requests * (1 - best.success_rate)
        worst_daily_failures = daily_requests * (1 - worst.success_rate)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                f"{best.metadata.platform} @ 10K req/day",
                f"{best_daily_failures:.0f} failures",
                delta="Best",
                delta_color="off",
            )
        
        with col2:
            st.metric(
                f"{worst.metadata.platform} @ 10K req/day",
                f"{worst_daily_failures:.0f} failures",
                delta=f"+{worst_daily_failures - best_daily_failures:.0f}",
                delta_color="inverse",
            )
        
        extra_tickets = (worst_daily_failures - best_daily_failures) * 30  # per month
        if extra_tickets > 100:
            st.error(f"""
            ⚠️ **Support Impact:** {worst.metadata.platform} would generate 
            **~{extra_tickets:.0f} extra support tickets per month**
            """)
    
    st.markdown("---")
    
    # Check for specific problems
    st.markdown("## 🔍 Any Red Flags?")
    
    all_good = True
    for benchmark in benchmarks:
        if benchmark.success_rate < 0.999:
            all_good = False
            st.warning(f"""
            ⚠️ **{benchmark.metadata.platform}** is below 99.9% reliability
            - Current: {benchmark.success_rate*100:.3f}%
            - Target: 99.900%+
            - Impact: Users will notice failures
            """)
        
        # Check if error rate is increasing over time
        if benchmark.metadata.failed_requests > 0:
            failed_df = benchmark.df[benchmark.df["status_code"] != 200]
            if len(failed_df) > 0:
                # Simple check: compare first half vs second half
                mid_point = len(benchmark.df) // 2
                first_half_errors = len(benchmark.df[:mid_point][benchmark.df[:mid_point]["status_code"] != 200])
                second_half_errors = len(benchmark.df[mid_point:][benchmark.df[mid_point:]["status_code"] != 200])
                
                if second_half_errors > first_half_errors * 1.5:
                    all_good = False
                    st.error(f"""
                    🔴 **{benchmark.metadata.platform}** error rate is increasing over time!
                    - First half: {first_half_errors} errors
                    - Second half: {second_half_errors} errors
                    - This suggests performance degradation or resource exhaustion
                    """)
    
    if all_good:
        st.success("✅ **All platforms meet or exceed 99.9% reliability!**")
    
    st.markdown("---")
    
    if st.button("🔬 Show Me The Technical Details", use_container_width=True):
        st.session_state["mode"] = "advanced"
        st.rerun()

else:
    # ============= ADVANCED MODE =============
    
    # Success rate comparison
    st.subheader("✅ Success Rate Comparison")
    fig = create_success_rate_chart(benchmarks)
    st.plotly_chart(fig, use_container_width=True)
    
    # Error breakdown
    st.markdown("---")
    st.subheader("📊 Detailed Error Analysis")
    
    for benchmark in benchmarks:
        with st.expander(
            f"{benchmark.metadata.platform} - "
            f"{benchmark.metadata.failed_requests} failures "
            f"({(1-benchmark.success_rate)*100:.2f}%)"
        ):
            failed_df = benchmark.df[benchmark.df["status_code"] != 200]
            
            if len(failed_df) == 0:
                st.success("🎉 No failures detected!")
                continue
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                error_rate = (len(failed_df) / len(benchmark.df)) * 100
                st.metric("Error Rate", f"{error_rate:.3f}%")
            
            with col2:
                st.metric("Total Failures", len(failed_df))
            
            with col3:
                st.metric("Successful", len(benchmark.df) - len(failed_df))
            
            # Status code distribution
            st.markdown("**Status Code Breakdown:**")
            status_counts = failed_df["status_code"].value_counts()
            
            for status_code, count in status_counts.items():
                pct = (count / len(failed_df)) * 100
                
                # Interpret status codes
                if status_code == 429:
                    status_name = "Rate Limit Exceeded"
                    color = "🟡"
                elif status_code >= 500:
                    status_name = "Server Error"
                    color = "🔴"
                elif status_code >= 400:
                    status_name = "Client Error"
                    color = "🟠"
                else:
                    status_name = "Unknown"
                    color = "⚪"
                
                st.markdown(
                    f"  {color} **{status_code}** ({status_name}): {count} "
                    f"({pct:.1f}% of failures)"
                )
            
            # Error timeline
            st.markdown("**Failures Over Time:**")
            
            # Group by time buckets
            failed_df_copy = failed_df.copy()
            failed_df_copy["time_bucket"] = (
                (failed_df_copy["request_id"] // 100) * 100
            )
            errors_over_time = failed_df_copy.groupby("time_bucket").size()
            
            if len(errors_over_time) > 0:
                st.line_chart(errors_over_time)
                
                # Check for patterns
                if errors_over_time.max() > errors_over_time.mean() * 3:
                    st.warning(
                        "⚠️ Error spikes detected - failures are not evenly distributed"
                    )
    
    # Comparative reliability table
    st.markdown("---")
    st.subheader("📋 Reliability Comparison Table")
    
    reliability_data = []
    for benchmark in benchmarks:
        failed_requests = benchmark.metadata.failed_requests
        total_requests = benchmark.metadata.total_requests
        success_rate = benchmark.success_rate
        
        # Calculate failures per 1K, 10K, 100K requests
        failures_per_1k = (1 - success_rate) * 1000
        failures_per_10k = (1 - success_rate) * 10000
        
        reliability_data.append({
            "Platform": benchmark.metadata.platform,
            "Success Rate": f"{success_rate*100:.3f}%",
            "Total Failures": failed_requests,
            "Per 1K Requests": f"{failures_per_1k:.1f}",
            "Per 10K Requests": f"{failures_per_10k:.0f}",
            "SLA Compliant": "✅ Yes" if success_rate >= 0.999 else "❌ No",
        })
    
    st.table(reliability_data)
    
    # Technical insights
    st.markdown("---")
    st.subheader("🔬 Technical Insights")
    
    if len(benchmarks) >= 2:
        # Find most reliable
        best_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)
        best = benchmarks[best_idx]
        
        st.success(
            f"🏆 **{best.metadata.platform}** has the best success rate: "
            f"**{best.success_rate*100:.3f}%** "
            f"({best.metadata.failed_requests} failures out of {best.metadata.total_requests} requests)"
        )
        
        # Check SLA compliance (99.9%)
        for benchmark in benchmarks:
            if benchmark.success_rate < 0.999:
                shortfall = (0.999 - benchmark.success_rate) * 100
                st.warning(
                    f"⚠️ **{benchmark.metadata.platform}** below 99.9% SLA: "
                    f"{benchmark.success_rate*100:.3f}% "
                    f"({shortfall:.3f}% below target)"
                )
        
        # Statistical significance
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
    else:
        st.info("Upload at least 2 benchmarks to see comparative insights")
    
    # Recommendations
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
        """)
