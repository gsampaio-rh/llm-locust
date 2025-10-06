"""
🎯 LLM Benchmark Comparison Dashboard

Professional analysis tool for comparing LLM serving platforms.
"""

from datetime import datetime

import streamlit as st

from config import (
    APP_ICON,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    MAX_FILES,
)
from lib.data_loader import load_multiple_benchmarks
from lib.export import export_summary_csv, export_summary_markdown
from lib.statistics import compare_benchmarks
from lib.visualizations import create_normalized_comparison_chart

# Page configuration
st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

# ============= SIDEBAR: File Upload =============
with st.sidebar:
    st.header("📁 Upload Benchmarks")

    uploaded_files = st.file_uploader(
        f"Drop CSV files here (max {MAX_FILES})",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload benchmark CSVs from llm-locust",
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"❌ Too many files! Maximum is {MAX_FILES}.")
            uploaded_files = uploaded_files[:MAX_FILES]

        # Load benchmarks
        with st.spinner("📊 Loading benchmarks..."):
            benchmarks, errors = load_multiple_benchmarks(uploaded_files)

        # Store in session state
        st.session_state["benchmarks"] = benchmarks
        st.session_state["load_errors"] = errors

        # Show summary
        if benchmarks:
            st.success(f"✅ {len(benchmarks)} benchmark(s) loaded")

        if errors:
            st.error(f"❌ {len(errors)} file(s) failed")
            with st.expander("View Errors"):
                for filename, error in errors.items():
                    st.text(f"• {filename}")
                    st.caption(f"  {error}")

        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ============= MAIN CONTENT =============

if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    # ============= WELCOME SCREEN =============
    st.title("🎯 LLM Benchmark Dashboard")
    st.markdown("### Professional analysis tool for comparing LLM serving platforms")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🚀 Quick Start")
        st.markdown("""
        1. **Run benchmarks** using llm-locust
        2. **Upload CSV files** via the sidebar
        3. **Analyze results** with statistical rigor
        
        Built for platform engineers, ML engineers, and SREs.
        """)

        st.info("👈 Upload your benchmark files to begin")

    with col2:
        st.markdown("#### 📊 Features")
        st.markdown("""
        - Statistical significance testing
        - Multi-platform comparison
        - Latency, throughput, reliability analysis
        - Interactive visualizations
        - Export-ready charts
        """)

    st.markdown("---")

    st.markdown("#### 📁 Expected File Format")
    st.code("""
Filename: {platform}-{YYYYMMDD}-{HHMMSS}-{benchmark-id}.csv
Example: vllm-20251003-175002-1a-chat-simulation.csv

Required columns:
- request_id, timestamp, user_id, user_request_num
- input_tokens, output_tokens
- ttft_ms, tpot_ms, end_to_end_s
- total_tokens_per_sec, output_tokens_per_sec
- status_code
    """, language="text")

else:
    # ============= EXECUTIVE SUMMARY =============
    benchmarks = st.session_state["benchmarks"]

    st.title("🎯 Executive Summary")
    st.caption(f"Analysis of {len(benchmarks)} platform(s)")

    # Metadata summary
    total_requests = sum(b.metadata.total_requests for b in benchmarks)
    avg_duration = sum(b.metadata.duration_seconds for b in benchmarks) / len(benchmarks)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Platforms", len(benchmarks))
    with col2:
        st.metric("Total Requests", f"{total_requests:,}")
    with col3:
        st.metric("Avg Duration", f"{avg_duration:.0f}s")
    with col4:
        oldest_benchmark = min(b.metadata.timestamp for b in benchmarks)
        st.metric("Date", oldest_benchmark.strftime("%b %d, %Y"))

    st.markdown("---")

    # Find winners
    winner_ttft_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].ttft_p50)
    winner_throughput_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg)
    winner_reliability_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)

    # Recommendation
    if len(benchmarks) >= 2:
        st.markdown("## 🏆 Recommendation")
        
        # Determine overall winner (prioritize latency + reliability)
        winner = benchmarks[winner_ttft_idx]
        
        # Check if statistically significant
        if len(benchmarks) == 2:
            comparison = compare_benchmarks(benchmarks[0], benchmarks[1])
            is_significant = comparison.ttft_significant
            winner_name = winner.metadata.platform
        else:
            # For 3+ platforms, just show fastest
            is_significant = True  # Assume significant for now
            winner_name = winner.metadata.platform
        
        if is_significant:
            st.success(f"""
            ### {winner_name}
            
            **Recommended based on:**
            - ⚡ Fastest TTFT: {winner.ttft_p50:.0f}ms (P50)
            - 📊 P99: {winner.ttft_p99:.0f}ms
            - ✅ Reliability: {winner.success_rate*100:.2f}%
            """)
        else:
            st.info(f"""
            ### {winner_name} (slight edge)
            
            **Note:** Differences are not statistically significant.
            All platforms perform similarly.
            """)
        
        if len(benchmarks) == 2:
            other = benchmarks[1] if winner == benchmarks[0] else benchmarks[0]
            diff_pct = ((other.ttft_p50 - winner.ttft_p50) / other.ttft_p50) * 100
            
            st.caption(f"→ {diff_pct:.1f}% faster than {other.metadata.platform}")
        
        st.markdown("[📊 View Detailed Comparison →](#)")
        st.caption("Click 'Comparison' in the sidebar for full analysis")

    st.markdown("---")

    # Key metrics at-a-glance
    st.markdown("## 📊 Key Metrics")
    
    cols = st.columns(len(benchmarks))
    
    for idx, benchmark in enumerate(benchmarks):
        with cols[idx]:
            # Platform name with winner badge (smaller font to prevent wrapping)
            is_winner = (
                idx == winner_ttft_idx or 
                idx == winner_throughput_idx or 
                idx == winner_reliability_idx
            )
            
            # Use smaller font size to prevent breaking alignment
            if is_winner:
                header = f'<p style="font-size: 14px; font-weight: bold; margin: 0;">🏆 {benchmark.metadata.platform}</p>'
            else:
                header = f'<p style="font-size: 14px; font-weight: bold; margin: 0;">{benchmark.metadata.platform}</p>'
            st.markdown(header, unsafe_allow_html=True)
            
            # TTFT
            ttft_color = "🟢" if benchmark.ttft_p50 < 500 else "🟡" if benchmark.ttft_p50 < 1000 else "🔴"
            st.metric(
                "TTFT P50",
                f"{benchmark.ttft_p50:.0f}ms",
                delta=f"{ttft_color} P99: {benchmark.ttft_p99:.0f}ms",
                delta_color="off",
            )
            
            # Throughput
            st.metric(
                "Throughput",
                f"{benchmark.throughput_avg:.0f}",
                delta=f"tok/s",
                delta_color="off",
            )
            
            # Reliability
            reliability_color = "🟢" if benchmark.success_rate >= 0.999 else "🟡" if benchmark.success_rate >= 0.99 else "🔴"
            st.metric(
                "Success Rate",
                f"{benchmark.success_rate*100:.2f}%",
                delta=f"{reliability_color} {benchmark.metadata.failed_requests} failures",
                delta_color="off",
            )
            
            st.caption(f"{benchmark.metadata.total_requests:,} requests")

    st.markdown("---")

    # ============= EXPORT SECTION =============
    st.markdown("## 💾 Export Summary Report")
    st.caption("Download executive summary with key metrics")

    col1, col2 = st.columns(2)

    with col1:
        # CSV Export
        csv_data = export_summary_csv(benchmarks)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = f"benchmark-summary-{timestamp}.csv"

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            help="Download summary metrics as CSV for analysis",
        )

    with col2:
        # Markdown Export
        md_data = export_summary_markdown(benchmarks)
        md_filename = f"benchmark-summary-{timestamp}.md"

        st.download_button(
            label="📥 Download Markdown",
            data=md_data,
            file_name=md_filename,
            mime="text/markdown",
            help="Download formatted report for documentation",
        )

    # Preview
    with st.expander("👀 Preview Markdown Report"):
        st.markdown(md_data)

    st.markdown("---")

    # Normalized comparison
    if len(benchmarks) >= 2:
        st.markdown("## 🎨 Normalized Performance")
        st.caption("All metrics scaled 0-100 for easy comparison (100 = best in category)")
        
        fig = create_normalized_comparison_chart(benchmarks)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ How to Read This Chart"):
            st.markdown("""
            **Normalized scoring:**
            - Each metric is scaled to 0-100 within the tested platforms
            - 100 = Best performer for that metric
            - 0 = Worst performer for that metric
            
            **Note:**
            - TTFT and TPOT: Lower latency = Higher score (inverted)
            - Throughput and Success Rate: Higher value = Higher score
            
            **Use this to:**
            - See at-a-glance which platform excels where
            - Identify trade-offs (e.g., fast but less reliable)
            - Make balanced decisions across multiple metrics
            """)

    st.markdown("---")

    # Navigation help
    st.markdown("## 📖 Deep Dive Analysis")
    st.markdown("""
    Navigate to detailed analysis pages in the sidebar:
    
    - **Comparison** - Side-by-side comparison with statistical tests
    - **Latency Analysis** - TTFT, TPOT, and end-to-end latency
    - **Throughput Analysis** - TPS, RPS, and stability metrics
    - **Reliability** - Success rates, error analysis, and failure patterns
    """)
