"""
🎯 LLM Benchmark Comparison Dashboard

Which LLM platform should you choose? Let the data decide.
"""

import streamlit as st

from config import (
    APP_ICON,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    MAX_FILES,
)
from lib.dashboard import (
    find_winners,
    render_advanced_mode_dashboard,
    render_recommendation_section,
    render_simple_mode_dashboard,
)
from lib.data_loader import load_multiple_benchmarks

# Page configuration
st.set_page_config(
    page_title="LLM Benchmark Comparison",
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

# Initialize session state
if "mode" not in st.session_state:
    st.session_state["mode"] = "simple"  # Default to simple mode

# ============= SIDEBAR: File Upload and Mode Toggle =============
with st.sidebar:
    st.header("📁 Upload Benchmarks")

    uploaded_files = st.file_uploader(
        f"Drop your CSV files here (max {MAX_FILES} files)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload benchmark CSVs from llm-locust",
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"❌ Too many files! Maximum is {MAX_FILES}.")
            uploaded_files = uploaded_files[:MAX_FILES]

        # Load benchmarks
        with st.spinner("📊 Analyzing your benchmarks..."):
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
            st.session_state["mode"] = "simple"
            st.rerun()

    # Mode toggle (only show if data is loaded)
    if "benchmarks" in st.session_state and st.session_state.get("benchmarks"):
        st.markdown("---")
        st.markdown("**View Mode:**")

        current_mode = st.session_state.get("mode", "simple")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💡 Simple",
                use_container_width=True,
                disabled=(current_mode == "simple"),
                type="primary" if current_mode == "simple" else "secondary",
            ):
                st.session_state["mode"] = "simple"
                st.rerun()

        with col2:
            if st.button(
                "🔬 Advanced",
                use_container_width=True,
                disabled=(current_mode == "advanced"),
                type="primary" if current_mode == "advanced" else "secondary",
            ):
                st.session_state["mode"] = "advanced"
                st.rerun()

        if current_mode == "simple":
            st.caption("📖 Easy-to-understand insights")
        else:
            st.caption("📊 Detailed technical analysis")

# ============= MAIN CONTENT =============

if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    # ============= WELCOME SCREEN =============
    st.title("🎯 Which LLM Platform Should You Choose?")
    st.markdown("### Let data guide your infrastructure decisions")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🚀 How It Works")
        st.markdown(
            """
        1. **Run benchmarks** using llm-locust on your platforms
        2. **Upload the CSV files** using the sidebar
        3. **Get clear answers** about which platform is best
        
        No PhD required. Just upload and understand.
        """
        )

        st.info("👈 Start by uploading your benchmark files in the sidebar")

    with col2:
        st.markdown("#### ✨ What You'll Learn")
        st.markdown(
            """
        - Which platform is **fastest**
        - Which is most **reliable**
        - What it means for **your users**
        - Whether the difference **matters**
        """
        )

    st.markdown("---")

    # Feature overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💡 Simple Mode")
        st.markdown(
            "Plain English explanations. Perfect for product managers, executives, and anyone making decisions."
        )

    with col2:
        st.markdown("##### 🔬 Advanced Mode")
        st.markdown(
            "Deep technical analysis. Statistical tests, distributions, and raw metrics for engineers."
        )

    with col3:
        st.markdown("##### 🎯 Both Modes")
        st.markdown("One click to switch between them. Start simple, go deep when needed.")

    st.markdown("---")

    # Sample insight
    st.markdown("#### 📖 Example Insight (Simple Mode)")
    st.info(
        """
    **🏆 Recommendation: Use vLLM**
    
    Why?
    - ✅ **19% faster** - Users get responses 200ms sooner
    - ✅ **99.8% success rate** - Only 2 failures per 1000 requests  
    - ✅ **Consistent** - Fast for everyone, not just sometimes
    
    **What this means:** Your users will have a noticeably better experience, and you'll handle support tickets less often.
    """
    )

else:
    # ============= DASHBOARD WITH DATA =============
    benchmarks = st.session_state["benchmarks"]
    mode = st.session_state.get("mode", "simple")

    # Find winners once for all views
    winners = find_winners(benchmarks)

    # Render appropriate dashboard based on mode
    if mode == "simple":
        render_simple_mode_dashboard(benchmarks, winners)
        render_recommendation_section(benchmarks)
    else:
        render_advanced_mode_dashboard(benchmarks, winners)

        # Additional technical sections for advanced mode
        st.markdown("## 📋 Comparative Analysis")

        if len(benchmarks) >= 2:
            comparison_data = []
            for benchmark in benchmarks:
                comparison_data.append(
                    {
                        "Platform": benchmark.metadata.platform,
                        "Requests": f"{benchmark.metadata.total_requests:,}",
                        "Success %": f"{benchmark.success_rate*100:.2f}",
                        "TTFT P50": f"{benchmark.ttft_p50:.1f}ms",
                        "TTFT P99": f"{benchmark.ttft_p99:.1f}ms",
                        "TPOT P50": f"{benchmark.tpot_p50:.1f}ms",
                        "Throughput": f"{benchmark.throughput_avg:.1f}",
                        "RPS": f"{benchmark.rps:.2f}",
                    }
                )

            st.table(comparison_data)

            # Winners summary
            st.markdown("### 🏆 Performance Leaders")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "⚡ Best TTFT P50",
                    benchmarks[winners["ttft"]].metadata.platform,
                    f"{benchmarks[winners['ttft']].ttft_p50:.1f}ms",
                )

            with col2:
                st.metric(
                    "⚡ Best TTFT P99",
                    benchmarks[winners["ttft"]].metadata.platform,
                    f"{benchmarks[winners['ttft']].ttft_p99:.1f}ms",
                )

            with col3:
                st.metric(
                    "🚀 Best Throughput",
                    benchmarks[winners["throughput"]].metadata.platform,
                    f"{benchmarks[winners['throughput']].throughput_avg:.1f}",
                )

            with col4:
                st.metric(
                    "✅ Best Reliability",
                    benchmarks[winners["reliability"]].metadata.platform,
                    f"{benchmarks[winners['reliability']].success_rate*100:.3f}%",
                )

        else:
            st.info("Upload at least 2 benchmarks for comparison")

        st.markdown("---")
        st.markdown("👈 Navigate to analysis pages in the sidebar for detailed charts")
