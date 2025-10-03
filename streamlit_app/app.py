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
    MAX_FILE_SIZE_MB,
)
from lib.data_loader import load_multiple_benchmarks
from lib.explanations import (
    GLOSSARY,
    generate_simple_recommendation,
    reliability_to_emoji_and_label,
    speed_to_emoji_and_label,
)

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

# Sidebar - File Upload and Mode Toggle
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

# Main content
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    # No data loaded - show welcome screen
    st.title("🎯 Which LLM Platform Should You Choose?")
    st.markdown("### Let data guide your infrastructure decisions")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 🚀 How It Works")
        st.markdown("""
        1. **Run benchmarks** using llm-locust on your platforms
        2. **Upload the CSV files** using the sidebar
        3. **Get clear answers** about which platform is best
        
        No PhD required. Just upload and understand.
        """)
        
        st.info("👈 Start by uploading your benchmark files in the sidebar")
    
    with col2:
        st.markdown("#### ✨ What You'll Learn")
        st.markdown("""
        - Which platform is **fastest**
        - Which is most **reliable**
        - What it means for **your users**
        - Whether the difference **matters**
        """)

    st.markdown("---")
    
    # Quick feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 💡 Simple Mode")
        st.markdown("Plain English explanations. Perfect for product managers, executives, and anyone making decisions.")
    
    with col2:
        st.markdown("##### 🔬 Advanced Mode")
        st.markdown("Deep technical analysis. Statistical tests, distributions, and raw metrics for engineers.")
    
    with col3:
        st.markdown("##### 🎯 Both Modes")
        st.markdown("One click to switch between them. Start simple, go deep when needed.")
    
    st.markdown("---")
    
    # Sample insight
    st.markdown("#### 📖 Example Insight (Simple Mode)")
    st.info("""
    **🏆 Recommendation: Use vLLM**
    
    Why?
    - ✅ **19% faster** - Users get responses 200ms sooner
    - ✅ **99.8% success rate** - Only 2 failures per 1000 requests  
    - ✅ **Consistent** - Fast for everyone, not just sometimes
    
    **What this means:** Your users will have a noticeably better experience, and you'll handle support tickets less often.
    """)

else:
    # Data loaded - show analysis
    benchmarks = st.session_state["benchmarks"]
    mode = st.session_state.get("mode", "simple")
    
    # ============= KEY METRICS AT THE TOP =============
    
    if mode == "simple":
        # SIMPLE MODE - Beautiful metric cards grid
        st.markdown("## 📊 Performance Comparison")
        st.caption("All the numbers that matter, side by side")
        
        # Find winners
        best_ttft_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].ttft_p50)
        best_tpot_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].tpot_p50)
        best_throughput_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg)
        best_reliability_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)
        
        # Create a column for each platform
        cols = st.columns(len(benchmarks))
        
        for idx, benchmark in enumerate(benchmarks):
            with cols[idx]:
                # Platform header with badge
                if idx == best_ttft_idx:
                    st.markdown(f"### 🏆 {benchmark.metadata.platform}")
                else:
                    st.markdown(f"### {benchmark.metadata.platform}")
                
                st.markdown("---")
                
                # TTFT - Response Speed
                speed_emoji, speed_label, _ = speed_to_emoji_and_label(benchmark.ttft_p50)
                st.markdown(f"**⚡ Response Speed**")
                st.markdown(f"## {benchmark.ttft_p50:.0f}ms")
                st.caption(f"{speed_emoji} {speed_label}" + (" 🏆" if idx == best_ttft_idx else ""))
                
                st.markdown("")
                
                # TPOT - Streaming
                st.markdown(f"**🔄 Streaming Speed**")
                st.markdown(f"## {benchmark.tpot_p50:.1f}ms")
                st.caption("per word" + (" 🏆" if idx == best_tpot_idx else ""))
                
                st.markdown("")
                
                # Throughput
                st.markdown(f"**🚀 Throughput**")
                st.markdown(f"## {benchmark.throughput_avg:.0f}")
                st.caption("tokens/sec" + (" 🏆" if idx == best_throughput_idx else ""))
                
                st.markdown("")
                
                # Reliability
                reliability_emoji, _, _ = reliability_to_emoji_and_label(benchmark.success_rate)
                st.markdown(f"**✅ Reliability**")
                st.markdown(f"## {benchmark.success_rate*100:.2f}%")
                st.caption(f"{reliability_emoji} Success rate" + (" 🏆" if idx == best_reliability_idx else ""))
                
                st.markdown("---")
                
                # Quick verdict
                if benchmark.ttft_p50 < 500 and benchmark.success_rate >= 0.999:
                    st.success("**Excellent** ✨")
                elif benchmark.ttft_p50 < 1000 and benchmark.success_rate >= 0.99:
                    st.info("**Good** 👍")
                else:
                    st.warning("**Needs work** ⚠️")
        
        # Quick help
        st.markdown("")
        with st.expander("🤔 What do these metrics mean?"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **⚡ Response Speed (TTFT)**
                - Time until first word appears
                - Target: Under 1 second
                - 🟢 <500ms • 🟡 <1s • 🔴 >1s
                
                **🔄 Streaming Speed (TPOT)**
                - Time per word generated
                - Target: Smooth & natural
                - Good: <15ms per word
                """)
            
            with col2:
                st.markdown("""
                **🚀 Throughput**
                - Words generated per second
                - Higher = more users
                - Typical: 500-2000 tok/s
                
                **✅ Reliability**
                - Requests that succeed
                - Target: 99.9%+ (production)
                - 🟢 >99.9% • 🟡 >99% • 🔴 <99%
                """)
    
    else:
        # ADVANCED MODE - Detailed metric table
        st.markdown("## 📊 Key Metrics at a Glance")
        st.caption("Core performance indicators across all platforms")
        
        # Create columns for each benchmark
        cols = st.columns(len(benchmarks))
        
        for idx, benchmark in enumerate(benchmarks):
            with cols[idx]:
                st.markdown(f"### {benchmark.metadata.platform}")
                
                st.metric(
                    "TTFT P50",
                    f"{benchmark.ttft_p50:.1f}ms",
                    delta=f"P99: {benchmark.ttft_p99:.0f}ms"
                )
                
                st.metric(
                    "TPOT P50",
                    f"{benchmark.tpot_p50:.2f}ms",
                    delta=f"P99: {benchmark.tpot_p99:.1f}ms"
                )
                
                st.metric(
                    "Throughput",
                    f"{benchmark.throughput_avg:.1f} tok/s",
                    delta=f"RPS: {benchmark.rps:.2f}"
                )
                
                st.metric(
                    "Success Rate",
                    f"{benchmark.success_rate*100:.3f}%",
                    delta=f"{benchmark.metadata.failed_requests} failures"
                )
    
    st.markdown("---")
    
    if mode == "simple":
        # ============= SIMPLE MODE =============
        st.title("🎯 Which Platform Should You Choose?")
        st.markdown("### Clear, data-driven comparison")
        
        st.markdown("---")
        
        # Generate recommendation
        recommendation = generate_simple_recommendation(benchmarks)
        
        if recommendation["recommended"]:
            st.markdown("## 🏆 The Answer")
            
            # Big recommendation box
            st.success(f"""
            ### We recommend: **{recommendation["recommended"]}**
            
            Based on {len(benchmarks)} platform comparison(s)
            """)
            
            # Why section
            st.markdown("#### Why?")
            for reason in recommendation["reasons"]:
                st.markdown(f"- {reason}")
            
            # Find the winner benchmark
            winner_benchmark = next(
                b for b in benchmarks 
                if b.metadata.platform == recommendation["recommended"]
            )
            
            st.markdown("---")
            
            # Visual comparison
            st.markdown("## 📊 Quick Comparison")
            
            st.markdown("### ⚡ Response Speed")
            st.caption("How long users wait before seeing text appear")
            
            for benchmark in sorted(benchmarks, key=lambda b: b.ttft_p50):
                emoji, label, description = speed_to_emoji_and_label(benchmark.ttft_p50)
                
                # Calculate bar length (relative to slowest)
                max_ttft = max(b.ttft_p50 for b in benchmarks)
                bar_length = int((benchmark.ttft_p50 / max_ttft) * 12)
                bar = "█" * bar_length + "░" * (12 - bar_length)
                
                col1, col2, col3 = st.columns([2, 3, 2])
                with col1:
                    st.markdown(f"**{emoji} {benchmark.metadata.platform}**")
                with col2:
                    st.markdown(f"`{bar}` {label}")
                with col3:
                    st.caption(f"{benchmark.ttft_p50:.0f}ms")
                
                if benchmark.metadata.platform == recommendation["recommended"]:
                    st.caption(f"→ {description} 🏆")
                else:
                    st.caption(f"→ {description}")
            
            st.markdown("")
            
            # Reliability comparison
            st.markdown("### 🎯 Reliability")
            st.caption("How often requests succeed without errors")
            
            for benchmark in sorted(benchmarks, key=lambda b: b.success_rate, reverse=True):
                emoji, label, description = reliability_to_emoji_and_label(benchmark.success_rate)
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown(f"**{emoji} {benchmark.metadata.platform}**")
                with col2:
                    st.markdown(f"{benchmark.success_rate*100:.2f}% {label}")
                
                st.caption(f"→ {description}")
            
            st.markdown("---")
            
            # What this means
            st.markdown("## 💡 What This Means For You")
            
            if len(benchmarks) >= 2:
                # Compare winner to second place
                sorted_by_ttft = sorted(benchmarks, key=lambda b: b.ttft_p50)
                winner = sorted_by_ttft[0]
                second = sorted_by_ttft[1] if len(sorted_by_ttft) > 1 else winner
                
                if winner.metadata.platform == recommendation["recommended"]:
                    time_diff_ms = second.ttft_p50 - winner.ttft_p50
                    time_diff_s = time_diff_ms / 1000
                    pct_faster = (time_diff_ms / second.ttft_p50) * 100
                    
                    st.markdown(f"""
                    - **Speed:** Users get responses **{time_diff_ms:.0f}ms ({pct_faster:.0f}%) faster** with {winner.metadata.platform}
                    - **Perception:** {time_diff_s:.2f} seconds might not sound like much, but users absolutely notice
                    - **Impact:** At 10,000 requests/day, that's **{(time_diff_s * 10000 / 3600):.1f} hours** of saved waiting time
                    """)
                
                # Reliability impact
                reliability_diff = (winner.success_rate - second.success_rate) * 100
                if abs(reliability_diff) > 0.1:
                    failures_per_1000 = (1 - winner.success_rate) * 1000
                    st.markdown(f"""
                    - **Reliability:** {winner.metadata.platform} has **{reliability_diff:.2f}% higher success rate**
                    - **Failures:** Only ~**{failures_per_1000:.0f} failures per 1,000 requests**
                    - **Support:** Fewer error tickets means less firefighting
                    """)
            
            st.markdown("---")
            
            # Show more details button
            if st.button("🔬 Show Me The Technical Details", use_container_width=True):
                st.session_state["mode"] = "advanced"
                st.rerun()
    
    else:
        # ============= ADVANCED MODE =============
        st.title("🎯 LLM Benchmark Comparison Dashboard")
        st.caption("Technical Analysis Mode")
        
        st.markdown("---")
        
        # Benchmark cards
        st.markdown("### 📂 Loaded Benchmarks")
        
        cols = st.columns(min(len(benchmarks), 4))
        for idx, benchmark in enumerate(benchmarks):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                quality_emoji = (
                    "🟢" if benchmark.quality_score >= 90
                    else "🟡" if benchmark.quality_score >= 70
                    else "🔴"
                )
                
                st.metric(
                    label=f"**{benchmark.metadata.platform}**",
                    value=f"{benchmark.metadata.total_requests:,} req",
                    delta=f"{benchmark.metadata.duration_seconds:.0f}s",
                )
                st.caption(
                    f"{quality_emoji} Q:{benchmark.quality_score:.0f}% • "
                    f"✅{benchmark.success_rate*100:.1f}%"
                )
        
        st.markdown("---")
        
        # Comparison table
        st.markdown("### 📊 Comparative Metrics")
        
        if len(benchmarks) >= 2:
            comparison_data = []
            for benchmark in benchmarks:
                comparison_data.append({
                    "Platform": benchmark.metadata.platform,
                    "Requests": f"{benchmark.metadata.total_requests:,}",
                    "Success %": f"{benchmark.success_rate*100:.2f}",
                    "TTFT P50": f"{benchmark.ttft_p50:.1f}ms",
                    "TTFT P99": f"{benchmark.ttft_p99:.1f}ms",
                    "TPOT P50": f"{benchmark.tpot_p50:.1f}ms",
                    "Throughput": f"{benchmark.throughput_avg:.1f}",
                    "RPS": f"{benchmark.rps:.2f}",
                })
            
            st.table(comparison_data)
            
            # Winners
            st.markdown("### 🏆 Performance Leaders")
            col1, col2, col3, col4 = st.columns(4)
            
            best_ttft_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].ttft_p50)
            with col1:
                st.metric(
                    "⚡ Best TTFT P50",
                    benchmarks[best_ttft_idx].metadata.platform,
                    f"{benchmarks[best_ttft_idx].ttft_p50:.1f}ms",
                )
            
            best_ttft_p99_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].ttft_p99)
            with col2:
                st.metric(
                    "⚡ Best TTFT P99",
                    benchmarks[best_ttft_p99_idx].metadata.platform,
                    f"{benchmarks[best_ttft_p99_idx].ttft_p99:.1f}ms",
                )
            
            best_throughput_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg)
            with col3:
                st.metric(
                    "🚀 Best Throughput",
                    benchmarks[best_throughput_idx].metadata.platform,
                    f"{benchmarks[best_throughput_idx].throughput_avg:.1f}",
                )
            
            best_success_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].success_rate)
            with col4:
                st.metric(
                    "✅ Best Reliability",
                    benchmarks[best_success_idx].metadata.platform,
                    f"{benchmarks[best_success_idx].success_rate*100:.3f}%",
                )
        
        else:
            st.info("Upload at least 2 benchmarks for comparison")
        
        st.markdown("---")
        st.markdown("👈 Navigate to analysis pages in the sidebar")
        
        # Quick glossary
        with st.expander("📖 Metrics Glossary"):
            for term, info in GLOSSARY.items():
                st.markdown(f"**{term}** ({info['full_name']})")
                st.caption(info['simple'])
                st.markdown("")
