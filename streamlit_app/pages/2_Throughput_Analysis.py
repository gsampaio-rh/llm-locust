"""
Throughput Analysis Page

Understand system capacity in plain English, with technical deep-dive on demand.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.visualizations import create_time_series_chart

st.set_page_config(page_title="Throughput Analysis", page_icon="🚀", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]
mode = st.session_state.get("mode", "simple")

# Title based on mode
if mode == "simple":
    st.title("🚀 System Capacity Analysis")
    st.markdown("### How many users can you handle?")
else:
    st.title("🚀 Throughput Analysis")
    st.caption("Tokens/second, requests/second, and stability metrics")

st.markdown("---")

if mode == "simple":
    # ============= SIMPLE MODE =============
    
    st.markdown("## 💡 What is Throughput?")
    st.info("""
    **Throughput** is how many users your system can serve at the same time.
    
    Think of it like a restaurant:
    - Higher throughput = more customers you can serve per hour
    - Lower throughput = longer wait times or turning customers away
    """)
    
    with st.expander("🤔 Tell me more"):
        st.markdown("""
        **Technical definition:** Throughput measures how many tokens (words) your system generates per second.
        
        **Why it matters:**
        - Determines your maximum concurrent users
        - Affects infrastructure costs (higher = more efficient)
        - Impacts scalability and growth capacity
        
        **Good targets:**
        - Small app: 100-500 tokens/second
        - Medium app: 500-2,000 tokens/second
        - Large app: 2,000+ tokens/second
        """)
    
    st.markdown("---")
    
    # Simple comparison
    st.markdown("## 📊 Which Platform Handles More Load?")
    
    # Sort by throughput
    sorted_benchmarks = sorted(benchmarks, key=lambda b: b.throughput_avg, reverse=True)
    
    for rank, benchmark in enumerate(sorted_benchmarks, 1):
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            if rank == 1:
                st.markdown("### 🏆")
            else:
                st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"### {benchmark.metadata.platform}")
            
            # Calculate concurrent users estimate
            # Rough estimate: 1 token/s per concurrent user (assuming 50-100 token responses)
            est_users = int(benchmark.throughput_avg / 10)  # Conservative estimate
            st.caption(f"Can handle ~**{est_users} concurrent users**")
        
        with col3:
            st.metric("Capacity", f"{benchmark.throughput_avg:.0f} tok/s")
            st.caption(f"{benchmark.rps:.1f} req/s")
        
        # Show comparison to leader
        if rank > 1:
            leader = sorted_benchmarks[0]
            diff_pct = ((leader.throughput_avg - benchmark.throughput_avg) / leader.throughput_avg) * 100
            st.caption(f"→ {diff_pct:.0f}% less capacity than {leader.metadata.platform}")
        
        st.markdown("")
    
    st.markdown("---")
    
    # What this means
    st.markdown("## 💭 What This Means For Your Business")
    
    if len(benchmarks) >= 2:
        best = sorted_benchmarks[0]
        worst = sorted_benchmarks[-1]
        
        capacity_diff_pct = ((best.throughput_avg - worst.throughput_avg) / worst.throughput_avg) * 100
        
        st.markdown(f"""
        **With {best.metadata.platform}:**
        - Handle **{capacity_diff_pct:.0f}% more users** than {worst.metadata.platform}
        - Process **{best.throughput_avg:.0f} tokens/second** vs {worst.metadata.platform}'s {worst.throughput_avg:.0f}
        - Better infrastructure efficiency = lower costs per user
        """)
        
        # Cost implications
        best_cost_per_1k_tokens = 1000 / best.throughput_avg
        worst_cost_per_1k_tokens = 1000 / worst.throughput_avg
        cost_diff = ((worst_cost_per_1k_tokens - best_cost_per_1k_tokens) / worst_cost_per_1k_tokens) * 100
        
        if cost_diff > 10:
            st.success(
                f"💰 **{best.metadata.platform}** is ~**{cost_diff:.0f}% more cost-efficient** "
                f"(serves more users with same hardware)"
            )
    
    st.markdown("---")
    
    # Stability insight
    st.markdown("## 📊 Performance Stability")
    st.caption("Does it stay fast under load?")
    
    for benchmark in sorted_benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) == 0:
            continue
        
        throughput_data = success_df["output_tokens_per_sec"].dropna()
        cv = (throughput_data.std() / throughput_data.mean()) * 100 if throughput_data.mean() > 0 else 0
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown(f"**{benchmark.metadata.platform}**")
            
            if cv < 20:
                st.caption("✅ Very stable - consistent performance")
            elif cv < 40:
                st.caption("⚠️ Some variation - mostly predictable")
            else:
                st.caption("❌ High variation - unpredictable performance")
        
        with col2:
            st.metric("Stability Score", f"{100-cv:.0f}%")
            st.caption(f"CV: {cv:.1f}%")
    
    with st.expander("ℹ️ What is stability?"):
        st.markdown("""
        **Stability** measures how consistent performance is over time.
        
        - **High stability:** Users get the same fast experience every time
        - **Low stability:** Sometimes fast, sometimes slow (frustrating!)
        
        **Coefficient of Variation (CV):**
        - CV < 20% = Very stable ✅
        - CV 20-40% = Moderately stable ⚠️
        - CV > 40% = Unstable ❌
        """)
    
    st.markdown("---")
    
    if st.button("🔬 Show Me The Technical Charts", use_container_width=True):
        st.session_state["mode"] = "advanced"
        st.rerun()

else:
    # ============= ADVANCED MODE =============
    
    # Overview metrics
    st.subheader("📊 Throughput Overview")
    
    cols = st.columns(len(benchmarks))
    for idx, benchmark in enumerate(benchmarks):
        with cols[idx]:
            st.metric(
                label=f"{benchmark.metadata.platform}",
                value=f"{benchmark.throughput_avg:.1f} tok/s",
                delta=f"{benchmark.rps:.2f} RPS",
            )
    
    # Time series analysis
    st.markdown("---")
    st.subheader("📈 Throughput Over Time")
    
    selected_platform = st.selectbox(
        "Select Platform",
        [b.metadata.platform for b in benchmarks],
    )
    
    selected_benchmark = next(b for b in benchmarks if b.metadata.platform == selected_platform)
    
    fig = create_time_series_chart(
        selected_benchmark,
        "output_tokens_per_sec",
        f"{selected_platform} - Tokens/Second Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **What to look for:**
    - Flat line = stable performance
    - Downward trend = performance degradation over time
    - Spikes/dips = inconsistent behavior
    """)
    
    # Stability analysis
    st.markdown("---")
    st.subheader("📊 Stability Metrics")
    
    stability_data = []
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) == 0:
            continue
        
        throughput_data = success_df["output_tokens_per_sec"].dropna()
        
        # Coefficient of variation
        cv = (throughput_data.std() / throughput_data.mean()) * 100 if throughput_data.mean() > 0 else 0
        
        stability_data.append({
            "Platform": benchmark.metadata.platform,
            "Mean Throughput": f"{throughput_data.mean():.1f} tok/s",
            "Std Dev": f"{throughput_data.std():.1f}",
            "CV (%)": f"{cv:.1f}%",
            "Min": f"{throughput_data.min():.1f}",
            "Max": f"{throughput_data.max():.1f}",
            "Range": f"{throughput_data.max() - throughput_data.min():.1f}",
        })
    
    st.table(stability_data)
    
    with st.expander("ℹ️ Understanding These Metrics"):
        st.markdown("""
        **Coefficient of Variation (CV):**
        - Measures relative variability
        - Lower is better (more consistent)
        - CV < 20% = excellent stability
        
        **Standard Deviation:**
        - Absolute measure of variation
        - Lower = more predictable performance
        
        **Range:**
        - Difference between fastest and slowest
        - Smaller range = more consistent
        """)
