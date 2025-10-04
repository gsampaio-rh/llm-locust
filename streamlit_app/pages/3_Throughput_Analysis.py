"""
Throughput Analysis - Deep Dive

Analysis of tokens per second, requests per second, and stability.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.visualizations import create_multi_platform_timeline

st.set_page_config(page_title="Throughput Analysis", page_icon="🚀", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("🚀 Throughput Analysis")
st.caption("Tokens/second, requests/second, and stability metrics")

st.markdown("---")

# ============= OVERVIEW METRICS =============
st.subheader("📊 Throughput Overview")

# Check if any benchmarks have zero throughput
zero_throughput_platforms = [b.metadata.platform for b in benchmarks if b.throughput_avg == 0]
if zero_throughput_platforms:
    st.warning(f"""
    ⚠️ **Data Issue Detected:** {', '.join(zero_throughput_platforms)} has zero throughput.
    
    This means the benchmark data has zero output tokens. Possible causes:
    - Streaming responses weren't captured properly
    - API returned empty responses
    - Benchmark script issue
    
    **Solution:** Re-run the benchmark or use a different dataset with complete data.
    """)

cols = st.columns(len(benchmarks))
for idx, benchmark in enumerate(benchmarks):
    with cols[idx]:
        if benchmark.throughput_avg == 0:
            st.metric(
                label=f"**{benchmark.metadata.platform}** ⚠️",
                value="0 tok/s",
                delta="No output data",
            )
            st.caption("❌ Zero output tokens in dataset")
        else:
            st.metric(
                label=f"**{benchmark.metadata.platform}**",
                value=f"{benchmark.throughput_avg:.0f} tok/s",
                delta=f"{benchmark.rps:.2f} RPS",
            )
            
            # Capacity estimate
            est_users = int(benchmark.throughput_avg / 10)
            st.caption(f"~{est_users} concurrent users")

st.markdown("---")

# ============= THROUGHPUT OVER TIME =============
st.subheader("📈 Throughput Over Time (All Platforms)")

fig = create_multi_platform_timeline(
    benchmarks,
    "output_tokens_per_sec",
    "Throughput (Tokens/Second) Over Time - All Platforms",
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
**What to look for:**
- Flat lines = stable performance
- Downward trends = performance degradation over time
- Spikes/dips = inconsistent behavior
- Compare platforms side-by-side to see which is more stable
""")

st.markdown("---")

# ============= STABILITY METRICS =============
st.subheader("📊 Stability Analysis")

stability_data = []
for benchmark in benchmarks:
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    if len(success_df) == 0:
        continue

    throughput_data = success_df["output_tokens_per_sec"].dropna()

    # Coefficient of variation
    cv = (throughput_data.std() / throughput_data.mean()) * 100 if throughput_data.mean() > 0 else 0
    
    # Stability assessment
    if cv < 20:
        stability = "Excellent"
    elif cv < 40:
        stability = "Good"
    else:
        stability = "Fair"

    stability_data.append(
        {
            "Platform": benchmark.metadata.platform,
            "Mean Throughput": f"{throughput_data.mean():.1f} tok/s",
            "Std Dev": f"{throughput_data.std():.1f}",
            "CV (%)": f"{cv:.1f}%",
            "Stability": stability,
            "Min": f"{throughput_data.min():.1f}",
            "Max": f"{throughput_data.max():.1f}",
            "Range": f"{throughput_data.max() - throughput_data.min():.1f}",
        }
    )

st.table(stability_data)

with st.expander("ℹ️ Understanding Stability Metrics"):
    st.markdown("""
    **Coefficient of Variation (CV):**
    - Measures relative variability (Std Dev / Mean × 100)
    - Lower is better (more consistent)
    - CV < 20% = Excellent stability
    - CV 20-40% = Good stability
    - CV > 40% = Fair/Poor stability
    
    **Standard Deviation:**
    - Absolute measure of variation
    - Lower = more predictable performance
    
    **Range:**
    - Difference between fastest and slowest
    - Smaller range = more consistent
    """)

st.markdown("---")

# ============= CAPACITY ANALYSIS =============
st.subheader("🎯 Capacity & Performance")

capacity_data = []
for benchmark in benchmarks:
    # Calculate sustained metrics
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    
    if len(success_df) > 0:
        # Estimate concurrent users (rough calculation)
        avg_latency = success_df["end_to_end_s"].mean()
        est_concurrent_users = int(benchmark.rps * avg_latency) if avg_latency > 0 else 0
        
        capacity_data.append({
            "Platform": benchmark.metadata.platform,
            "Sustained RPS": f"{benchmark.rps:.2f}",
            "Avg Latency": f"{avg_latency:.2f}s",
            "Est. Concurrent Users": est_concurrent_users,
            "Total Tokens": f"{success_df['output_tokens'].sum():,}",
            "Avg Tokens/Request": f"{success_df['output_tokens'].mean():.0f}",
        })

st.table(capacity_data)

st.caption("""
**Est. Concurrent Users** is calculated as: RPS × Avg Latency  
This represents how many users the system can handle simultaneously.
""")

st.markdown("---")

# ============= TECHNICAL INSIGHTS =============
st.subheader("🔬 Technical Insights")

if len(benchmarks) >= 2:
    # Find best performer
    best_idx = max(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg)
    worst_idx = min(range(len(benchmarks)), key=lambda i: benchmarks[i].throughput_avg)
    
    best = benchmarks[best_idx]
    worst = benchmarks[worst_idx]
    
    throughput_diff = best.throughput_avg - worst.throughput_avg
    throughput_diff_pct = (throughput_diff / worst.throughput_avg) * 100
    
    st.success(
        f"🏆 **{best.metadata.platform}** has the highest throughput: "
        f"**{best.throughput_avg:.0f} tok/s** ({throughput_diff_pct:.1f}% higher than {worst.metadata.platform})"
    )
    
    # RPS comparison
    st.info(
        f"📊 RPS comparison: **{best.metadata.platform}** at {best.rps:.2f} req/s vs "
        f"**{worst.metadata.platform}** at {worst.rps:.2f} req/s"
    )
    
    # Stability comparison
    best_stability = []
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 0:
            throughput_data = success_df["output_tokens_per_sec"].dropna()
            cv = (throughput_data.std() / throughput_data.mean()) * 100 if throughput_data.mean() > 0 else 0
            best_stability.append((benchmark.metadata.platform, cv))
    
    best_stability.sort(key=lambda x: x[1])
    most_stable = best_stability[0]
    
    st.markdown("**Stability Ranking:**")
    for rank, (platform, cv) in enumerate(best_stability, 1):
        if cv < 20:
            emoji = "✅"
            assessment = "Very stable"
        elif cv < 40:
            emoji = "📊"
            assessment = "Moderately stable"
        else:
            emoji = "⚠️"
            assessment = "Variable performance"
        
        st.markdown(f"{rank}. {emoji} **{platform}**: CV={cv:.1f}% ({assessment})")
    
    # Performance degradation check
    st.markdown("**Performance Degradation Check:**")
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 100:
            # Compare first half to second half
            mid_point = len(success_df) // 2
            first_half = success_df.iloc[:mid_point]["output_tokens_per_sec"].mean()
            second_half = success_df.iloc[mid_point:]["output_tokens_per_sec"].mean()
            
            change_pct = ((second_half - first_half) / first_half) * 100
            
            if abs(change_pct) < 5:
                st.markdown(f"- ✅ **{benchmark.metadata.platform}**: Stable throughout ({change_pct:+.1f}%)")
            elif change_pct < -10:
                st.markdown(f"- ⚠️ **{benchmark.metadata.platform}**: Performance degradation detected ({change_pct:+.1f}%)")
            elif change_pct > 10:
                st.markdown(f"- 📈 **{benchmark.metadata.platform}**: Performance improved over time ({change_pct:+.1f}%)")
            else:
                st.markdown(f"- 📊 **{benchmark.metadata.platform}**: Minor variation ({change_pct:+.1f}%)")

else:
    st.info("Upload at least 2 benchmarks to see comparative insights")
    
    # Show single platform insights
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]
        
        st.metric(
            "Throughput",
            f"{benchmark.throughput_avg:.0f} tok/s",
            delta=f"{benchmark.rps:.2f} RPS",
        )
        
        # Stability assessment
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 0:
            throughput_data = success_df["output_tokens_per_sec"].dropna()
            cv = (throughput_data.std() / throughput_data.mean()) * 100 if throughput_data.mean() > 0 else 0
            
            if cv < 20:
                st.success(f"✅ Excellent stability (CV={cv:.1f}%)")
            elif cv < 40:
                st.info(f"📊 Good stability (CV={cv:.1f}%)")
            else:
                st.warning(f"⚠️ Variable performance (CV={cv:.1f}%) - investigate")

