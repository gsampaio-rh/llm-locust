"""
Token Analysis

Understand how token counts affect performance and identify patterns.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.visualizations import (
    create_token_distribution_chart,
    create_token_scatter_plot,
)

st.set_page_config(page_title="Token Analysis", page_icon="🔤", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("🔤 Token Analysis")
st.caption("Understand how input/output token counts affect performance")

st.markdown("---")

# Chart controls
with st.expander("⚙️ Chart Settings"):
    col1, col2 = st.columns(2)
    with col1:
        marker_size = st.slider(
            "Marker Size",
            min_value=4,
            max_value=20,
            value=10,
            help="Adjust scatter plot marker size for better visibility",
        )
    with col2:
        sample_points = st.number_input(
            "Max Points to Display",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
            help="Sample data if more points than this (reduces clutter)",
        )

st.markdown("---")

# ============= TOKEN OVERVIEW =============
st.subheader("📊 Token Statistics Overview")

token_stats = []
for benchmark in benchmarks:
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    
    if len(success_df) > 0:
        input_data = success_df["input_tokens"].dropna()
        output_data = success_df["output_tokens"].dropna()
        
        token_stats.append({
            "Platform": benchmark.metadata.platform,
            "Avg Input": f"{input_data.mean():.0f}",
            "Min Input": f"{input_data.min():.0f}",
            "Max Input": f"{input_data.max():.0f}",
            "Avg Output": f"{output_data.mean():.0f}",
            "Min Output": f"{output_data.min():.0f}",
            "Max Output": f"{output_data.max():.0f}",
            "Total Tokens": f"{(input_data.sum() + output_data.sum()):,}",
        })

st.table(token_stats)

st.markdown("---")

# ============= INPUT VS OUTPUT =============
st.subheader("📈 Input vs Output Tokens")

fig = create_token_scatter_plot(
    benchmarks,
    "input_tokens",
    "output_tokens",
    "Input vs Output Tokens - Correlation Analysis",
    max_points=sample_points,
    marker_size=marker_size,
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
**What this shows:**
- Each dot is one request
- X-axis: Input tokens (prompt length)
- Y-axis: Output tokens (response length)
- Correlation coefficient (r) shows how related they are
- Trendline (dashed) shows if longer prompts → longer responses
""")

st.markdown("---")

# ============= TOKEN DISTRIBUTIONS =============
st.subheader("📊 Token Distribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Input Token Distribution**")
    fig = create_token_distribution_chart(
        benchmarks,
        "input_tokens",
        "Input Token Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows the range and frequency of input token counts")

with col2:
    st.markdown("**Output Token Distribution**")
    fig = create_token_distribution_chart(
        benchmarks,
        "output_tokens",
        "Output Token Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows the range and frequency of output token counts")

st.markdown("---")

# ============= PROMPT LENGTH IMPACT =============
st.subheader("⚡ Prompt Length Impact on TTFT")

fig = create_token_scatter_plot(
    benchmarks,
    "input_tokens",
    "ttft_ms",
    "TTFT vs Input Tokens - Does Prompt Length Slow Down Response?",
    max_points=sample_points,
    marker_size=marker_size,
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
**What to look for:**
- Upward trend = longer prompts take longer to start responding
- Flat = prompt length doesn't affect TTFT much
- Strong correlation (r > 0.7) = prompt length is a major factor
- Weak correlation (r < 0.3) = other factors matter more
""")

st.markdown("---")

# ============= GENERATION LENGTH IMPACT =============
st.subheader("🔄 Generation Length Impact on TPOT")

fig = create_token_scatter_plot(
    benchmarks,
    "output_tokens",
    "tpot_ms",
    "TPOT vs Output Tokens - Does Generation Length Affect Streaming Speed?",
    max_points=sample_points,
    marker_size=marker_size,
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
**What to look for:**
- Upward trend = longer generations slow down streaming
- Flat = generation length doesn't affect TPOT (ideal)
- High scatter = inconsistent TPOT regardless of length
""")

st.markdown("---")

# ============= TECHNICAL INSIGHTS =============
st.subheader("🔬 Technical Insights")

if len(benchmarks) >= 2:
    import numpy as np
    
    # Analyze correlations
    st.markdown("**Correlation Analysis:**")
    
    for benchmark in benchmarks:
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        
        if len(success_df) > 10:
            # Calculate correlations
            df_clean = success_df[["input_tokens", "output_tokens", "ttft_ms", "tpot_ms"]].dropna()
            
            if len(df_clean) > 1:
                # Input vs TTFT correlation
                r_input_ttft = np.corrcoef(df_clean["input_tokens"], df_clean["ttft_ms"])[0, 1]
                
                # Output vs TPOT correlation
                r_output_tpot = np.corrcoef(df_clean["output_tokens"], df_clean["tpot_ms"])[0, 1]
                
                # Input vs Output correlation
                r_input_output = np.corrcoef(df_clean["input_tokens"], df_clean["output_tokens"])[0, 1]
                
                st.markdown(f"**{benchmark.metadata.platform}:**")
                
                # Input → TTFT
                if abs(r_input_ttft) > 0.7:
                    st.markdown(f"- ⚠️ Strong correlation between prompt length and TTFT (r={r_input_ttft:.3f}) - longer prompts significantly slow response")
                elif abs(r_input_ttft) > 0.3:
                    st.markdown(f"- 📊 Moderate correlation between prompt length and TTFT (r={r_input_ttft:.3f})")
                else:
                    st.markdown(f"- ✅ Weak correlation between prompt length and TTFT (r={r_input_ttft:.3f}) - prompt length doesn't affect response time much")
                
                # Output → TPOT
                if abs(r_output_tpot) > 0.7:
                    st.markdown(f"- ⚠️ Strong correlation between generation length and TPOT (r={r_output_tpot:.3f}) - longer responses slow streaming")
                elif abs(r_output_tpot) > 0.3:
                    st.markdown(f"- 📊 Moderate correlation between generation length and TPOT (r={r_output_tpot:.3f})")
                else:
                    st.markdown(f"- ✅ Weak correlation between generation length and TPOT (r={r_output_tpot:.3f}) - consistent streaming speed")
                
                # Input → Output
                if abs(r_input_output) > 0.7:
                    st.markdown(f"- 📊 Input/output highly correlated (r={r_input_output:.3f}) - longer prompts → longer responses")
                elif abs(r_input_output) > 0.3:
                    st.markdown(f"- 📊 Input/output moderately correlated (r={r_input_output:.3f})")
                else:
                    st.markdown(f"- ℹ️ Input/output weakly correlated (r={r_input_output:.3f}) - response length is independent of prompt")
                
                st.markdown("")

else:
    st.info("Upload at least 2 benchmarks to see comparative insights")
    
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        
        if len(success_df) > 0:
            input_data = success_df["input_tokens"].dropna()
            output_data = success_df["output_tokens"].dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Input Tokens", f"{input_data.mean():.0f}")
                st.caption(f"Range: {input_data.min():.0f} - {input_data.max():.0f}")
            
            with col2:
                st.metric("Avg Output Tokens", f"{output_data.mean():.0f}")
                st.caption(f"Range: {output_data.min():.0f} - {output_data.max():.0f}")

