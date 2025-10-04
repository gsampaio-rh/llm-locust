"""
Cost Analysis Calculator

Simple TCO (Total Cost of Ownership) comparison across platforms.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INSTANCE_CONFIGS

st.set_page_config(page_title="Cost Analysis", page_icon="💰", layout="wide")

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("💰 Cost Analysis Calculator")
st.caption("Calculate and compare Total Cost of Ownership (TCO) across platforms")

st.markdown("---")

# ============= COST INPUT SECTION =============
st.subheader("⚙️ Select Instance Type")
st.caption("Just pick your machine - pricing auto-fills!")

# Initialize session state for cost config
if "cost_config" not in st.session_state:
    st.session_state["cost_config"] = {}
    # Default to first instance config for each platform
    default_instance = list(INSTANCE_CONFIGS.keys())[0]
    for benchmark in benchmarks:
        config = INSTANCE_CONFIGS[default_instance]
        st.session_state["cost_config"][benchmark.metadata.platform] = {
            "instance_name": default_instance,
            "gpu_type": config["gpu"],
            "gpu_count": config["gpu_count"],
            "cloud_provider": config["provider"],
            "cost_per_hour": config["cost_per_hour"],
        }

# Create tabs for each platform
tabs = st.tabs([b.metadata.platform for b in benchmarks])

instance_list = list(INSTANCE_CONFIGS.keys())

for idx, benchmark in enumerate(benchmarks):
    platform = benchmark.metadata.platform
    
    with tabs[idx]:
        # Simple instance dropdown
        current_instance = st.session_state["cost_config"][platform].get("instance_name", instance_list[0])
        
        selected_instance = st.selectbox(
            "Select Instance Type",
            instance_list,
            key=f"instance_{platform}",
            index=instance_list.index(current_instance) if current_instance in instance_list else 0,
            help="Choose from common cloud instances with pre-filled pricing",
        )
        
        # Get config from selection
        config = INSTANCE_CONFIGS[selected_instance]
        
        # Show details
        if selected_instance == "Custom (Enter Manually)":
            # Manual input mode
            col1, col2 = st.columns(2)
            
            with col1:
                gpu_type = st.text_input(
                    "GPU Type",
                    value=st.session_state["cost_config"][platform].get("gpu_type", "H100"),
                    key=f"gpu_manual_{platform}",
                )
                gpu_count = st.number_input(
                    "GPU Count",
                    min_value=1,
                    max_value=16,
                    value=st.session_state["cost_config"][platform].get("gpu_count", 1),
                    key=f"gpu_count_manual_{platform}",
                )
            
            with col2:
                cloud_provider = st.text_input(
                    "Cloud Provider",
                    value=st.session_state["cost_config"][platform].get("cloud_provider", "AWS"),
                    key=f"provider_manual_{platform}",
                )
                cost_per_hour = st.number_input(
                    "Cost per Hour ($)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state["cost_config"][platform].get("cost_per_hour", 0.0),
                    step=0.50,
                    key=f"cost_manual_{platform}",
                )
            
            # Update config
            st.session_state["cost_config"][platform] = {
                "instance_name": selected_instance,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "cloud_provider": cloud_provider,
                "cost_per_hour": cost_per_hour,
            }
        else:
            # Auto-filled from pre-configured instance
            st.session_state["cost_config"][platform] = {
                "instance_name": selected_instance,
                "gpu_type": config["gpu"],
                "gpu_count": config["gpu_count"],
                "cloud_provider": config["provider"],
                "cost_per_hour": config["cost_per_hour"],
            }
            
            # Show what was auto-filled
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GPU", f"{config['gpu_count']}x {config['gpu']}")
            with col2:
                st.metric("Provider", config["provider"])
            with col3:
                st.metric("Cost", f"${config['cost_per_hour']:.2f}/hr")
            
            st.caption(f"✅ All pricing auto-filled for {selected_instance}")

st.markdown("---")

# ============= COST CALCULATION =============
st.subheader("💵 Cost Efficiency Comparison")

# Calculate costs for each platform
cost_results = []

for benchmark in benchmarks:
    platform = benchmark.metadata.platform
    config = st.session_state["cost_config"][platform]
    
    # Get throughput metrics
    throughput_avg = benchmark.throughput_avg  # tokens/sec
    cost_per_hour = config["cost_per_hour"]
    gpu_type = config["gpu_type"]
    gpu_count = config.get("gpu_count", 1)
    
    if throughput_avg > 0:
        # Calculate costs
        tokens_per_hour = throughput_avg * 3600  # tok/s * 3600s
        cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1_000_000
        
        # Cost per 1K requests (using average tokens per request)
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 0:
            avg_tokens_per_request = success_df["output_tokens"].mean()
            if avg_tokens_per_request > 0:
                cost_per_1k_requests = (avg_tokens_per_request * 1000 / 1_000_000) * cost_per_million_tokens
            else:
                cost_per_1k_requests = 0
        else:
            cost_per_1k_requests = 0
        
        # Tokens per dollar (efficiency)
        tokens_per_dollar = tokens_per_hour / cost_per_hour if cost_per_hour > 0 else 0
        
        cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{gpu_count}x {gpu_type}",
            "Provider": config["cloud_provider"],
            "$/Hour": f"${cost_per_hour:.2f}",
            "$/1M Tokens": f"${cost_per_million_tokens:.2f}",
            "$/1K Requests": f"${cost_per_1k_requests:.3f}",
            "Tokens per $": f"{tokens_per_dollar:,.0f}",
        })
    else:
        cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{gpu_count}x {gpu_type}",
            "Provider": config["cloud_provider"],
            "$/Hour": f"${cost_per_hour:.2f}",
            "$/1M Tokens": "N/A (zero throughput)",
            "$/1K Requests": "N/A",
            "Tokens per $": "N/A",
        })

st.table(cost_results)

st.caption("""
**Key metrics explained:**
- **$/1M Tokens**: Cost to generate 1 million tokens (lower is better)
- **$/1K Requests**: Cost to serve 1,000 requests (based on average output tokens)
- **Tokens per $**: How many tokens you get for $1 (higher is better)
""")

st.markdown("---")

# ============= MONTHLY PROJECTIONS =============
st.subheader("📅 Monthly Cost Projections")

# QPS selector
col1, col2 = st.columns([2, 3])

with col1:
    target_qps = st.number_input(
        "Target QPS (Queries Per Second)",
        min_value=1,
        max_value=10000,
        value=100,
        step=10,
        help="Expected queries per second in production",
    )
    
    hours_per_month = st.number_input(
        "Operating Hours per Month",
        min_value=1,
        max_value=744,  # 31 days * 24 hours
        value=720,  # 30 days * 24 hours
        step=24,
        help="How many hours per month the system will run",
    )

with col2:
    st.metric("Total Requests per Month", f"{target_qps * 3600 * hours_per_month:,}")
    st.caption("Based on your QPS and operating hours")

st.markdown("---")

# Calculate monthly costs
monthly_projections = []

for benchmark in benchmarks:
    platform = benchmark.metadata.platform
    config = st.session_state["cost_config"][platform]
    cost_per_hour = config["cost_per_hour"]
    
    # Monthly infrastructure cost (fixed)
    monthly_infra_cost = cost_per_hour * hours_per_month
    
    # Calculate token volume at target QPS
    if benchmark.throughput_avg > 0:
        # How many concurrent instances needed?
        # throughput_avg is tok/s per instance
        # target_qps * avg_output_tokens = total tok/s needed
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        if len(success_df) > 0:
            avg_output_tokens = success_df["output_tokens"].mean()
            tokens_per_second_needed = target_qps * avg_output_tokens
            
            # Instances needed
            instances_needed = max(1, int(np.ceil(tokens_per_second_needed / benchmark.throughput_avg)))
            
            # Monthly cost = instances * cost/hr * hours
            monthly_total_cost = instances_needed * cost_per_hour * hours_per_month
            
            monthly_projections.append({
                "Platform": platform,
                "Instances Needed": instances_needed,
                "Cost per Instance": f"${cost_per_hour:.2f}/hr",
                "Monthly Infrastructure": f"${monthly_total_cost:,.0f}",
                "Cost per Request": f"${monthly_total_cost / (target_qps * 3600 * hours_per_month):.6f}",
            })
        else:
            monthly_projections.append({
                "Platform": platform,
                "Instances Needed": "N/A",
                "Cost per Instance": f"${cost_per_hour:.2f}/hr",
                "Monthly Infrastructure": f"${monthly_infra_cost:,.0f}",
                "Cost per Request": "N/A",
            })
    else:
        monthly_projections.append({
            "Platform": platform,
            "Instances Needed": "N/A (zero throughput)",
            "Cost per Instance": f"${cost_per_hour:.2f}/hr",
            "Monthly Infrastructure": f"${monthly_infra_cost:,.0f}",
            "Cost per Request": "N/A",
        })

st.table(monthly_projections)

st.caption("""
**How we calculate Instances Needed:**

We determine the minimum number of instances required to handle your target load:

1. **Calculate demand**: `Target QPS × Average Output Tokens = Total Tokens/Second Needed`
2. **Determine capacity**: Each instance can produce `Throughput (tokens/sec)` from benchmark results
3. **Scale to meet demand**: `Instances Needed = ⌈Total Tokens Needed ÷ Instance Throughput⌉` (rounded up)

**Example**: If you need 1,000 tokens/sec and each instance produces 250 tokens/sec, you need 4 instances.

*Note: This is a simplified calculation. In production, you'd also need to account for:*
- *Headroom for traffic spikes (typically 20-30% buffer)*
- *High availability (multi-AZ redundancy)*
- *Rolling updates and maintenance windows*
""")

# Show detailed calculation breakdown
with st.expander("🔢 Show Calculation Breakdown for Each Platform"):
    st.markdown(f"**Target Load:** {target_qps} QPS (queries per second)")
    st.markdown("---")
    
    for benchmark in benchmarks:
        platform = benchmark.metadata.platform
        
        if benchmark.throughput_avg > 0:
            success_df = benchmark.df[benchmark.df["status_code"] == 200]
            
            if len(success_df) > 0:
                avg_output_tokens = success_df["output_tokens"].mean()
                tokens_per_second_needed = target_qps * avg_output_tokens
                instances_needed = max(1, int(np.ceil(tokens_per_second_needed / benchmark.throughput_avg)))
                
                st.markdown(f"### {platform}")
                st.code(f"""
Step 1: Calculate Demand
    Target QPS                  = {target_qps} requests/sec
    × Avg Output Tokens         = {avg_output_tokens:.2f} tokens/request
    ─────────────────────────────────────────────────────
    = Tokens Needed             = {tokens_per_second_needed:.2f} tokens/sec

Step 2: Check Instance Capacity
    Instance Throughput         = {benchmark.throughput_avg:.2f} tokens/sec
    (from benchmark results)

Step 3: Calculate Instances Needed
    Tokens Needed               = {tokens_per_second_needed:.2f} tokens/sec
    ÷ Instance Throughput       = {benchmark.throughput_avg:.2f} tokens/sec
    ─────────────────────────────────────────────────────
    = Raw Result                = {tokens_per_second_needed / benchmark.throughput_avg:.4f}
    = Rounded Up (⌈⌉)           = {instances_needed} instance(s)

Result: You need {instances_needed} instance(s) to handle {target_qps} QPS
""", language="text")
            else:
                st.markdown(f"### {platform}")
                st.warning("No successful requests in benchmark data")
        else:
            st.markdown(f"### {platform}")
            st.warning("Zero throughput - cannot calculate instances needed")

st.markdown("---")

# ============= COST INSIGHTS =============
st.subheader("🎯 Cost Insights")

if len(benchmarks) >= 2:
    # Find most cost-efficient (lowest cost per 1M tokens)
    valid_platforms = []
    for benchmark in benchmarks:
        if benchmark.throughput_avg > 0:
            config = st.session_state["cost_config"][benchmark.metadata.platform]
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = benchmark.throughput_avg * 3600
            cost_per_million = (cost_per_hour / tokens_per_hour) * 1_000_000
            valid_platforms.append((benchmark.metadata.platform, cost_per_million, benchmark))
    
    if len(valid_platforms) >= 2:
        valid_platforms.sort(key=lambda x: x[1])  # Sort by cost
        
        best_platform, best_cost, best_benchmark = valid_platforms[0]
        worst_platform, worst_cost, worst_benchmark = valid_platforms[-1]
        
        savings = worst_cost - best_cost
        savings_pct = (savings / worst_cost) * 100
        
        st.success(f"""
        🏆 **{best_platform}** is most cost-efficient
        
        - **${best_cost:.2f}** per 1M tokens
        - **{savings_pct:.1f}% cheaper** than {worst_platform}
        - **Saves ${savings:.2f}** per 1M tokens
        """)
        
        # Calculate monthly savings at target QPS
        success_df_best = best_benchmark.df[best_benchmark.df["status_code"] == 200]
        success_df_worst = worst_benchmark.df[worst_benchmark.df["status_code"] == 200]
        
        if len(success_df_best) > 0 and len(success_df_worst) > 0:
            avg_tokens_best = success_df_best["output_tokens"].mean()
            avg_tokens_worst = success_df_worst["output_tokens"].mean()
            
            if avg_tokens_best > 0:
                monthly_requests = target_qps * 3600 * hours_per_month
                monthly_tokens = monthly_requests * avg_tokens_best
                monthly_savings = (monthly_tokens / 1_000_000) * savings
                
                if monthly_savings > 100:
                    st.info(f"""
                    💰 **Monthly Savings:** At {target_qps} QPS, using {best_platform} instead of {worst_platform} 
                    saves approximately **${monthly_savings:,.0f}/month**
                    """)
    
    # Cost vs Performance trade-off
    st.markdown("**Cost vs Performance Trade-offs:**")
    
    for benchmark in benchmarks:
        if benchmark.throughput_avg > 0:
            config = st.session_state["cost_config"][benchmark.metadata.platform]
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = benchmark.throughput_avg * 3600
            cost_per_million = (cost_per_hour / tokens_per_hour) * 1_000_000
            
            # Efficiency score: throughput per dollar per hour
            efficiency = benchmark.throughput_avg / cost_per_hour if cost_per_hour > 0 else 0
            
            st.markdown(f"- **{benchmark.metadata.platform}**: ${cost_per_million:.2f}/1M tok | "
                       f"TTFT P50: {benchmark.ttft_p50:.0f}ms | "
                       f"Efficiency: {efficiency:.1f} tok/s per $/hr")

else:
    st.info("Upload at least 2 benchmarks to see cost comparisons")
    
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]
        platform = benchmark.metadata.platform
        config = st.session_state["cost_config"][platform]
        
        if benchmark.throughput_avg > 0:
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = benchmark.throughput_avg * 3600
            cost_per_million = (cost_per_hour / tokens_per_hour) * 1_000_000
            
            st.metric(
                "Cost per 1M Tokens",
                f"${cost_per_million:.2f}",
                delta=f"{config['gpu_type']} on {config['cloud_provider']}",
            )

st.markdown("---")

# ============= HELP SECTION =============
with st.expander("💡 How to Use This Calculator"):
    st.markdown("""
    **Step 1: Select instance type for each platform**
    - Choose from pre-configured cloud instances (AWS, GCP, Azure)
    - Or select "Custom (Enter Manually)" to input your own pricing
    - Pricing auto-fills based on public cloud rates
    
    **Step 2: Review cost efficiency**
    - **$/1M Tokens**: Cost to generate 1 million tokens (lower is better)
    - **$/1K Requests**: Cost per 1,000 requests (lower is better)
    - **Tokens per $**: Efficiency metric (higher is better)
    
    **Step 3: Monthly projections**
    - Set your expected QPS (queries per second)
    - Set operating hours per month
    - **Instances Needed** shows the minimum number of instances to handle your target load
        - Calculated as: `⌈(Target QPS × Avg Output Tokens) ÷ Instance Throughput⌉`
        - This ensures you have enough capacity to meet demand
        - In production, add 20-30% buffer for spikes and redundancy
    - See total monthly infrastructure cost across all instances
    
    **Notes:**
    - Pricing is approximate and based on public cloud pricing as of Oct 2025
    - On-prem costs are amortized estimates (hardware + power + maintenance)
    - Actual costs may vary based on discounts, region, utilization, reserved instances, etc.
    - Multi-GPU instances are supported (pricing is per instance, not per GPU)
    """)

with st.expander("📋 Available Instance Types"):
    st.markdown("**Pre-configured instances with auto-filled pricing:**")
    
    for instance_name, config in INSTANCE_CONFIGS.items():
        if instance_name != "Custom (Enter Manually)":
            st.markdown(
                f"- **{instance_name}**: "
                f"{config['gpu_count']}x {config['gpu']} on {config['provider']} - "
                f"${config['cost_per_hour']:.2f}/hr"
            )

