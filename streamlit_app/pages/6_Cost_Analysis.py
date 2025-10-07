"""
Cost Analysis Calculator

Simple TCO (Total Cost of Ownership) comparison across platforms.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INSTANCE_CONFIGS
from lib.cloud_pricing_loader import (
    APIModelPricing,
    filter_models,
    get_models_by_provider,
    get_popular_models,
    get_unique_providers,
    get_unique_sizes,
    load_model_prices,
)
from lib.visualizations import create_breakeven_chart
from lib.yaml_parser import parse_all_yamls_in_directory

st.set_page_config(page_title="Cost Analysis", page_icon="💰", layout="wide")

# Load API pricing data
@st.cache_data
def load_api_pricing():
    """Load API pricing data (cached)."""
    return load_model_prices()

# Check if data is loaded
if "benchmarks" not in st.session_state or not st.session_state["benchmarks"]:
    st.warning("👈 Please upload benchmark CSV files in the main page first")
    st.stop()

benchmarks = st.session_state["benchmarks"]

st.title("💰 Cost Analysis Calculator")
st.caption("Calculate and compare Total Cost of Ownership (TCO) across platforms")

st.markdown("---")

# ============= BENCHMARK PERFORMANCE (AT THE TOP!) =============
st.subheader("📊 Your Actual Benchmark Performance")
st.caption("Single-instance throughput measured from your test data")

# Collect data for both tables
input_stats = []
output_stats = []

for benchmark in benchmarks:
    # Calculate actual tokens processed in test
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    total_output_tokens = success_df["output_tokens"].sum() if len(success_df) > 0 else 0
    total_input_tokens = success_df["input_tokens"].sum() if len(success_df) > 0 else 0
    
    # Get actual test duration (from first to last timestamp)
    test_duration_sec = benchmark.metadata.duration_seconds
    
    # Calculate INPUT throughput
    input_tokens_per_sec = total_input_tokens / test_duration_sec if test_duration_sec > 0 else 0
    input_tokens_per_hour = input_tokens_per_sec * 3600
    
    # Calculate time to process 1M INPUT tokens
    if input_tokens_per_sec > 0:
        seconds_for_1m_input = 1_000_000 / input_tokens_per_sec
        if seconds_for_1m_input < 60:
            time_for_1m_input = f"{seconds_for_1m_input:.1f}s"
        elif seconds_for_1m_input < 3600:
            time_for_1m_input = f"{seconds_for_1m_input/60:.1f}min"
        else:
            time_for_1m_input = f"{seconds_for_1m_input/3600:.1f}hr"
    else:
        time_for_1m_input = "N/A"
    
    # Calculate OUTPUT throughput
    output_tokens_per_sec = total_output_tokens / test_duration_sec if test_duration_sec > 0 else 0
    output_tokens_per_hour = output_tokens_per_sec * 3600
    
    # Calculate time to process 1M OUTPUT tokens
    if output_tokens_per_sec > 0:
        seconds_for_1m_output = 1_000_000 / output_tokens_per_sec
        if seconds_for_1m_output < 60:
            time_for_1m_output = f"{seconds_for_1m_output:.1f}s"
        elif seconds_for_1m_output < 3600:
            time_for_1m_output = f"{seconds_for_1m_output/60:.1f}min"
        else:
            time_for_1m_output = f"{seconds_for_1m_output/3600:.1f}hr"
    else:
        time_for_1m_output = "N/A"
    
    # Get total requests and tokens per request
    total_requests = benchmark.metadata.total_requests
    avg_input_per_request = success_df["input_tokens"].mean() if len(success_df) > 0 else 0
    avg_output_per_request = success_df["output_tokens"].mean() if len(success_df) > 0 else 0
    
    # Input table data
    input_stats.append({
        "Platform": benchmark.metadata.platform,
        "Test Duration": f"{test_duration_sec:.0f}s ({test_duration_sec/60:.1f}min)",
        "Total Requests": f"{total_requests:,}",
        "Avg Input/Request": f"{avg_input_per_request:.0f}",
        "Input Tokens Processed": f"{total_input_tokens:,}",
        "Input Throughput (tok/s)": f"{input_tokens_per_sec:.1f}",
        "Input Tokens/Hour": f"{input_tokens_per_hour:,.0f}",
        "Time for 1M Input Tokens": time_for_1m_input,
    })
    
    # Output table data
    output_stats.append({
        "Platform": benchmark.metadata.platform,
        "Test Duration": f"{test_duration_sec:.0f}s ({test_duration_sec/60:.1f}min)",
        "Total Requests": f"{total_requests:,}",
        "Avg Output/Request": f"{avg_output_per_request:.0f}",
        "Output Tokens Processed": f"{total_output_tokens:,}",
        "Output Throughput (tok/s)": f"{output_tokens_per_sec:.1f}",
        "Output Tokens/Hour": f"{output_tokens_per_hour:,.0f}",
        "Time for 1M Output Tokens": time_for_1m_output,
    })

# Display INPUT metrics table
st.markdown("#### 📥 Input Token Performance (Prompt Processing)")
st.table(input_stats)

st.caption("""
**Input metrics** - How fast the platform processes prompts:
- Input processing is typically very fast (parallel prefill)
- Larger prompts (more input tokens) increase TTFT but not throughput bottleneck
""")

st.markdown("---")

# Display OUTPUT metrics table
st.markdown("#### 📤 Output Token Performance (Generation) - **KEY FOR CAPACITY!**")
st.table(output_stats)

st.caption("""
**Output metrics** - How fast the platform generates responses:
- ⚡ **This is your capacity bottleneck!** Output generation is sequential (token-by-token)
- Use **Output Throughput** for capacity planning and instance scaling
- Output generation speed determines how many concurrent requests you can handle

⚠️ **Important**: Your benchmark = **1 instance** performance baseline.

💡 **For cost/capacity analysis**: Focus on **Output** metrics - that's what limits your throughput!
""")

st.markdown("---")

# ============= YAML AUTO-LOAD SECTION =============
st.subheader("📄 Auto-Load from Deployment YAMLs")

# Check if deployment YAMLs exist
configs_dir = Path(__file__).parent.parent.parent / "configs" / "engines"
yaml_configs_available = configs_dir.exists() and any(configs_dir.glob("*.yaml"))

if yaml_configs_available:
    if st.button("🔄 Load Instance Specs from Deployment YAMLs", type="primary"):
        with st.spinner("Parsing deployment YAMLs..."):
            yaml_configs = parse_all_yamls_in_directory(configs_dir)
            
            if yaml_configs:
                # Auto-populate cost configs from YAML
                if "yaml_loaded_configs" not in st.session_state:
                    st.session_state["yaml_loaded_configs"] = {}
                
                for platform_name, yaml_config in yaml_configs.items():
                    # Store the parsed config (just for reference)
                    st.session_state["yaml_loaded_configs"][platform_name] = yaml_config
                
                st.success(f"✅ Loaded {len(yaml_configs)} deployment config(s) from YAMLs!")
                st.rerun()
            else:
                st.error("❌ No valid deployment YAMLs found in configs/engines/")
    
    # Show what was loaded
    if "yaml_loaded_configs" in st.session_state and st.session_state["yaml_loaded_configs"]:
        with st.expander("📋 View Parsed Deployment Specs (from YAMLs)", expanded=True):
            st.caption("These specs were extracted from your deployment YAMLs. Select instance type below to set GPU type and pricing.")
            
            # Show platform name mapping debug info
            st.info(f"🔍 **Debug**: Found {len(st.session_state['yaml_loaded_configs'])} YAML config(s): {', '.join(st.session_state['yaml_loaded_configs'].keys())}")
            if benchmarks:
                benchmark_platforms = [b.metadata.platform for b in benchmarks]
                st.info(f"🔍 **Debug**: Benchmark platforms: {', '.join(benchmark_platforms)}")
            
            st.markdown("---")
            for platform_name, yaml_config in st.session_state["yaml_loaded_configs"].items():
                st.markdown(f"**{platform_name}** (from YAML)")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("GPU Count", yaml_config.gpu_count)
                with col2:
                    st.metric("CPU Cores", yaml_config.cpu_cores)
                with col3:
                    st.metric("Memory", yaml_config.memory_gi)
                with col4:
                    st.metric("Replicas", yaml_config.replicas)
                with col5:
                    if yaml_config.gpu_memory_utilization:
                        st.metric("GPU Util", f"{yaml_config.gpu_memory_utilization:.0%}")
                    else:
                        st.metric("GPU Util", "N/A")
                
                if yaml_config.model_name:
                    st.caption(f"📦 Model: `{yaml_config.model_name}`")
                
                if yaml_config.gpu_memory_utilization:
                    packing = 1 / yaml_config.gpu_memory_utilization
                    st.caption(f"💡 Theoretical packing: {packing:.2f}x models per GPU")
                
                st.markdown("---")
else:
    st.info(f"💡 **Tip:** Place your deployment YAMLs in `configs/engines/` directory to auto-load instance specifications!")

st.markdown("---")

# ============= COST INPUT SECTION =============
st.subheader("⚙️ Cost Calculation Settings")

# Cost allocation mode
allocation_mode = st.radio(
    "Cost Allocation Mode",
    options=["Proportional (GPU Memory Utilization)", "Full Instance Cost"],
    help="""
    **Proportional**: Allocate cost based on actual GPU memory used (from --gpu-memory-utilization).
    Example: If using 60% of GPU memory, cost = (Instance cost / GPUs) × 0.60
    
    **Full Instance Cost**: Charge full instance cost regardless of utilization.
    Use this if you have dedicated nodes for each deployment.
    """,
    horizontal=True,
)

use_proportional = allocation_mode.startswith("Proportional")

st.markdown("---")

st.subheader("⚙️ Select Instance Type")
st.caption("Just pick your machine - pricing auto-fills!")

# Initialize session state for cost config
if "cost_config" not in st.session_state:
    st.session_state["cost_config"] = {}

# Ensure all current benchmark platforms have cost config entries
default_instance = "AWS g6.4xlarge (1x L4)"  # Default to L4 instance
for benchmark in benchmarks:
    platform = benchmark.metadata.platform
    if platform not in st.session_state["cost_config"]:
        config = INSTANCE_CONFIGS[default_instance]
        st.session_state["cost_config"][platform] = {
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
input_cost_results = []
output_cost_results = []

for benchmark in benchmarks:
    platform = benchmark.metadata.platform
    config = st.session_state["cost_config"][platform]
    
    # Calculate SUSTAINED throughput for both input and output
    success_df_temp = benchmark.df[benchmark.df["status_code"] == 200]
    total_output_tokens_temp = success_df_temp["output_tokens"].sum() if len(success_df_temp) > 0 else 0
    total_input_tokens_temp = success_df_temp["input_tokens"].sum() if len(success_df_temp) > 0 else 0
    test_duration_temp = benchmark.metadata.duration_seconds
    
    output_throughput = total_output_tokens_temp / test_duration_temp if test_duration_temp > 0 else 0
    input_throughput = total_input_tokens_temp / test_duration_temp if test_duration_temp > 0 else 0
    
    base_cost_per_hour = config["cost_per_hour"]
    gpu_type = config["gpu_type"]
    instance_gpu_count = config.get("gpu_count", 1)
    
    # Check if we have YAML config for this platform
    # Normalize platform name for matching (case-insensitive)
    yaml_config = None
    if "yaml_loaded_configs" in st.session_state:
        # Try exact match first
        yaml_config = st.session_state["yaml_loaded_configs"].get(platform)
        
        # If not found, try case-insensitive match
        if not yaml_config:
            for yaml_platform, yaml_cfg in st.session_state["yaml_loaded_configs"].items():
                if yaml_platform.lower() == platform.lower():
                    yaml_config = yaml_cfg
                    break
    
    # Calculate actual cost based on allocation mode
    full_cost_per_hour = base_cost_per_hour  # Always show the full instance cost
    
    if use_proportional and yaml_config and yaml_config.gpu_memory_utilization:
        # Proportional allocation: cost per GPU × GPU util × replicas
        cost_per_gpu = base_cost_per_hour / instance_gpu_count
        gpu_util = yaml_config.gpu_memory_utilization
        deployment_gpu_count = yaml_config.gpu_count
        replicas = yaml_config.replicas
        
        cost_per_hour = cost_per_gpu * gpu_util * deployment_gpu_count * replicas
        cost_note = f"({deployment_gpu_count} GPU × {gpu_util:.0%} util × {replicas} replica)"
        
        # Calculate discount percentage
        discount_pct = ((full_cost_per_hour - cost_per_hour) / full_cost_per_hour * 100) if full_cost_per_hour > 0 else 0
    else:
        # Full instance cost
        cost_per_hour = base_cost_per_hour
        if yaml_config:
            replicas = yaml_config.replicas
            cost_per_hour = base_cost_per_hour * replicas
            full_cost_per_hour = base_cost_per_hour * replicas  # Update full cost for replicas
            cost_note = f"({replicas} replica × full instance)"
        else:
            cost_note = "(full instance cost)"
        
        discount_pct = 0  # No discount for full instance
    
    # Calculate INPUT costs
    if input_throughput > 0:
        input_tokens_per_hour = input_throughput * 3600
        input_cost_per_million = (cost_per_hour / input_tokens_per_hour) * 1_000_000
        
        # Time to process 1M input tokens
        seconds_for_1m_input = 1_000_000 / input_throughput
        if seconds_for_1m_input < 60:
            time_for_1m_input = f"{seconds_for_1m_input:.1f}s"
        elif seconds_for_1m_input < 3600:
            time_for_1m_input = f"{seconds_for_1m_input/60:.1f}min"
        else:
            time_for_1m_input = f"{seconds_for_1m_input/3600:.1f}hr"
        
        input_price_per_token = input_cost_per_million / 1_000_000
        input_tokens_per_dollar = input_tokens_per_hour / cost_per_hour if cost_per_hour > 0 else 0
        
        input_cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "Full $/Hour": f"${full_cost_per_hour:.2f}",
            "Discount %": f"{discount_pct:.0f}%" if discount_pct > 0 else "-",
            "Actual $/Hour": f"${cost_per_hour:.2f}",
            "Time for 1M Input Tokens": time_for_1m_input,
            "$/1M Input Tokens": f"${input_cost_per_million:.2f}",
            "Price per Input Token": f"${input_price_per_token:.6f}",
            "Input Tokens per $": f"{input_tokens_per_dollar:,.0f}",
        })
    else:
        input_cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "Full $/Hour": f"${full_cost_per_hour:.2f}",
            "Discount %": "-",
            "Actual $/Hour": f"${cost_per_hour:.2f}",
            "Time for 1M Input Tokens": "N/A",
            "$/1M Input Tokens": "N/A",
            "Price per Input Token": "N/A",
            "Input Tokens per $": "N/A",
        })
    
    # Calculate OUTPUT costs
    if output_throughput > 0:
        output_tokens_per_hour = output_throughput * 3600
        output_cost_per_million = (cost_per_hour / output_tokens_per_hour) * 1_000_000
        
        # Time to process 1M output tokens
        seconds_for_1m_output = 1_000_000 / output_throughput
        if seconds_for_1m_output < 60:
            time_for_1m_output = f"{seconds_for_1m_output:.1f}s"
        elif seconds_for_1m_output < 3600:
            time_for_1m_output = f"{seconds_for_1m_output/60:.1f}min"
        else:
            time_for_1m_output = f"{seconds_for_1m_output/3600:.1f}hr"
        
        output_price_per_token = output_cost_per_million / 1_000_000
        output_tokens_per_dollar = output_tokens_per_hour / cost_per_hour if cost_per_hour > 0 else 0
        
        output_cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "Full $/Hour": f"${full_cost_per_hour:.2f}",
            "Discount %": f"{discount_pct:.0f}%" if discount_pct > 0 else "-",
            "Actual $/Hour": f"${cost_per_hour:.2f}",
            "Time for 1M Output Tokens": time_for_1m_output,
            "$/1M Output Tokens": f"${output_cost_per_million:.2f}",
            "Price per Output Token": f"${output_price_per_token:.6f}",
            "Output Tokens per $": f"{output_tokens_per_dollar:,.0f}",
        })
    else:
        output_cost_results.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "Full $/Hour": f"${full_cost_per_hour:.2f}",
            "Discount %": "-",
            "Actual $/Hour": f"${cost_per_hour:.2f}",
            "Time for 1M Output Tokens": "N/A",
            "$/1M Output Tokens": "N/A",
            "Price per Output Token": "N/A",
            "Output Tokens per $": "N/A",
        })

# Display INPUT cost table
st.markdown("#### 📥 Input Token Costs (Prompt Processing)")
st.table(input_cost_results)

st.caption("""
**Input cost metrics explained:**
- **Full $/Hour**: Base instance cost before optimization
- **Discount %**: Savings from proportional GPU allocation
- **Actual $/Hour**: Real cost after optimization (what you pay!)
- **Time for 1M Input Tokens**: How long to process 1M input tokens (prompt handling)
- **$/1M Input Tokens**: Cost to process 1M input tokens
- **Price per Input Token**: Cost per single input token (for API comparison)
- **Input Tokens per dollar**: Efficiency metric

💡 Input processing is typically fast and cheap (parallel prefill), but APIs charge for it too!
""")

st.markdown("---")

# Display OUTPUT cost table
st.markdown("#### 📤 Output Token Costs (Generation) - **PRIMARY COST DRIVER!**")
st.table(output_cost_results)

st.caption("""
**Output cost metrics explained:**
- **Full $/Hour**: Base instance cost before any optimization
- **Discount %**: Savings from proportional GPU allocation (e.g., 60% GPU memory = 40% discount)
- **Actual $/Hour**: Real hourly cost after optimization (this is what you pay!)
- **Time for 1M Output Tokens**: How long 1 instance takes to generate 1M output tokens ⚡
- **$/1M Output Tokens**: Cost to generate 1M output tokens 💰 **KEY METRIC!**
- **Price per Output Token**: Cost per single output token (for API comparison)
- **Output Tokens per dollar**: Efficiency - how many output tokens for one dollar (higher is better)

💡 **Focus here**: Output generation is your bottleneck and main cost driver!
""")

st.markdown("---")

# ============= MONTHLY PROJECTIONS =============
st.subheader("📅 Monthly Cost Projections")
st.caption("💡 Based on your actual benchmark throughput (shown above) - no guessing needed!")

# Set defaults for projections (using 24/7 operation)
hours_per_month = 720  # 30 days × 24 hours
scale_multiplier = 1.0

# Calculate monthly tokens for first benchmark (for reference)
if benchmarks:
    first_bench = benchmarks[0]
    success_df = first_bench.df[first_bench.df["status_code"] == 200]
    total_output = success_df["output_tokens"].sum() if len(success_df) > 0 else 0
    test_duration = first_bench.metadata.duration_seconds
    actual_tok_per_sec = (total_output / test_duration) if test_duration > 0 else 0
    scaled_tok_per_sec = actual_tok_per_sec * scale_multiplier
    monthly_tokens = scaled_tok_per_sec * hours_per_month * 3600
    tokens_per_month = monthly_tokens
    tokens_per_second = scaled_tok_per_sec
else:
    tokens_per_month = 0
    tokens_per_second = 0

st.markdown("#### 💰 Capacity & Cost Projections")
st.caption("Single instance token capacity and costs at different time scales (24/7 operation)")

# Calculate capacity and costs at different time granularities
input_projection_data = []
output_projection_data = []

for benchmark in benchmarks:
    platform = benchmark.metadata.platform
    config = st.session_state["cost_config"][platform]
    
    # Calculate SUSTAINED throughput for both input and output
    success_df = benchmark.df[benchmark.df["status_code"] == 200]
    total_output_tokens = success_df["output_tokens"].sum() if len(success_df) > 0 else 0
    total_input_tokens = success_df["input_tokens"].sum() if len(success_df) > 0 else 0
    test_duration_sec = benchmark.metadata.duration_seconds
    
    output_sustained_throughput = total_output_tokens / test_duration_sec if test_duration_sec > 0 else 0
    input_sustained_throughput = total_input_tokens / test_duration_sec if test_duration_sec > 0 else 0
    
    # Get cost per hour (with any discounts applied)
    base_cost_per_hour = config["cost_per_hour"]
    yaml_config = None
    if "yaml_loaded_configs" in st.session_state:
        yaml_config = st.session_state["yaml_loaded_configs"].get(platform)
        if not yaml_config:
            for yaml_platform, yaml_cfg in st.session_state["yaml_loaded_configs"].items():
                if yaml_platform.lower() == platform.lower():
                    yaml_config = yaml_cfg
                    break
    
    # Calculate actual cost (with proportional allocation if applicable)
    instance_gpu_count = config.get("gpu_count", 1)
    gpu_type = config["gpu_type"]
    
    # Get GPU utilization if available
    gpu_util_deployed = 0
    if yaml_config and yaml_config.gpu_memory_utilization:
        gpu_util_deployed = yaml_config.gpu_memory_utilization
    
    if use_proportional and yaml_config and yaml_config.gpu_memory_utilization:
        cost_per_gpu = base_cost_per_hour / instance_gpu_count
        gpu_util = yaml_config.gpu_memory_utilization
        deployment_gpu_count = yaml_config.gpu_count
        replicas = yaml_config.replicas
        cost_per_hour = cost_per_gpu * gpu_util * deployment_gpu_count * replicas
    else:
        cost_per_hour = base_cost_per_hour
        if yaml_config:
            replicas = yaml_config.replicas
            cost_per_hour = base_cost_per_hour * replicas
    
    # INPUT projections
    if input_sustained_throughput > 0:
        input_tok_sec = input_sustained_throughput
        input_tok_min = input_sustained_throughput * 60
        input_tok_hour = input_sustained_throughput * 3600
        input_tok_day = input_sustained_throughput * 86400
        input_tok_month = input_sustained_throughput * hours_per_month * 3600
        
        # Costs (same for both input and output - same infrastructure)
        cost_per_second = cost_per_hour / 3600
        cost_per_minute = cost_per_hour / 60
        cost_per_day = cost_per_hour * 24
        cost_per_month = cost_per_hour * hours_per_month
        
        input_projection_data.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "GPU Util": f"{gpu_util_deployed:.0%}" if gpu_util_deployed > 0 else "-",
            "Input Tokens/Second": f"{input_tok_sec:.1f}",
            "Input Tokens/Minute": f"{input_tok_min:,.0f}",
            "Input Tokens/Hour": f"{input_tok_hour:,.0f}",
            "Input Tokens/Day (24hr)": f"{input_tok_day:,.0f}",
            "Input Tokens/Month (720hr)": f"{input_tok_month:,.0f}",
            "Cost/Second": f"${cost_per_second:.4f}",
            "Cost/Minute": f"${cost_per_minute:.3f}",
            "Cost/Hour": f"${cost_per_hour:.2f}",
            "Cost/Day (24hr)": f"${cost_per_day:.2f}",
            "Cost/Month (720hr)": f"${cost_per_month:,.0f}",
        })
    else:
        input_projection_data.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "GPU Util": f"{gpu_util_deployed:.0%}" if gpu_util_deployed > 0 else "-",
            "Input Tokens/Second": "0",
            "Input Tokens/Minute": "0",
            "Input Tokens/Hour": "0",
            "Input Tokens/Day (24hr)": "0",
            "Input Tokens/Month (720hr)": "0",
            "Cost/Second": "N/A",
            "Cost/Minute": "N/A",
            "Cost/Hour": f"${cost_per_hour:.2f}",
            "Cost/Day (24hr)": f"${cost_per_hour * 24:.2f}",
            "Cost/Month (720hr)": f"${cost_per_hour * hours_per_month:,.0f}",
        })
    
    # OUTPUT projections
    if output_sustained_throughput > 0:
        output_tok_sec = output_sustained_throughput
        output_tok_min = output_sustained_throughput * 60
        output_tok_hour = output_sustained_throughput * 3600
        output_tok_day = output_sustained_throughput * 86400
        output_tok_month = output_sustained_throughput * hours_per_month * 3600
        
        # Costs (same for both input and output - same infrastructure)
        cost_per_second = cost_per_hour / 3600
        cost_per_minute = cost_per_hour / 60
        cost_per_day = cost_per_hour * 24
        cost_per_month = cost_per_hour * hours_per_month
        
        output_projection_data.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "GPU Util": f"{gpu_util_deployed:.0%}" if gpu_util_deployed > 0 else "-",
            "Output Tokens/Second": f"{output_tok_sec:.1f}",
            "Output Tokens/Minute": f"{output_tok_min:,.0f}",
            "Output Tokens/Hour": f"{output_tok_hour:,.0f}",
            "Output Tokens/Day (24hr)": f"{output_tok_day:,.0f}",
            "Output Tokens/Month (720hr)": f"{output_tok_month:,.0f}",
            "Cost/Second": f"${cost_per_second:.4f}",
            "Cost/Minute": f"${cost_per_minute:.3f}",
            "Cost/Hour": f"${cost_per_hour:.2f}",
            "Cost/Day (24hr)": f"${cost_per_day:.2f}",
            "Cost/Month (720hr)": f"${cost_per_month:,.0f}",
        })
    else:
        output_projection_data.append({
            "Platform": platform,
            "Instance": config["instance_name"],
            "GPUs": f"{instance_gpu_count}x {gpu_type}",
            "GPU Util": f"{gpu_util_deployed:.0%}" if gpu_util_deployed > 0 else "-",
            "Output Tokens/Second": "0",
            "Output Tokens/Minute": "0",
            "Output Tokens/Hour": "0",
            "Output Tokens/Day (24hr)": "0",
            "Output Tokens/Month (720hr)": "0",
            "Cost/Second": "N/A",
            "Cost/Minute": "N/A",
            "Cost/Hour": f"${cost_per_hour:.2f}",
            "Cost/Day (24hr)": f"${cost_per_hour * 24:.2f}",
            "Cost/Month (720hr)": f"${cost_per_hour * hours_per_month:,.0f}",
        })

# Display INPUT projections
st.markdown("##### 📥 Input Token Capacity & Costs (Prompt Processing)")
input_proj_df = pd.DataFrame(input_projection_data)
st.dataframe(input_proj_df, use_container_width=True, hide_index=True)

st.caption("""
**Input token projections:**
- Shows how many input tokens (prompts) 1 instance can process
- Input processing is typically very fast (parallel prefill)
- Same infrastructure cost applies to both input and output
""")

st.markdown("---")

# Display OUTPUT projections
st.markdown("##### 📤 Output Token Capacity & Costs (Generation) - **KEY FOR CAPACITY!**")
output_proj_df = pd.DataFrame(output_projection_data)
st.dataframe(output_proj_df, use_container_width=True, hide_index=True)

st.caption(f"""
**Output token projections - USE THIS FOR CAPACITY PLANNING:**
- Shows how many **output** tokens 1 instance can generate over different time periods
- **GPU Util**: GPU memory utilization from deployment config (affects cost discount)
- **Month = 720 hours**: Assumes 24/7 operation for 30 days

💡 **Quick scaling math:**
- See "Output Tokens/Month (720hr)" for 1 instance capacity
- Divide your target monthly **output** volume by this number
- That's how many instances you need!

**Example**: Need 100M output tokens/month? Platform does 50M/month → Need 2 instances

*Note: In production, add 20-30% buffer for traffic spikes and high availability.*
""")

# Show detailed calculation breakdown
with st.expander("🔢 Show Scaling Calculation for Each Platform"):
    st.markdown(f"**📊 Target Load:** {tokens_per_month:,} tokens/month = {tokens_per_second:.1f} tok/s")
    st.markdown(f"**⚠️ Reminder:** Benchmark = 1 instance performance. We calculate how many instances to scale up.")
    st.markdown("---")
    
    for benchmark in benchmarks:
        platform = benchmark.metadata.platform
        
        # Calculate sustained throughput (same as first table)
        success_df = benchmark.df[benchmark.df["status_code"] == 200]
        total_output_tokens = success_df["output_tokens"].sum() if len(success_df) > 0 else 0
        test_duration_sec = benchmark.metadata.duration_seconds
        sustained_throughput = total_output_tokens / test_duration_sec if test_duration_sec > 0 else 0
        
        if sustained_throughput > 0:
            instances_needed = max(1, int(np.ceil(tokens_per_second / sustained_throughput)))
            
            st.markdown(f"### {platform}")
            st.code(f"""
🎯 YOUR TARGET MONTHLY LOAD:
    Monthly Token Volume        = {tokens_per_month:,} tokens
    ÷ Hours per Month           = {hours_per_month} hours × 3,600 seconds
    ─────────────────────────────────────────────────────
    = Throughput Needed         = {tokens_per_second:.2f} tok/s
    
📊 SINGLE INSTANCE CAPACITY (from your benchmark):
    1 Instance Throughput       = {sustained_throughput:.2f} tok/s
    (Total Tokens Processed ÷ Test Duration)
    
⚙️ CALCULATE HOW MANY INSTANCES TO DEPLOY:
    Throughput Needed           = {tokens_per_second:.2f} tok/s
    ÷ Single Instance Capacity  = {sustained_throughput:.2f} tok/s
    ─────────────────────────────────────────────────────
    = Raw Calculation           = {tokens_per_second / sustained_throughput:.4f}
    = Rounded Up (⌈⌉)           = {instances_needed} instance(s)
    
💰 TOTAL MONTHLY COST:
    {instances_needed} instance(s) × ${config["cost_per_hour"]:.2f}/hr × {hours_per_month} hrs
    = ${instances_needed * config["cost_per_hour"] * hours_per_month:,.0f}/month

✅ RESULT: Deploy {instances_needed} instance(s) to handle {tokens_per_month:,} tokens/month
""", language="text")
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
        # Calculate sustained throughput
        success_df_temp = benchmark.df[benchmark.df["status_code"] == 200]
        total_output_temp = success_df_temp["output_tokens"].sum() if len(success_df_temp) > 0 else 0
        sustained_throughput = total_output_temp / benchmark.metadata.duration_seconds if benchmark.metadata.duration_seconds > 0 else 0
        
        if sustained_throughput > 0:
            config = st.session_state["cost_config"][benchmark.metadata.platform]
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = sustained_throughput * 3600
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
        
        - **\\${best_cost:.2f}** per 1M tokens
        - **{savings_pct:.1f}% cheaper** than {worst_platform}
        - **Saves \\${savings:.2f}** per 1M tokens
        """)
        
        # Calculate monthly savings at target QPS
        success_df_best = best_benchmark.df[best_benchmark.df["status_code"] == 200]
        success_df_worst = worst_benchmark.df[worst_benchmark.df["status_code"] == 200]
        
        if len(success_df_best) > 0 and len(success_df_worst) > 0:
            # Calculate savings at current token volume
            # Using 10M tokens/month as default scenario
            scenario_tokens = 10_000_000
            monthly_savings = (scenario_tokens / 1_000_000) * savings
            
            if monthly_savings > 100:
                st.info(f"""
                💰 **Example Savings:** At 10M tokens/month, using {best_platform} instead of {worst_platform} 
                saves approximately **\\${monthly_savings:,.0f}/month** (\\${monthly_savings * 12:,.0f}/year)
                """)
    
    # Cost vs Performance trade-off
    st.markdown("**Cost vs Performance Trade-offs:**")
    
    for benchmark in benchmarks:
        # Calculate sustained throughput
        success_df_temp = benchmark.df[benchmark.df["status_code"] == 200]
        total_output_temp = success_df_temp["output_tokens"].sum() if len(success_df_temp) > 0 else 0
        sustained_throughput = total_output_temp / benchmark.metadata.duration_seconds if benchmark.metadata.duration_seconds > 0 else 0
        
        if sustained_throughput > 0:
            config = st.session_state["cost_config"][benchmark.metadata.platform]
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = sustained_throughput * 3600
            cost_per_million = (cost_per_hour / tokens_per_hour) * 1_000_000
            
            # Efficiency score: throughput per dollar per hour
            efficiency = sustained_throughput / cost_per_hour if cost_per_hour > 0 else 0
            
            st.markdown(f"- **{benchmark.metadata.platform}**: ${cost_per_million:.2f}/1M tok | "
                       f"TTFT P50: {benchmark.ttft_p50:.0f}ms | "
                       f"Efficiency: {efficiency:.1f} tok/s per $/hr")

else:
    st.info("Upload at least 2 benchmarks to see cost comparisons")
    
    if len(benchmarks) == 1:
        benchmark = benchmarks[0]
        platform = benchmark.metadata.platform
        config = st.session_state["cost_config"][platform]
        
        # Calculate sustained throughput
        success_df_temp = benchmark.df[benchmark.df["status_code"] == 200]
        total_output_temp = success_df_temp["output_tokens"].sum() if len(success_df_temp) > 0 else 0
        sustained_throughput = total_output_temp / benchmark.metadata.duration_seconds if benchmark.metadata.duration_seconds > 0 else 0
        
        if sustained_throughput > 0:
            cost_per_hour = config["cost_per_hour"]
            tokens_per_hour = sustained_throughput * 3600
            cost_per_million = (cost_per_hour / tokens_per_hour) * 1_000_000
            
            st.metric(
                "Cost per 1M Tokens",
                f"${cost_per_million:.2f}",
                delta=f"{config['gpu_type']} on {config['cloud_provider']}",
            )

st.markdown("---")

# ============= API PROVIDER COMPARISON =============
st.subheader("☁️ Compare with API Providers")
st.caption("See how self-hosted costs compare to managed API services")

# Load API pricing
api_pricing = load_api_pricing()

if not api_pricing:
    st.warning("⚠️ Could not load API pricing data. Make sure `model_prices.json` exists.")
else:
    st.success(f"✅ Loaded pricing for {len(api_pricing):,} API models")
    
    # Popular models quick selection
    st.markdown("#### Select API Models for Comparison")
    
    # Filter controls
    st.markdown("**🔍 Filter Models:**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Provider filter
        available_providers = get_unique_providers(api_pricing)
        selected_providers = st.multiselect(
            "By Provider",
            options=available_providers,
            default=[],
            help="Filter by API provider (OpenAI, Anthropic, etc.)",
            key="provider_filter",
        )
    
    with filter_col2:
        # Size filter
        available_sizes = get_unique_sizes(api_pricing)
        selected_sizes = st.multiselect(
            "By Size",
            options=available_sizes,
            default=[],
            help="Filter by model size (7B, 13B, 70B, mini, opus, etc.)",
            key="size_filter",
        )
    
    with filter_col3:
        # Search box
        search_term = st.text_input(
            "Search",
            placeholder="e.g., llama, gpt, claude",
            help="Search model names",
            key="search_filter",
        )
    
    # Apply filters
    if selected_providers or selected_sizes or search_term:
        filtered_pricing = filter_models(
            api_pricing,
            providers=selected_providers if selected_providers else None,
            sizes=selected_sizes if selected_sizes else None,
            search_term=search_term if search_term else None,
        )
        st.caption(f"✅ Showing {len(filtered_pricing)} models (filtered from {len(api_pricing):,})")
    else:
        filtered_pricing = api_pricing
        st.caption(f"Showing all {len(api_pricing):,} models")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get popular models for defaults
        popular = get_popular_models()
        available_popular = [m for m in popular if m in filtered_pricing]
        
        # Option to show all models or just popular ones
        show_all_in_dropdown = st.checkbox(
            f"Show all {len(filtered_pricing):,} filtered models in dropdown",
            value=len(filtered_pricing) <= 100,  # Auto-enable if filtered to <= 100 models
            help="By default, dropdown shows popular models. Check this to see all filtered models.",
        )
        
        if show_all_in_dropdown:
            # Show ALL filtered models (sorted alphabetically)
            all_models = sorted(filtered_pricing.keys())
            # Default to first 3 models if no popular ones available
            default_selection = available_popular[:3] if available_popular else all_models[:3]
        else:
            # Show only popular models from filtered set
            all_models = available_popular if available_popular else sorted(filtered_pricing.keys())[:25]
            default_selection = available_popular[:3] if available_popular else all_models[:3]
        
        # Multi-select for API models
        selected_api_models = st.multiselect(
            "Choose API models to compare",
            options=all_models,
            default=default_selection if default_selection else [],
            help="Search by typing model name. Select 1-5 API models to compare with your self-hosted platforms.",
            key="model_selection",
        )
    
    with col2:
        # Show breakdown by provider
        if st.button("📋 Browse by Provider", help="View all models organized by provider"):
            st.session_state["show_provider_browser"] = not st.session_state.get("show_provider_browser", False)
        
        if st.session_state.get("show_provider_browser", False):
            providers = get_models_by_provider(api_pricing)
            
            st.markdown("---")
            st.markdown(f"### All Available Models ({len(api_pricing):,})")
            
            # Provider filter
            selected_providers = st.multiselect(
                "Filter by provider",
                options=sorted(providers.keys()),
                default=[],
                help="Leave empty to see all providers"
            )
            
            display_providers = selected_providers if selected_providers else sorted(providers.keys())
            
            for provider in display_providers:
                models = providers[provider]
                with st.expander(f"**{provider.upper()}** ({len(models)} models)", expanded=False):
                    # Show first 20 models with pricing
                    for model in sorted(models)[:20]:
                        pricing = api_pricing[model]
                        st.caption(
                            f"• `{model}`: "
                            f"${pricing.input_cost_per_token*1e6:.2f}/1M in, "
                            f"${pricing.output_cost_per_token*1e6:.2f}/1M out"
                        )
                    if len(models) > 20:
                        st.caption(f"... and {len(models)-20} more (search in dropdown above)")
                st.markdown("---")
    
    if selected_api_models:
        st.markdown("---")
        
        # Simple pricing comparison - reuse costs from Cost Efficiency Comparison above
        st.markdown("#### 💰 Pricing Comparison: Self-Hosted vs API")
        
        # Build INPUT comparison table
        input_comparison = []
        
        # Add self-hosted (from input_cost_results already calculated)
        for item in input_cost_results:
            input_comparison.append({
                "Platform": f"🔧 {item['Platform']}",
                "Type": "Self-Hosted",
                "Provider": item.get("Provider", "N/A") if "Provider" in input_cost_results[0] else "N/A",
                "Input $/1M": item["$/1M Input Tokens"],
                "Input_Cost_Value": float(item["$/1M Input Tokens"].replace("$", "").replace(",", "")) if item["$/1M Input Tokens"] != "N/A" else 999999,
            })
        
        # Add APIs
        for model_name in selected_api_models:
            if model_name in api_pricing:
                pricing = api_pricing[model_name]
                cost_per_1m_input = pricing.input_cost_per_token * 1_000_000
                
                input_comparison.append({
                    "Platform": f"☁️ {model_name}",
                    "Type": "API",
                    "Provider": pricing.provider.upper(),
                    "Input $/1M": f"${cost_per_1m_input:.2f}",
                    "Input_Cost_Value": cost_per_1m_input,
                })
        
        # Display INPUT comparison
        st.markdown("##### 📥 Input Token Pricing")
        input_df = pd.DataFrame(input_comparison)
        input_display = input_df.drop(columns=["Input_Cost_Value"]).sort_values("Input $/1M")
        st.dataframe(input_display, use_container_width=True, hide_index=True)
        
        st.caption("""
        **Input token pricing** (prompts sent to the model):
        - Self-hosted: Infrastructure cost ÷ input throughput
        - API: Per-token charge for processing prompts
        """)
        
        st.markdown("---")
        
        # Build OUTPUT comparison table
        output_comparison = []
        
        # Add self-hosted (from output_cost_results already calculated)
        for item in output_cost_results:
            output_comparison.append({
                "Platform": f"🔧 {item['Platform']}",
                "Type": "Self-Hosted",
                "Provider": item.get("Provider", "N/A") if "Provider" in output_cost_results[0] else "N/A",
                "Output $/1M": item["$/1M Output Tokens"],
                "Output_Cost_Value": float(item["$/1M Output Tokens"].replace("$", "").replace(",", "")) if item["$/1M Output Tokens"] != "N/A" else 999999,
            })
        
        # Add APIs
        for model_name in selected_api_models:
            if model_name in api_pricing:
                pricing = api_pricing[model_name]
                cost_per_1m_output = pricing.output_cost_per_token * 1_000_000
                
                output_comparison.append({
                    "Platform": f"☁️ {model_name}",
                    "Type": "API",
                    "Provider": pricing.provider.upper(),
                    "Output $/1M": f"${cost_per_1m_output:.2f}",
                    "Output_Cost_Value": cost_per_1m_output,
                })
        
        # Display OUTPUT comparison
        st.markdown("##### 📤 Output Token Pricing - **PRIMARY COST DRIVER!**")
        output_df = pd.DataFrame(output_comparison)
        output_display = output_df.drop(columns=["Output_Cost_Value"]).sort_values("Output $/1M")
        st.dataframe(output_display, use_container_width=True, hide_index=True)
        
        st.caption("""
        **Output token pricing** (responses generated by the model) - **KEY METRIC:**
        - Self-hosted: Infrastructure cost ÷ output throughput (your bottleneck!)
        - API: Per-token charge for generating responses
        
        💡 **Focus here**: Output generation is the capacity bottleneck and main cost driver!
        🎯 **Simple math**: Need 10M output tokens? Multiply Output $/1M by 10
        """)
        
        # Store for break-even analysis
        comparison_data = output_comparison  # Use output data for break-even
        
        st.markdown("---")
        
        # Key insights (based on OUTPUT tokens - the bottleneck)
        st.markdown("#### 🎯 Cost Comparison Insights")
        
        # Find cheapest by output cost (the bottleneck)
        sorted_by_cost = sorted(comparison_data, key=lambda x: x["Output_Cost_Value"])
        cheapest = sorted_by_cost[0]
        most_expensive = sorted_by_cost[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **💰 Lowest Output Token Cost**
            
            {cheapest['Platform']}
            - Type: {cheapest['Type']}
            - Output: {cheapest['Output $/1M']} per 1M tokens
            """)
        
        with col2:
            st.warning(f"""
            **💸 Highest Output Token Cost**
            
            {most_expensive['Platform']}
            - Type: {most_expensive['Type']}
            - Output: {most_expensive['Output $/1M']} per 1M tokens
            """)
        
        # Cost savings analysis
        st.markdown("---")
        st.markdown("#### 💡 Self-Hosted vs API Trade-offs")
        st.caption("Select platforms to compare and see break-even analysis")
        
        # Let user select which platforms to compare
        col1, col2 = st.columns(2)
        
        with col1:
            # Get self-hosted platforms
            self_hosted_platforms = [x for x in comparison_data if x["Type"] == "Self-Hosted"]
            self_hosted_names = [x["Platform"] for x in self_hosted_platforms]
            
            if self_hosted_names:
                # Default to cheapest
                cheapest_self_idx = min(range(len(self_hosted_platforms)), 
                                       key=lambda i: self_hosted_platforms[i]["Output_Cost_Value"])
                
                selected_self_hosted = st.selectbox(
                    "Select Self-Hosted Platform",
                    options=self_hosted_names,
                    index=cheapest_self_idx,
                    key="compare_self_hosted",
                )
        
        with col2:
            # Get API platforms
            api_platforms = [x for x in comparison_data if x["Type"] == "API"]
            api_names = [x["Platform"] for x in api_platforms]
            
            if api_names:
                # Default to cheapest
                cheapest_api_idx = min(range(len(api_platforms)), 
                                      key=lambda i: api_platforms[i]["Output_Cost_Value"])
                
                selected_api = st.selectbox(
                    "Select API Model",
                    options=api_names,
                    index=cheapest_api_idx,
                    key="compare_api",
                )
        
        st.markdown("---")
        
        # Get selected platforms
        self_hosted_costs = [x["Output_Cost_Value"] for x in comparison_data if x["Type"] == "Self-Hosted"]
        api_costs = [x["Output_Cost_Value"] for x in comparison_data if x["Type"] == "API"]
        
        if self_hosted_costs and api_costs and self_hosted_names and api_names:
            # Get the selected platform data
            cheapest_self = next(x for x in comparison_data if x["Platform"] == selected_self_hosted)
            cheapest_api = next(x for x in comparison_data if x["Platform"] == selected_api)
            avg_self_hosted = sum(self_hosted_costs) / len(self_hosted_costs)
            avg_api = sum(api_costs) / len(api_costs)
            
            # Compare selected platforms
            self_cost = cheapest_self["Output_Cost_Value"]
            api_cost = cheapest_api["Output_Cost_Value"]
            
            if self_cost < api_cost:
                savings_pct = ((api_cost - self_cost) / api_cost) * 100
                st.success(f"""
                ✅ **{selected_self_hosted} is {savings_pct:.1f}% cheaper** than {selected_api}
                
                - Self-hosted: {cheapest_self['Output $/1M']}
                - API: {cheapest_api['Output $/1M']}
                
                **When to self-host:**
                - ✅ High volume workloads (>1M tokens/day)
                - ✅ Need for low latency (APIs have network overhead)
                - ✅ Data privacy and compliance requirements
                - ✅ Model customization (fine-tuning, quantization)
                - ✅ Predictable costs
                """)
            else:
                premium_pct = ((self_cost - api_cost) / api_cost) * 100
                st.warning(f"""
                ⚠️ **{selected_api} is {premium_pct:.1f}% cheaper** than {selected_self_hosted}
                
                - Self-hosted: {cheapest_self['Output $/1M']}
                - API: {cheapest_api['Output $/1M']}
                
                **When to use APIs:**
                - ✅ Low volume workloads (<100K tokens/day)
                - ✅ Variable/unpredictable load
                - ✅ No infrastructure management
                - ✅ Rapid prototyping and experimentation
                - ✅ Pay-per-use flexibility
                """)
            
            st.markdown("---")
            
            # Break-even analysis (ECONOMY OF SCALE)
            st.markdown("#### 📊 Economy of Scale Analysis")
            st.caption(f"Comparing: {selected_self_hosted} vs {selected_api}")
            
            # Use the selected platforms (already set above)
            
            # Calculate costs for different TOKEN quantities (pure token count!)
            token_levels = [
                100_000,      # 100K tokens
                1_000_000,    # 1M tokens
                10_000_000,   # 10M tokens
                50_000_000,   # 50M tokens
                100_000_000,  # 100M tokens
                500_000_000,  # 500M tokens
                1_000_000_000 # 1B tokens
            ]
            
            # Get self-hosted throughput and cost
            self_platform_name = cheapest_self["Platform"].replace("🔧 ", "")
            self_config = st.session_state["cost_config"][self_platform_name]
            
            # Find the benchmark for this platform
            self_benchmark = next((b for b in benchmarks if b.metadata.platform == self_platform_name), None)
            if self_benchmark:
                success_df_self = self_benchmark.df[self_benchmark.df["status_code"] == 200]
                total_output_self = success_df_self["output_tokens"].sum() if len(success_df_self) > 0 else 0
                self_throughput = total_output_self / self_benchmark.metadata.duration_seconds if self_benchmark.metadata.duration_seconds > 0 else 1
                
                # Get actual cost per hour (with discounts)
                yaml_config_self = None
                if "yaml_loaded_configs" in st.session_state:
                    yaml_config_self = st.session_state["yaml_loaded_configs"].get(self_platform_name)
                
                base_cost = self_config["cost_per_hour"]
                if use_proportional and yaml_config_self and yaml_config_self.gpu_memory_utilization:
                    instance_gpu_count = self_config.get("gpu_count", 1)
                    cost_per_gpu = base_cost / instance_gpu_count
                    gpu_util = yaml_config_self.gpu_memory_utilization
                    deployment_gpu_count = yaml_config_self.gpu_count
                    replicas = yaml_config_self.replicas
                    self_cost_per_hour = cost_per_gpu * gpu_util * deployment_gpu_count * replicas
                else:
                    self_cost_per_hour = base_cost
            else:
                self_throughput = 1
                self_cost_per_hour = self_config["cost_per_hour"]
            
            # Get API pricing
            api_pricing_obj = api_pricing[cheapest_api["Platform"].replace("☁️ ", "")]
            
            breakeven_data = []
            for tokens in token_levels:
                # Self-hosted cost = time to process tokens × cost per hour
                # Time = tokens / throughput (in seconds)
                # Cost = (time in hours) × cost per hour
                time_to_process_hours = (tokens / self_throughput) / 3600
                self_cost = time_to_process_hours * self_cost_per_hour
                
                # API cost = tokens × cost per token (simple!)
                api_cost = tokens * api_pricing_obj.output_cost_per_token
                
                # Format token levels nicely
                if tokens >= 1_000_000_000:
                    token_label = f"{tokens/1_000_000_000:.1f}B"
                elif tokens >= 1_000_000:
                    token_label = f"{tokens/1_000_000:.0f}M"
                else:
                    token_label = f"{tokens/1_000:.0f}K"
                
                breakeven_data.append({
                    "Output Token Quantity": token_label,
                    "Tokens (numeric)": tokens,  # Keep for chart
                    "Self-Hosted Cost": f"${self_cost:.2f}",
                    "API Cost": f"${api_cost:.2f}",
                    "Winner": "🏆 Self-Hosted" if self_cost < api_cost else "🏆 API",
                    "Savings": f"${abs(self_cost - api_cost):.2f}",
                })
            
            st.table([{k: v for k, v in item.items() if k != "Tokens (numeric)"} for item in breakeven_data])
            
            st.caption(f"""
            **Comparing cheapest options (OUTPUT tokens):**
            - 🔧 Self-Hosted: {cheapest_self['Platform'].replace('🔧 ', '')} @ {cheapest_self['Output $/1M']} - cost = time to process × infrastructure cost
            - ☁️ API: {cheapest_api['Platform'].replace('☁️ ', '')} @ {cheapest_api['Output $/1M']} - cost = tokens × per-token price
            
            💡 **How costs are calculated**: 
            - **Self-hosted**: Cost to process X tokens = (X ÷ throughput) × hourly infrastructure cost
            - **API**: Cost to process X tokens = X × price per token
            - **Break-even**: Where both methods cost the same for the same quantity
            """)
            
            # Add ECONOMY OF SCALE chart
            st.markdown("---")
            st.markdown("#### 📈 Economy of Scale Visualization")
            st.caption("See where the lines cross! 🎯 That's your break-even point.")
            
            # Extract cost values for chart
            token_volumes = [item["Tokens (numeric)"] for item in breakeven_data]
            self_costs = []
            api_costs = []
            for item in breakeven_data:
                # Parse costs from string format "$1,234.56" to float
                self_cost_str = item["Self-Hosted Cost"].replace("$", "").replace(",", "")
                api_cost_str = item["API Cost"].replace("$", "").replace(",", "")
                self_costs.append(float(self_cost_str))
                api_costs.append(float(api_cost_str))
            
            # Create TOTAL COST chart - shows the crossing point!
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Self-hosted line (relatively flat - fixed infrastructure cost)
            fig.add_trace(
                go.Scatter(
                    x=token_volumes,  # Token volume on X-axis
                    y=self_costs,  # Total cost on Y-axis
                    mode="lines+markers",
                    name=f"🔧 {cheapest_self['Platform'].replace('🔧 ', '')} (infrastructure)",
                    line=dict(color="#8B5CF6", width=5),
                    marker=dict(size=14),
                    hovertemplate="<b>Volume:</b> %{x:,.0f} tokens<br><b>Total Cost:</b> $%{y:.2f}<extra></extra>",
                )
            )
            
            # API line (rises with volume - pay per token)
            fig.add_trace(
                go.Scatter(
                    x=token_volumes,  # Token volume on X-axis
                    y=api_costs,  # Total cost on Y-axis
                    mode="lines+markers",
                    name=f"☁️ {cheapest_api['Platform'].replace('☁️ ', '')} (pay-per-use)",
                    line=dict(color="#3B82F6", width=5),
                    marker=dict(size=14),
                    hovertemplate="<b>Volume:</b> %{x:,.0f} tokens<br><b>Total Cost:</b> $%{y:.2f}<extra></extra>",
                )
            )
            
            # Find intersection (break-even point) with linear interpolation
            eq_found = False
            for i in range(len(token_volumes) - 1):
                if ((self_costs[i] >= api_costs[i] and self_costs[i+1] < api_costs[i+1]) or
                    (self_costs[i] <= api_costs[i] and self_costs[i+1] > api_costs[i+1])):
                    # Linear interpolation to find exact crossing point
                    x1, x2 = token_volumes[i], token_volumes[i+1]
                    y1_self, y2_self = self_costs[i], self_costs[i+1]
                    y1_api, y2_api = api_costs[i], api_costs[i+1]
                    
                    # Find t where lines intersect (linear interpolation)
                    if (y2_self - y1_self) != (y2_api - y1_api):  # Not parallel
                        t = (y1_api - y1_self) / ((y2_self - y1_self) - (y2_api - y1_api))
                        eq_tokens = x1 + t * (x2 - x1)
                        eq_cost = y1_self + t * (y2_self - y1_self)
                    else:
                        eq_tokens = (x1 + x2) / 2
                        eq_cost = (y1_self + y1_api) / 2
                    
                    eq_found = True
                    break
            
            # If no crossing found, use point where they're closest
            if not eq_found:
                min_diff_idx = min(range(len(token_volumes)), key=lambda i: abs(self_costs[i] - api_costs[i]))
                eq_tokens = token_volumes[min_diff_idx]
                eq_cost = (self_costs[min_diff_idx] + api_costs[min_diff_idx]) / 2
                eq_found = True
            
            if eq_found:
                
                # Format volume nicely
                if eq_tokens >= 1_000_000_000:
                    vol_label = f"{eq_tokens/1_000_000_000:.1f}B"
                elif eq_tokens >= 1_000_000:
                    vol_label = f"{eq_tokens/1_000_000:.0f}M"
                else:
                    vol_label = f"{eq_tokens:,.0f}"
                
                # Add equilibrium point (HUGE purple dot!)
                fig.add_trace(
                    go.Scatter(
                        x=[eq_tokens],
                        y=[eq_cost],
                        mode="markers+text",
                        marker=dict(size=35, color="#9333EA", symbol="circle", line=dict(color="black", width=4)),
                        text=["⚖️"],
                        textposition="middle center",
                        textfont=dict(size=20, color="white"),
                        name="🎯 Break-Even",
                        hovertemplate=f"<b>🎯 BREAK-EVEN POINT</b><br>Volume: {vol_label}<br>Cost: ${eq_cost:.2f}<extra></extra>",
                        showlegend=True,
                    )
                )
                
                # BOLD dashed lines to both axes
                fig.add_shape(type="line", x0=min(token_volumes), x1=eq_tokens, y0=eq_cost, y1=eq_cost,
                             line=dict(color="black", width=3, dash="dash"), layer="above")
                fig.add_shape(type="line", x0=eq_tokens, x1=eq_tokens, y0=0, y1=eq_cost,
                             line=dict(color="black", width=3, dash="dash"), layer="above")
                
                # Axis labels with boxes
                fig.add_annotation(x=min(token_volumes), y=eq_cost, text=f" ${eq_cost:.0f} ",
                                  showarrow=False, xanchor="right",
                                  font=dict(size=16, color="black", family="Arial Black"),
                                  bgcolor="yellow", bordercolor="black", borderwidth=2, borderpad=4)
                
                fig.add_annotation(x=eq_tokens, y=0, text=f" {vol_label} ",
                                  showarrow=False, yanchor="top",
                                  font=dict(size=16, color="black", family="Arial Black"),
                                  bgcolor="yellow", bordercolor="black", borderwidth=2, borderpad=4)
                
                # BIG Equilibrium annotation above the point
                fig.add_annotation(x=eq_tokens, y=eq_cost, text=f"  BREAK-EVEN\n  {vol_label} @ ${eq_cost:.0f}",
                                  showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, 
                                  ax=100, ay=-70,
                                  font=dict(size=18, color="black", family="Arial Black"),
                                  bgcolor="#FFD700", bordercolor="black", borderwidth=3, borderpad=6)
                
                # Add shaded regions with text
                fig.add_vrect(x0=min(token_volumes), x1=eq_tokens,
                             fillcolor="blue", opacity=0.12, layer="below", line_width=0,
                             annotation_text="API Cheaper", annotation_position="top left",
                             annotation=dict(font=dict(size=14, color="blue")))
                fig.add_vrect(x0=eq_tokens, x1=max(token_volumes),
                             fillcolor="purple", opacity=0.12, layer="below", line_width=0,
                             annotation_text="Self-Hosted Cheaper", annotation_position="top right",
                             annotation=dict(font=dict(size=14, color="purple")))
            
            fig.update_layout(
                title=dict(text="💰 Economy of Scale: Where Do the Lines Cross?",
                          font=dict(size=22, color="black", family="Arial Black")),
                xaxis_title="Output Token Volume",
                yaxis_title="Total Cost to Process ($)",
                height=700,  # Bigger!
                hovermode="closest",
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                           bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=2),
                xaxis_type="log",  # Log scale for volume
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="lightgray", showline=True, 
                          linewidth=2, linecolor="black", mirror=True,
                          tickformat=".2s", ticksuffix=" tokens"),
                yaxis=dict(showgrid=True, gridcolor="lightgray", showline=True,
                          linewidth=2, linecolor="black", mirror=True),
            )
            
            st.plotly_chart(fig, use_container_width=True, key="economy_of_scale_chart")
            
            # Make it super clear what this means
            st.success("""
            ### 🎯 Classic Supply & Demand Equilibrium Chart
            
            **The purple dot** shows where the lines cross - the **equilibrium point**.
            
            **How to read:**
            - **X-axis**: How many output tokens you need to process
            - **Y-axis**: Total cost to process those tokens
            - **Purple line** (self-hosted): FLAT! Fixed infrastructure cost (doesn't matter how many tokens)
            - **Blue line** (API): RISING! Cost increases linearly with token volume
            
            **Key insights:**
            - **Left of equilibrium** (low volume): Blue line is below purple → API is cheaper
            - **Right of equilibrium** (high volume): Purple line is below blue → Self-hosting is cheaper
            - **The crossing point** is your break-even volume!
            
            This is **economy of scale** in action! 🚀
            """)
            
            st.caption("""
            **Why they cross:**
            - 🟣 **Self-hosted (flat)**: Pay fixed infrastructure cost regardless of volume
              - Low volume: Expensive per token (infrastructure underutilized)
              - High volume: Cheap per token (cost spread over many tokens)
            
            - 🔵 **API (rising)**: Pay per token you use
              - Low volume: Cheap (only pay for what you use)
              - High volume: Expensive (costs add up linearly)
            
            💡 **Below the equilibrium**: APIs win. **Above the equilibrium**: Self-hosting wins!
            """)

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
    
    **Step 3: Smart Monthly Projections (NEW! 🎯)**
    - We automatically calculate throughput from your benchmark data:
        - Total tokens processed during test
        - Test duration (first to last timestamp)
        - Actual tokens/second = Total Tokens ÷ Duration
    - Set operating hours per month (default: 24/7 = 720 hours)
    - Use **Traffic Scale Multiplier** to model growth:
        - 1.0x = Same load as your test
        - 2.0x = Double the traffic
        - 0.5x = Half the traffic
    - See projected monthly costs at that scale
    
    **Step 4: Compare with API providers**
    - Filter by provider, size, or search
    - Select API models (OpenAI, Anthropic, Google, etc.) to compare
    - See cost per 1M tokens for both self-hosted and API options
    - View **Economy of Scale chart** showing where lines cross
    - Understand break-even point and when to self-host vs use APIs
    
    **Step 5: Economy of Scale Analysis**
    - See beautiful chart showing cost curves
    - Find the equilibrium point (where costs are equal)
    - Understand trade-offs at different volumes
    
    **Notes:**
    - All projections based on REAL measured performance from your benchmark
    - Pricing is approximate and based on public cloud pricing as of Oct 2025
    - On-prem costs are amortized estimates (hardware + power + maintenance)
    - Actual costs may vary based on discounts, region, utilization, reserved instances, etc.
    - Multi-GPU instances are supported (pricing is per instance, not per GPU)
    - API pricing based on pay-per-token model (no infrastructure costs)
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

