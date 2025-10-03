# LLM Benchmark Comparison Dashboard

A Streamlit-based dashboard for comparing LLM inference benchmark results across multiple serving platforms.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Upload Benchmark Data

1. Run benchmarks using llm-locust:
   ```bash
   python examples/benchmark_chat_simulation.py \
       --host https://your-endpoint.com \
       --model your-model \
       --engine vllm
   ```

2. Upload the generated CSV files from `results/` directory

3. Explore the analysis pages!

## 📁 Project Structure

```
streamlit_app/
├── app.py                  # Main entry point
├── config.py              # Configuration and constants
├── requirements.txt       # Dependencies
├── README.md             # This file
│
├── pages/                # Multi-page sections
│   ├── 1_📊_Overview.py
│   ├── 2_⚡_Latency_Analysis.py
│   ├── 3_🚀_Throughput_Analysis.py
│   └── ...
│
├── lib/                  # Business logic
│   ├── data_loader.py    # CSV loading and validation
│   ├── metrics.py        # Metric calculations
│   ├── statistics.py     # Statistical tests
│   └── visualizations.py # Chart generation
│
├── models/               # Data models
│   └── benchmark.py      # BenchmarkData, Metadata, etc.
│
└── utils/               # Utilities
    ├── validators.py    # Data validation
    └── formatters.py    # Display formatting
```

## 🎯 Features

### Current (MVP)

- ✅ **File Upload**: Drag-and-drop CSV files
- ✅ **Data Validation**: Schema checking and quality scoring
- ✅ **Overview Dashboard**: Quick comparison table and winner metrics
- ✅ **Metadata Extraction**: Auto-detect platform from filenames

### Coming Soon

- 🚧 **Latency Analysis**: Distributions, percentiles, statistical tests
- 🚧 **Throughput Analysis**: Time series, stability metrics
- 🚧 **Error Analysis**: Failure patterns and reliability scoring
- 🚧 **Cost Analysis**: TCO comparison (optional)
- 🚧 **Report Generation**: Export PDF summaries

## 📊 Expected CSV Format

The dashboard expects CSV files with the following columns:

**Required:**
- `request_id`, `timestamp`, `user_id`, `user_request_num`
- `input_tokens`, `output_tokens`
- `ttft_ms`, `tpot_ms`, `end_to_end_s`
- `total_tokens_per_sec`, `output_tokens_per_sec`
- `status_code`

**Optional:**
- `input_prompt`, `output_text`

**Filename Format:** `{platform}-{YYYYMMDD}-{HHMMSS}-{benchmark-id}.csv`

Example: `vllm-20251003-175002-1a-chat-simulation.csv`

## 🔧 Configuration

Edit `config.py` to customize:

- File upload limits
- Chart styling and colors
- Statistical thresholds
- SLA criteria
- Default percentiles

## 🧪 Development

### Run Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy streamlit_app/
```

### Linting

```bash
ruff check streamlit_app/
```

## 📝 Usage Example

1. **Run benchmarks** on multiple platforms:
   ```bash
   # vLLM
   python examples/benchmark_chat_simulation.py --engine vllm --host ...

   # TGI
   python examples/benchmark_chat_simulation.py --engine tgi --host ...
   ```

2. **Upload CSVs** to dashboard (both files)

3. **Compare results** across all analysis pages

4. **Make decisions** based on data (latency vs throughput trade-offs)

## 🤝 Contributing

Contributions welcome! Follow the project's coding standards and add tests for new features.

## 📄 License

Same as parent project (llm-locust)

## 🙏 Credits

Built with:
- [Streamlit](https://streamlit.io)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org)
- [SciPy](https://scipy.org)

