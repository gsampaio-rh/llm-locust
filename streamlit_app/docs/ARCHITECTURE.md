# Streamlit App Architecture

## 📐 Design Principles

This dashboard follows **SOLID principles** and **separation of concerns** for maintainability and testability.

---

## 🏗️ Module Structure

```
streamlit_app/
├── app.py                      # Entry point (orchestration only)
├── config.py                   # Configuration constants
│
├── lib/                        # Business logic (pure functions)
│   ├── components.py           # Reusable UI components
│   ├── dashboard.py            # Dashboard rendering logic
│   ├── data_loader.py          # CSV loading and validation
│   ├── explanations.py         # Plain English translations
│   ├── visualizations.py       # Chart generation
│   ├── metrics.py              # Metric calculations (future)
│   └── statistics.py           # Statistical tests (future)
│
├── models/                     # Data models (Pydantic)
│   └── benchmark.py            # BenchmarkData, Metadata, etc.
│
├── pages/                      # Multi-page sections
│   ├── 1_Latency_Analysis.py
│   ├── 2_Throughput_Analysis.py
│   └── 3_Error_Analysis.py
│
└── utils/                      # Utilities
    ├── validators.py           # Data validation (future)
    └── formatters.py           # Display formatting (future)
```

---

## 🎯 Separation of Concerns

### **app.py** (Orchestration Layer)
**Responsibility:** Application flow and user interaction
**Does:**
- Page configuration
- Sidebar rendering (file upload, mode toggle)
- Welcome screen
- Delegates rendering to dashboard modules

**Does NOT:**
- Contain HTML/CSS
- Calculate metrics
- Render individual components

**Lines of Code:** ~240 (down from 600+)

---

### **lib/components.py** (UI Component Layer)
**Responsibility:** Reusable UI components
**Exports:**
- `render_platform_header()` - Purple gradient platform cards
- `render_benchmark_metadata()` - Metadata info card
- `render_metric_card()` - Colored metric cards with status
- `render_help_expander()` - Expandable help sections
- `get_status_colors()` - Color logic for metrics

**Pattern:** Pure functions that take data, return UI
**Testability:** Easy to test rendering logic in isolation

---

### **lib/dashboard.py** (Dashboard Logic Layer)
**Responsibility:** Mode-specific dashboard rendering
**Exports:**
- `find_winners()` - Determine best performer for each metric
- `render_simple_mode_dashboard()` - Simple mode with 4 cards
- `render_advanced_mode_dashboard()` - Advanced mode with 6 cards + metadata
- `render_recommendation_section()` - Recommendation and impact analysis

**Pattern:** Compose components to create views
**Benefits:**
- Simple and Advanced modes share component library
- Easy to add new modes or modify existing ones
- Clear separation between modes

---

### **lib/data_loader.py** (Data Layer)
**Responsibility:** Data loading, validation, and transformation
**Exports:**
- `load_benchmark_csv()` - Load single CSV file
- `load_multiple_benchmarks()` - Load multiple files
- `extract_platform_from_filename()` - Parse metadata
- `validate_csv_schema()` - Ensure required columns exist
- `calculate_quality_score()` - Data quality assessment

**Pattern:** Input validation at the edge
**Benefits:**
- Fail fast with clear error messages
- Data quality visible to users
- Centralized validation logic

---

### **lib/explanations.py** (Translation Layer)
**Responsibility:** Technical → Plain English translation
**Exports:**
- `GLOSSARY` - Complete definitions for all metrics
- `get_simple_explanation()` - One-line explanations
- `speed_to_emoji_and_label()` - Visual status indicators
- `reliability_to_emoji_and_label()` - Visual reliability status
- `generate_simple_recommendation()` - AI-style recommendation engine
- `calculate_user_impact()` - Business impact calculations

**Pattern:** Domain-driven design - encapsulate domain knowledge
**Benefits:**
- Consistent messaging across the app
- Easy to update explanations without touching UI
- Testable recommendation logic

---

### **lib/visualizations.py** (Chart Layer)
**Responsibility:** Chart generation with Plotly
**Exports:**
- `create_latency_distribution_chart()` - Histogram + KDE
- `create_percentile_comparison_chart()` - Bar chart
- `create_box_plot_chart()` - Box plots
- `create_cdf_chart()` - Cumulative distribution
- `create_time_series_chart()` - Time series with rolling mean
- `create_success_rate_chart()` - Success rate bars
- `get_platform_color()` - Consistent color scheme

**Pattern:** Chart factory functions
**Benefits:**
- Reusable across pages
- Consistent styling
- Easy to swap visualization libraries

---

### **models/benchmark.py** (Data Model Layer)
**Responsibility:** Type-safe data structures
**Exports:**
- `BenchmarkMetadata` - Test metadata (platform, duration, etc.)
- `BenchmarkData` - Complete benchmark with df and metrics
- `ComparisonResult` - Comparison between two benchmarks

**Pattern:** Pydantic models for validation and type safety
**Benefits:**
- Runtime validation
- IDE autocomplete
- Clear contracts between layers

---

## 🔄 Data Flow

```
1. User uploads CSV
   ↓
2. data_loader.py validates and parses
   ↓
3. BenchmarkData models created
   ↓
4. dashboard.py finds winners
   ↓
5. components.py renders UI cards
   ↓
6. explanations.py adds plain English
   ↓
7. User sees beautiful dashboard
```

---

## 🎨 Component Hierarchy

```
app.py (orchestrator)
└── Mode: Simple or Advanced
    ├── dashboard.render_simple_mode_dashboard()
    │   └── components.render_metric_card() × 4
    │       ├── explanations.speed_to_emoji_and_label()
    │       └── components.get_status_colors()
    │
    └── dashboard.render_advanced_mode_dashboard()
        ├── components.render_platform_header()
        ├── components.render_benchmark_metadata()
        └── components.render_metric_card() × 6
```

---

## 🧪 Testability

### Unit Tests (Easy)
```python
# Test component rendering
def test_get_status_colors():
    bg, text, border = get_status_colors(400, {"excellent": 500}, lower_is_better=True)
    assert bg == "#d4edda"  # Green for excellent

# Test winner detection
def test_find_winners():
    winners = find_winners(mock_benchmarks)
    assert winners["ttft"] == 0  # First benchmark is fastest
```

### Integration Tests (Moderate)
```python
# Test full dashboard rendering
def test_render_simple_dashboard(mock_benchmarks):
    render_simple_mode_dashboard(mock_benchmarks, winners)
    # Verify st.markdown was called with expected content
```

---

## 🔧 Adding New Features

### Add a New Metric Card

1. **Add to components.py:**
   ```python
   def render_new_metric_card(...):
       render_metric_card(...)
   ```

2. **Use in dashboard.py:**
   ```python
   def render_advanced_mode_dashboard(...):
       # ... existing cards ...
       render_new_metric_card(...)
   ```

3. **Add explanation:**
   ```python
   # In explanations.py
   GLOSSARY["NewMetric"] = {
       "simple": "Plain English",
       "detailed": "Technical explanation",
       ...
   }
   ```

### Add a New Page

1. Create `pages/4_New_Analysis.py`
2. Import from `lib/` modules
3. Use existing components and visualizations
4. Follow Simple/Advanced mode pattern

---

## 📊 Benefits of This Architecture

### ✅ Maintainability
- **Single Responsibility**: Each module has one clear purpose
- **DRY**: No duplication between Simple and Advanced modes
- **Readable**: Clear function names and documentation

### ✅ Testability
- **Pure Functions**: Easy to unit test
- **Isolated Logic**: Components don't depend on Streamlit state
- **Mockable**: Easy to mock data for tests

### ✅ Extensibility
- **New metrics**: Just add to components and reuse
- **New modes**: Create new render functions
- **New pages**: Import existing components

### ✅ Consistency
- **Shared components**: Same look across all views
- **Centralized styling**: Change once, applies everywhere
- **Unified colors**: Platform colors in one place

---

## 🚀 Performance Considerations

### Lazy Loading
- Visualizations only created when needed
- Charts generated on-demand per page
- Session state caches loaded data

### Efficient Calculations
- Winners calculated once per view
- Percentiles pre-calculated in BenchmarkData
- Success filtering done once per benchmark

### Minimal Re-renders
- Components are pure functions
- Streamlit caches where appropriate
- Mode switching is instant (no recalculation)

---

## 📏 Code Quality Metrics

| Module | Lines | Complexity | Testability |
|--------|-------|------------|-------------|
| app.py | ~240 | Low | Medium (UI testing) |
| components.py | ~200 | Low | High |
| dashboard.py | ~180 | Medium | High |
| data_loader.py | ~180 | Medium | High |
| explanations.py | ~260 | Medium | High |
| visualizations.py | ~280 | Medium | High |

**Total:** ~1,340 lines (well-structured, readable)

---

## 🎓 Design Patterns Used

1. **Factory Pattern** - `lib/visualizations.py` creates charts
2. **Strategy Pattern** - Simple vs Advanced rendering strategies
3. **Composition** - Dashboard composed of smaller components
4. **Pure Functions** - No side effects in lib modules
5. **Dependency Injection** - Components receive data as parameters

---

## 🔮 Future Enhancements

### Phase 2 (Planned)
- `lib/metrics.py` - Advanced metric calculations
- `lib/statistics.py` - Statistical significance tests
- `utils/validators.py` - Extended validation rules
- `utils/exporters.py` - PDF/Excel export logic

### Phase 3 (Nice to Have)
- `lib/insights.py` - ML-powered insight generation
- `lib/cost_calculator.py` - TCO analysis
- `utils/formatters.py` - Number/date formatting
- `tests/` - Comprehensive test suite

---

## 📖 Developer Guide

### Making Changes

**Rule:** Keep business logic out of `app.py`

**Example - Adding a new metric:**

❌ **Don't do this:**
```python
# In app.py
st.markdown(f"""
<div style="...">
    <div>{metric_value}</div>
</div>
""")
```

✅ **Do this:**
```python
# In components.py
def render_new_metric_card(...):
    # ... rendering logic

# In dashboard.py
def render_advanced_mode_dashboard(...):
    render_new_metric_card(...)

# In app.py
render_advanced_mode_dashboard(benchmarks, winners)
```

### Code Review Checklist

- [ ] No HTML in app.py
- [ ] Components are reusable
- [ ] Functions are pure (no side effects)
- [ ] Type hints on all functions
- [ ] Docstrings with Args/Returns
- [ ] No magic numbers (use config.py)
- [ ] Consistent naming conventions

---

## 🎯 Summary

**Before Refactoring:**
- app.py: 600+ lines
- Mixed concerns (UI + logic + styling)
- Hard to test
- Duplication between modes

**After Refactoring:**
- app.py: ~240 lines (orchestration only)
- Clear separation of concerns
- Highly testable
- Reusable components
- Easy to extend

**Impact:**
- 60% reduction in app.py size
- 5 focused modules vs 1 monolith
- Better developer experience
- Easier to onboard new contributors

