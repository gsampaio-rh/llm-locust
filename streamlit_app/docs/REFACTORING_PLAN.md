# Dashboard Refactoring Plan: Streamlined Technical View

**Date:** October 3, 2025  
**Goal:** Eliminate duplication, remove Simple mode, create clean professional dashboard for technical users

---

## 🎯 Problems Identified

### 1. Dual-Mode Creates Duplication
- Same metrics shown twice (Simple + Advanced)
- Mode toggle adds cognitive load
- Simple mode patronizes technical users
- Maintenance burden (update two views)

### 2. Information Architecture Issues
- Charts scattered across multiple pages
- Unclear navigation flow
- Some charts isolated when they should be together
- No clear "story" as you navigate

### 3. Redundant Content
- Overview dashboard shows same metrics as individual pages
- Multiple ways to see the same data
- Explanation text repeated across pages

---

## ✅ Refactoring Goals

1. **Single, Professional View** - Technical users only, no dumbing down
2. **Clear Information Hierarchy** - Logical flow from overview → deep dive
3. **Eliminate Duplication** - Show each metric once, in the right place
4. **Better Navigation** - Clear story from high-level to detailed
5. **Faster Time to Insight** - Key findings up front

---

## 🏗️ New Architecture

### Page Structure (Streamlined)

```
streamlit_app/
├── app.py                          # Home: Executive Summary + Quick Actions
├── pages/
│   ├── 1_Comparison.py            # Side-by-side platform comparison
│   ├── 2_Latency_Deep_Dive.py     # TTFT, TPOT, E2E analysis
│   ├── 3_Throughput_Deep_Dive.py  # TPS, RPS, stability
│   └── 4_Reliability.py           # Errors, success rate, status codes
```

**Removed:**
- ❌ Simple/Advanced mode toggle
- ❌ Duplicate metric cards
- ❌ Redundant explanation text
- ❌ "What this means for you" business speak

**Added:**
- ✅ Clear executive summary on home
- ✅ Comparison page (consolidated overview)
- ✅ Deep dive pages (detailed analysis only)
- ✅ Statistical tests integrated naturally

---

## 📄 Page-by-Page Redesign

### Home (`app.py`) - Executive Summary
**Purpose:** Quick decision-making, then navigate to details

**Content:**
```
┌─────────────────────────────────────────────────┐
│ 🎯 LLM Benchmark Dashboard                     │
│ [Upload CSVs] [Clear] [Export Report]          │
├─────────────────────────────────────────────────┤
│                                                  │
│ 📊 EXECUTIVE SUMMARY                            │
│                                                  │
│ Benchmarks: 3 platforms, 10K requests each     │
│ Duration: 5 minutes each                        │
│ Date: Oct 3, 2025                               │
│                                                  │
│ ┌───────────────────────────────────────────┐  │
│ │ 🏆 RECOMMENDATION: vLLM                    │  │
│ │                                            │  │
│ │ • 19% faster TTFT (statistically significant)│
│ │ • 99.8% reliability (best)                │  │
│ │ • Stable performance over time            │  │
│ │                                            │  │
│ │ [View Detailed Comparison →]              │  │
│ └───────────────────────────────────────────┘  │
│                                                  │
│ 📈 KEY METRICS AT-A-GLANCE                     │
│                                                  │
│ [3-column metric cards: TTFT, TPOT, TPS]       │
│ - Show winner badges                            │
│ - Color-coded                                   │
│ - Click to navigate to deep dive                │
│                                                  │
│ 🎨 NORMALIZED PERFORMANCE                      │
│                                                  │
│ [Normalized comparison chart]                   │
│ - All metrics 0-100 scale                      │
│ - See trade-offs at a glance                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Removed:**
- ❌ Duplicate platform headers
- ❌ Mode toggle
- ❌ Welcome screen fluff
- ❌ "What you'll learn" marketing speak

**Changes:**
- Move recommendation to top
- Clickable metric cards → navigate to deep dive
- Keep only executive summary content
- Clean, professional tone

---

### Page 1: Platform Comparison (`1_Comparison.py`)
**Purpose:** Side-by-side detailed comparison of all platforms

**Content:**
```
┌─────────────────────────────────────────────────┐
│ 📊 Platform Comparison                          │
├─────────────────────────────────────────────────┤
│                                                  │
│ 📋 COMPARISON MATRIX                            │
│                                                  │
│ Platform  │ TTFT P50 │ TTFT P99 │ TPOT │ TPS  │ Success │
│ ─────────┼──────────┼──────────┼──────┼──────┼─────────│
│ vLLM  🏆 │ 234ms   │ 456ms   │ 12ms │ 1.2K │ 99.8%  │
│ TGI      │ 289ms   │ 578ms   │ 12ms │ 1.1K │ 99.7%  │
│ Ollama   │ 312ms   │ 623ms   │ 15ms │ 0.9K │ 98.9%  │
│                                                  │
│ 📊 STATISTICAL COMPARISON                       │
│                                                  │
│ [For each pair of platforms:]                   │
│                                                  │
│ vLLM vs TGI                                     │
│ • TTFT: 19% faster (p=0.001, highly significant)│
│ • TPOT: No significant difference (p=0.234)    │
│ • Throughput: 9% higher (significant)          │
│ Winner: vLLM                                    │
│                                                  │
│ [Normalized comparison chart]                   │
│                                                  │
│ 🎯 QUICK INSIGHTS                              │
│                                                  │
│ • vLLM: Best overall (fast + reliable)         │
│ • TGI: Competitive TPOT, slightly slower TTFT  │
│ • Ollama: Lower throughput, needs investigation│
│                                                  │
└─────────────────────────────────────────────────┘
```

**Key Features:**
- Comparison matrix (sortable table)
- Pairwise statistical tests
- Normalized chart
- Winner determination
- Quick insights (auto-generated)

**Removed:**
- ❌ Individual platform cards (use table instead)
- ❌ Duplicate charts
- ❌ Long explanations

---

### Page 2: Latency Deep Dive (`2_Latency_Deep_Dive.py`)
**Purpose:** Comprehensive latency analysis (TTFT, TPOT, E2E)

**Content:**
```
┌─────────────────────────────────────────────────┐
│ ⚡ Latency Analysis                             │
├─────────────────────────────────────────────────┤
│                                                  │
│ [Metric Selector: TTFT | TPOT | End-to-End]    │
│                                                  │
│ 📈 DISTRIBUTIONS                                │
│                                                  │
│ [Row 1: 2 columns]                              │
│ Col 1: Histogram + KDE (all platforms overlaid)│
│ Col 2: Box plot comparison                      │
│                                                  │
│ [Row 2: 2 columns]                              │
│ Col 1: CDF plot                                 │
│ Col 2: Percentile bar chart (P50/P90/P95/P99)  │
│                                                  │
│ 📊 OVER TIME (STABILITY)                        │
│                                                  │
│ [Multi-platform timeline - full width]          │
│ - Shows all platforms overlaid                  │
│ - Rolling average + raw points                  │
│ - Detect degradation visually                   │
│                                                  │
│ 📋 DETAILED STATISTICS                          │
│                                                  │
│ [Expandable per platform]                       │
│ Platform: vLLM                                  │
│ • Count: 10,000                                 │
│ • Mean: 245ms (±15ms 95% CI)                   │
│ • Median (P50): 234ms                           │
│ • P90: 342ms                                    │
│ • P95: 389ms                                    │
│ • P99: 456ms                                    │
│ • P99.9: 523ms                                  │
│ • Std Dev: 67ms                                 │
│ • CV: 27% (moderate variance)                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Layout:**
- 2x2 grid for distribution charts (compact)
- Full-width timeline
- Expandable stats tables (not always visible)

**Removed:**
- ❌ Separate "What is TTFT" sections (use tooltips)
- ❌ Speed rankings (that's on Comparison page)
- ❌ Duplicate percentile displays

---

### Page 3: Throughput Deep Dive (`3_Throughput_Deep_Dive.py`)
**Purpose:** TPS, RPS, stability, capacity analysis

**Content:**
```
┌─────────────────────────────────────────────────┐
│ 🚀 Throughput Analysis                          │
├─────────────────────────────────────────────────┤
│                                                  │
│ 📊 THROUGHPUT OVER TIME                         │
│                                                  │
│ [Multi-platform timeline - full width]          │
│ - TPS for all platforms                         │
│ - Detect degradation                            │
│                                                  │
│ 📈 STABILITY METRICS                            │
│                                                  │
│ Platform  │ Mean TPS │ Std Dev │ CV    │ Stability │
│ ─────────┼──────────┼─────────┼───────┼───────────│
│ vLLM     │ 1,234    │ 156     │ 12.6% │ Excellent │
│ TGI      │ 1,189    │ 201     │ 16.9% │ Good      │
│ Ollama   │ 912      │ 278     │ 30.5% │ Fair      │
│                                                  │
│ 🎯 CAPACITY ANALYSIS                           │
│                                                  │
│ Platform  │ Sustained RPS │ Est. Concurrent Users │
│ ─────────┼───────────────┼─────────────────────│
│ vLLM     │ 16.7          │ ~167                 │
│ TGI      │ 16.5          │ ~165                 │
│ Ollama   │ 2.5           │ ~25                  │
│                                                  │
│ 📉 DISTRIBUTION                                 │
│                                                  │
│ [Histogram of TPS values]                       │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Removed:**
- ❌ Single-platform selector (show all at once)
- ❌ "What is throughput" explanations
- ❌ Duplicate cards

---

### Page 4: Reliability (`4_Reliability.py`)
**Purpose:** Error analysis, success rates, failure patterns

**Content:**
```
┌─────────────────────────────────────────────────┐
│ ✅ Reliability & Error Analysis                 │
├─────────────────────────────────────────────────┤
│                                                  │
│ 📊 SUCCESS RATE COMPARISON                      │
│                                                  │
│ [Bar chart with 99.9% SLA line]                 │
│                                                  │
│ Platform  │ Success Rate │ Failures │ Per 1K │ SLA │
│ ─────────┼──────────────┼──────────┼────────┼─────│
│ vLLM     │ 99.82%      │ 18       │ 1.8    │ ✅  │
│ TGI      │ 99.73%      │ 27       │ 2.7    │ ✅  │
│ Ollama   │ 98.91%      │ 109      │ 10.9   │ ❌  │
│                                                  │
│ 🔴 ERROR BREAKDOWN                              │
│                                                  │
│ [Tabs for each platform]                        │
│                                                  │
│ Tab: vLLM (18 failures)                         │
│ [Pie chart: Status codes]                       │
│                                                  │
│ 500 - Server Error: 12 (67%)                    │
│ 429 - Rate Limit: 6 (33%)                       │
│                                                  │
│ [Timeline: Errors over time]                    │
│ - Shows when failures occurred                  │
│ - Detect patterns/clusters                      │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Removed:**
- ❌ "What is reliability" sections
- ❌ Duplicate success rate displays
- ❌ Expanders for each platform (use tabs instead)

---

## 🎨 Component Library (Shared)

Create reusable, composable components:

### `lib/components.py`
```python
# High-level components
def render_metric_summary_card(benchmark, metric_name)
def render_comparison_table(benchmarks, metrics)
def render_statistical_test_result(comparison)
def render_percentile_table(benchmark, metric)
def render_platform_tabs(benchmarks, content_fn)

# Layout helpers
def create_chart_grid(charts, cols=2)
def render_expandable_section(title, content_fn)
def render_metric_selector(metrics, default)
```

### `lib/visualizations.py`
Keep only unique chart functions:
```python
# Distribution charts
create_latency_distribution_chart()    # Histogram + KDE
create_box_plot_chart()                 # Box plot
create_cdf_chart()                      # CDF

# Time series
create_multi_platform_timeline()        # Already have this!

# Comparison
create_percentile_comparison_chart()    # Percentile bars
create_normalized_comparison_chart()    # Already have this!
create_success_rate_chart()             # Success rate bars
create_status_code_pie_chart()          # Already have this!
```

**Remove duplicates:**
- ❌ Single-platform time series (use multi-platform)
- ❌ Separate TTFT/TPOT functions (parameterize)

---

## 🔄 Migration Strategy

### Phase 1: Refactor Core (Day 1-2)
1. **Remove mode toggle** from `app.py`
2. **Consolidate home page** to executive summary only
3. **Remove duplicate cards/charts** from dashboard.py
4. **Keep statistics.py** (no changes needed)

### Phase 2: Reorganize Pages (Day 3-4)
5. **Create `1_Comparison.py`** - Merge overview + comparison logic
6. **Refactor `2_Latency_Deep_Dive.py`** - Keep advanced only
7. **Refactor `3_Throughput_Deep_Dive.py`** - Keep advanced only
8. **Refactor `4_Reliability.py`** - Consolidate error analysis

### Phase 3: Component Library (Day 5)
9. **Extract reusable components** to `lib/components.py`
10. **Consolidate visualization functions**
11. **Remove old files**: `dashboard.py`, `explanations.py`

### Phase 4: Polish (Day 6)
12. **Update navigation** (sidebar links)
13. **Add breadcrumbs** (show current page)
14. **Consistent styling** across all pages
15. **Performance optimization** (caching)

---

## 📊 Before/After Comparison

### Current State (Problems)
```
app.py (256 lines)
├── Mode toggle (Simple/Advanced)
├── Welcome screen
├── Duplicate metric cards (x2)
├── Recommendation section
└── Comparison table

pages/
├── 1_Latency_Analysis.py (322 lines)
│   ├── Simple mode section
│   ├── Advanced mode section
│   └── Duplicate explanations
├── 2_Throughput_Analysis.py (261 lines)
│   ├── Simple mode section
│   ├── Advanced mode section
│   └── Single + multi platform views
└── 3_Error_Analysis.py (369 lines)
    ├── Simple mode section
    ├── Advanced mode section
    └── Expandable platform sections

lib/
├── dashboard.py (385 lines)
│   ├── render_simple_mode_dashboard()
│   ├── render_advanced_mode_dashboard()
│   └── render_recommendation_section()
└── explanations.py (256 lines)
    └── Lots of "Simple English" helpers

Total: ~1,849 lines with heavy duplication
```

### After Refactor (Clean)
```
app.py (~120 lines)
├── Executive summary
├── Quick metric cards (clickable)
├── Normalized chart
└── Recommendation

pages/
├── 1_Comparison.py (~180 lines)
│   ├── Comparison matrix
│   ├── Statistical tests
│   └── Normalized chart
├── 2_Latency_Deep_Dive.py (~220 lines)
│   ├── Distribution grid (2x2)
│   ├── Multi-platform timeline
│   └── Stats tables
├── 3_Throughput_Deep_Dive.py (~150 lines)
│   ├── Multi-platform timeline
│   ├── Stability table
│   └── Capacity analysis
└── 4_Reliability.py (~140 lines)
    ├── Success rate comparison
    └── Platform tabs with pie charts

lib/
├── components.py (~200 lines)
│   └── Reusable UI components
└── visualizations.py (~400 lines)
    └── Chart functions (no duplicates)

Total: ~1,410 lines, no duplication
Reduction: 24% less code, 2x clearer
```

---

## ✅ Benefits

### For Users
- ✅ **Clearer navigation** - Know exactly where to go
- ✅ **Faster insights** - No mode switching
- ✅ **Professional tone** - Respects technical expertise
- ✅ **Less scrolling** - Denser, more efficient layouts
- ✅ **Better comparisons** - Side-by-side is natural

### For Developers
- ✅ **Less code** - 24% reduction
- ✅ **No duplication** - Single source of truth
- ✅ **Easier maintenance** - Change once, applies everywhere
- ✅ **Better organization** - Clear responsibilities
- ✅ **Faster development** - Reusable components

### For Quality
- ✅ **Consistency** - Same charts behave the same
- ✅ **Testing** - Test components once
- ✅ **Performance** - Less rendering overhead
- ✅ **Accessibility** - Standardized patterns

---

## 🎯 Success Metrics

| Metric | Current | Target | After Refactor |
|--------|---------|--------|----------------|
| Lines of Code | 1,849 | <1,500 | ~1,410 |
| Duplicate Functions | ~15 | 0 | 0 |
| Pages | 4 | 5 | 5 |
| Mode Toggle Clicks | 100% | 0% | 0% |
| Time to Insight | 3 clicks | 2 clicks | 2 clicks |
| Code Duplication | 40% | <10% | <5% |

---

## 🚀 Next Steps

### Immediate Actions
1. **Review this plan** - Agree on structure
2. **Prioritize changes** - Which pages first?
3. **Create backup branch** - Safety first
4. **Start with app.py** - Remove mode toggle

### Implementation Order
1. ✅ Create `1_Comparison.py` (new page)
2. ✅ Refactor `app.py` (remove duplication)
3. ✅ Simplify `2_Latency_Deep_Dive.py`
4. ✅ Simplify `3_Throughput_Deep_Dive.py`
5. ✅ Simplify `4_Reliability.py`
6. ✅ Extract components to `lib/components.py`
7. ✅ Remove `dashboard.py`, `explanations.py`
8. ✅ Update documentation

---

## 📝 Open Questions

1. **Keep glossary?** - Tooltips vs separate page?
   - **Recommendation:** Inline tooltips only (no separate page)

2. **Expandable sections?** - More or less?
   - **Recommendation:** Less - default to showing key info

3. **Export buttons?** - Per page or global?
   - **Recommendation:** Global in sidebar (coming in Phase 2)

4. **Statistical tests?** - Always show or hide by default?
   - **Recommendation:** Always show in Comparison page, hide in deep dives

5. **Navigation style?** - Sidebar only or breadcrumbs too?
   - **Recommendation:** Sidebar + page titles (no breadcrumbs needed)

---

## 🎊 Conclusion

This refactor will:
- ✅ Eliminate 24% of code
- ✅ Remove all duplication
- ✅ Create clear information architecture
- ✅ Respect technical users
- ✅ Maintain all functionality
- ✅ Improve maintainability

**Estimated effort:** 6 days for full implementation  
**Risk level:** Low (mostly reorganization, no new logic)  
**Impact:** High (much better UX and DX)

**Ready to proceed?** Let's start with Phase 1!

