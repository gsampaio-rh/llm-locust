# 🏁 Agile Plan: LLM Locust Interactive CLI - "The Great Model Race"

## 📋 Project Overview

**Project Name:** Interactive Racing CLI  
**Code Name:** "The Great Model Race"  
**Team Size:** 2-3 developers  
**Start Date:** TBD  
**Target Release:** v0.3.0

---

## 🎯 Product Vision

Transform LLM benchmarking from a passive batch process into an engaging, real-time experience where developers **race models head-to-head**, understand performance metrics intuitively, and learn optimization through interactive feedback.

**North Star Metric:** 3x increase in benchmark runs per user (from passive analysis to active exploration)

---

### Definition of Done
- [ ] Linter passes (ruff, mypy)
- [ ] Documentation updated
- [ ] Demo-able in terminal
- [ ] Peer reviewed and approved
- [ ] Merged to `main` branch

---

## 📦 Release Plan

### MVP (v0.3.0) - End of Sprint 2
**Goal:** Basic racing with live visualization  
**Features:** Multi-endpoint execution, progress bars, simple leaderboard  
**Success Criteria:** Can race 2+ endpoints with live updates

### Beta (v0.4.0) - End of Sprint 3
**Goal:** Polished UX with educational features  
**Features:** Sparklines, commentary, drill-downs, themes  
**Success Criteria:** User testing shows >70% understand metrics

### GA (v0.5.0) - End of Sprint 4
**Goal:** Production-ready with sharing  
**Features:** Replays, achievements, export, full docs  
**Success Criteria:** Zero P0 bugs, positive community feedback

---

## 🗓️ Sprint Breakdown

---

## SPRINT 1: Foundation

**Theme:** "Build the Racing Engine"  
**Goal:** Establish multi-process architecture and basic TUI  
**Story Points:** 21  
**Status:** ✅ COMPLETE (21/21 points - 100%)  
**Completed:** 2025-10-04

### Progress Summary
- ✅ US-1.1: Multi-Endpoint Race Configuration (3 pts)
- ✅ US-1.2: Parallel Benchmark Execution (5 pts)
- ✅ US-1.3: Basic TUI Framework (5 pts)
- ✅ US-1.4: Live Progress Bars (3 pts)
- ✅ US-1.5: Simple Leaderboard (2 pts)
- ✅ US-1.6: Basic Race Summary (3 pts)

**Files Created:**
- `llm_locust/race/config.py` - Configuration models & validation
- `llm_locust/race/orchestrator.py` - Multi-process orchestration with warm-up
- `llm_locust/race/runner.py` - Per-engine benchmark execution
- `llm_locust/race/tui.py` - Terminal UI with live updates
- `llm_locust/race/state.py` - Race state tracking & metrics
- `llm_locust/race/summary.py` - Race summary & export
- `llm_locust/cli/race.py` - CLI command
- `configs/races/*.yaml` - 6 race configurations
- `requirements.txt` - Main dependencies
- `requirements-dev.txt` - Development dependencies

### User Stories

#### US-1.1: Multi-Endpoint Race Configuration ✅ COMPLETE
**As a** developer  
**I want to** define multiple endpoints in a config file  
**So that** I can race them head-to-head

**Acceptance Criteria:**
- [x] Create `race.yaml` schema with YAML validation
- [x] Support 2-10 endpoints per race
- [x] Each endpoint has: name, URL, emoji, color
- [x] CLI reads config: `llm-locust race --config race.yaml`
- [x] Error handling for invalid configs
- [x] Example configs in `configs/races/` (moved from examples)

**Story Points:** 3  
**Priority:** P0  
**Dependencies:** None  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/config.py` with frozen dataclasses
- Comprehensive validation with clear error messages
- 5 example configs including real cluster endpoints
- CLI validation mode: `--validate-only`

**Technical Notes:**
```yaml
# Example race.yaml
race:
  name: "Production Candidates"
  engines:
    - name: vllm
      url: https://vllm.endpoint.com
      emoji: 🚀
      color: purple
    - name: tgi  
      url: https://tgi.endpoint.com
      emoji: 🤖
      color: pink
```

---

#### US-1.2: Parallel Benchmark Execution ✅ COMPLETE
**As a** benchmark runner  
**I want to** execute tests against multiple endpoints simultaneously  
**So that** results are comparable and fair

**Acceptance Criteria:**
- [x] Spawn separate process for each endpoint
- [x] Shared metrics queue for IPC
- [x] Synchronized start (countdown mechanism)
- [x] Same prompts sent to all endpoints (order-preserved)
- [x] Graceful shutdown of all processes
- [x] Handle process failures without killing race

**Story Points:** 5  
**Priority:** P0  
**Dependencies:** US-1.1  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/orchestrator.py` - Multi-process coordination
- Created `llm_locust/race/runner.py` - Per-engine benchmark execution
- Countdown: 3...2...1...GO! before spawning processes
- Signal handlers for graceful shutdown (SIGINT, SIGTERM)
- Tested successfully with real cluster endpoints

**Technical Design:**
```python
class RaceOrchestrator:
    def __init__(self, config: RaceConfig):
        self.engines = config.engines
        self.metrics_queue = Queue()
        self.processes: List[Process] = []
    
    def start_race(self):
        # Countdown: 3...2...1...GO!
        for engine in self.engines:
            p = Process(target=run_benchmark, args=(engine, self.metrics_queue))
            p.start()
            self.processes.append(p)
```

---

#### US-1.3: Basic TUI Framework ✅ COMPLETE
**As a** user  
**I want to** see a live terminal dashboard  
**So that** I can watch the race in real-time

**Acceptance Criteria:**
- [x] Set up Textual or Rich framework (chose Rich)
- [x] Render at 10 FPS minimum (10 FPS implemented)
- [x] Full-screen terminal mode
- [x] Graceful fallback for small terminals (min 80x24)
- [x] Clean exit on Ctrl+C
- [x] No visual artifacts or flicker

**Story Points:** 5  
**Priority:** P0  
**Dependencies:** None  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/tui.py` using Rich framework
- Layout with header, body, footer sections
- Live display with 10 FPS refresh rate
- Terminal size detection with warnings
- Demo script: `examples/demo_tui.py`

**Tech Stack Decision:**
- **Option A:** Textual (full framework, widgets, CSS-like styling)
- **Option B:** Rich (simpler, more control, proven)
- **Recommendation:** Start with Rich, migrate to Textual if needed

---

#### US-1.4: Live Progress Bars ✅ COMPLETE
**As a** user  
**I want to** see real-time progress for each engine  
**So that** I know which is ahead

**Acceptance Criteria:**
- [x] Progress bar per engine showing completion %
- [x] Live request counter (updates every 100ms)
- [x] Smooth visual updates (no jank)
- [x] Color-coded by engine
- [x] Show rate: "X reqs/sec"
- [ ] Responsive to terminal resize (pending)

**Story Points:** 3  
**Priority:** P0  
**Dependencies:** US-1.2, US-1.3  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/state.py` - Race state tracking
- `EngineState` dataclass tracks: requests, failures, users, tokens, RPS
- `RaceState` consumes metrics queue and updates engine states
- Progress bars with Rich Progress component
- Real-time metrics display with success rates

**Visual Mock:**
```
🚀 vLLM     [████████████████░░░░] 1,234/2,000  (61%)  12 req/s
🤖 TGI      [█████████░░░░░░░░░░░]   987/2,000  (49%)   9 req/s
🦙 Ollama   [█████░░░░░░░░░░░░░░░]   456/2,000  (23%)   4 req/s
```

---

#### US-1.5: Simple Leaderboard ✅ COMPLETE
**As a** user  
**I want to** see a live ranking of engines  
**So that** I know who's winning

**Acceptance Criteria:**
- [x] Ranked list (1st, 2nd, 3rd, etc.)
- [x] Medal emojis (🥇🥈🥉)
- [x] Show key metric: Request count
- [x] Updates every second (4 FPS)
- [ ] Smooth re-ranking animations (deferred to Sprint 2)
- [ ] Highlight position changes (deferred to Sprint 2)

**Story Points:** 2  
**Priority:** P0  
**Dependencies:** US-1.4  
**Status:** ✅ COMPLETE

**Implementation:**
- Added `render_leaderboard()` to TUI
- Live rankings sorted by request count
- Medal display (🥇🥈🥉) for top 3
- Color-coded: Gold, Silver, Bronze
- Updates in real-time with race state

---

#### US-1.6: Basic Race Summary ✅ COMPLETE
**As a** user  
**I want to** see final results after race  
**So that** I can compare performance

**Acceptance Criteria:**
- [x] Summary screen at race end
- [x] Winner announcement with emoji
- [x] Table with: Engine, Requests, Failures, Success Rate, Total Tokens
- [x] Race statistics (total requests, overall success rate)
- [x] Next steps guidance (Streamlit dashboard, run another race)
- [ ] Interactive options (deferred to Sprint 2)

**Story Points:** 3  
**Priority:** P0  
**Dependencies:** US-1.5  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/summary.py`
- `show_race_summary()` - Beautiful results table with winner
- `show_export_options()` - Next steps guidance
- Full statistics with medal rankings
- Integrated with orchestrator on race completion

---

### Sprint 1 Deliverables ✅ ALL COMPLETE
- [x] Working `llm-locust race` command
- [x] Multi-endpoint configuration via YAML
- [x] Live TUI with progress bars and leaderboard
- [x] Leaderboard with medals (🥇🥈🥉)
- [x] Basic race summary with winner announcement
- [x] Example race configs (6 configs including real cluster)
- [x] Updated CLI with race command
- [x] Warm-up phase for dataset loading
- [x] Real-time metrics display (4 FPS)
- [x] Type-safe implementation (mypy strict)
- [x] Zero linting errors (ruff)

**Completed:** 21/21 story points (100%) ✅  
**Status:** SPRINT 1 COMPLETE! 🎉

**Demo Script:**
```bash
# Sprint 1 Demo - Full System Test
llm-locust race --config configs/races/test-2min.yaml

# What you'll see:
# 1. Race header with configuration
# 2. Spawning engine processes
# 3. Warm-up phase (loading datasets)
# 4. Countdown: 3...2...1...GO!
# 5. Live TUI with:
#    - Real-time progress bars
#    - Request counters and req/s
#    - Live leaderboard with medals (🥇🥈🥉)
#    - Success rates
# 6. Final summary:
#    - Winner announcement
#    - Results table
#    - Race statistics
#    - Export options

# Test with real cluster endpoints:
llm-locust race --config configs/races/cluster-race.yaml
```

---

## SPRINT 2: Visual Polish (Weeks 3-4)

**Theme:** "Make It Beautiful"  
**Goal:** Add sparklines, charts, animations, and themes  
**Story Points:** 24  
**Status:** 🚧 IN PROGRESS (16/24 points complete - 67%)  
**Last Updated:** 2025-10-04

### Progress Summary
- ✅ US-2.1: Metric Sparklines (5 pts)
- ✅ US-2.2: Smooth Animations (3 pts)
- ⏳ US-2.3: Color Themes (2 pts) - PENDING (deferred)
- ✅ US-2.4: Status Indicators (3 pts)
- ✅ US-2.5: Time-Series Charts (5 pts)
- ⏳ US-2.6: Request Timeline View (5 pts) - PENDING
- ⏳ US-2.7: Keyboard Shortcuts (2 pts) - PENDING

### User Stories

#### US-2.1: Metric Sparklines ✅ COMPLETE
**As a** user  
**I want to** see mini-charts of metrics over time  
**So that** I can spot trends and issues

**Acceptance Criteria:**
- [x] Sparklines for TTFT, TPOT, throughput
- [x] Show last 60 data points (1 min @ 1 Hz)
- [x] Updates in real-time
- [x] Color-coded (green=good, yellow=ok, red=bad)
- [x] Scale automatically to data range
- [x] Fits in 20 character width

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-1.4  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/sparkline.py` - ASCII sparkline rendering
- 8-level characters: ▁▂▃▄▅▆▇█
- Color-coded based on thresholds (TTFT <300ms=green, >1000ms=red)
- Trend indicators: ↗ improving, ↘ degrading, → stable
- Updated `EngineState` to track metric history (last 60 points)
- Integrated into TUI with real-time updates

**Visual Mock:**
```
🚀 vLLM     [████████████████░░░░] 1,234 reqs
   TTFT:  234ms  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  P99: 456ms
   TPOT:   12ms  ▃▄▅▅▅▅▅▄▄▃▃▂▂▁  Stable
```

---

#### US-2.2: Smooth Animations ✅ COMPLETE
**As a** user  
**I want to** see smooth visual transitions  
**So that** the UI feels polished and professional

**Acceptance Criteria:**
- [x] Numbers count up/down (not jump) - AnimatedValue
- [x] Smooth counter interpolation
- [x] 20 FPS refresh rate (50ms frame time)
- [x] No blocking operations in render loop
- [ ] Progress bars animate smoothly (deferred - good enough)
- [ ] Leaderboard re-rankings slide (deferred - not needed yet)
- [ ] Fade in/out for alerts (deferred to Sprint 3)

**Story Points:** 3  
**Priority:** P1  
**Dependencies:** US-1.4, US-1.5  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/animation.py` - Animation utilities
- `AnimatedValue` - Smooth value interpolation
- `CounterAnimation` - Smooth counting (10 units/sec speed)
- Integrated into `EngineState` for request/token counters
- Increased refresh rate from 4 FPS to 20 FPS (smoother updates)
- Numbers now smoothly count up instead of jumping

---

#### US-2.4: Status Indicators ✅ COMPLETE
**As a** user  
**I want to** see health and error indicators  
**So that** I know when something goes wrong

**Acceptance Criteria:**
- [x] Health badges: ✅ Healthy, ⚠️ Warning, ❌ Error, ⏳ Initializing
- [x] Error counter per engine (shown inline)
- [x] Health calculation based on success rate, TTFT, recent errors
- [ ] Red flash on new error (deferred to Sprint 3)
- [x] Status icons displayed with each engine

**Story Points:** 3  
**Priority:** P1  
**Dependencies:** US-1.5  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/health.py` - Health monitoring system
- `HealthStatus` enum: HEALTHY, WARNING, ERROR, INITIALIZING
- `calculate_health_status()` - Smart health calculation
- Health badges: ✅ (healthy), ⚠️ (warning), ❌ (error), ⏳ (init)
- Color-coded by health status
- Error/warning counts tracked per engine
- Integrated into TUI with real-time updates

---

#### US-2.5: Time-Series Charts ✅ COMPLETE
**As a** user  
**I want to** see detailed metric charts  
**So that** I can understand performance over time

**Acceptance Criteria:**
- [x] Show TTFT/TPOT/Throughput charts
- [x] Multi-line chart (all engines overlaid)
- [x] Terminal-based rendering with plotext
- [x] Automatic scaling and theming
- [ ] Press `[c]` to open (requires keyboard shortcuts - US-2.7)
- [ ] Interactive zoom/pan (deferred to Sprint 3)
- [ ] Export as PNG (deferred to Sprint 3)

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-2.1  
**Status:** ✅ COMPLETE

**Implementation:**
- Created `llm_locust/race/charts.py` - Terminal chart rendering
- `render_metric_chart()` - Plot individual metrics  
- `show_charts_view()` - Full charts display
- Uses plotext for terminal-native charts
- Multi-engine overlays with legends
- Added plotext>=5.2.8 dependency
- Ready for keyboard shortcut integration (US-2.7)

---

#### US-2.6: Request Timeline View
**As a** user  
**I want to** see individual requests in flight  
**So that** I understand concurrency patterns

**Acceptance Criteria:**
- [ ] Press `[t]` to open timeline view
- [ ] Show last 50 requests as bars
- [ ] Color by status: green=success, red=error, yellow=slow
- [ ] Length = duration
- [ ] Stacked per engine
- [ ] Real-time updates (stream new requests)
- [ ] Click to see request details (if supported)

**Story Points:** 5  
**Priority:** P2  
**Dependencies:** US-1.4

---

#### US-2.7: Keyboard Shortcuts
**As a** user  
**I want to** control the race via keyboard  
**So that** I can interact efficiently

**Acceptance Criteria:**
- [ ] `[p]` - Pause/Resume race
- [ ] `[r]` - Restart race
- [ ] `[q]` - Quit
- [ ] `[h]` or `[?]` - Help overlay
- [ ] `[i]` - **Inspector view (live requests/responses)**
- [ ] `[d]` - Detailed view (drill-down)
- [ ] `[c]` - Charts view
- [ ] `[t]` - Timeline view
- [ ] `[s]` - Save snapshot
- [ ] Help overlay shows all shortcuts
- [ ] Footer bar shows key shortcuts

**Story Points:** 2  
**Priority:** P1  
**Dependencies:** US-1.3

---

### Sprint 2 Deliverables
- [ ] Sparklines for key metrics
- [ ] Smooth animations (60 FPS)
- [ ] 5 color themes
- [ ] Status indicators and error tracking
- [ ] Time-series chart view
- [ ] Keyboard controls and help
- [ ] Performance optimized (<1% overhead)
- [ ] Updated docs with screenshots

**Demo Script:**
```bash
# Sprint 2 Demo
llm-locust race --config demo-race.yaml --theme hacker

# Shows:
# - Sparklines in real-time
# - Smooth animations
# - Press 'c' for charts
# - Press 't' for timeline
# - Press '?' for help
# - Status indicators flashing on errors
```
---

#### US-2.3: Color Themes (MOVE THIS TO SPRINT 3)
**As a** user  
**I want to** choose a visual theme  
**So that** I can customize appearance

**Acceptance Criteria:**
- [ ] Built-in themes: Default, Dark, Light, Hacker, Retro
- [ ] CLI flag: `--theme dark`
- [ ] Environment variable: `LLM_LOCUST_THEME`
- [ ] Color-blind friendly palettes
- [ ] ASCII fallback for no-color terminals
- [ ] Themes defined in YAML (extensible)

**Story Points:** 2  
**Priority:** P2  
**Dependencies:** US-1.3

**Theme Example:**
```yaml
# themes/hacker.yaml
theme:
  name: "Hacker"
  colors:
    primary: "#00ff00"
    secondary: "#003300"
    accent: "#00ff00"
    background: "#000000"
    text: "#00ff00"
  style: "matrix"
```

---

## SPRINT 3: Education & Intelligence

**Theme:** "Teach, Don't Tell"  
**Goal:** Add live commentary, explanations, and intelligent insights  
**Story Points:** 34

### User Stories

#### US-3.1: Live Commentary System
**As a** user  
**I want to** see real-time explanations of what's happening  
**So that** I understand the race dynamics

**Acceptance Criteria:**
- [ ] Commentary panel (scrolling text area)
- [ ] Events: milestones, warnings, insights
- [ ] Templates for common events
- [ ] Color-coded by type (info/warning/error)
- [ ] Auto-scroll with pause on user scroll
- [ ] Configurable verbosity (--commentary quiet|normal|verbose)
- [ ] Max 100 messages in buffer (memory efficient)

**Story Points:** 5  
**Priority:** P0  
**Dependencies:** US-1.4

**Commentary Examples:**
```
🎙️  vLLM completed first 100 requests! (23 seconds)
💡  TGI's TTFT is spiking - possible memory pressure
⚠️  Ollama error rate increased to 2% - investigating...
🎉  Milestone: 1,000 total requests across all engines!
```

---

#### US-3.2: Metric Explainers
**As a** novice user  
**I want to** understand what metrics mean  
**So that** I can interpret results correctly

**Acceptance Criteria:**
- [ ] Press `[e]` on any metric to see explanation
- [ ] Explain: TTFT, TPOT, Throughput, RPS, P50/P90/P99
- [ ] Contextual explanations (e.g., "TTFT for chat should be <500ms")
- [ ] Simple language (avoid jargon)
- [ ] Links to documentation
- [ ] Examples of good vs bad values
- [ ] Close with any key

**Story Points:** 3  
**Priority:** P0  
**Dependencies:** US-2.7

**Explainer Example:**
```
┌─ 💡 METRIC EXPLAINED: Time to First Token (TTFT) ─────┐
│                                                        │
│ What is it?                                            │
│   The time from sending a request until receiving     │
│   the very first token of the response.                │
│                                                        │
│ Why does it matter?                                    │
│   TTFT directly impacts perceived responsiveness.     │
│   Users notice delays > 300ms as "sluggish".          │
│                                                        │
│ What's a good value?                                   │
│   • Excellent: < 200ms                                │
│   • Good:      200-500ms                              │
│   • Acceptable: 500-1000ms                            │
│   • Poor:      > 1000ms                               │
│                                                        │
│ Current value: 234ms (Excellent! ✅)                  │
│                                                        │
│ Learn more: docs.llm-locust.io/metrics/ttft          │
│                                                        │
│ [Press any key to close]                               │
└────────────────────────────────────────────────────────┘
```

---

#### US-3.3: Difficulty Modes
**As a** user with varying expertise  
**I want to** adjust the information density  
**So that** I see what's relevant to me

**Acceptance Criteria:**
- [ ] Three modes: Novice, Expert, Teacher
- [ ] CLI flag: `--mode novice|expert|teacher`
- [ ] **Novice:** Full explanations, tips, simplified metrics
- [ ] **Expert:** Raw numbers, no commentary, max density
- [ ] **Teacher:** Educational callouts, best practices, quizzes
- [ ] Switch modes mid-race with `[1]`, `[2]`, `[3]` keys
- [ ] Save preference in config file

**Story Points:** 3  
**Priority:** P1  
**Dependencies:** US-3.1, US-3.2

---

#### US-3.4: Anomaly Detection
**As a** user  
**I want to** be alerted to performance issues  
**So that** I can investigate problems

**Acceptance Criteria:**
- [ ] Detect: latency spikes (>50% increase)
- [ ] Detect: error rate increase (>1% per minute)
- [ ] Detect: throughput drops (>25% decrease)
- [ ] Detect: outlier requests (3σ from mean)
- [ ] Flash warning indicator
- [ ] Add to commentary with explanation
- [ ] Suggest possible causes
- [ ] Log to file for later analysis

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-3.1

**Anomaly Examples:**
```
⚠️  ANOMALY DETECTED: TGI TTFT spike
    • Previous avg: 280ms
    • Current:      450ms (+61%)
    • Possible causes:
      - Memory pressure (GPU memory > 90%)
      - Queue buildup (>10 pending requests)
      - Model loading/reloading
    • Recommendation: Check GPU memory usage
```

---

#### US-3.5: Performance Insights
**As a** user  
**I want to** get actionable recommendations  
**So that** I can optimize my deployments

**Acceptance Criteria:**
- [ ] Analyze patterns during race
- [ ] Generate insights at milestones (25%, 50%, 75%, 100%)
- [ ] Categories: Speed, Reliability, Cost, Efficiency
- [ ] Specific recommendations with reasons
- [ ] Prioritized by impact (high/medium/low)
- [ ] Save insights to summary report
- [ ] Link to relevant docs

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-3.1, US-3.4

**Insight Examples:**
```
💡 PERFORMANCE INSIGHT: vLLM Optimization Opportunity

   Your vLLM deployment is performing well, but:
   
   🔍 Observation:
      • GPU memory utilization: 60%
      • TTFT: 234ms (excellent)
      • Throughput: 45 tok/s
   
   💡 Recommendation:
      Increase --gpu-memory-utilization to 0.80 for:
      • +30% throughput potential
      • Still safe memory margin
      • No TTFT degradation expected
   
   📚 Learn more: docs.llm-locust.io/tuning/gpu-memory
   
   [Priority: MEDIUM] [Impact: +30% throughput]
```

---

#### US-3.6: Live Request/Response Inspector
**As a** user  
**I want to** see actual prompts and streaming responses in real-time  
**So that** I understand what's happening at the token level

**Acceptance Criteria:**
- [ ] Press `[i]` to open inspector view (live request viewer)
- [ ] Show last 5 active requests with live streaming
- [ ] Display: Prompt text (truncated to 200 chars)
- [ ] Display: Response text streaming token-by-token
- [ ] Highlight: First token arrival (TTFT moment)
- [ ] Show: Token counter incrementing
- [ ] Show: Timing annotations (0ms, 234ms, 456ms, etc.)
- [ ] Color-code: Input (blue), Output (green), Timing (yellow)
- [ ] Auto-scroll to follow active requests
- [ ] Click request to see full text in modal

**Story Points:** 8  
**Priority:** P0  
**Dependencies:** US-1.4, US-2.6

**Visual Mock:**
```
┌─ 🔍 LIVE REQUEST INSPECTOR ───────────────────────────────────┐
│                                                               │
│  Request #1234 | vLLM | Status: Streaming... | Elapsed: 2.3s│
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 📝 Prompt (256 tokens):                                  ││
│  │    "Explain the concept of neural networks in simple..." ││
│  │                                                           ││
│  │ 🎯 TTFT: 234ms ◄─ First token arrived here!            ││
│  │                                                           ││
│  │ 💬 Response (streaming):                                 ││
│  │    Neural networks are computational models inspired     ││
│  │    by the human brain. They consist of interconnected▓   ││
│  │    [29 tokens • 12ms/token • 2.3s elapsed]              ││
│  └──────────────────────────────────────────────────────────┘│
│                                                               │
│  Request #1235 | TGI | Status: Streaming... | Elapsed: 1.8s │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 📝 Prompt: "What is machine learning?"                   ││
│  │ 🎯 TTFT: 289ms                                           ││
│  │ 💬 Response: Machine learning is a subset of artificial▓││
│  │    [18 tokens • 15ms/token • 1.8s elapsed]              ││
│  └──────────────────────────────────────────────────────────┘│
│                                                               │
│  💡 LEARNING TIP:                                            │
│  Notice how TGI took longer to start (289ms TTFT) but       │
│  vLLM is generating faster (12ms vs 15ms TPOT)              │
│                                                               │
│  [←] Back  [p] Pause  [f] Full Text  [s] Save Sample        │
└───────────────────────────────────────────────────────────────┘
```

**Technical Implementation:**
```python
class RequestInspector:
    def __init__(self, metrics_queue: Queue):
        self.active_requests: Dict[str, StreamingRequest] = {}
        self.max_display = 5
    
    async def update(self):
        # Poll for token events from metrics queue
        while not metrics_queue.empty():
            event = metrics_queue.get_nowait()
            
            if event.type == "token_received":
                req = self.active_requests[event.request_id]
                req.append_token(event.token, event.timestamp)
                self.render_streaming_request(req)
            
            elif event.type == "request_complete":
                req = self.active_requests.pop(event.request_id)
                self.archive_request(req)
```

**Educational Value:**
- Users see **actual model behavior** not just numbers
- Understand **token-by-token generation** visually
- Connect **TTFT/TPOT metrics** to real streaming
- Observe **differences between engines** in real-time
- Learn **how prompts affect responses**

---

#### US-3.7: Interactive Drill-Downs
**As a** user  
**I want to** explore detailed metrics on demand  
**So that** I can investigate specific areas

**Acceptance Criteria:**
- [ ] Click/select any engine to drill down
- [ ] Show detailed dashboard for that engine
- [ ] Tabs: Overview, Latency, Throughput, Errors, Timeline
- [ ] Histograms for latency distributions
- [ ] Percentile tables (P50, P75, P90, P95, P99)
- [ ] Sample requests (success + failures)
- [ ] Back button to return to race view

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-2.5, US-2.6

---

### Sprint 3 Deliverables
- [ ] Live commentary system with 20+ event templates
- [ ] **Live request/response inspector** with streaming visualization
- [ ] Metric explainers for all key concepts
- [ ] Three difficulty modes (Novice, Expert, Teacher)
- [ ] Anomaly detection with alerts
- [ ] Performance insights engine
- [ ] Interactive drill-down views
- [ ] User testing with 5 beta testers
- [ ] Feedback incorporated

**Demo Script:**
```bash
# Sprint 3 Demo
llm-locust race --config demo-race.yaml --mode teacher

# Shows:
# - Live commentary explaining events
# - **Press 'i' to see live request inspector**
# - **Watch tokens streaming in real-time**
# - **See TTFT moment highlighted**
# - **Observe token-by-token generation**
# - Press 'e' on TTFT to see explanation
# - Anomaly detected and explained
# - Performance insights at 50% completion
# - Drill down into vLLM details
# - Switch to Expert mode mid-race
```

---

## SPRINT 4: Gamification & Sharing

**Theme:** "Share the Joy"  
**Goal:** Add achievements, replays, export, and sharing features  
**Story Points:** 23

### User Stories

#### US-4.1: Achievements System
**As a** user  
**I want to** unlock achievements  
**So that** I feel rewarded for exploration

**Acceptance Criteria:**
- [ ] 15+ achievements defined
- [ ] Categories: Speed, Reliability, Efficiency, Explorer
- [ ] Toast notification on unlock
- [ ] Achievement gallery (press `[a]`)
- [ ] Progress bars for partial achievements
- [ ] Save to local profile (`~/.llm-locust/profile.json`)
- [ ] Share achievements (Twitter, Slack)

**Story Points:** 5  
**Priority:** P2  
**Dependencies:** US-1.6

**Achievement Examples:**
```
🏆 ACHIEVEMENTS

✨ Speed Demon       - Maintain P50 TTFT < 200ms for 5 minutes
🎯 Consistency King  - Achieve σ < 20ms variance
💪 Throughput Beast  - Generate 10,000 tokens in 1 minute
🔥 Zero Failures     - Complete 1,000 requests with 0 errors
⚡ First Blood       - First to complete 100 requests
🎓 Quick Learner     - View 10 metric explanations
🌟 Completionist     - Unlock all achievements
```

---

#### US-4.2: Race Replays
**As a** user  
**I want to** record and replay races  
**So that** I can review and share results

**Acceptance Criteria:**
- [ ] CLI flag: `--save-replay`
- [ ] Save to JSON format with full event stream
- [ ] Replay command: `llm-locust replay results/race-xyz.json`
- [ ] Playback controls: play, pause, speed (0.5x - 10x)
- [ ] Jump to timestamps
- [ ] Export replay as GIF (optional, via terminalizer)
- [ ] Replay file size < 10MB for 10-min race

**Story Points:** 5  
**Priority:** P1  
**Dependencies:** US-1.6

**Replay Format:**
```json
{
  "race_id": "abc123",
  "timestamp": "2025-10-04T12:00:00Z",
  "config": {...},
  "events": [
    {"t": 0, "type": "start", "data": {...}},
    {"t": 1.2, "type": "request_complete", "engine": "vllm", ...},
    {"t": 2.5, "type": "anomaly_detected", "engine": "tgi", ...},
    ...
  ],
  "summary": {...}
}
```

---

#### US-4.3: Race Summary Cards
**As a** user  
**I want to** generate shareable summary images  
**So that** I can post results to social media

**Acceptance Criteria:**
- [ ] Generate PNG summary card at race end
- [ ] Beautiful design (branded, colorful)
- [ ] Show: winner, key metrics, podium, chart preview
- [ ] Export options: PNG, SVG, HTML
- [ ] Template system (customizable)
- [ ] Social media optimized sizes (Twitter, LinkedIn)
- [ ] Auto-copy sharing URL

**Story Points:** 3  
**Priority:** P2  
**Dependencies:** US-1.6

---

#### US-4.4: Export Formats
**As a** user  
**I want to** export results in multiple formats  
**So that** I can integrate with other tools

**Acceptance Criteria:**
- [ ] Export to: CSV, JSON, HTML, Markdown
- [ ] CLI flag: `--export-format csv,json,html`
- [ ] CSV: Compatible with Streamlit dashboard
- [ ] JSON: Full detail with metadata
- [ ] HTML: Interactive standalone report
- [ ] Markdown: GitHub-friendly summary
- [ ] All formats generated automatically

**Story Points:** 3  
**Priority:** P1  
**Dependencies:** US-1.6

---

#### US-4.5: Race Modifiers (Challenges)
**As a** power user  
**I want to** add difficulty modifiers  
**So that** I can test edge cases

**Acceptance Criteria:**
- [ ] `--sudden-death` - One error = elimination
- [ ] `--burst-mode` - Random traffic spikes (2x-5x)
- [ ] `--memory-limit X` - Restrict to X GB
- [ ] `--cost-challenge` - Optimize for $/1M tokens
- [ ] `--endurance` - 24-hour marathon mode
- [ ] `--chaos` - Random failures injected
- [ ] Modifiers affect scoring/achievements
- [ ] Shown in race config panel

**Story Points:** 3  
**Priority:** P2  
**Dependencies:** US-1.2

---

#### US-4.6: Cloud Sharing (Optional)
**As a** user  
**I want to** share races to a public URL  
**So that** others can view without installing

**Acceptance Criteria:**
- [ ] CLI flag: `--share`
- [ ] Upload replay to cloud storage
- [ ] Generate shareable URL: race.llm-locust.io/abc123
- [ ] Web viewer (read-only)
- [ ] Expire after 30 days (configurable)
- [ ] Privacy: public, unlisted, private
- [ ] Rate limiting (10 uploads/day)
- [ ] Optional feature (requires API key)

**Story Points:** 5  
**Priority:** P3 (Nice-to-have)  
**Dependencies:** US-4.2

---

#### US-4.7: Documentation Polish
**As a** new user  
**I want to** comprehensive documentation  
**So that** I can use all features

**Acceptance Criteria:**
- [ ] Quick Start guide (5 min to first race)
- [ ] CLI reference (all commands/flags)
- [ ] Race configuration guide (YAML reference)
- [ ] Metric glossary (all metrics explained)
- [ ] Best practices guide
- [ ] Troubleshooting FAQ
- [ ] Video tutorial (5-10 min)
- [ ] Screenshots and GIFs throughout

**Story Points:** 3  
**Priority:** P0  
**Dependencies:** All previous stories

---

### Sprint 4 Deliverables
- [ ] Achievements system with 15 achievements
- [ ] Race replay and playback
- [ ] Summary card generation
- [ ] Multi-format export (CSV, JSON, HTML, MD)
- [ ] 5 race modifiers
- [ ] Complete documentation
- [ ] Video tutorial
- [ ] Beta program complete (feedback incorporated)

**Demo Script:**
```bash
# Sprint 4 Demo (Final!)
llm-locust race --config demo-race.yaml --save-replay --sudden-death

# Shows:
# - Full race with all features
# - Achievement unlocked mid-race
# - Race summary with export options
# - Replay the race
# - Generate summary card
# - Share to Twitter

# Final result: Production-ready interactive CLI! 🎉
```

---

## 📊 Story Point Reference

| Points | Complexity | Time Estimate | Examples |
|--------|-----------|---------------|----------|
| 1 | Trivial | 1-2 hours | Config change, simple UI tweak |
| 2 | Simple | 2-4 hours | New color theme, keyboard shortcut |
| 3 | Moderate | 4-8 hours | Metric explainer, export format |
| 5 | Complex | 1-2 days | Sparklines, commentary system |
| 8 | Very Complex | 2-4 days | Multi-process architecture |
| 13 | Extremely Complex | 1 week | Cloud sharing platform |

---

## 🎯 Sprint Goals & Metrics

### Sprint 1 Success Criteria
- [ ] Demo complete: 2+ endpoints racing with live UI
- [ ] Performance: <50ms render latency
- [ ] Quality: Zero visual glitches
- [ ] Test coverage: >80% for core components

### Sprint 2 Success Criteria
- [ ] Demo complete: All visual features working
- [ ] Performance: 60 FPS animations
- [ ] Quality: Polished, professional look
- [ ] User feedback: "Looks awesome!"

### Sprint 3 Success Criteria
- [ ] Demo complete: Full educational experience
- [ ] User testing: 70% understand metrics without external docs
- [ ] Quality: Helpful, not annoying commentary
- [ ] Feedback: "I learned so much!"

### Sprint 4 Success Criteria
- [ ] Demo complete: Full feature set
- [ ] Beta program: 10 users testing
- [ ] Quality: Zero P0 bugs
- [ ] Feedback: "Ready for release!"

---

## 🏃 Velocity Tracking

### Historical Velocity (to be updated)
- Sprint 1: __ points completed
- Sprint 2: __ points completed
- Sprint 3: __ points completed
- Sprint 4: __ points completed

### Velocity Adjustment
- Start with planned 21-26 points per sprint
- Adjust after Sprint 1 based on actual velocity
- Account for holidays, PTO, external dependencies

---

## 🎨 Technical Debt Management

### Debt Accepted (Ship Fast)
- Basic error handling (improve in v0.6.0)
- Limited theme customization
- Manual testing heavy (automate later)
- Performance optimization deferred

### Debt Unacceptable (Pay Now)
- Memory leaks (fix immediately)
- Data races in IPC (fix before merge)
- Security issues (fix before beta)
- Breaking API changes (plan carefully)

---

## 🔧 Development Setup

### Prerequisites
```bash
# Python 3.11+
python --version

# Install dependencies
pip install -e ".[dev]"

# Install TUI frameworks
pip install textual rich plotext

# Install testing tools
pip install pytest pytest-asyncio pytest-cov
```

### Running Tests
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# TUI tests (requires textual-dev)
pytest tests/tui/

# Coverage
pytest --cov=llm_locust --cov-report=html
```

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/US-1.1-race-config

# Make changes, test locally
python -m llm_locust.cli.race --config test.yaml

# Run linters
ruff check .
mypy llm_locust/

# Commit with story ID
git commit -m "feat: Add race config schema (US-1.1)"

# Push and create PR
git push origin feature/US-1.1-race-config
```

---

## 📝 Definition of Ready (DoR)

User stories are ready for sprint when:
- [ ] Acceptance criteria are clear and testable
- [ ] Story is estimated (story points assigned)
- [ ] Dependencies identified and resolved
- [ ] Technical design discussed (if needed)
- [ ] No blockers
- [ ] Team understands and accepts story

---

## ✅ Definition of Done (DoD)

User stories are done when:
- [ ] Code complete and committed
- [ ] Unit tests written (>80% coverage)
- [ ] Integration tests written (where applicable)
- [ ] Linter passes (ruff, mypy)
- [ ] Code reviewed and approved
- [ ] Acceptance criteria met (demo-able)
- [ ] Documentation updated
- [ ] Merged to `main` branch
- [ ] No new bugs introduced

---

## 🔮 Future Roadmap (Post-v0.5.0)

### v0.6.0 (Month 3)
- Multi-user races (compete with others)
- Historical tracking (performance over time)
- CI/CD integration (auto-race on deploy)
- Advanced analytics dashboard

### v0.7.0 (Month 4)
- Web version (browser-based viewer)
- Mobile app (watch races on phone)
- AI commentary (GPT-powered announcer)
- Team leaderboards

### v1.0.0 (Month 6)
- Cloud service (hosted racing platform)
- Enterprise features (SSO, RBAC)
- Custom metrics plugins
- Production stability guarantees

---

## 📚 References

- [Textual Documentation](https://textual.textualize.io/)
- [Rich Documentation](https://rich.readthedocs.io/)
