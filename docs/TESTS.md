### **Benchmark Test 1a: Chat Simulation (256 input / 128 output tokens)**

* **Objective:**
  Evaluate system performance under short, interactive workloads representative of conversational AI.

* **Workload Profile:**

  * **Input tokens:** ~256 per request
  * **Output tokens:** ~128 per request
  * **Interaction type:** Compact prompts and concise responses, mimicking natural back-and-forth dialogue.

* **Test Parameters:**

  * **Duration:** 5–10 minutes (long enough to measure stability and response consistency).
  * **Concurrency:** ~50 parallel chat sessions.
  * **Rate:** Steady conversational pace (1–2 requests per user per minute).
  * **Number of Users Simulated:** Dozens of customer or assistant interactions in parallel.

* **Benchmark Focus:**

  * **Latency Sensitivity:** *Time-to-first-token (TTFT)* and *p99 latency* as indicators of responsiveness.
  * **Throughput:** Ability to sustain dozens of interactive conversations simultaneously.
  * **User Experience Impact:** Ensures responses remain conversational (<1s median, <2s p99).

* **Business Context:**
  Customer-facing assistants, support bots, or copilots where responsiveness is critical for usability and adoption.

---

### **Benchmark Test 1b: RAG Simulation (4096 input / 512 output tokens)**

* **Objective:**
  Assess performance when handling large input contexts and longer responses typical of retrieval-augmented generation (RAG) systems.

* **Workload Profile:**

  * **Input tokens:** ~4096 per request
  * **Output tokens:** ~512 per request
  * **Interaction type:** Long-form context ingestion with detailed answers.

* **Test Parameters:**

  * **Duration:** 10–15 minutes (longer runs needed for large context processing).
  * **Concurrency:** ~20 parallel sessions.
  * **Rate:** Moderate, with bursts representing multiple users querying documents simultaneously.
  * **Number of Users Simulated:** Enterprise-scale workloads, such as analysts querying knowledge bases.

* **Benchmark Focus:**

  * **Memory Load:** Stress-test KV cache growth and GPU memory usage.
  * **Latency Distribution:** Observe how latency scales with large token counts.
  * **Throughput Impact:** Identify drop-offs as request size increases.

* **Business Context:**
  Knowledge-base assistants, research copilots, or enterprise search systems requiring context-heavy queries.

---

### **Benchmark Test 1c: Code Generation Simulation (512 input / 512 output tokens)**

* **Objective:**
  Benchmark balanced input-output scenarios common in development assistance and code generation.

* **Workload Profile:**

  * **Input tokens:** ~512 per request
  * **Output tokens:** ~512 per request
  * **Interaction type:** Medium-sized prompts with equally long completions.

* **Test Parameters:**

  * **Duration:** 5–10 minutes.
  * **Concurrency:** ~30 developer sessions.
  * **Rate:** Constant flow of requests, reflecting active programming cycles.
  * **Number of Users Simulated:** Teams of developers using AI assistants concurrently.

* **Benchmark Focus:**

  * **Balanced Load:** Measures efficiency when both prompt parsing and response generation are significant.
  * **Latency:** Focus on *median* and *tail latencies* for developer workflow smoothness.
  * **Throughput:** Can the system sustain multiple code completions in parallel?

* **Business Context:**
  AI-powered coding copilots, auto-completion engines, or dev tool integrations where balanced input/output is typical.

---

### **Benchmark Test 2a: Constant Rate (Sustained Load)**

* **Objective:**
  Validate system reliability and performance under continuous, predictable workloads.

* **Workload Profile:**

  * **Input tokens:** ~512 per request
  * **Output tokens:** ~256 per request
  * **Interaction type:** Steady production-like traffic flow.

* **Test Parameters:**

  * **Duration:** 15–20 minutes (to reveal long-term degradation trends).
  * **Concurrency:** ~40 concurrent streams.
  * **Rate:** Fixed at ~2 requests/second across all users.
  * **Number of Users Simulated:** Dozens of sustained user sessions.

* **Benchmark Focus:**

  * **Sustained Performance:** Identify whether latency degrades over time.
  * **Stability:** Measure throughput consistency and error rates.
  * **SLA Readiness:** Ensures performance guarantees can be met under steady load.

* **Business Context:**
  Enterprise deployments with predictable usage patterns, such as internal productivity copilots or workflow automation tools.

---

### **Benchmark Test 2b: Poisson Rate (Bursty Traffic)**

* **Objective:**
  Evaluate system robustness under irregular, unpredictable bursts of traffic.

* **Workload Profile:**

  * **Input tokens:** ~512 per request
  * **Output tokens:** ~256 per request
  * **Interaction type:** Requests arrive in sudden spikes, modeled with Poisson distribution.

* **Test Parameters:**

  * **Duration:** 10–15 minutes (to capture multiple burst cycles).
  * **Concurrency:** Varies dynamically with traffic spikes.
  * **Rate:** Average ~2 requests/second, with unpredictable peaks above baseline.
  * **Number of Users Simulated:** Dozens to hundreds depending on burst profile.

* **Benchmark Focus:**

  * **Autoscaling:** Tests system’s ability to allocate resources dynamically.
  * **Queueing & Batching:** Reveals how the system manages traffic spikes.
  * **Tail Latency:** Identifies user experience risks under peak load.

* **Business Context:**
  Real-world enterprise apps with spiky traffic, such as e-commerce assistants during flash sales, or knowledge tools during peak work hours.

