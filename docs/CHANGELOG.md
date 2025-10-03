# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **[CRITICAL] Failed requests now logged to CSV** - Fixed major issue where failed requests (HTTP errors, timeouts, etc.) were not being written to CSV output files. This prevented accurate success rate calculations and failure pattern analysis.
  - Added `user_id` field to `RequestFailureLog` model
  - Updated `PerRequestLogger.log_request()` to handle both success and failure logs
  - Failed requests are logged with `status_code` ≠ 200 and zero values for token/latency metrics
  - Console output now shows failures with `[FAILURE] ❌` prefix and error details
  - This fix enables proper failure tracking in the Streamlit dashboard

### Technical Details
**Files Changed:**
- `llm_locust/core/models.py` - Added `user_id` to `RequestFailureLog`
- `llm_locust/core/user.py` - Include `user_id` when logging failures
- `llm_locust/metrics/per_request_logger.py` - Handle both success/failure in `log_request()`
- `llm_locust/metrics/collector.py` - Pass all request logs to per_request_logger

**Impact:**
- Benchmark CSV files now contain complete request history
- Success rate can be accurately calculated: `count(status_code=200) / count(*)`
- Failure patterns are visible in temporal analysis
- Dashboard can show error rate over time and by user

## [0.2.0] - 2025-10-03

### Added
- Per-request logging to CSV with full metrics
- Support for multiple benchmark types
- Comprehensive documentation (ARCHITECTURE, BENCHMARKS, METRICS guides)
- ShareGPT and Databricks Dolly dataset loaders

### Changed
- Refactored metrics collection architecture
- Improved error handling and logging

## [0.1.0] - Initial Release

### Added
- Basic load testing framework
- OpenAI-compatible client
- TTFT and TPOT metrics
- Multi-process architecture

