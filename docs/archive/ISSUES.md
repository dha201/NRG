# NRG Legislative Intelligence - Issues Report

**Generated:** Jan 8, 2026  

---

## 🔴 CRITICAL

### 1. Missing Bill-Level Change Analysis (Path A)
**Location:** `poc 2.py:3963-3966`

**Problem:** Bills with existing versions skip metadata change analysis when only metadata updates (no new versions).

**Occurs when:** 
- Bill already has versions cached
- Only metadata changes: summary updated, status changed
- System takes Path A (version tracking) but doesn't analyze metadata changes

**Impact:** Missing change detection for status transitions (Committee → Passed → Signed).

**Fix:** Add metadata change analysis before version loop in Path A.

---

### 2. State vs Federal Bill Version Terminology
**Location:** `poc 2.py:1183, 1371-1373`

**Problem:** JSON schema forces federal terminology for state bills.

**Example:**
```json
{
  "bill_version": "passed_house"  // LLM guesses federal term
  // Actual: "Engrossed" (Texas state term)
}
```

**Mapping gap:**
| State Term | Federal Equivalent | LLM Output |
|---|---|---|
| Engrossed | varies by chamber | passed_house ❌ |
| House Committee Report | house_committee | ? |

**Status:** May be resolved by custom prompt fix (Bug #2 in RUNTIME_ISSUES). Needs testing.

**Options:**
1. Override: Force `analysis['bill_version'] = version_type`
2. Expand schema with state terms
3. Pre-map state → federal before LLM

---

### 3. Version Clarification Needed
**Location:** `poc 2.py:332`

**TODO:** "Each version represents potential changes affecting NRG's operations. TODO: HOW?"

**Needs:** Document how version changes map to operational impact (regulatory compliance, cost implications, operational restrictions).

---

## 🚨 PRODUCTION BLOCKERS (from CHECKLIST)

### 4. Memory Exhaustion - Large Data
**Locations:** Lines 277-280, 1082-1084, 3182-3191

**Issues:**
- Full bill text (100KB-5MB) loaded into memory
- PDF extraction (500KB-10MB) accumulates strings
- SQLite TEXT columns store multi-MB content

**Risk:** 50+ bills = 50-250MB memory. OOM in containers.

**Fix:**
1. Stream large text to temp files
2. Use `''.join(list)` not `+=` for strings
3. Store full text in blob storage (S3), keep hashes in DB
4. Migrate to PostgreSQL with TOAST

---

### 5. No Rate Limiting
**Locations:** Line 311 (Congress), missing elsewhere

**Current:** Fixed 0.5s sleep only for Congress API. None for OpenStates, Gemini, OpenAI.

**API Limits:**
- Congress: 5,000/hr
- Regulations: 1,000/hr
- OpenStates: 500/day (free)
- Gemini: 360/min (paid)

**Impact:** 429 errors → silent data loss (fail-soft returns `[]`).

**Fix:** Token bucket algorithm per API with exponential backoff.

---

### 6. No LLM Retry Logic
**Locations:** Lines 1270, 1454

**Problem:** Single try → permanent failure on transient errors.

```python
except Exception as e:
    return {"error": str(e), "business_impact_score": 0}  # No retry
```

**Impact:** Wasted analyses, manual re-runs required.

**Fix:** Add `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`.

---

### 7. No Cost Tracking
**Location:** Entire codebase

**Problem:** Zero visibility into LLM token usage or costs.

**Estimated costs (unchecked):**
- 1,000 bills/month: $50-200
- With versions (3x-5x): $150-1,000/month

**Fix:** 
1. Track tokens per request
2. Real-time cost calculation
3. Budget circuit breaker
4. Cache responses (save 60%)

---

### 8. Synchronous LLM Calls
**Location:** `main()` lines 3814-3919

**Problem:** Sequential blocking calls.

**Impact:** 20 bills × 5 sec = 100 sec. With versions: 300 sec.

**Fix:** `asyncio.gather()` for concurrent LLM calls. 100s → 10s.

---

### 9. SQLite Not Production-Ready
**Location:** Line 3048

**Limitations:**
- Single writer lock
- No connection pooling
- No replication/HA
- File corruption risk
- Cannot scale horizontally

**Fix:** Migrate to PostgreSQL with connection pooling.

---

### 10. Missing Database Indexes
**Locations:** Lines 3130-3132

**Missing indexes:**
- `bills(text_hash)` - change detection
- `bills(last_checked)` - cleanup queries
- `bill_changes(change_detected_at)` - recent changes
- `bills(source, bill_number)` - deduplication

**Impact:** Full table scans on 1,000+ bills. 50ms → 5,000ms.

---

### 11. No Transaction Management
**Location:** Line 3317, 3922

**Problem:** Individual commits per bill (20 bills = 20 disk syncs).

**Impact:** 100x slower than batch transaction. Partial data on crash.

**Fix:** Wrap in `with db_conn:` transaction.

---

### 12. Silent Failures (Fail-Soft)
**Locations:** Lines 318, 568, 698, 1122

```python
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    return []  # Silent failure
```

**Impact:** API failures unnoticed, no alerts, no structured logs.

**Fix:** Structured logging + Sentry. Raise exceptions, don't swallow.

---

### 13. API Keys in Plain Text
**Location:** `.env` file

**Problem:** Keys in version control, no rotation, single point of compromise.

**Fix:** AWS Secrets Manager / Azure Key Vault. Rotate every 90 days.

---

### 14. No Input Validation
**Location:** API response handling

**Problem:** Direct use of API responses without validation.

**Impact:** Malformed data crashes system, JSON injection risks.

**Fix:** Add Pydantic models with validators.

---

## 🟠 HIGH PRIORITY

### 15. LLM Cached Analysis Not Used
**Location:** `poc 2.py:3952-3956`

**TODO:** "Implement cached version analysis retrieval"

**Problem:** Re-analyzing unchanged versions on every run.

**Impact:** Wasted LLM costs, API limits.

**Fix:** Check `get_version_analysis()` before calling `analyze_bill_version()`.

---

### 16. LLM Error Handling Silent Degradation
**Location:** `poc 2.py:3957-3961`

**NOTE:** "analyze_bill_version() returns error dict, execution continues with degraded data"

**Risk:** If all versions fail, `version_analyses = []` and `final analysis = {}`.

**Current behavior:** Silent degradation, no alerts.

---

### 17. IndexError Risk in Version Comparison
**Location:** `poc 2.py:4002-4008`

**NOTE:** "Assumes version_analyses has same length as versions processed"

**Risk:** If previous version analysis fails silently and doesn't append → IndexError.

**Current:** Safe because error dict always returned and appended. But fragile.

---

### 18. JSON Parsing Errors - Version Diffs
**Location:** `analyze_version_changes_with_llm()`

**Frequency:** 2-3 failures per run

**Errors:**
- `JSON object must be str, bytes or bytearray, not NoneType`
- `Unterminated string starting at: line 60 column 25`

**Impact:** Version-to-version change analysis lost.

**Fix:** Retry logic + log full LLM response for debugging.

---

### 19. Status Shows "Unknown" for Bills
**Location:** Report generation

**Example:** HB 4238 shows `Status: Unknown`

**Root cause:** Open States status field not mapped correctly in `fetch_specific_texas_bills()` (lines 700-1100).

---

### 20. Congress.gov Returns 0 Bills
**Observed:** Both runs returned 0 energy bills

**Possible causes:**
- Silent API failure (fail-soft returns `[]`)
- No new bills matching keywords
- Filter too restrictive

**Action:** Manual verification if API working.

---

### 21. High-Impact Items Aging Out
**Location:** Regulations.gov limit=3

**Issue:** New items push out older high-impact ones.

**Example:** EERE_FRDOC_0001-2165 (score 7/10) present in Run 1, missing in Run 2.

**Fix:** Archive items with `impact_score ≥ 7` to persistent storage.

---

## 🟡 MEDIUM PRIORITY

### 22. Monolithic 4,147-Line File
**Location:** Entire `poc 2.py`

**Problem:** Single file, 40+ functions, tight coupling, untestable.

**Fix:** Refactor to modules:
```
src/
  ├── api/
  ├── analysis/
  ├── storage/
  └── models/
```

---

### 23. No Unit Tests
**Location:** None exist

**Impact:** Cannot verify correctness or prevent regressions.

---

### 24. Global State - LLM Clients
**Locations:** Lines 132-133

```python
openai_client = OpenAI(...)  # Global singleton
gemini_client = genai.Client(...)
```

**Impact:** Cannot mock for testing, hard to swap implementations.

**Fix:** Dependency injection pattern.

---

### 25. Hard-Coded Values
**Examples:**
- `timeout=30.0`
- `"per_page": 20`
- `text[:20000]` (magic number)

**Fix:** Move to `config.yaml`.

---

### 26. No Full-Text Search
**Location:** Database design

**Problem:** Must load all bills to search. O(n) complexity.

**Fix:** PostgreSQL GIN index with `to_tsvector`.

---

### 27. No Semantic Search
**Location:** None

**Problem:** Cannot find similar bills or related legislation.

**Fix:** Generate embeddings, store in pgvector, similarity search.

---

### 28. No Caching Layer
**Location:** None

**Problem:** Repeated fetches of same data.

**Fix:** Redis cache with TTL.

---

### 29. No Async/Await for I/O
**Location:** All API calls use sync `httpx.Client`

**Fix:** Switch to `httpx.AsyncClient` + `asyncio.gather()`.

---

### 30. No Job Queue
**Location:** Main execution synchronous

**Fix:** Celery for background processing.

---

### 31. No Data Retention Policy
**Location:** Database schema

**Problem:** Data accumulates indefinitely. DB grows to 10GB+ after 6 months.

**Fix:** Archive bills >90 days, delete analyses >1 year, compress old versions.

---

### 32. No Health Checks
**Location:** None

**Fix:** Add `/health` endpoint checking DB, APIs, LLM availability, disk space.

---

### 33. No Performance Monitoring
**Location:** None

**Fix:** Prometheus metrics for API latency, LLM duration, memory usage.

---

### 34. No Circuit Breaker
**Location:** All API calls

**Problem:** System hammers failed APIs instead of backing off.

**Fix:** Three-state circuit breaker (CLOSED/OPEN/HALF-OPEN).

---

### 35. No Duplicate Bill Handling
**Location:** Data fetching

**Problem:** Same bill from multiple sources creates duplicates.

**Fix:** Deduplicate by `(bill_number, source)` key.

---

### 36. No Timeout Handling
**Location:** PDF extraction, LLM calls

**Problem:** Can hang indefinitely.

**Fix:** Add timeout context manager with signal.

---

## 📋 RESOLVED ISSUES (from RUNTIME_ISSUES)

### ✅ Wrong Version Index Selection (FIXED)
**Location:** Line 4032 (formerly 4023)

**Was:** `analysis = version_analyses[-1]` (got oldest)  
**Now:** `analysis = version_analyses[0]` (gets newest)

---

### ✅ Custom Prompt Never Used (FIXED)
**Location:** Lines 1591-1639

**Was:** Built `prompt_text` but never passed to LLM  
**Now:** Added `custom_prompt` parameter, version context reaches LLM

---

## 📊 SUMMARY

**Total Issues:** 36 (2 resolved, 34 active)

**By Priority:**
- 🔴 Critical: 3
- 🚨 Production Blockers: 11
- 🟠 High: 9
- 🟡 Medium: 11

**Quick Wins (implement first):**
1. Add rate limiting decorators (lines 867-891 in CHECKLIST)
2. Add retry logic with exponential backoff
3. Implement cost tracking class (lines 904-921 in CHECKLIST)
4. Add response validation function (lines 895-901 in CHECKLIST)
5. Wrap DB operations in transactions

**Cost Impact:**
- Current: $50-200/month (POC)
- Production (unoptimized): $2,280-4,670/month
- Production (optimized): $780-3,170/month

**Timeline:** 2-3 months for production-ready with testing/staging.

---

**Next Steps:**
1. Fix critical bugs (1-3)
2. Implement quick wins (1 week)
3. Address production blockers (1 month)
4. Refactor architecture (2-3 months)
