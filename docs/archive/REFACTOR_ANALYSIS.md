# NRG Legislative Tracker - Refactoring Analysis

**Date:** January 12, 2026  

---

## Decisions Summary

| Item | Decision |
|------|----------|
| Version terminology | Keep source-specific + add normalization layer |
| LegiScan code | Comment out, keep for future |
| Module split | **Option B** - api/, analysis/, reports/, db/, main.py |
| Testing | Data-centric tests + LLM eval framework |
| Azure Functions | Preserve current function_app.py |
| Error behavior | Retry with exponential backoff |
| Version caching | Hash-based cache check before LLM calls |
| Concurrency | SQLite locking (defer deployment design) |
| LLM cost tracking | Implement |
| Rate limiting | Placeholder with configurable values |
| Key Vault | Defer, add comments |
| PDF OCR | Detect if possible, otherwise placeholder |

---

## Current State Assessment

### 1. Major Logic Errors Found

#### **CRITICAL: Path A Missing Change Analysis (Lines 3992-4011, 4076)**
- **Problem:** When a bill has versions (`has_versions=True`) but only metadata changes (status update, no new version), Path A re-analyzes old versions but sets `change_impact = None`
- **Impact:** Misses critical intermediate signals like "Passed Committee" status changes
- **Example:** 
  - Day 1: Fetch bill with "Introduced" version → analyze
  - Day 5: Status changes "In Committee" → "Passed Committee", but no new version
  - Path A re-analyzes old "Introduced" version, ignores status change
- **Fix:** Add conditional to analyze bill-level changes even when versions exist

#### **CRITICAL: Version Analysis Cache Not Used (Lines 3981-3984)**
- **Problem:** `analyze_bill_version()` is always called, even for cached versions
- **Impact:** Wastes LLM API costs on every run for unchanged versions
- **Location:** Line 4012 - calls `analyze_bill_version()` without checking `get_version_analysis()` first
- **Fix:** Hash-based cache check: compute hash → query cache → use cached or call LLM

#### **HIGH: Bill Version Terminology Mismatch (Lines 1187, 1375-1377)**
- **Problem:** `bill_version` enum uses Federal terminology ("as_filed", "house_committee", etc.) but Open States returns state-specific terms ("Introduced", "Enrolled")
- **Impact:** LLM may return inconsistent/incorrect version labels in reports
- **Fix:** Keep source labels in `version_type_raw`, add `version_type_normalized`

#### **MEDIUM: Duplicate LLM Prompt (Lines 1188-1281 and 1378-1471)**
- **Problem:** 94-line identical prompt duplicated in `analyze_with_openai()` and `analyze_with_gemini()`
- **Impact:** Maintenance burden, risk of drift between providers
- **Fix:** Extract to shared constant in analysis/prompts.py

#### **MEDIUM: Empty `versions` When PDF Extraction Fails (Line 3298-3299)**
- **Problem:** If all PDFs fail to extract, `versions = []` but code proceeds
- **Impact:** `version_analyses` becomes empty, `analysis = {}` at line 4067
- **Fix:** Add fallback to use bill summary when no versions extracted

#### **LOW: Hardcoded Congress Number (Lines 216, 236, 259, 265)**
- **Problem:** `118` hardcoded for Congress.gov API calls
- **Impact:** Will break when 119th Congress convenes
- **Fix:** Make configurable or compute dynamically

#### **LOW: Bare `except` with `pass` (Lines 684, 800)**
- **Problem:** Silent failure on base64 decode errors in LegiScan
- **Impact:** No visibility into failures
- **Fix:** Log warning on decode failure

---

### 2. Architectural Issues

#### **Monolithic Structure**
The file contains 6 distinct functional domains mixed together:
1. **Data Fetching** (Lines 165-1131) - 4 API clients
2. **LLM Analysis** (Lines 1138-1541) - 2 providers + routing
3. **Version Tracking** (Lines 1543-1816) - Version comparison logic
4. **Report Generation** (Lines 1819-3012) - Console, MD, DOCX output
5. **Database/Caching** (Lines 3015-3659) - SQLite operations
6. **Main Orchestration** (Lines 3662-4176) - Pipeline coordination

#### **Tight Coupling Issues**
- Global `console`, `config`, `openai_client`, `gemini_client` at module level (lines 126-143)
- Functions directly access globals instead of receiving dependencies
- Makes testing impossible without mocking module state

#### **Code Duplication**
| Location | Description | Lines |
|----------|-------------|-------|
| Lines 2040-2331, 2333-2624, 2626-2916 | Markdown report sections (HIGH/MED/LOW) nearly identical | ~900 lines |
| Lines 1188-1281, 1378-1471 | LLM prompts duplicated | ~180 lines |
| Lines 905-1009, 1039-1114 | Open States bill processing duplicated | ~160 lines |

---

### 3. Production Readiness Gaps

#### **Error Handling**
- ✅ API calls have try/except at boundaries (good)
- ❌ No structured error logging (uses `console.print` only)
- ❌ No error aggregation or alerting mechanism
- ❌ LLM failures return minimal dict, may produce empty reports

#### **Configuration**
- ✅ Uses config.yaml for settings (good)
- ❌ No config validation at startup
- ❌ Hardcoded fallbacks scattered throughout code
- ❌ API keys loaded without validation (silent failures)

#### **Logging**
- ❌ Uses Rich console output only (not structured logs)
- ❌ No log levels (debug/info/warn/error)
- ❌ No request/response logging for debugging

#### **Testing**
- ❌ No test files present
- ❌ Functions tightly coupled to external services
- ❌ No dependency injection for mocking

#### **Type Safety**
- ❌ No type hints on any functions
- ❌ Dictionary access without validation throughout
- ❌ Implicit assumptions about API response shapes

---

### 4. Code Smells

| Smell | Location | Description |
|-------|----------|-------------|
| Magic Numbers | Lines 214, 220, 513 | `60` days, `20` limit, etc. |
| Long Functions | `main()` 488 lines, `generate_markdown_report()` 965 lines | Hard to test/maintain |
| Import inside function | Lines 315, 1023, 3194 | `import time`, `import io` |
| Inconsistent naming | `nrg_context` vs `NRG_CONTEXT`, `bill_text` vs `full_bill_text` | |
| Dead code | Lines 95 (commented import), Line 16 `DEPRECATED` LegiScan | |

---

## Approved Refactor Plan

### Phase 1: Critical Bug Fixes (No structural changes)
1. **Fix Path A change analysis bug** - Add change_impact analysis when versions exist
2. **Add hash-based version caching** - Check cache before LLM call
3. **Add retry with exponential backoff** for LLM failures
   - *Reasoning:* LLM APIs have transient failures; retry prevents data loss while backoff respects rate limits
4. **Validate config at startup** - Fail fast on missing API keys

### Phase 2: Module Split (Option B)
Target structure:
```
nrg_core/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── congress.py        # Congress.gov client
│   ├── regulations.py     # Regulations.gov client
│   ├── openstates.py      # Open States client
│   └── legiscan.py        # COMMENTED OUT - kept for future
├── analysis/
│   ├── __init__.py
│   ├── llm.py             # LLM routing + retry logic
│   ├── prompts.py         # Shared prompt constants
│   ├── versions.py        # Version tracking + normalization
│   └── changes.py         # Change detection
├── reports/
│   ├── __init__.py
│   ├── console.py         # Rich console output
│   ├── markdown.py        # MD report generation (deduplicated)
│   └── docx.py            # Word conversion
├── db/
│   ├── __init__.py
│   └── cache.py           # SQLite operations + locking
├── models.py              # Dataclasses for Bill, Analysis, Version
└── utils.py               # Hash, PDF extraction, rate limiting
```
(1. Extract LLM prompt to single constant
2. Extract markdown section generator (dedupe HIGH/MED/LOW sections)
3. Extract Open States bill processing to single function
1. Add type hints to public functions
2. Add dataclasses for Bill, Analysis, Version schemas
3. Validate API responses against expected shapes
1. Add structured logging (Python logging module)
2. Add config validation
3. Add basic integration tests
)

### Phase 3: Production Hardening
1. **LLM cost tracking** - Log token counts and estimated costs
2. **Rate limiting placeholder** - Configurable per-API rate limits
3. **PDF extraction detection** - Detect image-only PDFs, add placeholder for OCR
4. **Key Vault comments** - Add TODO comments for Azure Key Vault integration
5. **Version terminology normalization** - Map source labels to normalized terms

### Phase 4: Testing + Eval
1. **Unit tests** for pure logic (hash, diff, normalization, version mapping)
2. **Integration tests** with mocked API responses
3. **LLM eval framework** based on existing LLM_MODEL_COMPARISON.md template
   - Test fixtures with known ground-truth scores
   - Automated comparison between model outputs

---

## What Will NOT Be Changed

1. **Overall pipeline architecture** - Works correctly, just needs cleanup
2. **API response handling** - Functionally correct
3. **Database schema** - Already suitable for production
4. **Output format** - Reports are well-structured
5. **Config file structure** - Already flexible
6. **LLM prompt content** - Business logic, not code issue
7. **function_app.py** - Azure Functions integration preserved

---

## Implementation Order

| Priority | Task | Estimated Lines Changed |
|----------|------|------------------------|
| 1 | Fix Path A bug (lines 3992-4076) | ~20 |
| 2 | Add version hash caching (lines 3981-4027) | ~30 |
| 3 | Add LLM retry with backoff | ~40 |
| 4 | Extract module structure | ~200 (moves, not new) |
| 5 | Deduplicate markdown sections | -600 |
| 6 | Extract shared LLM prompt | -90 |
| 7 | Add cost tracking | ~50 |
| 8 | Add version normalization layer | ~60 |
| 9 | Add tests + eval framework | ~300 (new files) |

**Net result:** ~3000 lines (down from 4176) + ~300 lines tests

---

## Design Decisions Log

### Retry with Exponential Backoff (vs other options)
- **Continue with degraded data:** Risks empty reports if LLM fails
- **Fail entire run:** Too aggressive for transient failures
- **Retry with backoff:** Best balance - recovers from transient failures, respects rate limits, eventually fails gracefully if persistent issue

### Hash-based Version Caching
- Compare `text_hash` of current version against cached versions
- More robust than version_id matching (catches PDF republication with corrections)
- Existing infrastructure: `bill_versions.text_hash` column + `compute_bill_hash()` function

### SQLite Locking (deferred)
- Use `PRAGMA journal_mode=WAL` for concurrent reads
- Implement connection pooling with proper locking
- Full deployment design deferred until Azure architecture finalized

---

**Status:** APPROVED - Proceeding to Phase 2 Verification
