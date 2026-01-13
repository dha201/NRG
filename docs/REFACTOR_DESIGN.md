# NRG Legislative Tracker - Refactor Design Document

**Date:** January 12, 2026  
**Status:** Phase 2 - Awaiting Final Approval

---

## Target Module Structure

```
nrg_core/
├── __init__.py              # Package exports
├── models.py                # Dataclasses: Bill, Analysis, Version, ChangeData
├── config.py                # Config loading + validation
├── utils.py                 # Hash, rate limiting, cost tracking
│
├── api/
│   ├── __init__.py
│   ├── base.py              # Shared HTTP client, retry logic
│   ├── congress.py          # Congress.gov client
│   ├── regulations.py       # Regulations.gov client  
│   ├── openstates.py        # Open States client + PDF extraction
│   └── legiscan.py          # COMMENTED OUT - kept for future
│
├── analysis/
│   ├── __init__.py
│   ├── prompts.py           # Shared LLM prompt constant
│   ├── llm.py               # LLM routing, retry w/ backoff, cost tracking
│   ├── versions.py          # Version analysis + normalization
│   └── changes.py           # Bill-level change detection
│
├── reports/
│   ├── __init__.py
│   ├── console.py           # Rich console output
│   ├── markdown.py          # MD report (deduplicated section generator)
│   └── docx.py              # Pandoc Word conversion
│
├── db/
│   ├── __init__.py
│   └── cache.py             # SQLite operations, version caching
│
└── main.py                  # Pipeline orchestration (refactored from main())

tests/
├── __init__.py
├── test_utils.py            # Hash, diff tests
├── test_versions.py         # Version normalization tests
├── fixtures/                # Test data
│   └── bills/               # Sample bill JSON
└── eval/
    ├── __init__.py
    ├── llm_eval.py          # LLM accuracy evaluation framework
    └── test_cases/          # Ground-truth test cases
        └── hb4238.json      # From LLM_MODEL_COMPARISON.md
```

---

## Key Design Decisions

### 1. Dependency Injection Pattern

**Current (global state):**
```python
# Module-level globals
console = Console()
config = load_config()
openai_client = OpenAI(...)
gemini_client = genai.Client(...)

def analyze_with_llm(item, nrg_context):
    if config['llm']['provider'] == 'gemini':  # Accesses global
        ...
```

**Proposed (injected dependencies):**
```python
@dataclass
class AppContext:
    config: dict
    console: Console
    openai_client: OpenAI | None
    gemini_client: genai.Client | None
    db_conn: sqlite3.Connection | None

def analyze_with_llm(item: Bill, nrg_context: str, ctx: AppContext) -> Analysis:
    if ctx.config['llm']['provider'] == 'gemini':
        ...
```

**Why:** Enables testing without mocking module state; makes dependencies explicit.

### 2. Version Cache Lookup

**Current flow:**
```
For each version:
  → analyze_bill_version() → LLM call → save to cache
```

**Proposed flow:**
```
For each version:
  → compute_bill_hash(version.full_text)
  → query: SELECT * FROM bill_versions WHERE text_hash = ?
  → IF cached:
      → query: SELECT * FROM version_analyses WHERE version_id = ?
      → RETURN cached analysis
  → ELSE:
      → analyze_bill_version() → LLM call
      → save version + analysis to cache
      → RETURN new analysis
```

### 3. Path A Bug Fix

**Current (lines 3943-4050):**
```python
if has_versions and analyze_all_versions:
    # Path A: Version analysis
    ...
    result = {
        ...
        "change_impact": None  # BUG: Always None
    }
else:
    # Path B: Regular analysis
    if change_data and change_data['has_changes']:
        change_impact = analyze_changes_with_llm(...)  # Only here
```

**Proposed fix:**
```python
if has_versions and analyze_all_versions:
    # Path A: Version analysis
    ...
    # FIX: Also analyze bill-level changes in Path A
    change_impact = None
    if change_data and change_data['has_changes'] and not change_data['is_new']:
        if config.get('change_tracking', {}).get('analyze_changes_with_llm', True):
            change_impact = analyze_changes_with_llm(item, change_data, nrg_context)
    
    result = {
        ...
        "change_impact": change_impact  # Now populated
    }
```

### 4. LLM Retry with Exponential Backoff

```python
def call_llm_with_retry(
    prompt: str,
    provider: str,
    ctx: AppContext,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict:
    """
    Retry LLM calls with exponential backoff.
    
    Reasoning: LLM APIs have transient failures (rate limits, timeouts).
    Retry recovers from transient issues while backoff respects rate limits.
    Eventually fails gracefully after max_retries if persistent issue.
    """
    for attempt in range(max_retries):
        try:
            return _call_llm(prompt, provider, ctx)
        except (RateLimitError, TimeoutError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
            ctx.console.print(f"[yellow]LLM call failed, retrying in {delay}s...[/yellow]")
            time.sleep(delay)
```

### 5. Version Terminology Normalization

```python
VERSION_NORMALIZATION = {
    # Open States (Texas) → Normalized
    "Introduced": "introduced",
    "Engrossed": "passed_originating_chamber",
    "Enrolled": "enrolled",
    "House Committee Report": "committee_report",
    "Senate Committee Report": "committee_report",
    
    # Congress.gov → Normalized  
    "IH": "introduced",  # Introduced in House
    "IS": "introduced",  # Introduced in Senate
    "RH": "committee_report",
    "EH": "passed_originating_chamber",
    "ENR": "enrolled",
}

@dataclass
class BillVersion:
    version_type_raw: str      # Original: "Engrossed"
    version_type_normalized: str  # Normalized: "passed_originating_chamber"
    ...
```

### 6. PDF Image Detection

```python
def extract_pdf_text(pdf_url: str) -> tuple[str | None, bool]:
    """
    Extract text from PDF.
    
    Returns:
        (text, is_image_based): Text content, and flag if PDF appears to be scanned
    """
    with pdfplumber.open(pdf_bytes) as pdf:
        text = ""
        char_count = 0
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                char_count += len(page_text)
        
        # Heuristic: If PDF has pages but very little text, likely scanned images
        # Typical bill page has 2000-4000 chars; <100 chars/page suggests images
        is_image_based = len(pdf.pages) > 0 and (char_count / len(pdf.pages)) < 100
        
        if is_image_based:
            # TODO: Implement OCR fallback (Azure Computer Vision or Tesseract)
            console.print("[yellow]⚠ PDF appears to be scanned images, OCR not implemented[/yellow]")
        
        return (text if text.strip() else None, is_image_based)
```

---

## Implementation Sequence

| Step | Files Changed | Lines | Risk |
|------|---------------|-------|------|
| 1. Fix Path A bug | legislative_tracker.py | ~10 | Low |
| 2. Add version hash caching | legislative_tracker.py | ~25 | Low |
| 3. Add LLM retry | legislative_tracker.py | ~30 | Low |
| 4. Create models.py | new file | ~80 | None |
| 5. Extract api/ modules | new files, move code | ~800 | Medium |
| 6. Extract analysis/ modules | new files, move code | ~400 | Medium |
| 7. Extract reports/ modules | new files, move code | ~1000 | Medium |
| 8. Extract db/ module | new file, move code | ~300 | Low |
| 9. Deduplicate markdown | reports/markdown.py | -600 | Low |
| 10. Extract shared prompt | analysis/prompts.py | -90 | Low |
| 11. Add cost tracking | analysis/llm.py | ~40 | Low |
| 12. Add version normalization | analysis/versions.py | ~50 | Low |
| 13. Add PDF detection | api/openstates.py | ~20 | Low |
| 14. Create tests | tests/ | ~300 | None |

---

## Verification Approach

After each step:
1. Run existing code to verify no regressions
2. Check imports resolve correctly
3. Verify function_app.py still works

Final verification:
- Run full pipeline with test config
- Compare output to baseline run
- Check LLM cost tracking logs

---

**Status:** Ready for final approval to begin implementation.

Confirm to proceed with **Step 1: Fix Path A bug**?
