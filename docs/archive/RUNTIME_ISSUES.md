## ✅ RESOLVED

### 1. ~~Wrong Version Index Selection~~ **FIXED**
**Location:** `poc 2.py:4032` (formerly 4023)

**Original Bug:**
```python
analysis = version_analyses[-1]['analysis']  # Got Introduced (oldest) ❌
```

**Fix Applied:**
```python
analysis = version_analyses[0]['analysis']  # Gets Enrolled (newest) ✓
```

**Verification:**
```
DEBUG: version_analyses order:
  [0] Enrolled (score: 2)      ← Now selected ✓
  [1] Senate (score: 5)
  [2] Engrossed (score: 5)
  [3] House (score: 4)
  [4] Introduced (score: 4)    ← Was incorrectly selected
```

**Status:** RESOLVED - Code now uses correct index `[0]` for newest version

---

### 2. ~~Custom Prompt Never Used in Version Analysis~~ **FIXED**
**Location:** `poc 2.py:1591-1639` (`analyze_bill_version`)

**Original Bug:**
```python
# Line 1591: Built version-specific prompt
prompt_text = f"""...
**Version Being Analyzed: {version_type}**  <-- THIS IS THE VERSION
..."""

# Line 1629: Never used it ❌
return analyze_with_gemini(temp_item, nrg_context)  # No prompt_text passed
```

**Root Cause:** `analyze_bill_version()` built custom `prompt_text` with version context but called `analyze_with_gemini()`/`analyze_with_openai()` without passing it. LLM functions built their own generic prompts with zero version context.

**Impact:** LLM inferred version labels from text content, causing state terminology → federal terminology mismatches.

**Fix Applied:**
1. Added `custom_prompt=None` parameter to both LLM functions
2. Pass `prompt_text` when calling from `analyze_bill_version()`
3. LLM now receives explicit version context

**Status:** RESOLVED - Version-specific prompts now reach the LLM

---

## 🔴 Critical Bugs

### 3. State vs Federal Terminology Mismatch - Bill Versions (May Be Resolved by Bug #2 Fix)
**Location:** `poc 2.py:1194` - JSON schema in Gemini/OpenAI prompts

**Issue:** LLM returns federal terminology for state bills:
```json
{
  "bill_version": "passed_house",   // ❌ LLM inference
  // Actual version_type: "Engrossed" (Texas state term)
}
```

**Root Cause - Schema Forces Federal Terms:**
```python
# Line 1194 - JSON schema constrains to federal terminology only:
"bill_version": "as_filed | house_committee | passed_house | ..."

# But Open States provides state-specific terms:
version_type = "Engrossed"  # Texas House bill passed originating chamber
```

**State vs Federal Terminology Gap:**
| Open States (State) | What It Means | Federal Equivalent | LLM Guesses |
|---|---|---|---|
| Introduced | Filed | as_filed | ✓ |
| House Committee Report | In committee | house_committee | ? |
| **Engrossed** | Passed originating chamber | **varies by chamber** | passed_house ❌ |
| Senate Committee Report | In committee | senate_committee | ? |
| Enrolled | Final passed version | enrolled | ✓ |

**Impact:** Reports show misleading version labels. "passed_house" for Engrossed creates inconsistency with actual metadata.

**Status:** NEEDS TESTING - Bug #2 fix now provides version context to LLM. May still need fallback:
1. **Override after parsing:** Force `analysis['bill_version'] = version_type` (ignore LLM output)
2. **Expand schema:** Add state terms to JSON schema
3. **Pre-map in code:** Convert state → federal terms before LLM

---

### 4. Status Shows "Unknown" for HB 4238
**Location:** Report generation

**Issue:**
```
Status: Unknown
```

**Expected:** Should show actual bill status (e.g., "Enrolled", "Passed", "Signed")

**Root Cause:** Bill metadata from Open States may not include status, or status field not mapped correctly.

**Impact:** Missing critical tracking information in reports.

**Needs Investigation:**
1. Check what status data Open States API returns
2. Verify status mapping in `fetch_specific_texas_bills` (line 700-1100)
3. Check if status exists in `item` dict when passed to report

---

## 🟠 LLM Analysis Failures

### 2. JSON Parsing Errors - Version Diffs
**Frequency:** 2-3 failures per run

**Errors:**
```
Error analyzing with Gemini: the JSON object must be str, bytes or bytearray, not NoneType
Error parsing JSON from Gemini: Unterminated string starting at: line 60 column 25
Error parsing JSON from Gemini: Expecting value: line 84 column 5
```

**Impact:** Version-to-version semantic change analysis lost. System continues with degraded data.

**Location:** `analyze_version_changes_with_llm()` calls fail silently.

**Needs:** Retry logic + log full response for debugging.

---

## ⚠️ Data Collection Issues

### 3. Congress.gov Returns 0 Bills
**Observed:** Both runs returned 0 energy bills from Congress.gov

**Possible causes:**
- API timing (no new bills matching keywords)
- Silent API failure (fail-soft returns `[]`)
- Filter too restrictive

**Action needed:** Manual verification if Congress.gov API is working.

---

## 📊 Architecture Limitations

### 4. High-Impact Items Aging Out
**Issue:** Regulations.gov limit=3. New items push out older high-impact ones.

**Example:** 
- Run 1: EERE_FRDOC_0001-2165 (impact score 7/10) - present
- Run 2: Same item missing (replaced by 3 new items)

**Impact:** Lose historical high-impact tracking.

**Solution:** Archive items with impact_score ≥ 7 to persistent storage.

---