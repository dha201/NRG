# LLM Quality Testing (DeepEval)

**The Problem:** Manual review doesn't scale. We can't read every analysis.
**The Solution:** Use a second LLM to grade the first one against expert human standards.

---

## How It Works

Simple pipeline:
`Bill Text` -> `Analysis LLM` -> `JSON Output` -> `Judge LLM` -> `Quality Score`

1.  **Analysis LLM** (Gemini 3 Pro) reads the bill and generates the analysis.
2.  **Judge LLM** (Gemini 3 Flash) compares that analysis to expert "ground truth."
    *   Did the score match what a human would say?
    *   Are the business verticals correct?
3.  **Judge** returns a pass/fail verdict and explains its reasoning.

---

## Why It Matters

### For Product Managers

**Quality at Scale**
*   Catches score inflation (claiming 8/10 impact when it's actually 2/10).
*   Stops hallucinations (flagging "EVs" for a debt collection bill).
*   Prevents false alarms.

**Cost & Confidence**
*   Cheaper than human review ($2.40/month for 10% sampling).
*   Deploys with confidence—run tests before shipping.
*   Catches regression bugs instantly.

### For Developers

**Better Testing**
*   Change a prompt? Run the judge. Know immediately if you broke something.
*   Upgrade models safely.
*   Golden dataset prevents "whack-a-mole" bug fixing.

**Faster Debugging**
*   Judge explains *why* it failed.
*   "Score failed because analysis missed the pay-at-pump exemption."
*   Root cause analysis takes minutes, not hours.

---

## Current Status

### Working
*   **Framework:** DeepEval G-Eval (solid, research-backed).
*   **Pipeline:** Tests real production code, not mocks.
*   **Detection:** successfully catches bad scores and hallucinated verticals.
*   **Output:** Clear, color-coded console logs.

### Gaps
*   **Data Starved:** Only one "golden" bill (HB 4238). Need 50+ to be robust.
*   **Manual Tuning:** Thresholds set by trial/error, not calibrated data.
*   **Offline:** Runs in tests, not yet monitoring live production traffic.

---

## The Roadmap

### 1. Expand Golden Dataset
We need 50+ bills validated by the legal team.
*   **Mix of impact:** Low (2/10), Medium (5/10), High (8/10).
*   **Mix of types:** Regulatory, tax, operational.
*   **Mix of verticals:** Generation, Retail, EVs.

**How:** Add bill text + expert judgment to `tests/test_llm_quality/test_llm_as_judge.py`.

### 2. Calibrate
Run the judge on those 50 bills.
*   Compare judge scores to human scores.
*   Tune thresholds until they agree >70% of the time.

### 3. Production Sampling
Add to `orchestrator.py`.
*   Sample 10% of daily analyses.
*   Log scores.
*   Alert if quality dips.

---

## Running Tests

**Quick Check (Full Output):**
```bash
pytest tests/test_llm_quality/test_llm_as_judge.py -v -s
```
*Expect: Green panel with analysis, followed by judge reasoning.*

**When to Run:**
*   **Pre-deploy:** Prompt or model changes.
*   **Weekly:** Catch drift.
*   **Diagnostics:** When scores look weird.

---

## Technical Specs

**Analysis:** Gemini 3 Pro (Smart, Temp 0.2)
**Judge:** Gemini 3 Flash (Fast, Temp 0.1)

**Metrics:**
*   **Score Accuracy:** Matches expert score within ±1? (Threshold: 0.5)
*   **Vertical Accuracy:** Selected the right business units? (Threshold: 0.3)
*   *Note: Split metrics let us debug exactly what went wrong.*

**Cost:**
*   ~$0.008 per bill.
*   10% daily sampling = ~$2.40/month.
*   Negligible.

---