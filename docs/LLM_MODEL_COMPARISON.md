# LLM Model Accuracy Assessment
**Date:** January 9, 2026
**Subject:** GPT-5 vs Gemini 3 Pro - Impact Scoring Accuracy Analysis

---

## Executive Summary

We tested two AI models (GPT-5 and Gemini 3 Pro) on the same Texas bill to evaluate accuracy in assessing business impact. **GPT-5 demonstrated superior accuracy** by correctly identifying minimal business exposure, while Gemini 3 overstated the impact by 2x.

**Recommendation:** Continue using GPT-5 for legislative analysis due to better understanding of BP's business model and risk assessment.

---

## Test Case: Texas HB 4238 (Identity Theft Debt Collection)

### Bill Summary (Plain English)
Texas passed a law requiring companies to stop collecting debts if a consumer provides a court order proving they were an identity theft victim. Companies must:
- Stop collection within 7 business days
- Notify credit bureaus
- Not sell the debt to collectors

### Model Comparison Results

| Metric | GPT-5 | Gemini 3 Pro | Winner |
|--------|-------|--------------|--------|
| **Impact Score** | 2/10 (LOW) | 4/10 (MEDIUM) |  GPT-5 |
| **Impact Type** | Regulatory Compliance | Regulatory Compliance | Tie |
| **Risk Assessment** | Risk | Mixed (risk + opportunity) |  GPT-5 |
| **Financial Impact** | <$250K | "Minimal, potential bad debt increase" |  GPT-5 |
| **Business Verticals** | General Business, Retail Non-commodity | Retail Commodity, Services, General Business, Electric Vehicles |  GPT-5 |

---

## Why GPT-5 Was More Accurate

### 1. Understanding of BP's Business Model

**GPT-5 recognized:**
- BP is primarily an **integrated oil & gas company** (upstream production, refining, wholesale)
- Retail operations use **immediate payment** (pay-at-pump), not credit accounts
- Consumer debt exposure is **minimal and rare**

**Gemini 3 missed:**
- Incorrectly flagged "Retail Commodity" (fuel sales are immediate payment, not credit)
- Overstated EV charging impact (BP pulse is <1% of business)
- Suggested "mixed" risk/opportunity (no opportunity exists, only compliance burden)

### 2. Scope Assessment

**GPT-5 correctly identified limited exposure:**
- **Geography:** Only Texas (1 of 46 states)
- **Customer type:** Only consumer accounts (excludes B2B fleet cards)
- **Scenario:** Only identity theft cases (rare: ~10-50 cases/year estimated)
- **Business line:** Only credit accounts (BP barely extends consumer credit)

**Impact calculation:**
```
Total BP revenue: $194.6 billion/year
Compliance cost: <$250,000
Percentage impact: 0.0001%
```

### 3. Precise Vertical Mapping

**GPT-5's verticals (correct):**
-  General Business (catch-all compliance)
-  Retail Non-commodity (station operations, not fuel sales)

**Gemini 3's verticals (overstated):**
- ❌ Retail Commodity (implies fuel sales affected - incorrect)
- ❌ Electric Vehicles (BP pulse too small for MEDIUM impact)
- ❌ Services (not relevant to debt collection)
---

## Business Context:

### BP's Revenue Model
BP makes money from:
1. **Drilling oil/gas** (upstream) - 55% of profit
2. **Refining crude into fuel** (downstream) - variable profit
3. **Selling fuel at stations** - mostly **immediate payment**

### Consumer Credit Exposure (Minimal)
- **Retail fuel:** Pay-at-pump (credit card processed instantly)
- **Fleet cards:** B2B accounts (not consumer debt, excluded from bill)
- **BP pulse:** Small subscription base (~3,000 charge points)
- **TravelCenters:** Mostly cash/immediate payment
---

## Impact

### Current State: GPT-5 is Working Well 
- Correctly identifies low-impact bills (avoids alert fatigue)
- Understands BP's business model nuances
- Provides accurate financial impact estimates

### Risk of Using Gemini 3
- **Over-alerting:** More false positives(Scoring 4/10 instead of 2/10)
- **False Positives** Real threats might get ignored
---









# Legal AI Models for Legislative Analysis
**Updated:** Jan 9, 2026  
**Focus:** Latest benchmarks for bill analysis

---

## LegalBench Leaderboard (Dec 24, 2025)

| Rank | Model | Legal Accuracy | Release |
|------|-------|----------------|---------|
| 1 | **Gemini 3 Pro** | 87.04% | Nov 2025 |
| 2 | **Gemini 3 Flash** | 86.86% | Dec 2025 |
| 3 | **GPT-5** | 86.02% | Aug 2025 |
| 4 | **GPT-5.1** | 85.68% | Nov 2025 |
| - | **Claude Opus 4.5** | Not evaluated | Nov 2025 |

**Source:** https://www.vals.ai/benchmarks/legal_bench

---

## AI vs Lawyers (VLAIR Oct 2025)

Legal research accuracy:

| System | Accuracy | Type |
|--------|----------|------|
| **Lawyers** | 71% | Baseline |
| **Counsel Stack** | 81% | Legal-specific |
| **ChatGPT/Alexi** | 80% | General/Legal |
| **Midpage** | 79% | Legal-specific |

**Key findings:**
- AI outperforms lawyers by 9 points
- Legal-specific AI: 76% authoritativeness vs ChatGPT 70%
- All struggle with multi-jurisdictional (-11 points)

**Source:** https://www.lawnext.com/2025/10/vals-ais-latest-benchmark-finds-legal-and-general-ai-now-outperform-lawyers-in-legal-research-accuracy.html

---
