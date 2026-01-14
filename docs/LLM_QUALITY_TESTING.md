# LLM Quality Assurance: G-Eval Framework

## Overview

As the volume of legislative analysis scales, manual review of every LLM output becomes infeasible. To ensure consistent quality without human bottlenecks, we have implemented an automated **LLM-as-Judge** evaluation pipeline using the **G-Eval framework**.

This system employs a secondary "Judge" LLM to audit the outputs of our primary "Analyst" LLM, grading them against expert human baselines ("Golden Datasets"). This allows for continuous regression testing, detection of hallucinations, and calibration of model performance.

## Methodology

We utilize the **G-Eval** framework (based on *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*), which leverages Chain-of-Thought (CoT) reasoning to perform reference-based evaluation.

### The Pipeline

The evaluation pipeline follows a strict flow:

1.  **Input:** Raw legislative text.
2.  **Analysis (Candidate):** The production model (`gemini-3-pro-preview`) generates the legislative analysis JSON.
3.  **Evaluation (Judge):** The judge model (`gemini-3-flash-preview`) compares the Candidate output against an **Expert Reference** (ground truth provided by legal/policy teams).
4.  **Scoring:** The judge outputs a score (0-1), a pass/fail verdict, and a reasoning trace explaining the decision.

### Architecture

| Component | Model | Configuration | Role |
|-----------|-------|---------------|------|
| **Analyst** | Gemini 3 Pro | Temp 0.2 | Generates business impact scores and vertical classifications. |
| **Judge** | Gemini 3 Flash | Temp 0.1 | Evaluates accuracy of the Analyst's output. |

## Evaluation Metrics

We employ split-metric evaluation to diagnose specific failure modes. A single monolithic score is often too opaque for debugging.

### 1. Score Accuracy
*   **Objective:** Validate that the `business_impact_score` aligns with expert judgment.
*   **Logic:** Checks if the LLM score is within **±1** of the expert reference score.
*   **Threshold:** `0.5` (Pass/Fail).
*   **Failure Mode:** Detects score inflation (e.g., scoring a procedural bill as 8/10) or deflation.

### 2. Vertical Accuracy
*   **Objective:** Ensure business verticals (e.g., "Retail", "Generation") are correctly identified.
*   **Logic:** Compares the list of flagged verticals against the ground truth.
*   **Threshold:** `0.3` (Pass/Fail).
*   **Failure Mode:** Detects hallucinations (e.g., flagging "EVs" for a debt collection bill) or missed exposure.

## The Golden Dataset

The core of this framework is the **Golden Dataset**—a curated set of bills with pre-validated expert analyses. The Judge does not rely on its own intuition alone; it relies on its ability to measure distance from this expert ground truth.

*   **Current State:** Initial calibration using HB 4238 (Identity Theft/Debt Collection).
*   **Requirement:** A robust dataset requires **50+ examples** covering:
    *   **Impact Variance:** Low (2/10), Medium (5/10), and High (8/10) impact bills.
    *   **Domain Variance:** Regulatory, Tax, Operational, and Market bills.

## Usage Guide

### Running Evaluations

Tests are implemented via `pytest` and `DeepEval`.

**Quick Test (Console Output):**
```bash
pytest tests/test_llm_quality/test_llm_as_judge.py -v -s
```
*The `-s` flag is required to view the rich console output containing the Judge's reasoning.*

### Integration Strategy

1.  **Regression Testing:** Run evaluations before any prompt modification or model version upgrade.
2.  **Calibration:** When adding new bills to the Golden Dataset, run the judge to ensure it agrees with human experts. If it disagrees, tune the evaluation prompts or thresholds.
3.  **Production Monitoring (Roadmap):** Sample 10% of live traffic for asynchronous evaluation to track quality drift over time.

## Roadmap

1.  **Dataset Expansion:** Scale the Golden Dataset from 1 to 50+ validated bills.
2.  **Threshold Calibration:** Tune metric thresholds to achieve >70% correlation with human reviewers across the full dataset.
3.  **Production Sampling:** Integrate sampling logic into the `orchestrator.py` pipeline for live quality alerts.

## FAQ

**Why use a Judge LLM instead of rules?**
Legislative analysis is subjective and semantic. Rule-based systems cannot evaluate if a "summary is accurate" or if a "score is reasonable." LLMs with CoT reasoning can.

**Can the Judge be wrong?**
Yes. However, the system is designed for **calibration**. If the Judge consistently disagrees with human experts on the Golden Dataset, we refine the Judge's prompt or the ground truth until alignment is achieved.

**Why Gemini Flash for the Judge?**
Evaluation is a classification and reasoning task that requires less "creative" capacity than generation. Gemini Flash provides the necessary reasoning capabilities at significantly lower latency and cost.
