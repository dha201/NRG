"""
LLM-as-Judge tests for legislative analysis quality evaluation.

Tests the REAL system end-to-end:
1. analyze_with_gemini (production function from nrg_core.analysis.llm)
2. Real NRG business context (nrg_business_context.txt)
3. Real prompt construction (build_analysis_prompt)
4. Real response parsing (extract_json_from_gemini_response)
5. LLM-as-Judge evaluation of the analysis

Calibrated against Texas HB 4238 case from LLM_MODEL_COMPARISON.md.
Expert judgment: Score 2/10, verticals = [General Business, Retail Non-commodity]

=== HOW TO RUN ===
Quick test (see full output with judge reasoning):
  pytest tests/test_llm_quality/test_llm_as_judge.py -v -s

Run all LLM quality tests:
  pytest -m llm -v -s

=== WHEN TO RUN ===
- Before deploying model/prompt changes (regression check)
- After adding new bills to golden dataset (calibration)
- Weekly in CI/CD for quality monitoring
- When analysis scores seem off (diagnostic)

=== EXPANDING THE GOLDEN DATASET ===
Currently testing 1 bill (HB 4238, low-impact). Need 50+ bills across:
- Impact levels: low (2/10), medium (5/10), high (8/10)
- Bill types: regulatory, financial, operational, market
- Verticals: Power Gen, EV Charging, Retail, Tax, Environmental

To add a bill:
1. Get expert human judgment (score, verticals, rationale)
2. Add to GOLDEN_BILLS dict below
3. Add corresponding EXPERT_JUDGMENT entry
4. Run test to calibrate judge thresholds
5. Adjust thresholds if judge consistently disagrees with experts

Goal: Judge correlation >0.7 with human experts across all bills
"""
import pytest
import json
import os
from typing import Any

from google import genai
from rich.console import Console
from rich.panel import Panel

from deepeval import assert_test
from deepeval.models import GeminiModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from nrg_core.analysis.llm import analyze_with_gemini
from nrg_core.config import load_nrg_context, load_config

console = Console()


# =============================================================================
# GOLDEN TEST CASES - Expert-Validated Bills for Judge Calibration
# =============================================================================
# These are bills where human experts have provided ground truth judgments
# Judge accuracy is measured by correlation with these expert assessments
# TODO: Expand from 1 bill to 50+ covering diverse impact levels and verticals

# Low-impact bill: Identity theft debt collection (minimal NRG exposure)
GOLDEN_BILL_HB4238 = {
    "source": "OpenStates",
    "type": "State Bill",
    "number": "HB 4238",
    "title": "Identity Theft Debt Collection",
    "status": "Enacted",
    "state": "TX",
    "summary": """Texas passed a law requiring companies to stop collecting debts 
    if a consumer provides a court order proving they were an identity theft victim. 
    Companies must: stop collection within 7 business days, notify credit bureaus, 
    and not sell the debt to collectors.""",
    "url": "https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=HB4238",
}

# Expert judgment from legal/policy team - this is ground truth for judge calibration
EXPERT_JUDGMENT_HB4238 = {
    "expected_score": 2,
    "score_tolerance": 1,
    "expected_impact_type": "regulatory_compliance",
    "expected_risk_assessment": "risk",
    "correct_verticals": ["General Business", "Retail Non-commodity"],
    "incorrect_verticals": ["Retail Commodity", "Electric Vehicles", "Services"],
    "rationale": """NRG has minimal consumer debt exposure. Retail fuel uses 
    pay-at-pump (immediate payment). Fleet cards are B2B (excluded from bill). 
    Identity theft cases are rare (~10-50/year). Compliance cost <$250K."""
}


# =============================================================================
# G-EVAL METRIC FACTORIES - Split Evaluation for Better Diagnostics
# =============================================================================
# Why split metrics? Single monolithic judge can't pinpoint failure cause
# Score accuracy vs vertical accuracy need different thresholds and evaluation logic
# Each metric uses chain-of-thought reasoning for explainability
#
# References:
# - G-Eval paper: https://arxiv.org/pdf/2303.16634
# - DeepEval docs: https://deepeval.com/docs/metrics-llm-evals
# - Gemini integration: https://deepeval.com/integrations/models/gemini

def create_score_accuracy_metric(api_key: str) -> GEval:
    """
    Judges if analysis score matches expert judgment (±1 tolerance)
    
    Threshold 0.5 = lenient, allows some LLM scoring variance
    Catches overstatement (e.g., scoring 8/10 for minimal-impact bills)
    """
    judge_model = GeminiModel(
        model="gemini-3-flash-preview",
        api_key=api_key,
        temperature=0.1
    )
    
    # Hardcoded steps (not generated) for consistency across runs
    # Each step is explicit instruction for judge's chain-of-thought reasoning
    return GEval(
        name="Business Impact Score Accuracy",
        evaluation_steps=[
            "Extract the business_impact_score from the actual_output JSON",
            "Extract the expected_score from the expected_output JSON",
            "Check if the score difference is within ±1 tolerance",
            "Verify the score reflects NRG's minimal debt collection exposure (pay-at-pump model)",
            "Penalize scores that overstate impact for identity theft debt laws"
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=judge_model,
        threshold=0.5,  # Tuned via trial - may need adjustment with more golden bills
        verbose_mode=True  # Shows reasoning but doesn't print to console (use metric.reason)
    )


def create_vertical_accuracy_metric(api_key: str) -> GEval:
    """
    Judges if analysis flags correct business verticals (not unrelated ones)
    
    Threshold 0.3 = very lenient, only fails on egregious errors
    Catches hallucinations like "Electric Vehicles" for debt collection bills
    Lower threshold because vertical taxonomy is fuzzy (Commodity vs Non-commodity)
    """
    judge_model = GeminiModel(
        model="gemini-3-flash-preview",
        api_key=api_key,
        temperature=0.1
    )
    
    return GEval(
        name="Vertical Classification Accuracy",
        evaluation_steps=[
            "Extract impacted_verticals from actual_output JSON",
            "Extract correct_verticals from expected_output JSON",
            "Check if reasonable verticals are selected for a debt collection bill",
            "Minor penalize if 'Retail Commodity' vs 'Retail Non-commodity' distinction is wrong",
            "Major penalize only if completely unrelated verticals are flagged (e.g., 'Electric Vehicles', 'Power Generation')"
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=judge_model,
        threshold=0.3,  # Lower than score metric - vertical classification is harder
        verbose_mode=True
    )


# =============================================================================
# END-TO-END ANALYSIS QUALITY TEST - Real Production Pipeline
# =============================================================================
# Tests actual analyze_with_gemini() with real config, prompts, schemas
# Not mocked - calls live LLM APIs (Gemini 3 Pro for analysis, 3 Flash for judge)
# Takes ~40s per bill due to two LLM calls (analysis + judge evaluation)

@pytest.mark.slow
@pytest.mark.llm
class TestAnalysisQuality:
    """End-to-end tests using real system components."""
    
    def test_analyze_and_judge(self):
        """
        Validates analysis quality by comparing LLM output to expert judgment
        
        Flow:
        1. Run real analysis (Gemini 3 Pro, production config)
        2. Judge evaluates analysis (Gemini 3 Flash, G-Eval framework)
        3. Compare judge scores to thresholds (0.5 for score, 0.3 for verticals)
        
        Why:
        - Catches quality regressions when changing models/prompts
        - Detects overstatement (scoring high-impact when actually low)
        - Identifies hallucinations (flagging unrelated business verticals)
        
        Current limitation: Only 1 golden bill - need 50+ for robust calibration
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # === STEP 1: Run REAL analysis ===
        # Uses production analyze_with_gemini() - not mocked, calls live API
        # Loads actual config.yaml (model, temp, tokens) and nrg_business_context.txt
        console.print("\n" + "="*80, style="bold cyan")
        console.print("STEP 1: Running analyze_with_gemini (real system)", style="bold cyan")
        console.print("="*80, style="bold cyan")
        
        client = genai.Client(api_key=api_key)
        nrg_context = load_nrg_context()
        config = load_config()  # Real config - currently Gemini 3 Pro at temp 0.2
        
        console.print(f"[dim]Using analysis model: {config['llm']['gemini']['model']}[/dim]")
        
        analysis = analyze_with_gemini(
            item=GOLDEN_BILL_HB4238,
            nrg_context=nrg_context,
            gemini_client=client,
            config=config
        )
        
        console.print("\n[bold green]Analysis Result:[/bold green]")
        console.print(Panel(
            json.dumps(analysis, indent=2),
            border_style="green",
            title="[bold]LLM Analysis Output[/bold]"
        ))
        
        assert "error" not in analysis, f"Analysis failed: {analysis}"
        assert "business_impact_score" in analysis
        
        # === STEP 2: Create DeepEval test case ===
        # Packages analysis output + expert judgment for judge evaluation
        # Judge compares actual_output (LLM analysis) to expected_output (expert ground truth)
        console.print("\n" + "="*80, style="bold magenta")
        console.print("STEP 2: Creating DeepEval test case", style="bold magenta")
        console.print("="*80, style="bold magenta")
        
        test_case = LLMTestCase(
            input=GOLDEN_BILL_HB4238["summary"],
            actual_output=json.dumps(analysis),
            expected_output=json.dumps({
                "expected_score": EXPERT_JUDGMENT_HB4238["expected_score"],
                "score_tolerance": EXPERT_JUDGMENT_HB4238["score_tolerance"],
                "correct_verticals": EXPERT_JUDGMENT_HB4238.get("correct_verticals", []),
                "rationale": EXPERT_JUDGMENT_HB4238["rationale"]
            })
        )
        
        # === STEP 3: Create G-Eval metrics ===
        # Split metrics = better diagnostics than single monolithic judge
        # Can tune thresholds independently (score needs 0.5, verticals only need 0.3)
        console.print("\n" + "="*80, style="bold yellow")
        console.print("STEP 3: Evaluating with G-Eval metrics (Gemini 3 Flash judge)", style="bold yellow")
        console.print("="*80, style="bold yellow")
        
        score_metric = create_score_accuracy_metric(api_key)
        vertical_metric = create_vertical_accuracy_metric(api_key)
        
        # === STEP 4: Manually measure metrics to get reasoning ===
        # DeepEval quirk: assert_test() doesn't populate metric.score/reason attributes
        # Must call measure() explicitly to capture judge's chain-of-thought reasoning
        console.print("\n[cyan]Evaluating metrics manually to capture reasoning...[/cyan]")
        
        score_metric.measure(test_case)
        vertical_metric.measure(test_case)
        
        # === STEP 4.5: Display judge reasoning ===
        # Shows judge's chain-of-thought explanation for scores
        # This is the "why" - helps debug when judge disagrees with expectations
        console.print("\n" + "="*80, style="bold magenta")
        console.print("Judge Reasoning (G-Eval)", style="bold magenta")
        console.print("="*80, style="bold magenta")
        
        console.print("\n[bold cyan]Score Accuracy Metric:[/bold cyan]")
        if score_metric.score is not None:
            console.print(f"  [green]Score:[/green] {score_metric.score:.2f}")
        else:
            console.print(f"  [yellow]Score:[/yellow] N/A")
            
        if score_metric.reason:
            console.print(f"  [yellow]Reason:[/yellow] {score_metric.reason}")
        else:
            console.print(f"  [yellow]Reason:[/yellow] Not available")
            
        success = score_metric.is_successful()
        status = "✓ PASS" if success else "✗ FAIL"
        color = "green" if success else "red"
        console.print(f"  [bold {color}]{status}[/bold {color}] (threshold: {score_metric.threshold})")
        
        console.print("\n[bold cyan]Vertical Accuracy Metric:[/bold cyan]")
        if vertical_metric.score is not None:
            console.print(f"  [green]Score:[/green] {vertical_metric.score:.2f}")
        else:
            console.print(f"  [yellow]Score:[/yellow] N/A")
            
        if vertical_metric.reason:
            console.print(f"  [yellow]Reason:[/yellow] {vertical_metric.reason}")
        else:
            console.print(f"  [yellow]Reason:[/yellow] Not available")
            
        success = vertical_metric.is_successful()
        status = "✓ PASS" if success else "✗ FAIL"
        color = "green" if success else "red"
        console.print(f"  [bold {color}]{status}[/bold {color}] (threshold: {vertical_metric.threshold})")
        
        # === STEP 4.6: Run DeepEval assertion ===
        # Final validation - fails test if either metric below threshold
        # Redundant with measure() above but required for DeepEval test framework
        console.print("\n[cyan]Running DeepEval assert_test for final validation...[/cyan]")
        assert_test(test_case, [score_metric, vertical_metric])
        
        # === STEP 5: Display results ===
        # Summary for quick scan - detailed reasoning already shown in Step 4.5
        console.print("\n" + "="*80, style="bold green")
        console.print("STEP 4: Results Summary", style="bold green")
        console.print("="*80, style="bold green")
        
        actual_score = analysis.get("business_impact_score", 0)
        expected_score = EXPERT_JUDGMENT_HB4238["expected_score"]
        
        console.print(f"\n[cyan]Expected Score:[/cyan] {expected_score}")
        console.print(f"[cyan]Actual Score:[/cyan] {actual_score}")
        
        # Display metric scores if available
        score_val = score_metric.score if score_metric.score is not None else "N/A"
        vertical_val = vertical_metric.score if vertical_metric.score is not None else "N/A"
        
        if isinstance(score_val, float):
            console.print(f"[cyan]Score Metric:[/cyan] {score_val:.2f} (threshold: {score_metric.threshold})")
        else:
            console.print(f"[cyan]Score Metric:[/cyan] {score_val} (threshold: {score_metric.threshold})")
            
        if isinstance(vertical_val, float):
            console.print(f"[cyan]Vertical Metric:[/cyan] {vertical_val:.2f} (threshold: {vertical_metric.threshold})")
        else:
            console.print(f"[cyan]Vertical Metric:[/cyan] {vertical_val} (threshold: {vertical_metric.threshold})")
        
        console.print("\n[bold green]✓ Test PASSED: DeepEval assert_test succeeded[/bold green]")
