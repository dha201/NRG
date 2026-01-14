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

Run with: pytest tests/test_llm_quality/test_llm_as_judge.py -v -s
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
# GOLDEN TEST CASE: Texas HB 4238 (from LLM_MODEL_COMPARISON.md)
# =============================================================================

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
# G-EVAL METRIC FACTORIES
# References:
# - G-Eval paper: https://arxiv.org/pdf/2303.16634
# - DeepEval docs: https://deepeval.com/docs/metrics-llm-evals
# - Gemini integration: https://deepeval.com/integrations/models/gemini
# =============================================================================

def create_score_accuracy_metric(api_key: str) -> GEval:
    """
    Create G-Eval metric for business impact score accuracy.
    
    Uses Gemini 3 Flash as judge to evaluate if the analysis score
    matches expert judgment and reflects NRG's actual business exposure.
    """
    judge_model = GeminiModel(
        model="gemini-3-flash-preview",
        api_key=api_key,
        temperature=0.1
    )
    
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
        threshold=0.5,
        verbose_mode=True
    )


def create_vertical_accuracy_metric(api_key: str) -> GEval:
    """
    Create G-Eval metric for vertical classification accuracy.
    
    Evaluates if the analysis correctly identifies affected business verticals
    without overstating impact (e.g., pay-at-pump != debt collection).
    
    Reference: https://www.evidentlyai.com/llm-guide/llm-as-a-judge
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
        threshold=0.3,
        verbose_mode=True
    )


# =============================================================================
# END-TO-END ANALYSIS QUALITY TEST
# =============================================================================

@pytest.mark.slow
@pytest.mark.llm
class TestAnalysisQuality:
    """End-to-end tests using real system components."""
    
    def test_analyze_and_judge(self):
        """
        End-to-end test: Real analysis + DeepEval G-Eval judge.
        
        Tests the REAL production pipeline:
        1. Runs analyze_with_gemini with actual NRG business context
        2. Uses DeepEval G-Eval metrics with Gemini 2.5 Pro judge
        3. Splits evaluation into score accuracy + vertical accuracy
        
        Architecture:
        - Analysis LLM: Uses production config from config.yaml
        - Judge LLM: gemini-3-flash-preview (Gemini 3 Flash)
        
        References:
        - DeepEval pytest integration: https://deepeval.com/docs/metrics-llm-evals
        - G-Eval methodology: https://www.confident-ai.com/blog/g-eval-the-definitive-guide
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # === STEP 1: Run REAL analysis ===
        console.print("\n" + "="*80, style="bold cyan")
        console.print("STEP 1: Running analyze_with_gemini (real system)", style="bold cyan")
        console.print("="*80, style="bold cyan")
        
        client = genai.Client(api_key=api_key)
        nrg_context = load_nrg_context()
        config = load_config()  # Use production config from config.yaml
        
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
        console.print("\n" + "="*80, style="bold yellow")
        console.print("STEP 3: Evaluating with G-Eval metrics (Gemini 3 Flash judge)", style="bold yellow")
        console.print("="*80, style="bold yellow")
        
        score_metric = create_score_accuracy_metric(api_key)
        vertical_metric = create_vertical_accuracy_metric(api_key)
        
        # === STEP 4: Manually measure metrics to get reasoning ===
        console.print("\n[cyan]Evaluating metrics manually to capture reasoning...[/cyan]")
        
        score_metric.measure(test_case)
        vertical_metric.measure(test_case)
        
        # === STEP 4.5: Display judge reasoning ===
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
        console.print("\n[cyan]Running DeepEval assert_test for final validation...[/cyan]")
        assert_test(test_case, [score_metric, vertical_metric])
        
        # === STEP 5: Display results ===
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
