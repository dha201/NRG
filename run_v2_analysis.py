#!/usr/bin/env python3
# run_v2_analysis.py
"""
CLI for Architecture v2.0 Two-Tier Analysis.

Usage:
    python run_v2_analysis.py --bill-id HB123 --bill-text-file path/to/bill.txt

Output:
    - Console tables showing findings, validations, and rubric scores
    - JSON file with complete analysis results

Design:
    - Uses supervisor for complexity routing (informational in Phase 1)
    - Runs full two-tier pipeline: primary analyst → judge → rubrics
    - Saves structured output for downstream processing
"""
import os
import sys
import argparse
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nrg_core.v2.supervisor import SupervisorRouter
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.config import load_nrg_context

load_dotenv()
console = Console()


def main():
    """Main CLI entry point for v2 analysis."""
    parser = argparse.ArgumentParser(
        description="Run Architecture v2.0 Two-Tier Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_v2_analysis.py --bill-id HB123 --bill-text-file bill.txt
    python run_v2_analysis.py --bill-id SB456 --bill-text-file senate_bill.txt --output results.json
        """
    )
    parser.add_argument("--bill-id", required=True, help="Bill identifier (e.g., HB123)")
    parser.add_argument("--bill-text-file", required=True, help="Path to bill text file")
    parser.add_argument("--output", default="v2_analysis.json", help="Output JSON file (default: v2_analysis.json)")
    
    args = parser.parse_args()
    
    # Validate bill text file exists
    if not os.path.exists(args.bill_text_file):
        console.print(f"[red]Error: Bill text file not found: {args.bill_text_file}[/red]")
        sys.exit(1)
    
    # Load bill text
    with open(args.bill_text_file, 'r') as f:
        bill_text = f.read()
    
    # Load NRG context
    nrg_context = load_nrg_context()
    
    # Step 1: Supervisor routing (informational in Phase 1)
    console.print("\n[cyan]Step 1: Assessing Complexity...[/cyan]")
    router = SupervisorRouter()
    
    # Estimate page count from text length (rough: ~3000 chars/page)
    bill_metadata = {
        "bill_id": args.bill_id,
        "page_count": max(1, len(bill_text) // 3000),
        "version_count": 1,  # TODO: detect from metadata
        "domain": "general"  # TODO: classify from content
    }
    
    route = router.assess_complexity(bill_metadata)
    console.print(f"Route: [bold]{route.value}[/bold]")
    console.print(f"Complexity Score: {router.complexity_score}")
    console.print(f"Breakdown: {router.score_breakdown}")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set in environment[/red]")
        sys.exit(1)
    
    # Step 2: Two-tier analysis
    console.print("\n[cyan]Step 2: Running Two-Tier Analysis...[/cyan]")
    
    orchestrator = TwoTierOrchestrator(
        primary_api_key=api_key,
        judge_api_key=api_key
    )
    
    try:
        result = orchestrator.analyze(
            bill_id=args.bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        sys.exit(1)
    
    # Display results
    console.print(f"\n[green]✓ Analysis Complete[/green]")
    console.print(f"Findings: {len(result.primary_analysis.findings)}")
    console.print(f"Validations: {len(result.judge_validations)}")
    console.print(f"Rubric Scores: {len(result.rubric_scores)}")
    
    # Table of findings
    if result.primary_analysis.findings:
        table = Table(title="Findings", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Statement", style="white", max_width=60)
        table.add_column("Impact", justify="center", width=7)
        table.add_column("Conf", justify="center", width=6)
        table.add_column("Verified", justify="center", width=8)
        
        for idx, finding in enumerate(result.primary_analysis.findings):
            validation = result.judge_validations[idx]
            
            # Determine verification status
            if validation.hallucination_detected:
                verified = "[red]✗ HAL[/red]"  # Hallucination
            elif not validation.quote_verified:
                verified = "[yellow]✗ QV[/yellow]"  # Quote not verified
            else:
                verified = "[green]✓[/green]"
            
            # Truncate long statements
            statement = finding.statement
            if len(statement) > 60:
                statement = statement[:57] + "..."
            
            table.add_row(
                str(idx),
                statement,
                str(finding.impact_estimate),
                f"{finding.confidence:.2f}",
                verified
            )
        
        console.print(table)
    else:
        console.print("[yellow]No findings extracted from bill.[/yellow]")
    
    # Rubric scores table
    if result.rubric_scores:
        rubric_table = Table(title="Rubric Scores")
        rubric_table.add_column("Dimension", style="cyan", width=18)
        rubric_table.add_column("Score", justify="center", width=6)
        rubric_table.add_column("Anchor", style="dim", width=35)
        rubric_table.add_column("Rationale", style="white", max_width=50)
        
        for score in result.rubric_scores:
            # Truncate rationale for display
            rationale = score.rationale
            if len(rationale) > 50:
                rationale = rationale[:47] + "..."
            
            rubric_table.add_row(
                score.dimension,
                str(score.score),
                score.rubric_anchor[:35] if len(score.rubric_anchor) > 35 else score.rubric_anchor,
                rationale
            )
        
        console.print(rubric_table)
    
    # Save to JSON
    output_data = {
        "bill_id": result.bill_id,
        "route": result.route,
        "complexity_score": router.complexity_score,
        "complexity_breakdown": router.score_breakdown,
        "findings_count": len(result.primary_analysis.findings),
        "findings": [
            {
                "statement": f.statement,
                "quotes": [{"text": q.text, "section": q.section, "page": q.page} for q in f.quotes],
                "confidence": f.confidence,
                "impact_estimate": f.impact_estimate
            }
            for f in result.primary_analysis.findings
        ],
        "validations": [
            {
                "finding_id": v.finding_id,
                "quote_verified": v.quote_verified,
                "hallucination_detected": v.hallucination_detected,
                "evidence_quality": v.evidence_quality,
                "ambiguity": v.ambiguity,
                "judge_confidence": v.judge_confidence
            }
            for v in result.judge_validations
        ],
        "rubric_scores": [
            {
                "dimension": s.dimension,
                "score": s.score,
                "rationale": s.rationale,
                "rubric_anchor": s.rubric_anchor,
                "evidence": [{"text": e.text, "section": e.section, "page": e.page} for e in s.evidence]
            }
            for s in result.rubric_scores
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
