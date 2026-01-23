#!/usr/bin/env python3
# run_v2_analysis.py
"""
CLI for Architecture v2.0 Hybrid Discovery-Analysis Pipeline.

Usage:
    python run_v2_analysis.py --bill-id HB123 --bill-text-file path/to/bill.txt

Output:
    - Console tables showing findings, validations, and rubric scores
    - JSON file with complete analysis results

Design:
    - Step 1: Sequential Evolution extracts findings with stability tracking
    - Step 2: Two-Tier validates findings (judge → rubrics)
    - Saves structured output for downstream processing
"""
import os
import sys
import argparse
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.v2.sequential_evolution import SequentialEvolutionAgent, BillVersion
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
    parser.add_argument("--output", default="v2_analysis.json",
                        help="Output JSON file (default: v2_analysis.json)")

    args = parser.parse_args()

    # Validate bill text file exists
    if not os.path.exists(args.bill_text_file):
        console.print(f"[red]Error: Bill text file not found: {args.bill_text_file}[/red]")
        sys.exit(1)

    # Load bill text
    with open(args.bill_text_file, 'r', encoding='utf-8') as f:
        bill_text = f.read()

    # Load NRG context
    nrg_context = load_nrg_context()

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set in environment[/red]")
        sys.exit(1)

    # Step 1: Extract findings with Sequential Evolution
    console.print("\n[cyan]Step 1: Extracting Findings (Sequential Evolution)...[/cyan]")

    try:
        evolution_agent = SequentialEvolutionAgent(api_key=api_key)
        versions = [BillVersion(version_number=1, text=bill_text, name="Current")]
        evolution_result = evolution_agent.walk_versions(bill_id=args.bill_id, versions=versions)
        console.print(f"Findings extracted: {len(evolution_result.findings_registry)}")
    except Exception as e:
        console.print(f"[red]Error during findings extraction: {e}[/red]")
        sys.exit(1)

    # Step 2: Validate findings with Two-Tier
    console.print("\n[cyan]Step 2: Validating Findings (Two-Tier)...[/cyan]")

    orchestrator = TwoTierOrchestrator(judge_api_key=api_key)

    try:
        result = orchestrator.validate(
            bill_id=args.bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context,
            findings_registry=evolution_result.findings_registry,
            stability_scores=evolution_result.stability_scores
        )
    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        sys.exit(1)

    # Display results
    console.print("\n[green]✓ Analysis Complete[/green]")
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

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[green]✓ Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
