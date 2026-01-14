from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()


def display_analysis(item: dict[str, Any], analysis: dict[str, Any]) -> None:
    """
    Display legislative analysis in the console with color-coded formatting.

    Uses Rich library to create visually organized panels showing:
    - Bill/regulation metadata (source, number, sponsor, dates)
    - Impact score with color coding (red/yellow/green)
    - Business impact summary and affected verticals

    Args:
        item: Legislative item with metadata (title, number, source, etc.)
        analysis: LLM analysis results with impact scoring

    Color Coding:
        - Score 7-10: RED panel border (HIGH IMPACT)
        - Score 4-6: YELLOW panel border (MEDIUM IMPACT)
        - Score 0-3: GREEN panel border (LOW IMPACT)
    """
    score = analysis.get("business_impact_score", 0)

    # Color code by impact
    if score >= 7:
        score_color = "red"
        priority = "HIGH IMPACT"
    elif score >= 4:
        score_color = "yellow"
        priority = "MEDIUM IMPACT"
    else:
        score_color = "green"
        priority = "LOW IMPACT"

    # Build metadata section
    metadata_lines = [
        f"[cyan]Source:[/cyan] {item['source']}",
        f"[cyan]Number:[/cyan] {item['number']}",
        f"[cyan]Type:[/cyan] {item['type']}",
    ]

    if 'sponsor' in item and item['sponsor'] != 'Unknown':
        metadata_lines.append(f"[cyan]Sponsor:[/cyan] {item['sponsor']}")
    elif 'agency' in item and item['agency'] != 'Unknown':
        metadata_lines.append(f"[cyan]Agency:[/cyan] {item['agency'].upper()}")

    if 'policy_area' in item and item['policy_area'] != 'Unknown':
        metadata_lines.append(f"[cyan]Policy Area:[/cyan] {item['policy_area']}")

    metadata_lines.append(f"[cyan]Status:[/cyan] {item['status']}")

    if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
        metadata_lines.append(f"[cyan]Introduced:[/cyan] {item['introduced_date']}")
    if 'posted_date' in item and item['posted_date'] != 'Unknown':
        metadata_lines.append(f"[cyan]Posted:[/cyan] {item['posted_date']}")
    if 'comment_end_date' in item and item['comment_end_date'] not in ['N/A', 'Unknown']:
        metadata_lines.append(f"[cyan]Comment Deadline:[/cyan] {item['comment_end_date']}")
    if 'effective_date' in item and item['effective_date'] not in ['N/A', 'Unknown']:
        metadata_lines.append(f"[cyan]Effective:[/cyan] {item['effective_date']}")

    metadata_lines.append(f"[cyan]Link:[/cyan] {item['url']}")
    metadata_lines.insert(3, f"[cyan]Bill Version:[/cyan] {analysis.get('bill_version', 'unknown')}")

    content = f"""
[bold]{item['title']}[/bold]

{chr(10).join(metadata_lines)}

[bold {score_color}]Impact Score: {score}/10[/bold {score_color}]
[cyan]Impact Type:[/cyan] {analysis.get('impact_type', 'N/A')}
[cyan]Risk/Opportunity:[/cyan] {analysis.get('risk_or_opportunity', 'N/A')}

[bold]Business Impact Summary:[/bold]
{analysis.get('impact_summary', 'No summary available')}

[bold cyan]NRG Business Verticals:[/bold cyan]
{', '.join(analysis.get('nrg_business_verticals', ['None identified']))}"""

    # Add legal code changes
    legal_changes = analysis.get('legal_code_changes', {})
    if legal_changes and any(legal_changes.values()):
        content += "\n\n[bold cyan]Legal Code Changes:[/bold cyan]"
        if legal_changes.get('sections_amended'):
            content += f"\n[yellow]Amended:[/yellow] {', '.join(legal_changes['sections_amended'][:3])}"
        if legal_changes.get('sections_added'):
            content += f"\n[green]Added:[/green] {', '.join(legal_changes['sections_added'][:3])}"
        if legal_changes.get('sections_deleted'):
            content += f"\n[red]Deleted:[/red] {', '.join(legal_changes['sections_deleted'][:3])}"

    # Add application scope
    app_scope = analysis.get('application_scope', {})
    if app_scope and (app_scope.get('applies_to') or app_scope.get('exclusions')):
        if app_scope.get('applies_to'):
            content += f"\n\n[bold]Applies To:[/bold] {', '.join(app_scope.get('applies_to', [])[:2])}"
        if app_scope.get('exclusions'):
            content += f"\n[bold]Exclusions:[/bold] {', '.join(app_scope.get('exclusions', [])[:2])}"

    # Add effective dates
    effective_dates = analysis.get('effective_dates', [])
    if effective_dates:
        content += "\n\n[bold]Effective Dates:[/bold]"
        for ed in effective_dates[:2]:
            content += f"\n  • {ed.get('date', 'Unknown')}: {ed.get('applies_to', 'All provisions')}"

    # Affected NRG assets
    affected_assets = analysis.get('affected_nrg_assets', {})
    if affected_assets:
        if affected_assets.get('generation_facilities'):
            content += f"\n\n[bold]Affected Generation:[/bold] {', '.join(affected_assets['generation_facilities'][:3])}"
        if affected_assets.get('geographic_exposure'):
            content += f"\n[bold]Geographic Impact:[/bold] {', '.join(affected_assets['geographic_exposure'][:3])}"

    content += f"""

[cyan]Financial Impact:[/cyan] {analysis.get('financial_impact', 'Unknown')}
[cyan]Timeline:[/cyan] {analysis.get('timeline', 'Unknown')}

[bold]Recommended Action:[/bold] {analysis.get('recommended_action', 'N/A').upper()}
[cyan]Stakeholders:[/cyan] {', '.join(analysis.get('internal_stakeholders', analysis.get('stakeholders', ['None']))[:5])}
"""

    console.print(Panel(content, title=f"[{score_color}]{priority}[/{score_color}]", border_style=score_color))
