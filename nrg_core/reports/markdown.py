import os
from datetime import datetime
from typing import Any, TextIO


def write_item_section(
    f: TextIO,
    item: dict[str, Any],
    analysis: dict[str, Any],
    result: dict[str, Any],
    include_version_timeline: bool = True
) -> None:
    """
    Write a single item section to markdown file.
    Used by high, medium, and low impact sections.
    
    Args:
        f: File handle to write to
        item: Legislative item dictionary
        analysis: LLM analysis results
        result: Full result dictionary (may contain version_analyses, change_data)
        include_version_timeline: Whether to include detailed version timeline
    """
    # Impact Assessment
    f.write("**Impact Assessment:**\n\n")
    f.write(f"- **Score:** {analysis.get('business_impact_score', 0)}/10")
    if analysis.get('business_impact_score', 0) >= 7:
        f.write(" ⚠️")
    f.write("\n")
    f.write(f"- **Type:** {analysis.get('impact_type', 'N/A').replace('_', ' ').title()}\n")
    f.write(f"- **Risk Level:** {analysis.get('risk_or_opportunity', 'N/A').upper()}\n\n")

    # Bill Information
    f.write("**Bill Information:**\n\n")
    f.write(f"- **Source:** {item['source']}\n")
    f.write(f"- **Number:** {item['number']}\n")

    if 'sponsor' in item and item['sponsor'] != 'Unknown':
        f.write(f"- **Sponsor:** {item['sponsor']}\n")
    elif 'agency' in item and item['agency'] != 'Unknown':
        f.write(f"- **Agency:** {item['agency'].upper()}\n")

    if 'policy_area' in item and item['policy_area'] != 'Unknown':
        f.write(f"- **Policy Area:** {item['policy_area']}\n")

    f.write(f"- **Status:** {item['status']}\n")

    if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
        f.write(f"- **Introduced:** {item['introduced_date']}\n")
    if 'posted_date' in item and item['posted_date'] != 'Unknown':
        f.write(f"- **Posted:** {item['posted_date']}\n")
    if 'comment_end_date' in item and item.get('comment_end_date') not in ['N/A', 'Unknown', None]:
        f.write(f"- **Comment Deadline:** {item['comment_end_date']}\n")
    if 'effective_date' in item and item.get('effective_date') not in ['N/A', 'Unknown', None]:
        f.write(f"- **Effective Date:** {item['effective_date']}\n")

    f.write(f"- **Bill Version:** {analysis.get('bill_version', 'unknown')}\n")
    f.write(f"- **Link:** {item['url']}\n\n")

    # Version Timeline (if available and requested)
    if include_version_timeline:
        version_analyses = result.get('version_analyses', [])
        version_diffs = result.get('version_diffs', [])

        if version_analyses and len(version_analyses) > 1:
            _write_version_timeline(f, version_analyses, version_diffs)

    # Change Information
    change_data = result.get('change_data')
    change_impact = result.get('change_impact')

    if change_data and change_data.get('has_changes'):
        if change_data.get('is_new'):
            f.write("**📍 NEW BILL - First Time Analyzed**\n\n")
        else:
            f.write("**⚠️ CHANGES DETECTED**\n\n")
            for change in change_data.get('changes', []):
                f.write(f"- **{change['type'].replace('_', ' ').title()}:** {change.get('summary', 'Change detected')}\n")

            if change_impact:
                f.write("\n**Change Impact Analysis:**\n\n")
                f.write(f"- **Change Impact Score:** {change_impact.get('change_impact_score', 'N/A')}/10\n")
                f.write(f"- **Impact Trend:** {'INCREASED ⬆️' if change_impact.get('impact_increased') else 'DECREASED ⬇️'}\n")
                f.write(f"- **Summary:** {change_impact.get('change_summary', 'N/A')}\n")
                f.write(f"- **Recommended Action:** {change_impact.get('recommended_action', 'N/A').upper()}\n")

            f.write("\n")

    # NRG Business Verticals
    verticals = analysis.get('nrg_business_verticals', [])
    if verticals:
        f.write("**NRG Business Verticals:**\n\n")
        for vertical in verticals:
            f.write(f"- {vertical}\n")
        f.write("\n")

    # Impact Summary
    f.write("**Why This Matters to NRG:**\n")
    f.write(f"{analysis.get('impact_summary', 'No summary available')}\n\n")

    # Legal Code Changes
    legal_changes = analysis.get('legal_code_changes', {})
    if legal_changes and any(legal_changes.values()):
        f.write("**Legal Code Changes:**\n\n")
        if legal_changes.get('sections_amended'):
            f.write("- **Amended:**\n\n")
            for section in legal_changes['sections_amended']:
                f.write(f"    - {section}\n")
        if legal_changes.get('sections_added'):
            f.write("\n- **Added:**\n\n")
            for section in legal_changes['sections_added']:
                f.write(f"    - {section}\n")
        if legal_changes.get('sections_deleted'):
            f.write("\n- **Deleted:**\n\n")
            for section in legal_changes['sections_deleted']:
                f.write(f"    - {section}\n")
        if legal_changes.get('substance_of_changes'):
            f.write(f"\n- **Substance:** {legal_changes['substance_of_changes']}\n")
        f.write("\n")

    # Application Scope
    app_scope = analysis.get('application_scope', {})
    if app_scope and any(app_scope.values()):
        f.write("**Application Scope:**\n\n")
        if app_scope.get('applies_to'):
            f.write("- **Applies To:**\n\n")
            for entity in app_scope['applies_to']:
                f.write(f"    - {entity}\n")
        if app_scope.get('exclusions'):
            f.write("\n- **Exclusions:**\n\n")
            for exclusion in app_scope['exclusions']:
                f.write(f"    - {exclusion}\n")
        if app_scope.get('geographic_scope'):
            f.write("\n- **Geographic:**\n\n")
            for geo in app_scope['geographic_scope']:
                f.write(f"    - {geo}\n")
        f.write("\n")

    # Effective Dates
    effective_dates = analysis.get('effective_dates', [])
    if effective_dates:
        f.write("**Effective Dates:**\n\n")
        for ed in effective_dates:
            f.write(f"- {ed.get('date', 'Unknown')}: {ed.get('applies_to', 'All provisions')}\n")
        f.write("\n")

    # Affected NRG Assets
    affected_assets = analysis.get('affected_nrg_assets', {})
    if affected_assets and any(affected_assets.values()):
        f.write("**Affected NRG Assets:**\n\n")
        if affected_assets.get('generation_facilities'):
            f.write("- **Generation:**\n\n")
            for facility in affected_assets['generation_facilities']:
                f.write(f"    - {facility}\n")
        if affected_assets.get('geographic_exposure'):
            f.write("\n- **Markets:**\n\n")
            for market in affected_assets['geographic_exposure']:
                f.write(f"    - {market}\n")
        if affected_assets.get('business_units'):
            f.write("\n- **Business Units:**\n\n")
            for unit in affected_assets['business_units']:
                f.write(f"    - {unit}\n")
        f.write("\n")

    # Financial and Timeline
    f.write(f"**Financial Estimate:** {analysis.get('financial_impact', 'Unknown')}\n\n")
    f.write(f"**Timeline:** {analysis.get('timeline', 'Unknown')}\n\n")

    # Recommended Actions
    f.write("**Recommended Actions:**\n\n")
    action = analysis.get('recommended_action', 'N/A').upper()
    f.write(f"- ✅ **{action}**")
    if analysis.get('business_impact_score', 0) >= 7:
        f.write(" - Immediate attention required")
    f.write("\n")
    f.write("- Track legislative progress closely\n")
    f.write("- Coordinate with stakeholders (see below)\n\n")

    # Internal Stakeholders
    stakeholders = analysis.get('internal_stakeholders', analysis.get('stakeholders', []))
    if stakeholders:
        f.write("**Internal Stakeholders:**\n\n")
        for stakeholder in stakeholders:
            f.write(f"- {stakeholder}\n")
        f.write("\n")

    f.write("---\n\n")


def _write_version_timeline(
    f: TextIO,
    version_analyses: list[dict[str, Any]],
    version_diffs: list[dict[str, Any]]
) -> None:
    f.write(f"**📚 VERSION TIMELINE ({len(version_analyses)} versions analyzed)**\n\n")
    f.write(f"This bill has evolved through {len(version_analyses)} legislative versions:\n\n")

    for idx, va in enumerate(version_analyses, 1):
        version = va['version']
        v_analysis = va['analysis']
        v_type = version.get('version_type', 'Unknown')
        v_date = version.get('version_date', 'N/A')
        impact_score = v_analysis.get('business_impact_score', 0)
        f.write(f"{idx}. **{v_type}** ({v_date}) - Impact Score: {impact_score}/10\n")

    f.write("\n**Detailed Version Analysis:**\n\n")

    for idx, va in enumerate(version_analyses, 1):
        version = va['version']
        v_analysis = va['analysis']
        v_type = version.get('version_type', 'Unknown')
        v_date = version.get('version_date', 'N/A')
        word_count = version.get('word_count', 0)
        impact_score = v_analysis.get('business_impact_score', 0)
        impact_summary = v_analysis.get('impact_summary', 'No summary available')

        f.write(f"**Version {idx}: {v_type}** ({v_date})\n\n")
        f.write(f"- **Impact Score:** {impact_score}/10\n")
        f.write(f"- **Word Count:** {word_count:,} words\n")
        f.write(f"- **Impact Summary:** {impact_summary[:300]}{'...' if len(impact_summary) > 300 else ''}\n")

        if idx > 1 and len(version_diffs) >= idx - 1:
            diff = version_diffs[idx - 2]
            if diff.get('changed'):
                semantic = diff.get('semantic_analysis', {})
                f.write(f"- **Changes from {diff.get('from_version')}:**\n\n")

                if semantic.get('summary'):
                    f.write(f"  *{semantic['summary']}*\n\n")

                if semantic.get('key_provisions_added'):
                    f.write("  **Provisions Added:**\n")
                    for prov in semantic['key_provisions_added'][:3]:
                        f.write(f"  - {prov}\n")
                    f.write("\n")

                if semantic.get('key_provisions_removed'):
                    f.write("  **Provisions Removed:**\n")
                    for prov in semantic['key_provisions_removed'][:3]:
                        f.write(f"  - {prov}\n")
                    f.write("\n")

                if semantic.get('impact_evolution'):
                    f.write(f"  **Impact Evolution:** {semantic['impact_evolution']}\n\n")

                f.write(f"  *Text changes: {diff.get('lines_added', 0)} lines added, {diff.get('lines_removed', 0)} removed*\n")
            else:
                f.write("- **No substantive changes from previous version**\n")

        f.write("\n")

    f.write("---\n\n")


def generate_markdown_report(
    results: list[dict[str, Any]],
    timestamp: str,
    output_dir: str = None
) -> str:
    """
    Generate comprehensive Markdown report for legislative analysis results.
    
    Args:
        results: List of analysis results
        timestamp: Timestamp for filename (YYYYMMDD_HHMMSS format)
        output_dir: Optional output directory
        
    Returns:
        str: Path to generated markdown file
    """
    # Group by impact level
    high_impact = [r for r in results if r["analysis"].get("business_impact_score", 0) >= 7]
    medium_impact = [r for r in results if 4 <= r["analysis"].get("business_impact_score", 0) < 7]
    low_impact = [r for r in results if r["analysis"].get("business_impact_score", 0) < 4]

    if output_dir:
        md_file = os.path.join(output_dir, f"nrg_analysis_{timestamp}.md")
    else:
        md_file = f"nrg_analysis_{timestamp}.md"

    with open(md_file, "w", encoding="utf-8") as f:
        # Header
        f.write("# NRG Energy Legislative Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Items Analyzed:** {len(results)}\n")
        f.write(f"- **High Impact (7-10):** {len(high_impact)} items - IMMEDIATE ACTION REQUIRED\n")
        f.write(f"- **Medium Impact (4-6):** {len(medium_impact)} items - MONITOR CLOSELY\n")
        f.write(f"- **Low Impact (0-3):** {len(low_impact)} items - AWARENESS ONLY\n\n")
        f.write("---\n\n")

        # High Impact Section
        if high_impact:
            f.write("## 🔴 HIGH IMPACT ITEMS (Score: 7-10)\n\n")
            f.write("*These items require immediate engagement with Government Affairs.*\n\n")

            for i, result in enumerate(high_impact, 1):
                item = result["item"]
                analysis = result["analysis"]
                f.write(f"### {i}. {item['number']} - {item['title']}\n\n")
                write_item_section(f, item, analysis, result, include_version_timeline=True)

        # Medium Impact Section
        if medium_impact:
            f.write("## 🟡 MEDIUM IMPACT ITEMS (Score: 4-6)\n\n")
            f.write("*Monitor these items and prepare response strategies.*\n\n")

            for i, result in enumerate(medium_impact, 1):
                item = result["item"]
                analysis = result["analysis"]
                f.write(f"### {i}. {item['number']} - {item['title']}\n\n")
                write_item_section(f, item, analysis, result, include_version_timeline=True)

        # Low Impact Section
        if low_impact:
            f.write("## 🟢 LOW IMPACT ITEMS (Score: 0-3)\n\n")
            f.write("*For awareness only - minimal action required.*\n\n")

            for i, result in enumerate(low_impact, 1):
                item = result["item"]
                analysis = result["analysis"]
                f.write(f"### {i}. {item['number']} - {item['title']}\n\n")
                write_item_section(f, item, analysis, result, include_version_timeline=False)

    return md_file
