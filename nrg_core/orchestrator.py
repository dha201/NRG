import os
import json
import tempfile
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

from nrg_core.config import load_config, load_nrg_context
from nrg_core.db.cache import (
    init_database,
    get_cached_bill,
    save_bill_to_cache,
    save_bill_version,
    save_version_analysis,
    get_cached_analysis_by_hash,
)
from nrg_core.api import (
    fetch_congress_bills,
    fetch_openstates_bills,
    fetch_bill_versions_from_openstates,
    fetch_bill_versions_from_congress,
    fetch_regulations,
)
from nrg_core.analysis import (
    analyze_with_openai,
    analyze_with_gemini,
    analyze_bill_version,
    detect_bill_changes,
    analyze_changes_with_llm,
    compare_consecutive_versions,
    analyze_version_changes_with_llm,
)
from nrg_core.reports import (
    display_analysis,
    generate_markdown_report,
    convert_markdown_to_word,
)
from nrg_core.utils import get_cost_summary

load_dotenv()
console = Console()


def _init_llm_clients() -> tuple[Any, Any]:
    # Setup LLM clients for both OpenAI and Gemini (we'll pick one later)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    os.environ["GOOGLE_GENAI_DISABLE_NON_TEXT_WARNINGS"] = "true"
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return openai_client, gemini_client


def _fetch_all_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    # Gather bills/regs from all configured sources (Congress, Regulations, Open States)
    all_items = []

    # Congress.gov
    if config.get('sources', {}).get('congress', {}).get('enabled', True):
        limit = config['sources']['congress'].get('limit', 3)
        all_items.extend(fetch_congress_bills(limit=limit))

    # Regulations.gov
    if config.get('sources', {}).get('regulations', {}).get('enabled', True):
        limit = config['sources']['regulations'].get('limit', 3)
        all_items.extend(fetch_regulations(limit=limit))

    # Open States (Texas bills)
    openstates_config = config.get('sources', {}).get('openstates', {})
    if openstates_config.get('enabled', False):
        texas_config = openstates_config.get('texas_bills', {})
        if texas_config.get('enabled', False):
            jurisdiction = texas_config.get('jurisdiction', 'TX')
            bill_numbers = texas_config.get('bills', [])
            if bill_numbers:
                texas_bills = fetch_openstates_bills(
                    jurisdiction=jurisdiction,
                    bill_numbers=bill_numbers,
                    limit=len(bill_numbers)
                )
                all_items.extend(texas_bills)

    return all_items


def _fetch_versions(items: list[dict[str, Any]], config: dict[str, Any]) -> None:
    # Pull full version history for each bill (if tracking enabled) for change analysis
    version_tracking_enabled = config.get('version_tracking', {}).get('enabled', False)
    version_tracking_openstates = config.get('version_tracking', {}).get('sources', {}).get('openstates', True)
    version_tracking_congress = config.get('version_tracking', {}).get('sources', {}).get('congress', False)

    if not version_tracking_enabled:
        return

    if not (version_tracking_openstates or version_tracking_congress):
        return

    console.print("\n[bold cyan]📚 Fetching bill versions...[/bold cyan]\n")

    for item in items:
        # Open States bills
        if version_tracking_openstates and item['source'] == 'Open States' and 'openstates_id' in item:
            bill_number = item['number']
            openstates_id = item['openstates_id']
            console.print(f"[cyan]Processing {bill_number}...[/cyan]")

            versions = fetch_bill_versions_from_openstates(openstates_id, bill_number)
            item['versions'] = versions if versions else []

            if versions:
                console.print(f"[green]✓ Loaded {len(versions)} versions for {bill_number}[/green]\n")
            else:
                console.print(f"[yellow]⚠ No versions found for {bill_number}[/yellow]\n")

        # Congress.gov bills
        elif version_tracking_congress and item['source'] == 'Congress.gov':
            if 'congress_num' in item and 'bill_type' in item and 'bill_number' in item:
                bill_number = item['number']
                console.print(f"[cyan]Processing {bill_number}...[/cyan]")

                versions = fetch_bill_versions_from_congress(
                    item['congress_num'],
                    item['bill_type'],
                    item['bill_number'],
                    bill_number
                )
                item['versions'] = versions if versions else []

                if versions:
                    console.print(f"[green]✓ Loaded {len(versions)} versions for {bill_number}[/green]\n")
                else:
                    console.print(f"[yellow]⚠ No versions found for {bill_number}[/yellow]\n")


def _analyze_with_llm(item: dict[str, Any], nrg_context: str, config: dict[str, Any],
                      openai_client: Any, gemini_client: Any) -> dict[str, Any]:
    # Router: delegate to configured LLM provider (gemini or openai) based on config
    provider = config['llm']['provider']
    if provider == 'gemini':
        return analyze_with_gemini(
            item, nrg_context,
            gemini_client=gemini_client,
            config=config
        )
    else:
        return analyze_with_openai(
            item, nrg_context,
            openai_client=openai_client
        )


def _analyze_item_with_versions(
    item: dict[str, Any],
    nrg_context: str,
    config: dict[str, Any],
    db_conn: Any,
    change_data: Optional[dict[str, Any]],
    openai_client: Any,
    gemini_client: Any
) -> dict[str, Any]:
    
    llm_provider = config['llm']['provider']
    change_tracking_enabled = config.get('change_tracking', {}).get('enabled', False)
    bill_id = f"{item['source']}:{item['number']}"

    console.print(f"[cyan]  Analyzing {item['number']} ({len(item['versions'])} versions) with {llm_provider.upper()}...[/cyan]")

    version_analyses = []
    version_diffs = []

    for v_idx, version in enumerate(item['versions'], 1):
        console.print(f"[dim]    Version {v_idx}/{len(item['versions'])}: {version['version_type']}[/dim]")

    # Cache check: skip LLM call if we've seen this exact text before (saves money/time)
        text_hash = version.get('text_hash')
        cached_analysis = None

        if change_tracking_enabled and db_conn and text_hash:
            cached_analysis = get_cached_analysis_by_hash(text_hash, db_conn)

        if cached_analysis:
            console.print(f"[green]      ✓ Using cached analysis (hash match)[/green]")
            version_analysis = cached_analysis
        else:
            # Run LLM analysis on this version (only if not cached)
            version_analysis = analyze_bill_version(
                version['full_text'],
                version['version_type'],
                item,
                nrg_context,
                config=config,
                openai_client=openai_client,
                gemini_client=gemini_client
            )

        version_analyses.append({
            'version': version,
            'analysis': version_analysis
        })

        # Persist version + analysis to DB for future cache hits and change tracking
        if change_tracking_enabled and db_conn:
            version_id = save_bill_version(bill_id, version, db_conn)
            save_version_analysis(version_id, version_analysis, db_conn)

        # Diff analysis: compare against previous version to spot substantive changes
        if v_idx > 1:
            prev_version = item['versions'][v_idx - 2]
            prev_analysis = version_analyses[v_idx - 2]['analysis']

            diff = compare_consecutive_versions(prev_version, version)

            # Only do LLM semantic analysis if text actually changed
            if diff.get('changed', False):
                console.print(f"[dim]      Analyzing substantive changes from {prev_version.get('version_type')} to {version.get('version_type')}...[/dim]")
                semantic_changes = analyze_version_changes_with_llm(
                    prev_version, version,
                    prev_analysis, version_analysis,
                    item, nrg_context, config,
                    gemini_client=gemini_client,
                    openai_client=openai_client
                )
                diff['semantic_analysis'] = semantic_changes
            else:
                console.print(f"[dim]      ✓ No text changes between {prev_version.get('version_type')} and {version.get('version_type')}[/dim]")
                diff['semantic_analysis'] = {
                    "summary": "No text changes detected between versions",
                    "key_provisions_added": [],
                    "key_provisions_removed": [],
                    "key_provisions_modified": [],
                    "impact_evolution": "Impact score unchanged",
                    "compliance_changes": "No compliance changes",
                    "strategic_significance": "No strategic changes"
                }
            
            version_diffs.append(diff)

    # Primary analysis = most recent version (first in list) for main report
    analysis = version_analyses[0]['analysis'] if version_analyses else {}

    # Bill-level change impact (if bill was modified) for executive summary
    change_impact = None
    if change_data and change_data['has_changes'] and not change_data.get('is_new', False):
        if config.get('change_tracking', {}).get('analyze_changes_with_llm', True):
            console.print(f"[dim]    Analyzing bill-level change impact...[/dim]")
            change_impact = analyze_changes_with_llm(
                item, change_data, nrg_context, config,
                gemini_client=gemini_client,
                openai_client=openai_client
            )

    # Return combined results for downstream processing and report generation
    return {
        "item": item,
        "analysis": analysis,
        "version_analyses": version_analyses,
        "version_diffs": version_diffs,
        "change_data": change_data,
        "change_impact": change_impact
    }


def _analyze_item_simple(
    item: dict[str, Any],
    nrg_context: str,
    config: dict[str, Any],
    change_data: Optional[dict[str, Any]],
    openai_client: Any,
    gemini_client: Any
) -> dict[str, Any]:
    # Analyze single-version item (no version history) - much faster path
    analysis = _analyze_with_llm(item, nrg_context, config, openai_client, gemini_client)

    change_impact = None
    if change_data and change_data['has_changes']:
        if config.get('change_tracking', {}).get('analyze_changes_with_llm', True):
            console.print(f"[dim]    Analyzing change impact...[/dim]")
            change_impact = analyze_changes_with_llm(
                item, change_data, nrg_context, config,
                gemini_client=gemini_client,
                openai_client=openai_client
            )

    return {
        "item": item,
        "analysis": analysis,
        "change_data": change_data,
        "change_impact": change_impact
    }


def _generate_reports(results: list[dict[str, Any]], timestamp: str) -> tuple[str, str, Optional[str]]:
    # Write output files: JSON (raw data), Markdown (readable), DOCX (Word) for stakeholders
    output_dir = f"nrg_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[cyan]Created output directory: {output_dir}/[/cyan]")

    # JSON report - machine-readable for downstream processing
    json_file = os.path.join(output_dir, f"nrg_analysis_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]✓ JSON results saved to {json_file}[/green]")

    # Markdown report - human-readable for business users
    console.print("[cyan]Generating markdown report...[/cyan]")
    md_file = generate_markdown_report(results, timestamp, output_dir)
    console.print(f"[green]✓ Markdown report saved to {md_file}[/green]")

    # Word document - executive-friendly format for meetings
    console.print("[cyan]Generating Word document...[/cyan]")
    docx_file = convert_markdown_to_word(md_file)
    if docx_file:
        console.print(f"[green]✓ Word document saved to {docx_file}[/green]\n")
    else:
        console.print("[dim]  (Word document generation skipped)[/dim]\n")

    return json_file, md_file, docx_file


def _print_summary(results: list[dict[str, Any]], change_tracking_enabled: bool) -> None:
    # Print impact breakdown + cost summary (if LLM was used) for quick insights
    high_impact = sum(1 for r in results if r["analysis"].get("business_impact_score", 0) >= 7)
    medium_impact = sum(1 for r in results if 4 <= r["analysis"].get("business_impact_score", 0) < 7)
    low_impact = sum(1 for r in results if r["analysis"].get("business_impact_score", 0) < 4)

    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  High Impact (7-10):   {high_impact}")
    console.print(f"  Medium Impact (4-6):  {medium_impact}")
    console.print(f"  Low Impact (0-3):     {low_impact}")

    if change_tracking_enabled:
        new_bills = sum(1 for r in results if r.get('change_data', {}).get('is_new', False))
        modified_bills = sum(1 for r in results if r.get('change_data', {}).get('has_changes', False) and not r.get('change_data', {}).get('is_new', False))
        unchanged_bills = sum(1 for r in results if not r.get('change_data', {}).get('has_changes', False))
        console.print(f"\n[bold]Change Tracking:[/bold]")
        console.print(f"  New Bills:        {new_bills}")
        console.print(f"  Modified Bills:   {modified_bills}")
        console.print(f"  Unchanged Bills:  {unchanged_bills}")

    cost_summary = get_cost_summary()
    if cost_summary['total_calls'] > 0:
        console.print(f"\n[bold]LLM Cost Estimate:[/bold]")
        console.print(f"  Total Calls:      {cost_summary['total_calls']}")
        console.print(f"  Input Tokens:     {cost_summary['total_input_tokens']:,}")
        console.print(f"  Output Tokens:    {cost_summary['total_output_tokens']:,}")
        console.print(f"  Estimated Cost:   ${cost_summary['estimated_cost_usd']:.4f}")


def run_analysis() -> None:
    """
    1. Configuration - Load config.yaml, init database, load context
    2. Data Collection - Fetch from Congress.gov, Regulations.gov, Open States
    3. Version Tracking - Fetch all versions of each bill
    4. Change Detection - Compare current vs cached data
    5. LLM Analysis - Analyze each item/version with configured LLM
    6. Report Generation - JSON, Markdown, Word output
    """
    console.print("\n[bold magenta]═══════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  NRG Energy Legislative Tracker  [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/bold magenta]\n")

    # Load LLM config and display settings so user knows what's running
    config = load_config()
    llm_provider = config['llm']['provider']
    llm_model = config['llm'][llm_provider]['model']
    change_tracking_enabled = config.get('change_tracking', {}).get('enabled', False)

    console.print(f"[bold cyan]🔧 Configuration:[/bold cyan]")
    console.print(f"  LLM Provider: {llm_provider.upper()} ({llm_model})")
    console.print(f"  Change Tracking: {'ENABLED' if change_tracking_enabled else 'DISABLED'}")
    console.print()

    # LLM client init - we need both ready since config might change
    openai_client, gemini_client = _init_llm_clients()

    # DB init for change tracking - only if enabled to save resources
    db_conn = None
    if change_tracking_enabled:
        console.print("[cyan]Initializing change tracking database...[/cyan]")
        temp_dir = os.environ.get('TEMP', tempfile.gettempdir())
        db_filename = config.get('change_tracking', {}).get('database', 'bill_cache.db')
        db_path = os.path.join(temp_dir, db_filename)
        db_conn = init_database(db_path)
        console.print(f"[green]✓ Database ready: {db_path}[/green]")

    # Load NRG context - gives LLM the business background for relevant analysis
    console.print("[cyan]Loading NRG business context...[/cyan]")
    nrg_context = load_nrg_context()
    console.print("[green]✓ Context loaded[/green]")

    # Fetch all items from configured sources (Congress, Regulations, Open States)
    all_items = _fetch_all_items(config)
    console.print(f"\n[bold green]Total items collected: {len(all_items)}[/bold green]\n")

    if not all_items:
        console.print("[red]No items found. Please check API keys and try again.[/red]")
        if db_conn:
            db_conn.close()
        return

    _fetch_versions(all_items, config)

    # Analyze items
    if change_tracking_enabled and db_conn:
        console.print("\n[bold cyan]🔍 Checking for changes...[/bold cyan]\n")
    console.print(f"\n[bold cyan]🤖 Analyzing with {llm_provider.upper()} ({llm_model})...[/bold cyan]\n")

    results = []
    analyze_all_versions = config.get('version_tracking', {}).get('analyze_all_versions', True)

    for i, item in enumerate(all_items, 1):
        bill_id = f"{item['source']}:{item['number']}"

        # Check for changes against cache to avoid re-analyzing unchanged bills
        change_data = None
        if change_tracking_enabled and db_conn:
            cached_bill = get_cached_bill(bill_id, db_conn)
            change_data = detect_bill_changes(cached_bill, item)

            if change_data['has_changes']:
                if change_data['is_new']:
                    console.print(f"[green]  ✨ NEW: {item['number']} ({item['source']})[/green]")
                else:
                    change_types = [c['type'] for c in change_data['changes']]
                    console.print(f"[yellow]  ⚠️  CHANGED: {item['number']} - {', '.join(change_types)}[/yellow]")
            else:
                console.print(f"[dim]  ✓ No changes: {item['number']}[/dim]")

        # Analyze - pick fast path (simple) or thorough path (with versions)
        has_versions = 'versions' in item and len(item.get('versions', [])) > 0

        if has_versions and analyze_all_versions:
            result = _analyze_item_with_versions(
                item, nrg_context, config, db_conn, change_data,
                openai_client, gemini_client
            )
        else:
            console.print(f"[dim]  Analyzing {i}/{len(all_items)} with {llm_provider.upper()}...[/dim]")
            result = _analyze_item_simple(
                item, nrg_context, config, change_data,
                openai_client, gemini_client
            )

        results.append(result)

        # Save to cache for next run's change detection
        if change_tracking_enabled and db_conn:
            save_bill_to_cache(item, db_conn)

    console.print("\n[bold magenta]═══════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  ANALYSIS RESULTS  [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/bold magenta]\n")

    # Sort by business impact so high-priority items appear first in reports
    results.sort(
        key=lambda x: x["analysis"].get("business_impact_score", 0) if isinstance(x["analysis"], dict) else 0,
        reverse=True
    )

    for result in results:
        display_analysis(result["item"], result["analysis"])

    # Generate reports in all formats for different stakeholder needs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _generate_reports(results, timestamp)

    _print_summary(results, change_tracking_enabled)

    console.print()

    if db_conn:
        db_conn.close()
