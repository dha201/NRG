"""
Gemini Deep Research Agent integration for comprehensive bill analysis.

Uses the Interactions API (Beta) with the deep-research-pro-preview-12-2025 agent
to perform autonomous, long-running research tasks that iterate on web search
and document analysis.

Key characteristics:
- Agentic workflow: Single request triggers autonomous loop of planning, searching, reading
- Long-running: Research tasks take minutes (max 60 min)
- Citation-rich: Provides granular sourcing for all claims
- Can combine web search with provided documents (bill text, NRG context)
"""

import json
import os
import time
from typing import Any, Optional

from rich.console import Console

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

console = Console()

DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"
DEFAULT_POLL_INTERVAL = 10  # seconds
DEFAULT_MAX_WAIT = 3600     # 60 minutes max


def create_deep_research_client():
    """
    Returns:
        google.genai.Client instance
        
    Raises:
        ImportError: If google-genai package not installed
        ValueError: If GOOGLE_API_KEY not set
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package required for Deep Research. "
            "Install with: pip install google-genai"
        )
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return genai.Client(api_key=api_key)


def build_deep_research_prompt(
    bills: list[dict[str, Any]],
    nrg_context: str,
    research_focus: Optional[str] = None
) -> str:
    """
    Args:
        bills: List of bill dicts with number, title, versions, etc.
        nrg_context: Full NRG business context
        research_focus: Optional specific focus area for research
        
    Returns:
        Complete prompt string for Deep Research
    """
    # Build bill summaries
    bill_sections = []
    for bill in bills:
        versions_text = ""
        if bill.get("versions"):
            versions_text = "\n".join([
                f"  - {v.get('version_type', 'Unknown')}: {len(v.get('full_text', ''))} chars"
                for v in bill["versions"]
            ])
        
        # Get the most recent version's full text
        full_text = ""
        if bill.get("versions") and bill["versions"]:
            full_text = bill["versions"][0].get("full_text", "")
        elif bill.get("summary"):
            full_text = bill.get("summary", "")
        
        bill_section = f"""
### {bill.get('number', 'Unknown Bill')} - {bill.get('title', 'No Title')}
**Source:** {bill.get('source', 'Unknown')}
**Status:** {bill.get('status', 'Unknown')}
**Sponsor:** {bill.get('sponsor', 'Unknown')}
**Versions Available:**
{versions_text or '  - No version history'}

**Full Text (Most Recent Version):**
{full_text}
"""
        bill_sections.append(bill_section)
    
    bills_content = "\n---\n".join(bill_sections)
    
    # Default research focus if not provided
    if not research_focus:
        research_focus = """
1. **Legal Analysis**: What specific code sections are being amended, added, or deleted?
2. **Compliance Requirements**: What mandatory (SHALL/MUST) vs permissive (MAY) provisions exist?
3. **Business Impact Assessment**: How does this affect NRG's operations, financials, and strategy?
4. **Market Context**: What industry trends, competitor actions, or market conditions are relevant?
5. **Regulatory Landscape**: What related regulations or pending rules should be considered?
6. **Timeline Analysis**: What are the effective dates and implementation deadlines?
7. **Stakeholder Mapping**: Which NRG business units and internal stakeholders are affected?
"""

    prompt = f"""You are conducting deep research for NRG Energy's Government Affairs team on legislative bills that may impact their business operations.

## RESEARCH OBJECTIVE

Analyze the following legislation and produce a comprehensive research report that combines:
1. Deep analysis of the bill text and legal implications
2. Web research on related regulatory context, industry impacts, and market trends
3. Assessment of specific impacts on NRG Energy's business operations

## NRG ENERGY BUSINESS CONTEXT

{nrg_context}

## BILLS TO ANALYZE

{bills_content}

## RESEARCH FOCUS AREAS

{research_focus}

## OUTPUT REQUIREMENTS

Generate a SEPARATE section for EACH bill using this EXACT structure. This format is critical for downstream processing.

---

## Executive Summary

- **Total Items Analyzed:** [count]
- **High Impact (7-10):** [count] items - IMMEDIATE ACTION REQUIRED
- **Medium Impact (4-6):** [count] items - MONITOR CLOSELY  
- **Low Impact (0-3):** [count] items - AWARENESS ONLY

---

## [IMPACT TIER EMOJI] [BILL NUMBER] - [BILL TITLE]

### Impact Assessment

| Field | Value |
|-------|-------|
| **Score** | [X]/10 |
| **Type** | [Regulatory Compliance / Operational / Tax / Environmental / etc.] |
| **Risk Level** | [RISK / OPPORTUNITY / NEUTRAL] |

### Bill Information

| Field | Value |
|-------|-------|
| **Source** | [Open States / Congress.gov / etc.] |
| **Number** | [Bill Number] |
| **Status** | [Current Status] |
| **Introduced** | [Date] |
| **Link** | [URL to bill] |

### NRG Business Verticals

List the affected verticals:
- [Vertical 1]
- [Vertical 2]

### Why This Matters to NRG

[2-3 paragraph narrative explaining the business relevance in plain language]

### Legal Code Changes

**Added:**
- [New code sections]

**Amended:**
- [Modified code sections]

**Substance:** [Summary of what the legal changes actually do]

### Application Scope

**Applies To:**
- [Entity type 1]
- [Entity type 2]

**Exclusions:**
- [What/who is excluded]

**Geographic:**
- [Jurisdiction/region]

### Effective Dates

- [Date]: [What takes effect]

### Provision Types

- **Mandatory ([count]):** [Key SHALL/MUST provisions]
- **Permissive ([count]):** [Key MAY provisions]

### Exceptions & Exemptions

**Exceptions:**
- [List exceptions]

**Exemptions:**
- [List exemptions]

### Affected NRG Assets

**Markets:**
- [Market 1]
- [Market 2]

**Business Units:**
- [Unit 1]
- [Unit 2]

### Key Provisions Relevant to NRG

- [Quoted provision 1 with section reference]
- [Quoted provision 2 with section reference]

### Financial Estimate

[Cost/revenue impact estimate with rationale]

### Timeline

[Key dates and implementation milestones]

### Recommended Actions

- [🔴/🟡/🟢/ℹ️] **[ACTION]** - [Description]
- [Additional action items]

### Internal Stakeholders

- [Stakeholder 1]
- [Stakeholder 2]

### Impact by Business Vertical

- **[Vertical 1]:** [Specific impact description]
- **[Vertical 2]:** [Specific impact description]

---

**Use these impact tier emojis:**
- 🔴 HIGH IMPACT (Score 7-10)
- 🟡 MEDIUM IMPACT (Score 4-6)
- 🟢 LOW IMPACT (Score 0-3)

**Important**: Generate a COMPLETE section for EACH bill following this exact structure. Be thorough but focus on NRG-relevant impacts. Use web search to find current market context and regulatory developments. If specific information is unavailable, state "Unknown" rather than speculating.
"""
    
    return prompt


def start_deep_research(
    client,
    prompt: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_wait: int = DEFAULT_MAX_WAIT,
    verbose: bool = True
) -> dict[str, Any]:
    """    
    Args:
        client: Google GenAI client
        prompt: Research prompt
        poll_interval: Not used in streaming mode (kept for API compatibility)
        max_wait: Maximum seconds to wait (default 3600 = 60 min)
        verbose: Print progress updates and stream content to console
        
    Returns:
        Dict with:
        - status: "completed" or "failed" or "timeout"
        - report: Final research report text (if completed)
        - interaction_id: ID for follow-up questions
        - duration_seconds: Total time taken
        - error: Error message (if failed)
        
    Note:
        Uses streaming for real-time console output and citation display.
        All citations from Deep Research are preserved in the report.
    """
    start_time = time.time()
    
    if verbose:
        console.print(f"[cyan]Starting Deep Research task with streaming...[/cyan]")
        console.print(f"[dim]Agent: {DEEP_RESEARCH_AGENT}[/dim]")
        console.print(f"[dim]Prompt length: {len(prompt):,} characters[/dim]")
    
    interaction_id = None
    last_event_id = None
    report_chunks = []
    is_complete = False
    
    try:
        # Start streaming research task
        # Ref: https://ai.google.dev/gemini-api/docs/deep-research#streaming
        stream = client.interactions.create(
            input=prompt,
            agent=DEEP_RESEARCH_AGENT,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "auto"  # Show research progress
            }
        )
        
        if verbose:
            console.print("[dim]═══ Streaming Research Output ═══[/dim]\n")
        
        # Process stream events
        for chunk in stream:
            # Capture interaction ID from start event
            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                if verbose:
                    console.print(f"[green]Research started: {interaction_id}[/green]\n")
            
            # Track event ID for reconnection capability
            if chunk.event_id:
                last_event_id = chunk.event_id
            
            # Handle content deltas (text and thinking summaries)
            if chunk.event_type == "content.delta":
                if chunk.delta.type == "text":
                    # Stream text to console with citations intact
                    text_chunk = chunk.delta.text
                    report_chunks.append(text_chunk)
                    if verbose:
                        print(text_chunk, end="", flush=True)
                        
                elif chunk.delta.type == "thought_summary":
                    # Display thinking process
                    if verbose:
                        thought = chunk.delta.content.text
                        console.print(f"\n[dim]💭 {thought}[/dim]\n")
            
            # Check for completion
            elif chunk.event_type == "interaction.complete":
                is_complete = True
                if verbose:
                    console.print("\n\n[green]✓ Research complete[/green]")
                # Don't break - continue to fetch final report below
                
            elif chunk.event_type == "error":
                error_msg = getattr(chunk, 'error', 'Unknown error')
                if verbose:
                    console.print(f"\n[yellow]Stream error: {error_msg}[/yellow]")
                # Don't return early - try fetching report anyway if we have an interaction_id
                # Gateway timeouts often occur after research completes but before stream ends
                break
        
        # Streaming only provides thinking summaries - fetch final report from interaction
        report_text = "".join(report_chunks)  # Will be empty if text didn't stream
        
        if interaction_id:
            # Stream may have timed out before research completed - poll until done
            # Then fetch the actual report from the completed interaction
            poll_start = time.time()
            remaining_wait = max_wait - (poll_start - start_time)
            
            while remaining_wait > 0:
                try:
                    interaction = client.interactions.get(interaction_id)
                    status = getattr(interaction, 'status', None)
                    
                    if verbose:
                        console.print(f"[dim]Interaction status: {status}[/dim]")
                    
                    # Check if research is complete
                    if status in ('COMPLETED', 'completed', 'DONE', 'done'):
                        is_complete = True
                        # Extract report from outputs
                        if interaction.outputs:
                            last_output = interaction.outputs[-1]
                            if hasattr(last_output, 'text') and last_output.text:
                                report_text = last_output.text
                            elif hasattr(last_output, 'content') and last_output.content:
                                report_text = str(last_output.content)
                        
                        if report_text:
                            if verbose:
                                console.print(f"[green]✓ Retrieved report: {len(report_text):,} characters[/green]")
                        break
                    
                    elif status in ('FAILED', 'failed', 'ERROR', 'error'):
                        if verbose:
                            console.print(f"[red]Research failed on server[/red]")
                        break
                    
                    # Still running - wait and poll again
                    if verbose:
                        elapsed = time.time() - start_time
                        console.print(f"[dim]Research still running... ({elapsed:.0f}s elapsed)[/dim]")
                    time.sleep(poll_interval)
                    remaining_wait = max_wait - (time.time() - start_time)
                    
                except Exception as poll_err:
                    if verbose:
                        console.print(f"[yellow]Poll error: {poll_err}[/yellow]")
                    time.sleep(poll_interval)
                    remaining_wait = max_wait - (time.time() - start_time)
        
        duration = time.time() - start_time
        
        if verbose:
            console.print(f"[dim]Report length: {len(report_text):,} characters[/dim]")
            console.print(f"[dim]Duration: {duration:.1f}s[/dim]")
        
        return {
            "status": "completed" if is_complete else "incomplete",
            "report": report_text,
            "interaction_id": interaction_id,
            "duration_seconds": duration,
            "last_event_id": last_event_id
        }
        
    except Exception as e:
        error_msg = f"Streaming failed: {str(e)}"
        if verbose:
            console.print(f"[red]{error_msg}[/red]")
        
        return {
            "status": "failed",
            "error": error_msg,
            "interaction_id": interaction_id,
            "duration_seconds": time.time() - start_time
        }


def follow_up_research(
    client,
    interaction_id: str,
    question: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_wait: int = DEFAULT_MAX_WAIT,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Ask a follow-up question on a completed Deep Research interaction.
    
    Args:
        client: Google GenAI client
        interaction_id: ID from previous research task
        question: Follow-up question
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait
        verbose: Print progress updates
        
    Returns:
        Same structure as start_deep_research()
    """
    start_time = time.time()
    
    if verbose:
        console.print(f"[cyan]Asking follow-up question...[/cyan]")
    
    try:
        interaction = client.interactions.create(
            input=question,
            agent=DEEP_RESEARCH_AGENT,
            previous_interaction_id=interaction_id,
            background=True
        )
        
        new_interaction_id = interaction.id
        if verbose:
            console.print(f"[green]Follow-up started: {new_interaction_id}[/green]")
            
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Failed to start follow-up: {str(e)}",
            "duration_seconds": time.time() - start_time
        }
    
    return _poll_interaction(
        client, new_interaction_id, poll_interval, max_wait, verbose, start_time
    )


def _poll_interaction(
    client,
    interaction_id: str,
    poll_interval: int,
    max_wait: int,
    verbose: bool,
    start_time: float
) -> dict[str, Any]:
    # Shared polling logic reused by start_deep_research and follow_up_research
    elapsed = 0
    last_status = None
    
    while elapsed < max_wait:
        try:
            interaction = client.interactions.get(interaction_id)
            status = interaction.status
            
            if status != last_status and verbose:
                console.print(f"[dim]Status: {status} ({int(elapsed)}s elapsed)[/dim]")
                last_status = status
            
            if status == "completed":
                report_text = ""
                if interaction.outputs:
                    report_text = interaction.outputs[-1].text
                
                duration = time.time() - start_time
                if verbose:
                    console.print(f"[green]✓ Completed in {duration:.1f}s[/green]")
                
                return {
                    "status": "completed",
                    "report": report_text,
                    "interaction_id": interaction_id,
                    "duration_seconds": duration
                }
                
            elif status == "failed":
                error_msg = getattr(interaction, 'error', 'Unknown error')
                return {
                    "status": "failed",
                    "error": str(error_msg),
                    "interaction_id": interaction_id,
                    "duration_seconds": time.time() - start_time
                }
            
            time.sleep(poll_interval)
            elapsed = time.time() - start_time
            
        except Exception as e:
            if verbose:
                console.print(f"[yellow]Poll error: {e}, retrying...[/yellow]")
            time.sleep(poll_interval)
            elapsed = time.time() - start_time
    
    return {
        "status": "timeout",
        "error": f"Did not complete within {max_wait}s",
        "interaction_id": interaction_id,
        "duration_seconds": max_wait
    }


def analyze_bills_with_deep_research(
    bills: list[dict[str, Any]],
    nrg_context: str,
    config: dict[str, Any],
    research_focus: Optional[str] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    High-level function to analyze bills using Deep Research.
    
    This is the main entry point for the Deep Research integration.
    Pass all bills at once - the agent will iterate autonomously.
    
    Args:
        bills: List of bill dicts with versions, text, metadata
        nrg_context: NRG business context string
        config: Configuration dict with deep_research settings
        research_focus: Optional specific focus areas
        verbose: Print progress updates
        
    Returns:
        Dict with:
        - status: "completed", "failed", or "timeout"
        - report: Research report text
        - interaction_id: For follow-up questions
        - duration_seconds: Time taken
        - bills_analyzed: List of bill numbers analyzed
        - error: Error message if failed
    """
    if not bills:
        return {
            "status": "failed",
            "error": "No bills provided for analysis",
            "duration_seconds": 0
        }
    
    # Get Deep Research config
    dr_config = config.get("deep_research", {})
    poll_interval = dr_config.get("poll_interval", DEFAULT_POLL_INTERVAL)
    max_wait = dr_config.get("max_wait", DEFAULT_MAX_WAIT)
    
    if verbose:
        console.print(f"\n[bold cyan]═══ Deep Research Analysis ═══[/bold cyan]")
        console.print(f"[cyan]Analyzing {len(bills)} bill(s) with Gemini Deep Research[/cyan]")
        for bill in bills:
            console.print(f"[dim]  • {bill.get('number', 'Unknown')}: {bill.get('title', 'No title')[:60]}...[/dim]")
    
    # Create client
    try:
        client = create_deep_research_client()
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "duration_seconds": 0
        }
    
    # Build prompt
    prompt = build_deep_research_prompt(bills, nrg_context, research_focus)
    
    # Run research
    result = start_deep_research(
        client=client,
        prompt=prompt,
        poll_interval=poll_interval,
        max_wait=max_wait,
        verbose=verbose
    )
    
    # Add metadata
    result["bills_analyzed"] = [b.get("number", "Unknown") for b in bills]
    result["prompt_length"] = len(prompt)
    
    return result


def save_deep_research_report(
    result: dict[str, Any],
    output_dir: str,
    timestamp: str
) -> Optional[str]:
    """
    Save Deep Research report to file.
    
    Args:
        result: Result dict from analyze_bills_with_deep_research
        output_dir: Directory to save to
        timestamp: Timestamp string for filename
        
    Returns:
        Path to saved file, or None if failed
    """
    if result.get("status") != "completed":
        console.print(f"[yellow]Cannot save report - status: {result.get('status')}[/yellow]")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save markdown report
    report_path = os.path.join(output_dir, f"deep_research_report_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(f"# Deep Research Report\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Duration:** {result.get('duration_seconds', 0):.1f} seconds\n")
        f.write(f"**Bills Analyzed:** {', '.join(result.get('bills_analyzed', []))}\n")
        f.write(f"**Interaction ID:** {result.get('interaction_id', 'N/A')}\n\n")
        f.write("---\n\n")
        f.write(result.get("report", "No report content"))
    
    console.print(f"[green]✓ Deep Research report saved to {report_path}[/green]")
    
    # Save JSON metadata
    meta_path = os.path.join(output_dir, f"deep_research_meta_{timestamp}.json")
    meta = {
        "status": result.get("status"),
        "duration_seconds": result.get("duration_seconds"),
        "interaction_id": result.get("interaction_id"),
        "bills_analyzed": result.get("bills_analyzed"),
        "prompt_length": result.get("prompt_length"),
        "report_length": len(result.get("report", ""))
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return report_path
