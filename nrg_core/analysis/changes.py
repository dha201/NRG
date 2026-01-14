import json
import difflib
import time
from typing import Any, Optional

from rich.console import Console

from nrg_core.db.cache import compute_bill_hash
from nrg_core.analysis.llm import extract_json_from_gemini_response
from nrg_core.config import load_config

console = Console()
_config = load_config()
_debug_llm_responses = _config.get('debug', {}).get('llm_responses', False)

# Change type constants
CHANGE_TYPE_TEXT: str = "text_change"
CHANGE_TYPE_STATUS: str = "status_change"
CHANGE_TYPE_AMENDMENTS: str = "new_amendments"

# Gemini response schemas for change analysis
CHANGE_IMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "change_impact_score": {"type": "integer"},
        "impact_increased": {"type": "boolean"},
        "change_summary": {"type": "string"},
        "nrg_impact": {"type": "string"},
        "recommended_action": {"type": "string"},
        "key_concerns": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["change_impact_score", "change_summary", "recommended_action"]
}

VERSION_CHANGES_SCHEMA = {
    "type": "object",
    "properties": {
        "key_provisions_added": {"type": "array", "items": {"type": "string"}},
        "key_provisions_removed": {"type": "array", "items": {"type": "string"}},
        "key_provisions_modified": {"type": "array", "items": {"type": "string"}},
        "impact_evolution": {"type": "string"},
        "compliance_changes": {"type": "string"},
        "strategic_significance": {"type": "string"},
        "summary": {"type": "string"}
    },
    "required": ["key_provisions_added", "key_provisions_removed", "key_provisions_modified", "summary"]
}


def compute_text_diff(old_text: str, new_text: str) -> str:
    """
    Generate unified diff between two texts.
    
    Args:
        old_text: Previous version text
        new_text: Current version text
        
    Returns:
        Unified diff string
    """
    if not old_text or not new_text:
        return "Full text comparison not available"

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        lineterm='',
        fromfile='Previous Version',
        tofile='Current Version'
    )

    return '\n'.join(diff)


def detect_bill_changes(
    cached_bill: Optional[dict[str, Any]],
    current_bill: dict[str, Any]
) -> dict[str, Any]:
    """
    Compare cached and current bill data to detect changes.
    
    Args:
        cached_bill: Previously cached bill data (None if new bill)
        current_bill: Current bill data
        
    Returns:
        Change detection results with has_changes, is_new, change_type, and changes list
    """
    # New bill - no cached data to compare
    if not cached_bill:
        return {
            "has_changes": False,
            "is_new": True,
            "change_type": "new_bill",
            "changes": []
        }

    changes: list[dict[str, Any]] = []
    cached_data = json.loads(cached_bill['full_data_json'])
    current_text = current_bill.get('summary', '')
    cached_text = cached_data.get('summary', '')

    # Check text changes
    current_hash = compute_bill_hash(current_text)
    if current_hash != cached_bill['text_hash']:
        diff = compute_text_diff(cached_text, current_text)
        changes.append({
            "type": CHANGE_TYPE_TEXT,
            "diff": diff,
            "summary": "Bill text has been modified"
        })
    
    # Check status changes
    current_status = current_bill.get('status')
    cached_status = cached_bill['status']
    if current_status != cached_status:
        changes.append({
            "type": CHANGE_TYPE_STATUS,
            "old_value": cached_status,
            "new_value": current_status,
            "summary": f"Status changed from '{cached_status}' to '{current_status}'"
        })
    
    # Check amendment changes
    current_amendments = current_bill.get('amendments', [])
    cached_amendments = cached_data.get('amendments', [])
    if len(current_amendments) > len(cached_amendments):
        new_amendments = current_amendments[len(cached_amendments):]
        changes.append({
            "type": CHANGE_TYPE_AMENDMENTS,
            "count": len(new_amendments),
            "amendments": new_amendments,
            "summary": f"{len(new_amendments)} new amendment(s) added"
        })

    return {
        "has_changes": len(changes) > 0,
        "is_new": False,
        "change_type": "modified" if changes else "unchanged",
        "changes": changes
    }


def analyze_changes_with_llm(
    bill: dict[str, Any],
    change_data: dict[str, Any],
    nrg_context: str,
    config: dict[str, Any],
    gemini_client: Optional[Any] = None,
    openai_client: Optional[Any] = None
) -> Optional[dict[str, Any]]:
    """    
    Args:
        bill: Bill dictionary
        change_data: Change detection results
        nrg_context: NRG business context
        config: Configuration dictionary
        gemini_client: Gemini client instance
        openai_client: OpenAI client instance
        
    Returns:
        Change impact analysis or None if no changes
    """
    if not change_data.get("has_changes"):
        return None

    # Build change description from detected changes
    changes_description = "\n\n".join([
        f"**{change['type'].upper()}:**\n{change.get('summary', 'No summary')}\n{change.get('diff', '')[:1000]}"
        for change in change_data['changes']
    ])

    prompt = f"""You are analyzing CHANGES to a bill that NRG Energy is tracking.

NRG BUSINESS CONTEXT:
{nrg_context}

BILL INFORMATION:
Number: {bill['number']}
Title: {bill['title']}
Source: {bill['source']}

CHANGES DETECTED:
{changes_description}

Analyze the impact of these changes on NRG Energy. Provide JSON response:
{{
  "change_impact_score": <0-10 integer, how significant are these changes?>,
  "impact_increased": <true/false, did changes increase business impact?>,
  "change_summary": "<brief summary of what changed>",
  "nrg_impact": "<how do these changes affect NRG's business?>",
  "recommended_action": "ignore | monitor | review | urgent",
  "key_concerns": ["<list any new concerns from changes>"]
}}"""

    max_retries = 3
    base_delay = 1.0
    last_error = None

    for attempt in range(max_retries):
        try:
            if config['llm']['provider'] == 'gemini':
                response = gemini_client.models.generate_content(
                    model=config['llm']['gemini']['model'],
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 8192,  # Increased for thinking model token budget
                        "response_mime_type": "application/json",
                        "response_schema": CHANGE_IMPACT_SCHEMA
                    }
                )
                
                json_text = response.text
                
                if json_text is None:
                    json_text = extract_json_from_gemini_response(response)
                
                try:
                    return json.loads(json_text)
                except (json.JSONDecodeError, TypeError):
                    json_text = extract_json_from_gemini_response(response)
                    return json.loads(json_text)
            else:
                response = openai_client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": "low"},
                    text={"verbosity": "medium"}
                )
                return json.loads(response.output_text)

        except Exception as e:
            last_error = str(e)
            if attempt == max_retries - 1:
                console.print(f"[red]Error analyzing changes after {max_retries} attempts: {e}[/red]")
                return {
                    "change_impact_score": 0,
                    "change_summary": "Analysis failed",
                    "error": last_error
                }
            
            delay = base_delay * (2 ** attempt)
            console.print(
                f"[yellow]Change analysis failed ({last_error}), retrying in {delay}s... "
                f"(attempt {attempt + 1}/{max_retries})[/yellow]"
            )
            time.sleep(delay)
    
    return {
        "change_impact_score": 0,
        "change_summary": "Analysis failed",
        "error": str(last_error)
    }


def compare_consecutive_versions(
    old_version: dict[str, Any],
    new_version: dict[str, Any]
) -> dict[str, Any]:
    """
    Generate diff between two consecutive bill versions.

    Args:
        old_version: Previous version dict with full_text, version_type, etc.
        new_version: Current version dict with full_text, version_type, etc.

    Returns:
        Dictionary with diff statistics and summary
    """
    old_text = old_version.get('full_text', '')
    new_text = new_version.get('full_text', '')

    if not old_text or not new_text:
        return {
            "changed": False,
            "error": "Missing text for one or both versions"
        }

    # Compute hash to check if changed
    old_hash = compute_bill_hash(old_text)
    new_hash = compute_bill_hash(new_text)

    if old_hash == new_hash:
        return {
            "changed": False,
            "from_version": old_version.get('version_type'),
            "to_version": new_version.get('version_type')
        }

    # Generate unified diff
    diff_summary = compute_text_diff(old_text, new_text)

    # Count changes
    diff_lines = diff_summary.split('\n')
    lines_added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))

    # Extract key changes (first 500 characters of diff)
    key_changes = diff_summary[:500] if len(diff_summary) > 500 else diff_summary

    return {
        "changed": True,
        "from_version": old_version.get('version_type'),
        "to_version": new_version.get('version_type'),
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "diff_summary": diff_summary,
        "key_changes_preview": key_changes,
        "word_count_change": new_version.get('word_count', 0) - old_version.get('word_count', 0)
    }


def analyze_version_changes_with_llm(
    old_version: dict[str, Any],
    new_version: dict[str, Any],
    old_analysis: dict[str, Any],
    new_analysis: dict[str, Any],
    bill_info: dict[str, Any],
    nrg_context: str,
    config: dict[str, Any],
    gemini_client: Optional[Any] = None,
    openai_client: Optional[Any] = None
) -> dict[str, Any]:
    """
    Use LLM to analyze substantive changes between two consecutive bill versions.

    Args:
        old_version: Previous version dict with full_text, version_type, etc.
        new_version: Current version dict with full_text, version_type, etc.
        old_analysis: LLM analysis of old version
        new_analysis: LLM analysis of new version
        bill_info: Bill metadata (number, title, etc.)
        nrg_context: NRG business context
        config: Configuration dictionary
        gemini_client: Gemini client instance
        openai_client: OpenAI client instance

    Returns:
        Dictionary with semantic change analysis
    """
    provider = config['llm']['provider']

    # Get abbreviated versions of each text (first 15K chars to stay within context limits)
    old_text_sample = old_version.get('full_text', '')[:15000]
    new_text_sample = new_version.get('full_text', '')[:15000]

    prompt = f"""You are analyzing legislative changes for NRG Energy's Government Affairs team.

**BILL INFORMATION:**
- Bill Number: {bill_info.get('number', 'Unknown')}
- Title: {bill_info.get('title', 'Unknown')}
- Source: {bill_info.get('source', 'Unknown')}

**VERSION TRANSITION:**
- FROM: {old_version.get('version_type')} ({old_version.get('version_date', 'N/A')}) - Impact Score: {old_analysis.get('business_impact_score', 'N/A')}/10
- TO: {new_version.get('version_type')} ({new_version.get('version_date', 'N/A')}) - Impact Score: {new_analysis.get('business_impact_score', 'N/A')}/10

**PREVIOUS VERSION TEXT (first 15K chars):**
{old_text_sample}

**CURRENT VERSION TEXT (first 15K chars):**
{new_text_sample}

**NRG BUSINESS CONTEXT:**
{nrg_context[:2000]}

**TASK:**
Analyze the substantive changes between these two versions from NRG's perspective.

Return ONLY a JSON object with this structure:
{{
  "key_provisions_added": ["List of major provisions added"],
  "key_provisions_removed": ["List of major provisions removed"],
  "key_provisions_modified": ["List of major provisions modified"],
  "impact_evolution": "Explanation of how NRG impact changed",
  "compliance_changes": "Summary of new/removed compliance requirements",
  "strategic_significance": "What these changes mean for NRG strategy",
  "summary": "2-3 sentence executive summary of changes"
}}"""

    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if provider == 'gemini':
                # Use high token budget for thinking models (8192) to prevent truncation
                response = gemini_client.models.generate_content(
                    model=config['llm']['gemini']['model'],
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 8192,  # Increased: thinking models use ~4K tokens for thoughts
                        "response_mime_type": "application/json",
                        "response_schema": VERSION_CHANGES_SCHEMA
                    }
                )
                
                # Debug: Log response state before extraction
                if _debug_llm_responses:
                    console.print(f"[dim]DEBUG version_changes: Got response, type={type(response)}[/dim]")
                    console.print(f"[dim]DEBUG version_changes: response.text is None? {response.text is None}[/dim]")
                    if response.text:
                        console.print(f"[dim]DEBUG version_changes: response.text length={len(response.text)}, first 200 chars={response.text[:200]}[/dim]")
                    console.print(f"[dim]DEBUG version_changes: has candidates? {hasattr(response, 'candidates') and response.candidates}[/dim]")
                    if hasattr(response, 'candidates') and response.candidates:
                        console.print(f"[dim]DEBUG version_changes: candidates[0].content? {response.candidates[0].content}[/dim]")
                        if response.candidates[0].content:
                            console.print(f"[dim]DEBUG version_changes: content.parts? {response.candidates[0].content.parts}[/dim]")
                            if response.candidates[0].content.parts:
                                console.print(f"[dim]DEBUG version_changes: parts count={len(response.candidates[0].content.parts)}[/dim]")
                
                # Try SDK's response.text first (preserves all content)
                # Fall back to extract_json_from_gemini_response for edge cases:
                #   - response.text is None (thought-only responses)
                #   - response.text has concatenated JSON (multi-part responses with malformed boundaries)
                json_text = response.text
                
                if json_text is None:
                    if _debug_llm_responses:
                        console.print(f"[yellow]DEBUG version_changes: response.text is None, calling extract_json_from_gemini_response[/yellow]")
                    # Gemini 3 thinking models may return None for thought-only responses
                    json_text = extract_json_from_gemini_response(response)
                
                try:
                    result = json.loads(json_text)
                    if _debug_llm_responses:
                        console.print(f"[dim]DEBUG version_changes: Successfully parsed JSON from response.text[/dim]")
                except (json.JSONDecodeError, TypeError) as e:
                    if _debug_llm_responses:
                        console.print(f"[yellow]DEBUG version_changes: JSON parse failed ({e}), calling extract_json_from_gemini_response[/yellow]")
                    # SDK concatenation bug: {"json1"}{"json2"} or unterminated strings at boundaries
                    json_text = extract_json_from_gemini_response(response)
                    result = json.loads(json_text)
                    if _debug_llm_responses:
                        console.print(f"[dim]DEBUG version_changes: Successfully parsed JSON from extracted part[/dim]")
            else:
                response = openai_client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": "low"},
                    text={"verbosity": "medium"}
                )
                result = json.loads(response.output_text)

            return {
                "key_provisions_added": result.get("key_provisions_added", []),
                "key_provisions_removed": result.get("key_provisions_removed", []),
                "key_provisions_modified": result.get("key_provisions_modified", []),
                "impact_evolution": result.get("impact_evolution", "No analysis available"),
                "compliance_changes": result.get("compliance_changes", "No analysis available"),
                "strategic_significance": result.get("strategic_significance", "No analysis available"),
                "summary": result.get("summary", result.get("impact_summary", "No summary available"))
            }
        
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                console.print(f"[yellow]  ⚠ JSON parse error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...[/yellow]")
                time.sleep(delay)
                continue
            else:
                console.print(f"[red]Error analyzing version changes with LLM: {e}[/red]")
                return {
                    "summary": f"Error during analysis: {str(e)}",
                    "key_provisions_added": [],
                    "key_provisions_removed": [],
                    "key_provisions_modified": [],
                    "impact_evolution": "Error",
                    "compliance_changes": "Error",
                    "strategic_significance": "Error"
                }
        
        except Exception as e:
            console.print(f"[red]Error analyzing version changes with LLM: {e}[/red]")
            return {
                "summary": f"Error during analysis: {str(e)}",
                "key_provisions_added": [],
                "key_provisions_removed": [],
                "key_provisions_modified": [],
                "impact_evolution": "Error",
                "compliance_changes": "Error",
                "strategic_significance": "Error"
            }
