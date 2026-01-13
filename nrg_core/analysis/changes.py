import json
import difflib
from typing import Any, Optional

from rich.console import Console

from nrg_core.db.cache import compute_bill_hash

console = Console()

# Change type constants
CHANGE_TYPE_TEXT: str = "text_change"
CHANGE_TYPE_STATUS: str = "status_change"
CHANGE_TYPE_AMENDMENTS: str = "new_amendments"


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

    try:
        if config['llm']['provider'] == 'gemini':
            response = gemini_client.models.generate_content(
                model=config['llm']['gemini']['model'],
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json"
                }
            )
            return json.loads(response.text)
        else:
            response = openai_client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": "low"},
                text={"verbosity": "medium"}
            )
            return json.loads(response.output_text)

    except Exception as e:
        console.print(f"[red]Error analyzing changes: {e}[/red]")
        return {
            "change_impact_score": 0,
            "change_summary": "Analysis failed",
            "error": str(e)
        }
