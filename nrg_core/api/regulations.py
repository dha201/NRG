"""
Regulations.gov API Integration

Fetches energy-related federal regulations from the Regulations.gov API.

Regulations.gov is the official federal rulemaking portal containing:
- Proposed Rules: Regulations under consideration
- Final Rules: Enacted regulations
- Notices: Informational documents
- Supporting Materials: Studies, analyses, comments

API Details:
    - Base URL: https://api.regulations.gov/v4
    - Rate Limit: 1,000 requests/hour
    - Authentication: API key via query parameter

LIMITATION: Regulations.gov API v4 does NOT support version tracking.
Only lastModifiedDate and withdrawn status are available.
"""

import os
from typing import Any

import httpx
from rich.console import Console

console = Console()

REGULATIONS_API_KEY = os.getenv("REGULATIONS_API_KEY", "")

# Energy-related keywords for filtering
ENERGY_KEYWORDS = [
    "oil", "gas", "petroleum", "natural gas", "fossil fuel",
    "energy", "pipeline", "drilling", "lng", "emissions"
]

# Energy-related agencies
ENERGY_AGENCIES = ["epa", "doe", "ferc", "phmsa"]


def fetch_regulations(limit: int = 3) -> list[dict[str, Any]]:
    """
    Fetch recent energy-related regulations from Regulations.gov API.

    Filtering Strategy:
        Due to API limitations with complex filters, this function:
        1. Fetches 20 most recent documents (sorted by posted date)
        2. Filters client-side for energy keywords in title/agency
        3. Prioritizes EPA, DOE, FERC documents

    Args:
        limit: Maximum number of regulations to return (default: 3)

    Returns:
        List of regulation dictionaries with keys:
            - source: "Regulations.gov"
            - type: "Regulation"
            - number: Docket ID
            - title: Document title
            - url: Link to regulation page
            - status: Document type (Rule, Proposed Rule, etc.)
            - agency: Issuing agency
            - posted_date: Publication date
            - comment_end_date: Comment deadline (if open)
            - effective_date: When regulation takes effect
            - summary: Document summary for LLM analysis
    """
    console.print("\n[bold cyan]Fetching from Regulations.gov...[/bold cyan]")

    url = "https://api.regulations.gov/v4/documents"

    try:
        with httpx.Client(timeout=30.0) as http:
            params = {
                "api_key": REGULATIONS_API_KEY,
                "page[size]": 20,  # Get more to filter client-side
                "sort": "-postedDate"  # Most recent first
            }

            response = http.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                regulations = []

                for doc in data.get("data", []):
                    if len(regulations) >= limit:
                        break

                    attrs = doc.get("attributes", {})
                    title = attrs.get("title", "").lower()
                    summary = attrs.get("summary", "").lower()
                    agency = attrs.get("agencyId", "").lower()

                    # Check if energy-related
                    is_energy = (
                        any(kw in title for kw in ENERGY_KEYWORDS) or
                        any(kw in summary for kw in ENERGY_KEYWORDS) or
                        agency in ENERGY_AGENCIES
                    )

                    if is_energy:
                        doc_id = doc.get("id", "Unknown")
                        regulations.append({
                            "source": "Regulations.gov",
                            "type": "Regulation",
                            "number": doc_id,
                            "title": attrs.get("title", "No title"),
                            "url": f"https://www.regulations.gov/document/{doc_id}",
                            "status": attrs.get("documentType", "Unknown"),
                            "agency": attrs.get("agencyId", "Unknown"),
                            "posted_date": attrs.get("postedDate", "Unknown"),
                            "comment_end_date": attrs.get("commentEndDate", "N/A"),
                            "effective_date": attrs.get("effectiveDate", "N/A"),
                            "summary": attrs.get("summary", attrs.get("title", "No summary available"))
                        })

                if regulations:
                    console.print(f"[green]✓ Found {len(regulations)} energy regulations from Regulations.gov[/green]")
                else:
                    console.print("[yellow]⚠ No energy regulations found in recent documents[/yellow]")

                return regulations

            else:
                console.print(f"[yellow]⚠ Regulations.gov returned {response.status_code}[/yellow]")
                return []

    except Exception as e:
        console.print(f"[red]Error fetching from Regulations.gov: {e}[/red]")
        return []
