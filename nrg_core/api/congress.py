import os
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
from rich.console import Console

from nrg_core.db.cache import compute_bill_hash

console = Console()

CONGRESS_API_KEY: Optional[str] = os.getenv("CONGRESS_API_KEY")
CONGRESS_BASE_URL: str = "https://api.congress.gov/v3"
CONGRESS_NUMBER: str = "118"
API_TIMEOUT: float = 30.0
API_RATE_LIMIT_DELAY: float = 0.5

ENERGY_KEYWORDS: list[str] = [
    "oil", "gas", "petroleum", "natural gas", "fossil fuel",
    "energy", "drilling", "fracking", "pipeline", "lng"
]


def _is_energy_related(
    bill: dict[str, any],
    subjects_data: dict[str, any]
) -> bool:
    """Check if bill is energy-related based on policy area and subjects."""
    policy_area = subjects_data.get("subjects", {}).get("policyArea", {}).get("name", "")
    
    if policy_area == "Energy":
        return True
    
    if any(keyword in policy_area.lower() for keyword in ENERGY_KEYWORDS):
        return True
    
    leg_subjects = subjects_data.get("subjects", {}).get("legislativeSubjects", [])
    subject_names = [s.get("name", "").lower() for s in leg_subjects]
    if any(keyword in " ".join(subject_names) for keyword in ENERGY_KEYWORDS):
        return True
    
    bill_title = bill.get("title", "").lower()
    if any(keyword in bill_title for keyword in ENERGY_KEYWORDS):
        return True
    
    return False


def _fetch_bill_text(
    http: httpx.Client,
    congress: str,
    bill_type: str,
    bill_number: str
) -> str:
    """Fetch full text of a bill, returning title as fallback."""
    text_url = f"{CONGRESS_BASE_URL}/bill/{congress}/{bill_type}/{bill_number}/text"
    params = {"api_key": CONGRESS_API_KEY, "format": "json"}
    
    try:
        text_resp = http.get(text_url, params=params)
        if text_resp.status_code != 200:
            return "No text available"
        
        text_data = text_resp.json()
        text_versions = text_data.get("textVersions", [])
        
        if not text_versions:
            return "No text available"
        
        latest_version = text_versions[0]
        for text_format in latest_version.get("formats", []):
            if text_format.get("type") == "Formatted Text":
                content_url = text_format.get("url")
                if content_url:
                    content_resp = http.get(content_url, params={"api_key": CONGRESS_API_KEY})
                    if content_resp.status_code == 200:
                        return content_resp.text
        
        return "No text available"
    except Exception:
        return "No text available"




def fetch_congress_bills(limit: int = 3) -> list[dict[str, any]]:
    """
    Fetch recent energy-related bills from Congress.gov API.
    
    Args:
        limit: Maximum number of energy bills to return
        
    Returns:
        List of normalized bill dictionaries
    """
    if not CONGRESS_API_KEY:
        console.print("[yellow] Congress.gov API key not configured[/yellow]")
        return []
    
    console.print("\n[bold cyan]Fetching from Congress.gov...[/bold cyan]")

    from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z")
    url = f"{CONGRESS_BASE_URL}/bill/{CONGRESS_NUMBER}/hr"
    params = {
        "api_key": CONGRESS_API_KEY,
        "format": "json",
        "limit": 20,
        "fromDateTime": from_date
    }

    try:
        with httpx.Client(timeout=API_TIMEOUT) as http:
            response = http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            energy_bills: list[dict[str, any]] = []

            for bill in data.get("bills", []):
                bill_num = bill["number"]

                # Fetch bill subjects
                subjects_url = f"{CONGRESS_BASE_URL}/bill/{CONGRESS_NUMBER}/hr/{bill_num}/subjects"
                subjects_resp = http.get(
                    subjects_url,
                    params={"api_key": CONGRESS_API_KEY, "format": "json"}
                )

                if subjects_resp.status_code == 200:
                    subjects_data = subjects_resp.json()

                    if _is_energy_related(bill, subjects_data):
                        # Fetch detailed bill info
                        detail_url = f"{CONGRESS_BASE_URL}/bill/{CONGRESS_NUMBER}/hr/{bill_num}"
                        detail_resp = http.get(
                            detail_url,
                            params={"api_key": CONGRESS_API_KEY, "format": "json"}
                        )
                        detail_data = detail_resp.json()
                        bill_info = detail_data.get("bill", {})

                        # Fetch bill text
                        bill_text = _fetch_bill_text(http, CONGRESS_NUMBER, "hr", bill_num)

                        # Build normalized bill dictionary
                        sponsor = bill_info.get("sponsor", {})
                        sponsor_name = sponsor.get("fullName", "Unknown") if sponsor else "Unknown"
                        policy_area = subjects_data.get("subjects", {}).get("policyArea", {}).get("name", "Unknown")
                        
                        energy_bills.append({
                            "source": "Congress.gov",
                            "type": "Federal Bill",
                            "number": f"H.R. {bill_num}",
                            "title": bill_info.get("title", "No title"),
                            "url": f"https://www.congress.gov/bill/{CONGRESS_NUMBER}th-congress/house-bill/{bill_num}",
                            "status": bill_info.get("latestAction", {}).get("text", "Unknown"),
                            "sponsor": sponsor_name,
                            "introduced_date": bill_info.get("introducedDate", "Unknown"),
                            "policy_area": policy_area,
                            "summary": bill_text,
                            "congress_num": CONGRESS_NUMBER,
                            "bill_type": "hr",
                            "bill_number": bill_num
                        })

                        if len(energy_bills) >= limit:
                            break

                time.sleep(API_RATE_LIMIT_DELAY)

            console.print(f"[green]✓ Found {len(energy_bills)} energy bills from Congress.gov[/green]")
            return energy_bills

    except Exception as e:
        console.print(f"[red]Error fetching from Congress.gov: {e}[/red]")
        return []


def fetch_bill_versions_from_congress(
    congress: str,
    bill_type: str,
    bill_number: str,
    bill_display_number: str
) -> list[dict[str, any]]:
    """
    Fetch all text versions of a federal bill from Congress.gov API.
    
    Args:
        congress: Congress number (e.g., "118")
        bill_type: Bill type code (hr, s, hjres, sjres)
        bill_number: Numeric bill identifier
        bill_display_number: Human-readable format for logging
        
    Returns:
        List of version dictionaries with full_text, hash, and metadata
    """
    if not CONGRESS_API_KEY:
        console.print("[yellow] Congress.gov API key not configured[/yellow]")
        return []

    try:
        console.print(f"[dim]  Fetching versions for {bill_display_number}...[/dim]")

        with httpx.Client(timeout=API_TIMEOUT) as http:
            text_url = f"{CONGRESS_BASE_URL}/bill/{congress}/{bill_type}/{bill_number}/text"
            params = {
                "api_key": CONGRESS_API_KEY,
                "format": "json"
            }

            response = http.get(text_url, params=params)
            response.raise_for_status()
            data = response.json()

            text_versions_raw = data.get("textVersions", [])

            if not text_versions_raw:
                console.print(f"[yellow]  ⚠ No versions found for {bill_display_number}[/yellow]")
                return []

            console.print(f"[green]  ✓ Found {len(text_versions_raw)} versions[/green]")

            versions: list[dict[str, any]] = []
            for idx, version_data in enumerate(text_versions_raw, 1):
                version_type: str = version_data.get("type", "Unknown")
                version_date: str = version_data.get("date", "")

                # Find best text URL from available formats
                formats = version_data.get("formats", [])
                txt_url: Optional[str] = None
                for fmt in formats:
                    if fmt.get("type") in ["Formatted Text", "TXT"]:
                        txt_url = fmt.get("url")
                        break
                if not txt_url and formats:
                    txt_url = formats[0].get("url")

                if not txt_url:
                    console.print(
                        f"[yellow]    Version {idx} ({version_type}): "
                        f"No text format found, skipping[/yellow]"
                    )
                    continue

                console.print(f"[cyan]  Version {idx}/{len(text_versions_raw)}: {version_type}[/cyan]")

                try:
                    text_response = http.get(txt_url, params={"api_key": CONGRESS_API_KEY})
                    text_response.raise_for_status()

                    full_text = text_response.text

                    if full_text:
                        text_hash = compute_bill_hash(full_text)
                        word_count = len(full_text.split())

                        versions.append({
                            "version_number": idx,
                            "version_type": version_type,
                            "version_date": version_date,
                            "text_url": txt_url,
                            "full_text": full_text,
                            "text_hash": text_hash,
                            "word_count": word_count
                        })
                    else:
                        console.print(f"[yellow]    Version {idx}: Empty text, skipping[/yellow]")

                except httpx.HTTPStatusError as e:
                    console.print(f"[yellow]    Version {idx}: HTTP {e.response.status_code}, skipping[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[yellow]    Version {idx}: Error extracting text: {e}[/yellow]")
                    continue

            console.print(f"[green]  ✓ Extracted text from {len(versions)}/{len(text_versions_raw)} versions[/green]")
            return versions

    except httpx.HTTPStatusError as e:
        console.print(f"[red]  HTTP Error fetching versions: {e.response.status_code}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]  Error fetching versions: {e}[/red]")
        return []
