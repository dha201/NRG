import io
import os
import time
from typing import Optional

import httpx
from rich.console import Console

from nrg_core.db.cache import compute_bill_hash
from nrg_core.utils import normalize_version_type

console = Console()

# API Configuration
OPENSTATES_BASE_URL: str = "https://v3.openstates.org"
API_TIMEOUT: float = 30.0
MAX_RETRIES: int = 3
RETRY_DELAY: float = 2.0

# PDF text density threshold (chars per page) to determine if there are images
PDF_IMAGE_THRESHOLD: int = 100


def extract_pdf_text(pdf_url: str) -> tuple[Optional[str], bool]:
    try:
        import pdfplumber

        console.print(f"[dim]    Downloading PDF from {pdf_url[:60]}...[/dim]")

        with httpx.Client(timeout=API_TIMEOUT) as http:
            response = http.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            pdf_bytes = io.BytesIO(response.content)

        text: str = ""
        char_count: int = 0
        with pdfplumber.open(pdf_bytes) as pdf:
            num_pages = len(pdf.pages)
            console.print(f"[dim]    Extracting text from {num_pages} pages...[/dim]")
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    char_count += len(page_text)
            
            avg_chars_per_page = char_count / max(num_pages, 1)
            is_image_based = num_pages > 0 and avg_chars_per_page < PDF_IMAGE_THRESHOLD

        if is_image_based:
            console.print("[yellow]    ⚠ PDF appears to be scanned images (low text density)[/yellow]")
            return (None, True)

        if not text.strip():
            console.print("[yellow]    ⚠ PDF appears empty or text extraction failed[/yellow]")
            return (None, False)

        console.print(f"[green]    ✓ Extracted {len(text)} characters ({len(text.split())} words)[/green]")
        return (text, False)

    except httpx.HTTPStatusError as e:
        console.print(f"[red]    HTTP Error downloading PDF: {e.response.status_code}[/red]")
        return (None, False)
    except Exception as e:
        console.print(f"[red]    Error extracting PDF text: {e}[/red]")
        return (None, False)


def _get_api_headers() -> dict[str, str]:
    api_key = os.getenv("OPENSTATES_API_KEY")
    return {
        "X-API-KEY": api_key if api_key else "",
        "User-Agent": "NRG-Energy-Legislative-Tracker/1.0"
    }


def _extract_bill_metadata(
    result: dict[str, any],
    full_text: Optional[str],
    abstract: str,
    jurisdiction: str
) -> dict[str, any]:
    actions = result.get("actions", [])
    last_action = actions[-1] if actions and isinstance(actions[-1], dict) else {}
    
    sponsorships = result.get("sponsorships", [])
    primary_sponsor = sponsorships[0] if sponsorships and isinstance(sponsorships[0], dict) else {}
    
    sources = result.get("sources", [])
    bill_url = sources[0].get("url", "") if sources and isinstance(sources[0], dict) else ""
    
    # Fallback URL for Texas bills
    if not bill_url and jurisdiction.upper() == "TX":
        session_data = result.get("session", {})
        session = session_data.get("identifier", "89R") if isinstance(session_data, dict) else "89R"
        identifier = result.get("identifier", "").replace(" ", "")
        bill_url = f"https://capitol.texas.gov/BillLookup/History.aspx?LegSess={session}&Bill={identifier}"
    
    final_summary = full_text if full_text else abstract
    
    return {
        "source": "Open States",
        "type": "Federal Bill" if jurisdiction == "US" else "Texas State Bill",
        "number": result.get("identifier", "Unknown"),
        "title": result.get("title", "No title"),
        "url": bill_url,
        "status": last_action.get("description", "Unknown") if last_action else "Unknown",
        "sponsor": primary_sponsor.get("name", "Unknown") if primary_sponsor else "Unknown",
        "introduced_date": result.get("created_at", "Unknown")[:10],
        "summary": final_summary,
        "abstract": abstract,
        "has_full_text": full_text is not None,
        "updated_at": result.get("updated_at", ""),
        "openstates_id": result.get("id", ""),
    }


def fetch_openstates_bills(
    jurisdiction: str = "TX",
    bill_numbers: Optional[list[str]] = None,
    search_query: Optional[str] = None,
    limit: int = 3
) -> list[dict[str, any]]:
    if not os.getenv("OPENSTATES_API_KEY"):
        console.print("[yellow]⚠ Open States API key not found in environment[/yellow]")
        return []

    jurisdiction_name = "Federal" if jurisdiction == "US" else jurisdiction.upper()
    console.print(f"\n[bold cyan]Fetching from Open States ({jurisdiction_name})...[/bold cyan]")

    headers = _get_api_headers()
    bills: list[dict[str, any]] = []

    try:
        with httpx.Client(timeout=API_TIMEOUT) as http:
            if bill_numbers:
                for bill_num in bill_numbers[:limit]:
                    console.print(f"[dim]  Searching for {jurisdiction} {bill_num}...[/dim]")

                    for attempt in range(MAX_RETRIES):
                        try:
                            search_url = f"{OPENSTATES_BASE_URL}/bills"
                            params = {
                                "jurisdiction": jurisdiction.lower(),
                                "q": bill_num,
                                "per_page": 5
                            }

                            response = http.get(search_url, headers=headers, params=params)
                            response.raise_for_status()
                            data = response.json()

                            bill_found = False
                            for result in data.get("results", []):
                                identifier = result.get("identifier", "")
                                if identifier.replace(" ", "").upper() == bill_num.replace(" ", "").upper():
                                    bill_found = True

                                    actions = result.get("actions", [])
                                    last_action = actions[-1] if actions and isinstance(actions[-1], dict) else {}

                                    sponsorships = result.get("sponsorships", [])
                                    primary_sponsor = sponsorships[0] if sponsorships and isinstance(sponsorships[0], dict) else {}

                                    sources = result.get("sources", [])
                                    abstracts = result.get("abstracts", [])

                                    bill_url = ""
                                    if sources and isinstance(sources[0], dict):
                                        bill_url = sources[0].get("url", "")
                                    if not bill_url and jurisdiction.upper() == "TX":
                                        session_data = result.get("session", {})
                                        session = session_data.get("identifier", "89R") if isinstance(session_data, dict) else "89R"
                                        bill_id = identifier.replace(" ", "")
                                        bill_url = f"https://capitol.texas.gov/BillLookup/History.aspx?LegSess={session}&Bill={bill_id}"

                                    summary = "No summary available"
                                    if abstracts and isinstance(abstracts[0], dict):
                                        summary = abstracts[0].get("abstract", "No summary available")

                                    full_bill_text = None
                                    openstates_id = result.get("id", "")
                                    if openstates_id:
                                        try:
                                            console.print(f"[dim]    Fetching full text for {identifier}...[/dim]")
                                            bill_detail_url = f"{OPENSTATES_BASE_URL}/bills/{openstates_id}"
                                            detail_params = {"include": "versions"}
                                            detail_response = http.get(bill_detail_url, headers=headers, params=detail_params)
                                            detail_response.raise_for_status()
                                            detail_data = detail_response.json()

                                            versions = detail_data.get("versions", [])
                                            
                                            if versions:
                                                latest_version = versions[-1] if versions else None
                                                if latest_version:
                                                    links = latest_version.get("links", [])
                                                    pdf_url = None
                                                    for link in links:
                                                        if link.get("media_type") == "application/pdf":
                                                            pdf_url = link.get("url")
                                                            break

                                                    if pdf_url:
                                                        full_bill_text, is_image = extract_pdf_text(pdf_url)
                                                        if is_image:
                                                            console.print("[yellow]    ⚠ PDF is image-based, OCR not implemented[/yellow]")
                                                        elif full_bill_text:
                                                            console.print(f"[green]    ✓ Extracted {len(full_bill_text.split())} words of full text[/green]")
                                        except Exception as e:
                                            console.print(f"[yellow]    ⚠ Could not fetch full text: {e}[/yellow]")

                                    bills.append(_extract_bill_metadata(
                                        result, full_bill_text, summary, jurisdiction
                                    ))

                                    console.print(f"[green]    ✓ Found {identifier}[/green]")
                                    break

                            if not bill_found:
                                console.print(f"[yellow]    ⚠ {bill_num} not found[/yellow]")

                            break

                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 429:
                                console.print(f"[yellow]    Rate limit hit, retrying (attempt {attempt + 1}/{MAX_RETRIES})...[/yellow]")
                                if attempt < MAX_RETRIES - 1:
                                    time.sleep(RETRY_DELAY ** attempt)
                                else:
                                    console.print(f"[red]    Failed after {MAX_RETRIES} attempts[/red]")
                            elif e.response.status_code == 401:
                                console.print("[red]Authentication failed - check OPENSTATES_API_KEY[/red]")
                                return bills
                            else:
                                console.print(f"[red]    HTTP {e.response.status_code}: {e}[/red]")
                                break
                        except Exception as e:
                            console.print(f"[red]    Error: {e}[/red]")
                            if attempt == MAX_RETRIES - 1:
                                break

            elif search_query:
                console.print(f"[dim]  Searching for: '{search_query}'...[/dim]")

                search_url = f"{OPENSTATES_BASE_URL}/bills"
                params = {
                    "jurisdiction": jurisdiction.lower(),
                    "q": search_query,
                    "per_page": limit,
                    "sort": "updated_at"
                }

                response = http.get(search_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                for result in data.get("results", [])[:limit]:
                    actions = result.get("actions", [])
                    last_action = actions[-1] if actions else {}
                    sponsorships = result.get("sponsorships", [])
                    primary_sponsor = sponsorships[0] if sponsorships else {}
                    sources = result.get("sources", [])
                    abstracts = result.get("abstracts", [])
                    identifier = result.get("identifier", "Unknown")

                    summary = abstracts[0].get("abstract", "No summary available") if abstracts else "No summary available"

                    full_bill_text = None
                    openstates_id = result.get("id", "")
                    if openstates_id:
                        try:
                            console.print(f"[dim]    Fetching full text for {identifier}...[/dim]")
                            bill_detail_url = f"{OPENSTATES_BASE_URL}/bills/{openstates_id}"
                            detail_params = {"include": "versions"}
                            detail_response = http.get(bill_detail_url, headers=headers, params=detail_params)
                            detail_response.raise_for_status()
                            detail_data = detail_response.json()

                            versions = detail_data.get("versions", [])
                            if versions:
                                latest_version = versions[-1] if versions else None
                                if latest_version:
                                    links = latest_version.get("links", [])
                                    pdf_url = None
                                    for link in links:
                                        if link.get("media_type") == "application/pdf":
                                            pdf_url = link.get("url")
                                            break

                                    if pdf_url:
                                        full_bill_text, is_image = extract_pdf_text(pdf_url)
                                        if is_image:
                                            console.print("[yellow]    ⚠ PDF is image-based, OCR not implemented[/yellow]")
                                        elif full_bill_text:
                                            console.print(f"[green]    ✓ Extracted {len(full_bill_text.split())} words of full text[/green]")
                        except Exception as e:
                            console.print(f"[yellow]    ⚠ Could not fetch full text: {e}[/yellow]")

                    final_summary = full_bill_text if full_bill_text else summary

                    bills.append({
                        "source": "Open States",
                        "type": "Federal Bill" if jurisdiction == "US" else "Texas State Bill",
                        "number": identifier,
                        "title": result.get("title", "No title"),
                        "url": sources[0].get("url", "") if sources else "",
                        "status": last_action.get("description", "Unknown"),
                        "sponsor": primary_sponsor.get("name", "Unknown"),
                        "introduced_date": result.get("created_at", "Unknown")[:10],
                        "summary": final_summary,
                        "abstract": summary,
                        "has_full_text": full_bill_text is not None,
                        "updated_at": result.get("updated_at", ""),
                        "openstates_id": openstates_id,
                    })

        console.print(f"[green]✓ Fetched {len(bills)}/{len(bill_numbers) if bill_numbers else limit} bills from Open States[/green]")
        return bills

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            console.print("[red]Rate limit exceeded (429) - You've hit the daily quota[/red]")
        elif e.response.status_code == 401:
            console.print("[red]Authentication failed (401) - Check API key[/red]")
        elif e.response.status_code == 403:
            console.print("[red]Forbidden (403) - Check API permissions[/red]")
        else:
            console.print(f"[red]HTTP Error {e.response.status_code}: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Error fetching from Open States: {e}[/red]")
        return []


def fetch_bill_versions_from_openstates(
    openstates_id: str,
    bill_number: str
) -> list[dict[str, any]]:
    if not os.getenv("OPENSTATES_API_KEY"):
        return []

    try:
        console.print(f"[dim]  Fetching versions for {bill_number}...[/dim]")

        headers = _get_api_headers()

        with httpx.Client(timeout=API_TIMEOUT) as http:
            bill_url = f"{OPENSTATES_BASE_URL}/bills/{openstates_id}"
            params = {"include": "versions"}

            response = http.get(bill_url, headers=headers, params=params)
            response.raise_for_status()
            bill_data = response.json()
        versions_raw = bill_data.get("versions", [])

        if not versions_raw:
            console.print(f"[yellow]  ⚠ No versions found for {bill_number}[/yellow]")
            return []
        console.print(f"[green]  ✓ Found {len(versions_raw)} versions[/green]")
        
        versions: list[dict[str, any]] = []
        for idx, version_data in enumerate(versions_raw, 1):
            raw_version_type: str = version_data.get("note", "Unknown")
            version_type: str = normalize_version_type(raw_version_type, source="Open States")
            version_date: str = version_data.get("date", "")
            links = version_data.get("links", [])

            pdf_url: Optional[str] = None
            for link in links:
                if link.get("media_type") == "application/pdf":
                    pdf_url = link.get("url")
                    break

            if not pdf_url:
                console.print(f"[yellow]    Version {idx} ({version_type}): No PDF found, skipping[/yellow]")
                continue

            console.print(f"[cyan]  Version {idx}/{len(versions_raw)}: {version_type}[/cyan]")

            full_text, is_image = extract_pdf_text(pdf_url)

            if is_image:
                console.print("[yellow]    ⚠ PDF is image-based, skipping version[/yellow]")
                continue

            if full_text:
                text_hash = compute_bill_hash(full_text)
                word_count = len(full_text.split())

                versions.append({
                    "version_number": idx,
                    "version_type": version_type,
                    "version_date": version_date,
                    "pdf_url": pdf_url,
                    "full_text": full_text,
                    "text_hash": text_hash,
                    "word_count": word_count
                })
            else:
                console.print(f"[yellow]    Version {idx}: Empty text, skipping[/yellow]")

        console.print(f"[green]  ✓ Extracted text from {len(versions)}/{len(versions_raw)} versions[/green]")
        return versions

    except httpx.HTTPStatusError as e:
        console.print(f"[red]  HTTP Error fetching versions: {e.response.status_code}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]  Error fetching versions: {e}[/red]")
        return []
