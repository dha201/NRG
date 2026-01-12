"""
================================================================================
NRG ENERGY LEGISLATIVE INTELLIGENCE POC
================================================================================

PURPOSE:
    Automated monitoring and analysis of federal/state legislation and regulations
    affecting NRG Energy's oil/gas and power generation operations.

ARCHITECTURE:
    This is a single-file monolithic POC demonstrating the full pipeline:

    1. DATA COLLECTION (Lines ~70-860)
       - fetch_congress_bills()      : Federal bills from Congress.gov API
       - fetch_regulations()         : Federal regulations from Regulations.gov API
       - fetch_legiscan_bills()      : Multi-state bills from LegiScan API (DEPRECATED)
       - fetch_openstates_bills()    : State bills from Open States API (PRIMARY)
       - fetch_bill_versions_*()     : Full bill version history with text

    2. LLM ANALYSIS (Lines ~860-1370)
       - analyze_with_openai()       : GPT-5 analysis with structured JSON output
       - analyze_with_gemini()       : Gemini Pro analysis (current default)
       - analyze_bill_version()      : Per-version analysis for change tracking
       - analyze_version_changes_with_llm() : Semantic diff analysis between versions

    3. CHANGE TRACKING (Lines ~2450-2990)
       - SQLite database for caching bill data and detecting changes
       - Version-by-version tracking with PDF text extraction
       - Diff computation and LLM-powered change impact analysis

    4. OUTPUT GENERATION (Lines ~1370-2450)
       - display_analysis()          : Rich console output with color coding
       - generate_markdown_report()  : Comprehensive MD report with executive summary
       - convert_markdown_to_word()  : DOCX generation via pandoc

DATA FLOW:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  API Sources    │────>│  LLM Analysis   │────>│  Report Output  │
    │  (Congress.gov  │     │  (Gemini/GPT-5) │     │  (JSON/MD/DOCX) │
    │   Open States   │     │                 │     │                 │
    │   Regulations)  │     │                 │     │                 │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                        │
           v                        v                        v
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  SQLite Cache   │<───>│  Version Track  │────>│  Change Alerts  │
    │  (bill_cache.db)│     │  (PDF Extract)  │     │  (LLM Analysis) │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

CONFIGURATION:
    - config.yaml      : Data sources, LLM provider, tracking settings
    - .env             : API keys (CONGRESS_API_KEY, GOOGLE_API_KEY, etc.)
    - nrg_business_context.txt : NRG business context for LLM analysis

EXECUTION:
    $ uv run poc.py    # Recommended (uses uv package manager)
    $ python poc.py    # Alternative (requires activated venv)

OUTPUT FILES:
    - nrg_analysis_YYYYMMDD_HHMMSS.json  : Structured analysis data
    - nrg_analysis_YYYYMMDD_HHMMSS.md    : Human-readable report
    - nrg_analysis_YYYYMMDD_HHMMSS.docx  : Word document (via pandoc)
    - bill_cache.db                       : SQLite cache for change tracking

COST ESTIMATE:
    ~$0.10-0.50 per run depending on number of bills and LLM provider

AUTHOR: Bytemethod (POC for NRG Energy)
LAST UPDATED: November 2025
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library
import os              # Environment variables and file operations
import json            # JSON parsing for API responses and output
import base64          # Decoding base64-encoded bill text (LegiScan)
import subprocess      # Running pandoc for DOCX conversion
import yaml            # Configuration file parsing
import sqlite3         # Local database for change tracking cache
import hashlib         # Computing hashes for change detection
import difflib         # Text diff computation between bill versions
import tempfile        # Temporary directory access for Azure Functions
from datetime import datetime, timedelta  # Date handling for API queries

# Third-party libraries
import httpx           # HTTP client for API requests (async-capable)
from openai import OpenAI      # OpenAI GPT-5 client
from google import genai       # Google Gemini client
from dotenv import load_dotenv  # Load .env file for API keys
from rich.console import Console  # Rich terminal output
from rich.panel import Panel      # Rich panels for formatted display
# from rich.json import JSON        # Rich JSON syntax highlighting

# =============================================================================
# INITIALIZATION
# =============================================================================
# Load environment variables from .env file (API keys)
load_dotenv()


def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        console.print("[red]Error: config.yaml not found. Creating default config...[/red]")
        # Return default config if file doesn't exist
        return {
            "llm": {"provider": "openai"},
            "sources": {
                "congress": {"enabled": True, "limit": 3},
                "regulations": {"enabled": True, "limit": 3},
                "legiscan_federal": {"enabled": True, "limit": 3},
                "texas_bills": {"enabled": False, "bills": []}
            },
            "change_tracking": {"enabled": False}
        }


# Initialize Rich console for formatted terminal output
console = Console()

# Load configuration from config.yaml (LLM provider, data sources, etc.)
config = load_config()

# Initialize LLM clients - both are available, config.yaml determines which is used
# Provider selection: config['llm']['provider'] = "gemini" or "openai"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # GPT-5 Responses API

# Disable non-text warnings for Gemini
os.environ["GOOGLE_GENAI_DISABLE_NON_TEXT_WARNINGS"] = "true"
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # Gemini Pro

# API Keys for legislative data sources (loaded from .env file)
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")      # Congress.gov (also works for Regulations.gov)
REGULATIONS_API_KEY = os.getenv("REGULATIONS_API_KEY") # Regulations.gov (same as Congress key)
LEGISCAN_API_KEY = os.getenv("LEGISCAN_API_KEY")      # LegiScan (deprecated, using Open States)
# Note: OPENSTATES_API_KEY is loaded directly in fetch_openstates_bills()


# =============================================================================
# CONTEXT LOADING
# =============================================================================

def load_nrg_context():
    """Load NRG business context from file"""
    context_file = "nrg_business_context.txt"
    try:
        with open(context_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        console.print(f"[red]Warning: {context_file} not found. Using minimal context.[/red]")
        return "NRG Energy is a major US electricity generator with natural gas and oil-fired power plants."


# =============================================================================
# DATA FETCHING - FEDERAL LEGISLATION (Congress.gov)
# =============================================================================

def fetch_congress_bills(limit=3):
    """
    Fetch recent energy-related bills from Congress.gov API.

    This function queries the Congress.gov API for House bills (H.R.) from the
    118th Congress, filters for energy-related content, and fetches full bill
    text for LLM analysis.

    API Details:
        - Base URL: https://api.congress.gov/v3
        - Rate Limit: 5,000 requests/hour
        - Authentication: API key via query parameter

    Filtering Logic:
        1. Fetch last 20 bills from past 60 days
        2. For each bill, check policy area and legislative subjects
        3. Match against energy keywords (oil, gas, energy, pipeline, etc.)
        4. Fetch full text of matching bills from govinfo.gov

    Args:
        limit (int): Maximum number of energy bills to return (default: 3)

    Returns:
        list[dict]: List of bill dictionaries with keys:
            - source: "Congress.gov"
            - type: "Federal Bill"
            - number: Bill number (e.g., "H.R. 1234")
            - title: Bill title
            - url: Link to Congress.gov page
            - status: Latest action text
            - sponsor: Sponsor's full name
            - introduced_date: Date introduced
            - policy_area: Policy area (e.g., "Energy")
            - summary: Full bill text for LLM analysis
            - congress_num, bill_type, bill_number: Metadata for version tracking

    Note:
        - Uses 0.5s sleep between API calls to respect rate limits
        - Returns empty list on error (fail-soft design)
    """
    console.print("\n[bold cyan]Fetching from Congress.gov...[/bold cyan]")

    # Energy-related keywords to filter for
    energy_keywords = [
        "oil", "gas", "petroleum", "natural gas", "fossil fuel",
        "energy", "drilling", "fracking", "pipeline", "lng"
    ]

    # Get bills from last 60 days (fetch more to find energy bills)
    from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z")

    url = "https://api.congress.gov/v3/bill/118/hr"
    params = {
        "api_key": CONGRESS_API_KEY,
        "format": "json",
        "limit": 20,  # Fetch more bills to filter
        "fromDateTime": from_date
    }

    try:
        with httpx.Client(timeout=30.0) as http:
            response = http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            energy_bills = []

            for bill in data.get("bills", []):
                bill_num = bill["number"]

                # Get subjects to filter for energy
                subjects_url = f"https://api.congress.gov/v3/bill/118/hr/{bill_num}/subjects"
                subjects_resp = http.get(subjects_url, params={"api_key": CONGRESS_API_KEY, "format": "json"})

                if subjects_resp.status_code == 200:
                    subjects_data = subjects_resp.json()

                    # Check policy area
                    policy_area = subjects_data.get("subjects", {}).get("policyArea", {}).get("name", "")

                    # Check legislative subjects
                    leg_subjects = subjects_data.get("subjects", {}).get("legislativeSubjects", [])
                    subject_names = [s.get("name", "").lower() for s in leg_subjects]

                    # Check if energy-related
                    is_energy = (
                        policy_area == "Energy" or
                        any(keyword in policy_area.lower() for keyword in energy_keywords) or
                        any(keyword in " ".join(subject_names) for keyword in energy_keywords) or
                        any(keyword in bill.get("title", "").lower() for keyword in energy_keywords)
                    )

                    if is_energy:
                        # Get full bill details
                        detail_url = f"https://api.congress.gov/v3/bill/118/hr/{bill_num}"
                        detail_resp = http.get(detail_url, params={"api_key": CONGRESS_API_KEY, "format": "json"})
                        detail_data = detail_resp.json()
                        bill_info = detail_data.get("bill", {})

                        # Fetch full bill text for deep analysis
                        text_url = f"https://api.congress.gov/v3/bill/118/hr/{bill_num}/text"
                        text_resp = http.get(text_url, params={"api_key": CONGRESS_API_KEY, "format": "json"})

                        bill_text = bill_info.get("title", "No text available")
                        if text_resp.status_code == 200:
                            text_data = text_resp.json()
                            text_versions = text_data.get("textVersions", [])
                            # Get the most recent text version (usually first)
                            if text_versions and len(text_versions) > 0:
                                latest_text_version = text_versions[0]
                                # Get the actual text content URL
                                for text_format in latest_text_version.get("formats", []):
                                    if text_format.get("type") == "Formatted Text":
                                        text_content_url = text_format.get("url")
                                        if text_content_url:
                                            # Fetch the actual bill text
                                            text_content_resp = http.get(text_content_url, params={"api_key": CONGRESS_API_KEY})
                                            if text_content_resp.status_code == 200:
                                                bill_text = text_content_resp.text
                                                break

                        # Extract sponsor information
                        sponsor = bill_info.get("sponsor", {})
                        sponsor_name = sponsor.get("fullName", "Unknown") if sponsor else "Unknown"

                        # Extract policy area from subjects
                        policy_area = subjects_data.get("subjects", {}).get("policyArea", {}).get("name", "Unknown")

                        # Bill Schema Definition for full_data_json storage
                        energy_bills.append({
                            "source": "Congress.gov",
                            "type": "Federal Bill",
                            "number": f"H.R. {bill_num}",
                            "title": bill_info.get("title", "No title"),
                            "url": f"https://www.congress.gov/bill/118th-congress/house-bill/{bill_num}",
                            "status": bill_info.get("latestAction", {}).get("text", "Unknown"),
                            "sponsor": sponsor_name,
                            "introduced_date": bill_info.get("introducedDate", "Unknown"),
                            "policy_area": policy_area,
                            "summary": bill_text,
                            # Metadata for version tracking
                            "congress_num": "118",
                            "bill_type": "hr",
                            "bill_number": bill_num
                        })

                        if len(energy_bills) >= limit:
                            break

                # Rate limiting
                import time
                time.sleep(0.5)  # Be nice to the API

            console.print(f"[green]✓ Found {len(energy_bills)} energy bills from Congress.gov[/green]")
            return energy_bills

    except Exception as e:
        console.print(f"[red]Error fetching from Congress.gov: {e}[/red]")
        return []


def fetch_bill_versions_from_congress(congress, bill_type, bill_number, bill_display_number):
    """
    Fetch ALL text versions of a federal bill from Congress.gov API.

    Federal bills go through multiple versions as they progress:
    - IH (Introduced in House) / IS (Introduced in Senate)
    - RH (Reported in House) / RS (Reported in Senate)
    - EH (Engrossed in House) / ES (Engrossed in Senate)
    - ENR (Enrolled Bill - final version sent to President)
    
    Each version represents potential changes affecting NRG's operations. TODO: HOW?

    This function retrieves all available versions with their full text,
    enabling version-by-version analysis and change tracking.

    API Flow:
        1. GET /bill/{congress}/{type}/{number}/text → List of text versions
        2. For each version, GET the "Formatted Text" URL to fetch actual content
        3. Parse HTML/XML text into clean format for LLM analysis

    Args:
        congress (str): Congress number (e.g., "118" for 118th Congress)
        bill_type (str): Bill type code - "hr", "s", "hjres", "sjres"
        bill_number (str): Numeric bill identifier (e.g., "3076")
        bill_display_number (str): Human-readable format for logging (e.g., "H.R. 3076")

    Returns:
        list[dict]: List of version dictionaries, each containing:
            - version_type: Version code (e.g., "IH", "EH", "ENR")
            - date: Date this version was published
            - full_text: Complete text content for LLM analysis
            - url: Direct URL to the version text
            - formats: Available format types

    Note:
        Returns empty list if API key missing or on error (fail-soft design)
    """
    if not CONGRESS_API_KEY:
        return []

    try:
        console.print(f"[dim]  Fetching versions for {bill_display_number}...[/dim]")

        with httpx.Client(timeout=30.0) as http:
            # Fetch text versions endpoint
            text_url = f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_number}/text"
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

            # Process each version
            versions = []
            for idx, version_data in enumerate(text_versions_raw, 1):
                version_type = version_data.get("type", "Unknown")
                version_date = version_data.get("date", "")

                # Find TXT format (easiest to extract)
                txt_url = None
                formats = version_data.get("formats", [])

                for fmt in formats:
                    if fmt.get("type") in ["Formatted Text", "TXT"]:
                        txt_url = fmt.get("url")
                        break

                # Fallback to any available format
                if not txt_url and formats:
                    txt_url = formats[0].get("url")

                if not txt_url:
                    console.print(f"[yellow]    Version {idx} ({version_type}): No text format found, skipping[/yellow]")
                    continue

                console.print(f"[cyan]  Version {idx}/{len(text_versions_raw)}: {version_type}[/cyan]")

                # Fetch the actual bill text
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


# =============================================================================
# DATA FETCHING - FEDERAL REGULATIONS (Regulations.gov)
# =============================================================================

def fetch_regulations(limit=3):
    """
    Fetch recent energy-related regulations from Regulations.gov API.

    Regulations.gov is the official federal rulemaking portal. It contains:
    - Proposed Rules: Regulations under consideration
    - Final Rules: Enacted regulations
    - Notices: Informational documents
    - Supporting Materials: Studies, analyses, comments

    API Details:
        - Base URL: https://api.regulations.gov/v4
        - Rate Limit: 1,000 requests/hour
        - Authentication: API key via query parameter (same as Congress.gov)

    Filtering Strategy:
        Due to API limitations with complex filters, this function:
        1. Fetches 20 most recent documents (sorted by posted date)
        2. Filters client-side for energy keywords in title/agency
        3. Prioritizes EPA, DOE, FERC documents

    IMPORTANT LIMITATION:
        Regulations.gov API v4 does NOT support version tracking.
        The API only provides lastModifiedDate and withdrawn status.
        Historical document versions and revision diffs are NOT available.
        For NRG's purposes, this limits change detection to monitoring
        lastModifiedDate changes only.

    Args:
        limit (int): Maximum number of regulations to return (default: 3)

    Returns:
        list[dict]: List of regulation dictionaries with keys:
            - source: "Regulations.gov"
            - type: "Regulation"
            - number: Docket ID
            - title: Document title
            - url: Link to regulation page
            - status: Document type (Rule, Proposed Rule, etc.)
            - agency: Issuing agency
            - posted_date: Publication date
            - comment_period: Comment deadline (if open)
            - summary: Document summary for LLM analysis
    """
    console.print("\n[bold cyan]Fetching from Regulations.gov...[/bold cyan]")

    url = "https://api.regulations.gov/v4/documents"

    try:
        with httpx.Client(timeout=30.0) as http:
            # Just get recent documents without complex filtering
            # The Regulations.gov API v4 seems to have issues with filter syntax
            params = {
                "api_key": REGULATIONS_API_KEY,
                "page[size]": 20,  # Get more to filter client-side
                "sort": "-postedDate"  # Most recent first
            }

            response = http.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                all_regulations = []

                # Filter for energy-related regulations client-side
                energy_keywords = [
                    "oil", "gas", "petroleum", "natural gas", "fossil fuel",
                    "energy", "pipeline", "drilling", "lng", "emissions"
                ]

                for doc in data.get("data", []):
                    if len(all_regulations) >= limit:
                        break

                    attrs = doc.get("attributes", {})
                    title = attrs.get("title", "").lower()
                    summary = attrs.get("summary", "").lower()
                    agency = attrs.get("agencyId", "").lower()

                    # Check if energy-related
                    is_energy = (
                        any(keyword in title for keyword in energy_keywords) or
                        any(keyword in summary for keyword in energy_keywords) or
                        agency in ["epa", "doe", "ferc", "phmsa"]  # Energy agencies
                    )

                    if is_energy:
                        doc_id = doc.get("id", "Unknown")
                        all_regulations.append({
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

                if all_regulations:
                    console.print(f"[green]✓ Found {len(all_regulations)} energy regulations from Regulations.gov[/green]")
                else:
                    console.print("[yellow]⚠ No energy regulations found in recent documents[/yellow]")

                return all_regulations

            else:
                console.print(f"[yellow]⚠ Regulations.gov returned {response.status_code}[/yellow]")
                # If even the simple query fails, return empty
                return []

    except Exception as e:
        console.print(f"[red]Error fetching from Regulations.gov: {e}[/red]")
        return []


# =============================================================================
# DATA FETCHING - STATE LEGISLATION (LegiScan - DEPRECATED)
# =============================================================================

def fetch_legiscan_bills(limit=3):
    """
    Fetch bills from LegiScan API using full-text search.

    ⚠️ DEPRECATED: This function is kept as a backup but Open States is now
    the primary source for state legislation. LegiScan is disabled in config.yaml.

    LegiScan provides multi-state bill tracking with full-text search capabilities.
    However, for NRG's use case, Open States offers better:
    - Bill version tracking (all versions with URLs)
    - Transparent pricing ($59/mo vs contact-sales)
    - Modern REST API design

    API Details:
        - Base URL: https://api.legiscan.com/
        - Rate Limit: 30,000 requests/month (free tier)
        - Authentication: API key via query parameter

    Args:
        limit (int): Maximum number of bills to return (default: 3)

    Returns:
        list[dict]: List of bill dictionaries with standard fields

    Note:
        Bill text is base64-encoded in LegiScan API responses and must be decoded.
    """
    console.print("\n[bold cyan]Fetching from LegiScan...[/bold cyan]")

    # Use getSearch operation
    url = "https://api.legiscan.com/"
    params = {
        "key": LEGISCAN_API_KEY,
        "op": "getSearch",
        "state": "US",  # Federal bills only
        "query": "oil and gas",
    }

    try:
        with httpx.Client(timeout=30.0) as http:
            response = http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            bills = []

            # Check if search was successful
            if data.get("status") == "OK":
                search_results = data.get("searchresult", {})

                # Skip the summary key, iterate over bill results
                count = 0
                for key, bill_data in search_results.items():
                    if key == "summary":
                        continue
                    if count >= limit:
                        break

                    # Fetch full bill text and metadata using getBillText and getBill operations
                    bill_id = bill_data.get("bill_id")
                    bill_text = bill_data.get("description", bill_data.get("title", "No text available"))
                    sponsor_name = "Unknown"
                    introduced_date = "Unknown"

                    if bill_id:
                        # Fetch full bill details for sponsor and dates
                        bill_params = {
                            "key": LEGISCAN_API_KEY,
                            "op": "getBill",
                            "id": bill_id
                        }
                        bill_response = http.get(url, params=bill_params)
                        if bill_response.status_code == 200:
                            bill_detail_data = bill_response.json()
                            if bill_detail_data.get("status") == "OK":
                                bill_detail = bill_detail_data.get("bill", {})

                                # Get sponsor information
                                sponsors = bill_detail.get("sponsors", [])
                                if sponsors and len(sponsors) > 0:
                                    primary_sponsor = sponsors[0]
                                    sponsor_name = primary_sponsor.get("name", "Unknown")

                                # Get introduction date
                                introduced_date = bill_detail.get("history", [{}])[0].get("date", "Unknown") if bill_detail.get("history") else "Unknown"

                        # Fetch full bill text
                        text_params = {
                            "key": LEGISCAN_API_KEY,
                            "op": "getBillText",
                            "id": bill_id
                        }
                        text_response = http.get(url, params=text_params)
                        if text_response.status_code == 200:
                            text_data = text_response.json()
                            if text_data.get("status") == "OK":
                                text_obj = text_data.get("text", {})
                                # Get the doc field which contains base64-encoded document
                                # or the text field if available
                                if "doc" in text_obj:
                                    try:
                                        bill_text = base64.b64decode(text_obj["doc"]).decode('utf-8')
                                    except:
                                        pass  # Keep using description if decode fails

                    bills.append({
                        "source": "LegiScan",
                        "type": "Federal Bill",
                        "number": bill_data.get("bill_number", "Unknown"),
                        "title": bill_data.get("title", "No title"),
                        "url": bill_data.get("url", ""),
                        "status": bill_data.get("status_desc", "Unknown"),
                        "sponsor": sponsor_name,
                        "introduced_date": introduced_date,
                        "summary": bill_text
                    })
                    count += 1

            console.print(f"[green]✓ Found {len(bills)} bills from LegiScan[/green]")
            return bills

    except Exception as e:
        console.print(f"[red]Error fetching from LegiScan: {e}[/red]")
        return []


def fetch_specific_texas_bills(bill_numbers, state="TX"):
    """Fetch specific Texas bills by bill number from LegiScan"""
    if not bill_numbers:
        return []

    console.print(f"\n[bold cyan]Fetching {len(bill_numbers)} Texas bills from LegiScan...[/bold cyan]")

    url = "https://api.legiscan.com/"
    bills = []

    try:
        with httpx.Client(timeout=30.0) as http:
            for bill_number in bill_numbers:
                console.print(f"[dim]  Searching for {state} {bill_number}...[/dim]")

                # Search for specific bill
                search_params = {
                    "key": LEGISCAN_API_KEY,
                    "op": "getSearch",
                    "state": state,
                    "query": bill_number
                }

                response = http.get(url, params=search_params)
                response.raise_for_status()
                data = response.json()

                if data.get("status") != "OK":
                    console.print(f"[yellow]    ⚠ {bill_number} not found[/yellow]")
                    continue

                search_results = data.get("searchresult", {})

                # Find matching bill (exact match on bill_number)
                bill_found = False
                for key, bill_data in search_results.items():
                    if key == "summary":
                        continue

                    # Check if this is the exact bill we're looking for
                    if bill_data.get("bill_number", "").upper() == bill_number.upper():
                        bill_found = True
                        bill_id = bill_data.get("bill_id")
                        bill_text = bill_data.get("description", bill_data.get("title", "No text available"))
                        sponsor_name = "Unknown"
                        introduced_date = "Unknown"
                        status = bill_data.get("status_desc", "Unknown")
                        amendments = []

                        if bill_id:
                            # Fetch full bill details
                            bill_params = {
                                "key": LEGISCAN_API_KEY,
                                "op": "getBill",
                                "id": bill_id
                            }
                            bill_response = http.get(url, params=bill_params)
                            if bill_response.status_code == 200:
                                bill_detail_data = bill_response.json()
                                if bill_detail_data.get("status") == "OK":
                                    bill_detail = bill_detail_data.get("bill", {})

                                    # Get sponsor information
                                    sponsors = bill_detail.get("sponsors", [])
                                    if sponsors and len(sponsors) > 0:
                                        primary_sponsor = sponsors[0]
                                        sponsor_name = primary_sponsor.get("name", "Unknown")

                                    # Get introduction date
                                    introduced_date = bill_detail.get("history", [{}])[0].get("date", "Unknown") if bill_detail.get("history") else "Unknown"

                                    # Get amendments
                                    amendments = bill_detail.get("amendments", [])

                                    # Get current status
                                    status = bill_detail.get("status_desc", status)

                            # Fetch full bill text
                            text_params = {
                                "key": LEGISCAN_API_KEY,
                                "op": "getBillText",
                                "id": bill_id
                            }
                            text_response = http.get(url, params=text_params)
                            if text_response.status_code == 200:
                                text_data = text_response.json()
                                if text_data.get("status") == "OK":
                                    text_obj = text_data.get("text", {})
                                    if "doc" in text_obj:
                                        try:
                                            bill_text = base64.b64decode(text_obj["doc"]).decode('utf-8')
                                        except:
                                            pass  # Keep using description if decode fails

                        bills.append({
                            "source": "LegiScan (Texas)",
                            "type": "Texas State Bill",
                            "number": bill_number,
                            "title": bill_data.get("title", "No title"),
                            "url": bill_data.get("url", ""),
                            "status": status,
                            "sponsor": sponsor_name,
                            "introduced_date": introduced_date,
                            "summary": bill_text,
                            "amendments": amendments,  # Track amendments for change detection
                            "state": state
                        })

                        console.print(f"[green]    ✓ Found {bill_number}[/green]")
                        break

                if not bill_found:
                    console.print(f"[yellow]    ⚠ {bill_number} not found in search results[/yellow]")

        console.print(f"[green]✓ Fetched {len(bills)}/{len(bill_numbers)} Texas bills[/green]")
        return bills

    except Exception as e:
        console.print(f"[red]Error fetching Texas bills: {e}[/red]")
        return bills  # Return whatever we managed to fetch


# =============================================================================
# DATA FETCHING - STATE LEGISLATION (Open States - PRIMARY)
# =============================================================================

def fetch_openstates_bills(jurisdiction="TX", bill_numbers=None, search_query=None, limit=3):
    """
    Fetch state bills from Open States API (Plural Policy).

    ✅ PRIMARY SOURCE FOR STATE LEGISLATION
    Open States is the recommended API for state bill tracking. It offers:
    - Coverage of all 50 states + DC + Puerto Rico + municipalities
    - Full bill version history with PDF links
    - Clear pricing tiers ($0/$59/$499/month)
    - Modern REST API (v3) with good documentation

    API Details:
        - Base URL: https://v3.openstates.org
        - Rate Limits: Tiered (500/day free, up to unlimited on paid plans)
        - Authentication: X-API-KEY header

    Fetching Modes:
        1. Specific bills (bill_numbers): Fetch exact bills by identifier
        2. Search query: Full-text search across bill content
        3. Default: Returns most recent bills for the jurisdiction

    Texas Bill Versions:
        Texas bills typically have these version stages:
        - Introduced: Original filed version
        - House Committee Report: After committee review
        - Engrossed: Passed the originating chamber
        - Senate Committee Report: After second chamber committee
        - Enrolled: Final version passed by both chambers

    Args:
        jurisdiction (str): State code (e.g., "TX", "CA") or "US" for federal
        bill_numbers (list): Specific bill IDs to fetch (e.g., ["HB4238", "SB1091"])
        search_query (str): Full-text search query (e.g., "oil and gas")
        limit (int): Maximum number of bills to return (default: 3)

    Returns:
        list[dict]: List of normalized bill dictionaries with keys:
            - source: "Open States"
            - type: "Texas State Bill" (or appropriate state)
            - number: Bill identifier (e.g., "HB4238")
            - title: Bill title
            - url: Link to bill page
            - status: Latest action description
            - sponsor: Primary sponsor name
            - summary: Bill text for LLM analysis (from PDF extraction)
            - openstates_id: Open States internal ID (for version fetching)
            - versions: List of bill version metadata

    Note:
        - PDF text extraction uses pdfplumber (see extract_pdf_text function)
        - Includes retry logic (3 attempts) for API resilience
        - Returns empty list if API key missing or on error
    """
    if not os.getenv("OPENSTATES_API_KEY"):
        console.print("[yellow]⚠ Open States API key not found in environment[/yellow]")
        return []

    jurisdiction_name = "Federal" if jurisdiction == "US" else jurisdiction.upper()
    console.print(f"\n[bold cyan]Fetching from Open States ({jurisdiction_name})...[/bold cyan]")

    base_url = "https://v3.openstates.org"
    headers = {
        "X-API-KEY": os.getenv("OPENSTATES_API_KEY"),
        "User-Agent": "NRG-Energy-Legislative-Tracker/1.0"
    }
    bills = []

    try:
        with httpx.Client(timeout=30.0) as http:
            # If specific bills requested, fetch each by identifier
            if bill_numbers:
                for bill_num in bill_numbers[:limit]: # NOTE: # If limit exceeds list length, Python returns the entire list without error
                    console.print(f"[dim]  Searching for {jurisdiction} {bill_num}...[/dim]")

                    # Retry logic for robustness
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Search for bill by identifier
                            search_url = f"{base_url}/bills"
                            params = {
                                "jurisdiction": jurisdiction.lower(),
                                "q": bill_num,  # Search query
                                "per_page": 5   # Get a few results to find exact match
                            }

                            response = http.get(search_url, headers=headers, params=params)
                            response.raise_for_status()
                            data = response.json()

                            # Find exact match (Open States uses "HB 49" with space)
                            bill_found = False
                            for result in data.get("results", []):
                                # Match bill number (with or without space)
                                identifier = result.get("identifier", "")
                                if identifier.replace(" ", "").upper() == bill_num.replace(" ", "").upper():
                                    bill_found = True

                                    # Extract data with type safety
                                    actions = result.get("actions", [])
                                    last_action = actions[-1] if actions and isinstance(actions[-1], dict) else {}

                                    sponsorships = result.get("sponsorships", [])
                                    primary_sponsor = sponsorships[0] if sponsorships and isinstance(sponsorships[0], dict) else {}

                                    sources = result.get("sources", [])
                                    abstracts = result.get("abstracts", [])

                                    # Get URL from sources or construct fallback for Texas bills
                                    bill_url = ""
                                    if sources and isinstance(sources[0], dict):
                                        bill_url = sources[0].get("url", "")
                                    if not bill_url and jurisdiction.upper() == "TX":
                                        # Construct Texas Legislature URL
                                        # Format: https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=HB553
                                        session_data = result.get("session", {})
                                        session = session_data.get("identifier", "89R") if isinstance(session_data, dict) else "89R"
                                        bill_id = identifier.replace(" ", "")  # Remove space: "HB 553" -> "HB553"
                                        bill_url = f"https://capitol.texas.gov/BillLookup/History.aspx?LegSess={session}&Bill={bill_id}"

                                    # Extract summary safely (fallback if full text extraction fails)
                                    summary = "No summary available"
                                    if abstracts and isinstance(abstracts[0], dict):
                                        summary = abstracts[0].get("abstract", "No summary available")

                                    # Fetch FULL BILL TEXT from versions endpoint
                                    full_bill_text = None
                                    openstates_id = result.get("id", "")
                                    if openstates_id:
                                        try:
                                            console.print(f"[dim]    Fetching full text for {identifier}...[/dim]")
                                            bill_detail_url = f"{base_url}/bills/{openstates_id}"
                                            detail_params = {"include": "versions"}
                                            detail_response = http.get(bill_detail_url, headers=headers, params=detail_params)
                                            detail_response.raise_for_status()
                                            detail_data = detail_response.json()

                                            versions = detail_data.get("versions", [])
                                            
                                            if versions:
                                                # Get the most recent version (last in list)
                                                latest_version = versions[-1] if versions else None
                                                if latest_version:
                                                    links = latest_version.get("links", [])
                                                    pdf_url = None
                                                    for link in links:
                                                        if link.get("media_type") == "application/pdf":
                                                            pdf_url = link.get("url")
                                                            break

                                                    if pdf_url:
                                                        full_bill_text = extract_pdf_text(pdf_url)
                                                        if full_bill_text:
                                                            console.print(f"[green]    ✓ Extracted {len(full_bill_text.split())} words of full text[/green]")
                                        except Exception as e:
                                            console.print(f"[yellow]    ⚠ Could not fetch full text: {e}[/yellow]")

                                    # Use full text if available, otherwise fall back to abstract
                                    final_summary = full_bill_text if full_bill_text else summary

                                    # Bill Schema Definition for full_data_json storage
                                    bills.append({
                                        "source": "Open States",
                                        "type": "Federal Bill" if jurisdiction == "US" else "Texas State Bill",
                                        "number": identifier,
                                        "title": result.get("title", "No title"),
                                        "url": bill_url,
                                        "status": last_action.get("description", "Unknown") if last_action else "Unknown",
                                        "sponsor": primary_sponsor.get("name", "Unknown") if primary_sponsor else "Unknown",
                                        "introduced_date": result.get("created_at", "Unknown")[:10],  # YYYY-MM-DD
                                        "summary": final_summary,
                                        "abstract": summary,  # Keep original abstract for reference
                                        "has_full_text": full_bill_text is not None,
                                        "updated_at": result.get("updated_at", ""),  # For change tracking
                                        "openstates_id": openstates_id,
                                    })

                                    console.print(f"[green]    ✓ Found {identifier}[/green]")
                                    break

                            if not bill_found:
                                console.print(f"[yellow]    ⚠ {bill_num} not found[/yellow]")

                            break  # Success, exit retry loop

                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 429:
                                console.print(f"[yellow]    Rate limit hit, retrying (attempt {attempt + 1}/{max_retries})...[/yellow]")
                                if attempt < max_retries - 1:
                                    import time
                                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                                else:
                                    console.print(f"[red]    Failed after {max_retries} attempts[/red]")
                            elif e.response.status_code == 401:
                                console.print("[red]Authentication failed - check OPENSTATES_API_KEY[/red]")
                                return bills
                            else:
                                console.print(f"[red]    HTTP {e.response.status_code}: {e}[/red]")
                                break
                        except Exception as e:
                            console.print(f"[red]    Error: {e}[/red]")
                            if attempt == max_retries - 1:
                                break

            # If search query provided (and no specific bills, or in addition to them)
            elif search_query:
                console.print(f"[dim]  Searching for: '{search_query}'...[/dim]")

                search_url = f"{base_url}/bills"
                params = {
                    "jurisdiction": jurisdiction.lower(),
                    "q": search_query,
                    "per_page": limit,
                    "sort": "updated_at"  # Most recently updated first
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

                    # Extract abstract as fallback
                    summary = abstracts[0].get("abstract", "No summary available") if abstracts else "No summary available"

                    # Fetch FULL BILL TEXT from versions endpoint
                    full_bill_text = None
                    openstates_id = result.get("id", "")
                    if openstates_id:
                        try:
                            console.print(f"[dim]    Fetching full text for {identifier}...[/dim]")
                            bill_detail_url = f"{base_url}/bills/{openstates_id}"
                            detail_params = {"include": "versions"}
                            detail_response = http.get(bill_detail_url, headers=headers, params=detail_params)
                            detail_response.raise_for_status()
                            detail_data = detail_response.json()

                            versions = detail_data.get("versions", [])
                            if versions:
                                # Get the most recent version (last in list)
                                latest_version = versions[-1] if versions else None
                                if latest_version:
                                    links = latest_version.get("links", [])
                                    pdf_url = None
                                    for link in links:
                                        if link.get("media_type") == "application/pdf":
                                            pdf_url = link.get("url")
                                            break

                                    if pdf_url:
                                        full_bill_text = extract_pdf_text(pdf_url)
                                        if full_bill_text:
                                            console.print(f"[green]    ✓ Extracted {len(full_bill_text.split())} words of full text[/green]")
                        except Exception as e:
                            console.print(f"[yellow]    ⚠ Could not fetch full text: {e}[/yellow]")

                    # Use full text if available, otherwise fall back to abstract
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
                        "abstract": summary,  # Keep original abstract for reference
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


# =============================================================================
# LLM ANALYSIS - OPENAI GPT-5
# =============================================================================

def analyze_with_openai(item, nrg_context, custom_prompt=None):
    """
    Analyze a bill/regulation using OpenAI GPT-5 Responses API.

    This function sends the bill text along with NRG business context to GPT-5
    for comprehensive legislative analysis. The output is a structured JSON
    response following a specific schema defined in the prompt.

    Model Configuration:
        - Model: gpt-5 (via OpenAI Responses API)
        - Reasoning Effort: "medium" (balanced depth/cost)
        - Text Verbosity: "high" (detailed explanations)

    Analysis Output Schema:
        The LLM returns a JSON object containing:
        - bill_version: Version stage (as_filed, enrolled, etc.)
        - business_impact_score: 0-10 impact rating
        - impact_type: regulatory_compliance | financial | operational | etc.
        - legal_code_changes: Sections amended/added/deleted
        - application_scope: Who the bill applies to, exclusions
        - effective_dates: Implementation timeline
        - mandatory_vs_permissive: SHALL vs MAY provisions
        - nrg_business_verticals: Affected business areas (mapped to 18 verticals)
        - affected_nrg_assets: Specific facilities/markets impacted
        - financial_impact: Estimated cost/revenue impact
        - recommended_action: ignore | monitor | engage | urgent

    Args:
        item (dict): Bill/regulation dictionary with at minimum:
            - type: "Federal Bill" | "Regulation" | "Texas State Bill"
            - source: API source name
            - number: Bill identifier
            - title: Bill title
            - status: Current status
            - summary: Full bill text for analysis
        nrg_context (str): NRG business context from nrg_business_context.txt

    Returns:
        dict: Structured analysis with all fields from schema above,
              or error dict with business_impact_score=0 on failure

    Note:
        - Cost: ~$0.05-0.15 per analysis depending on text length
        - Uses fail-soft design: returns minimal valid dict on error
    """
    # Use custom prompt if provided, otherwise build default
    if custom_prompt:
        combined_input = custom_prompt
    else:
        # TODO: the following bill_version is using Federal terminology. This cause inconsistency with OpenStates as it focuses on States Bills. Results in misleading (might be incorrect) bill version in reports. 
        combined_input = f"""You are a legislative analyst for NRG Energy's Government Affairs team. Your role is to provide deep, professional-grade legislative analysis similar to what a law firm would provide.

NRG BUSINESS CONTEXT:
{nrg_context}

CRITICAL INSTRUCTIONS:
1. Analyze this legislation with LEGAL PRECISION - identify specific code sections, mandatory vs permissive language, exceptions vs exemptions
2. Focus ONLY on portions relevant to NRG Energy - ignore unrelated sections
3. Map to NRG's business verticals (may be multiple)
4. Extract the substance of legal changes, not just summaries
5. Respond ONLY with valid JSON - no explanatory text before or after

REQUIRED JSON STRUCTURE:
{{
  "bill_version": "as_filed | house_committee | passed_house | senate_committee | passed_senate | conference_committee | enrolled | unknown",
  "business_impact_score": <0-10 integer>,
  "impact_type": "regulatory_compliance | financial | operational | market | strategic | minimal",
  "impact_summary": "<2-3 sentences on NRG business impact>",

  "legal_code_changes": {{
    "sections_amended": ["<Code Name> §<section>", ...],
    "sections_added": ["New Chapter X of <Code Name>", ...],
    "sections_deleted": ["<Code Name> §<section>", ...],
    "chapters_repealed": ["Chapter X of <Code Name>", ...],
    "substance_of_changes": "<Detailed explanation of what's changing from current law>"
  }},

  "application_scope": {{
    "applies_to": ["<who this bill applies to>", ...],
    "exclusions": ["<who/what is excluded>", ...],
    "geographic_scope": ["<states/regions>", ...]
  }},

  "effective_dates": [
    {{
      "date": "<date or 'upon passage' or 'unknown'>",
      "applies_to": "<which provisions>"
    }}
  ],

  "mandatory_vs_permissive": {{
    "mandatory_provisions": ["<SHALL/MUST/REQUIRED provisions>", ...],
    "permissive_provisions": ["<MAY/CAN/OPTIONAL provisions>", ...]
  }},

  "exceptions_and_exemptions": {{
    "exceptions": ["<exceptions - bill applies, person must prove exception applies>", ...],
    "exemptions": ["<exemptions - person is exempt, must be proven otherwise>", ...]
  }},

  "nrg_business_verticals": ["<select from: Cyber and Grid Security, Data Privacy/Public Information Act, Disaster/Business Continuity, Electric Vehicles, Environmental/Water/Sustainability, Natural Gas, General Business, General Government, Renewables/Distributed Generation/Demand Response, Retail Commodity, Retail Non-commodity, Services, Tax, Transmission-Distribution, Wholesale Market Reforms, Artificial Intelligence, Economic Development/Workforce Development, Electric Generation, Public Utility Commission of Texas>"],

  "nrg_vertical_impact_details": {{
    "<vertical_name>": "<specific impact on this vertical>",
    ...
  }},

  "nrg_relevant_excerpts": ["<Section X: Direct quote of NRG-relevant provision>", ...],

  "affected_nrg_assets": {{
    "generation_facilities": ["<specific facilities or types affected>", ...],
    "geographic_exposure": ["<markets/states>", ...],
    "business_units": ["<NRG business units>", ...]
  }},

  "financial_impact": "<estimated cost/revenue impact or 'unknown'>",
  "timeline": "<when this matters>",
  "risk_or_opportunity": "risk | opportunity | mixed | neutral",
  "recommended_action": "ignore | monitor | engage | urgent",

  "internal_stakeholders": ["<NRG departments/teams to involve>", ...]
}}

LEGISLATION TO ANALYZE:

Type: {item['type']}
Source: {item['source']}
Number: {item['number']}
Title: {item['title']}
Status: {item['status']}

Full Text/Summary:
{item['summary']}

ANALYSIS FOCUS AREAS:
1. What code sections are being amended/added/deleted?
2. Is this mandatory (SHALL/MUST) or permissive (MAY)?
3. Who does this apply to? Are there exclusions or exemptions?
4. What are the effective dates?
5. Which NRG business verticals are affected?
6. What are the NRG-relevant provisions (ignore irrelevant sections)?
7. What is the business impact to NRG specifically?

Provide comprehensive JSON analysis following the exact structure above."""

    try:
        # Use GPT-5 Responses API
        response = openai_client.responses.create(
            model="gpt-5",
            input=combined_input,
            reasoning={"effort": "medium"},  # Balanced analysis
            text={"verbosity": "high"}       # Detailed, verbose explanations
        )

        # Parse JSON from output_text
        analysis = json.loads(response.output_text)
        return analysis

    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON from GPT-5: {e}[/red]")
        console.print(f"[dim]Raw output: {response.output_text[:200]}...[/dim]")
        return {
            "error": f"JSON parse error: {str(e)}",
            "business_impact_score": 0,
            "impact_summary": "Analysis failed - invalid JSON response"
        }
    except Exception as e:
        console.print(f"[red]Error analyzing with GPT-5: {e}[/red]")
        return {
            "error": str(e),
            "business_impact_score": 0,
            "impact_summary": "Analysis failed"
        }


# -----------------------------------------------------------------------------
# GOOGLE GEMINI ANALYSIS
# -----------------------------------------------------------------------------
# Alternative LLM provider using Google Gemini Pro (gemini-3-pro-preview)
# Functionally equivalent to OpenAI analysis but uses Google's API
# Enabled by setting llm.provider: "gemini" in config.yaml
#
# Key differences from OpenAI:
# - Uses response_mime_type: "application/json" for structured output
# - Temperature-based creativity control (0.2 = deterministic)
# - No equivalent to reasoning_effort parameter
# - Generally faster response times, competitive accuracy
# -----------------------------------------------------------------------------

def analyze_with_gemini(item, nrg_context, custom_prompt=None):
    """
    Analyze legislative item using Google Gemini Pro for NRG business impact.

    This function is functionally equivalent to analyze_with_openai() but uses
    Google's Gemini API instead. The prompt structure and JSON output schema
    are identical to ensure consistent analysis regardless of LLM provider.

    API Details:
        - Model: gemini-3-pro-preview (configurable in config.yaml)
        - Output Mode: application/json (forced JSON response)
        - Temperature: 0.2 (deterministic, consistent analysis)
        - Max Tokens: 8192 (supports comprehensive analysis)

    Args:
        item (dict): Legislative item with required fields:
            - type: "Federal Bill" | "State Bill" | "Regulation"
            - source: API source name
            - number: Bill/regulation identifier
            - title: Official title
            - status: Current legislative status
            - summary: Full text or description for analysis
        nrg_context (str): Full NRG business context from nrg_business_context.txt

    Returns:
        dict: Structured analysis with identical schema to OpenAI analysis:
            - business_impact_score: 0-10 integer
            - impact_type: regulatory_compliance|financial|operational|market|strategic|minimal
            - legal_code_changes: sections amended/added/deleted
            - application_scope: who this applies to
            - nrg_business_verticals: list of affected verticals
            - recommended_action: ignore|monitor|engage|urgent
            - ... (see full JSON schema in prompt)

        On error, returns minimal dict with error message and score of 0.

    Provider Selection:
        The LLM provider is selected in config.yaml:
            llm:
              provider: "gemini"  # or "openai"

        Main execution calls the appropriate function based on this setting.
    """

    # Use custom prompt if provided, otherwise build default
    if custom_prompt:
        combined_input = custom_prompt
    else:
        # TODO: the following bill_version is using Federal terminology. 
        # This cause inconsistency with OpenStates as it focuses on States Bills. 
        # Results in misleading (might be incorrect) bill version in reports. 
        combined_input = f"""You are a legislative analyst for NRG Energy's Government Affairs team. Your role is to provide deep, professional-grade legislative analysis similar to what a law firm would provide.

NRG BUSINESS CONTEXT:
{nrg_context}

CRITICAL INSTRUCTIONS:
1. Analyze this legislation with LEGAL PRECISION - identify specific code sections, mandatory vs permissive language, exceptions vs exemptions
2. Focus ONLY on portions relevant to NRG Energy - ignore unrelated sections
3. Map to NRG's business verticals (may be multiple)
4. Extract the substance of legal changes, not just summaries
5. Respond ONLY with valid JSON - no explanatory text before or after

REQUIRED JSON STRUCTURE:
{{
  "bill_version": "as_filed | house_committee | passed_house | senate_committee | passed_senate | conference_committee | enrolled | unknown",
  "business_impact_score": <0-10 integer>,
  "impact_type": "regulatory_compliance | financial | operational | market | strategic | minimal",
  "impact_summary": "<2-3 sentences on NRG business impact>",

  "legal_code_changes": {{
    "sections_amended": ["<Code Name> §<section>", ...],
    "sections_added": ["New Chapter X of <Code Name>", ...],
    "sections_deleted": ["<Code Name> §<section>", ...],
    "chapters_repealed": ["Chapter X of <Code Name>", ...],
    "substance_of_changes": "<Detailed explanation of what's changing from current law>"
  }},

  "application_scope": {{
    "applies_to": ["<who this bill applies to>", ...],
    "exclusions": ["<who/what is excluded>", ...],
    "geographic_scope": ["<states/regions>", ...]
  }},

  "effective_dates": [
    {{
      "date": "<date or 'upon passage' or 'unknown'>",
      "applies_to": "<which provisions>"
    }}
  ],

  "mandatory_vs_permissive": {{
    "mandatory_provisions": ["<SHALL/MUST/REQUIRED provisions>", ...],
    "permissive_provisions": ["<MAY/CAN/OPTIONAL provisions>", ...]
  }},

  "exceptions_and_exemptions": {{
    "exceptions": ["<exceptions - bill applies, person must prove exception applies>", ...],
    "exemptions": ["<exemptions - person is exempt, must be proven otherwise>", ...]
  }},

  "nrg_business_verticals": ["<select from: Cyber and Grid Security, Data Privacy/Public Information Act, Disaster/Business Continuity, Electric Vehicles, Environmental/Water/Sustainability, Natural Gas, General Business, General Government, Renewables/Distributed Generation/Demand Response, Retail Commodity, Retail Non-commodity, Services, Tax, Transmission-Distribution, Wholesale Market Reforms, Artificial Intelligence, Economic Development/Workforce Development, Electric Generation, Public Utility Commission of Texas>"],

  "nrg_vertical_impact_details": {{
    "<vertical_name>": "<specific impact on this vertical>",
    ...
  }},

  "nrg_relevant_excerpts": ["<Section X: Direct quote of NRG-relevant provision>", ...],

  "affected_nrg_assets": {{
    "generation_facilities": ["<specific facilities or types affected>", ...],
    "geographic_exposure": ["<markets/states>", ...],
    "business_units": ["<NRG business units>", ...]
  }},

  "financial_impact": "<estimated cost/revenue impact or 'unknown'>",
  "timeline": "<when this matters>",
  "risk_or_opportunity": "risk | opportunity | mixed | neutral",
  "recommended_action": "ignore | monitor | engage | urgent",

  "internal_stakeholders": ["<NRG departments/teams to involve>", ...]
}}

LEGISLATION TO ANALYZE:

Type: {item['type']}
Source: {item['source']}
Number: {item['number']}
Title: {item['title']}
Status: {item['status']}

Full Text/Summary:
{item['summary']}

ANALYSIS FOCUS AREAS:
1. What code sections are being amended/added/deleted?
2. Is this mandatory (SHALL/MUST) or permissive (MAY)?
3. Who does this apply to? Are there exclusions or exemptions?
4. What are the effective dates?
5. Which NRG business verticals are affected?
6. What are the NRG-relevant provisions (ignore irrelevant sections)?
7. What is the business impact to NRG specifically?

Provide comprehensive, detailed JSON analysis following the exact structure above. Be thorough and verbose in your analysis."""

    try:
        # Use Gemini API with JSON response mode
        response = gemini_client.models.generate_content(
            model=config['llm']['gemini']['model'],
            contents=combined_input,
            config={
                "temperature": config['llm']['gemini'].get('temperature', 0.2),
                "max_output_tokens": config['llm']['gemini'].get('max_output_tokens', 8192),
                "response_mime_type": "application/json"  # Force JSON output
            }
        )

        # Parse JSON from response
        analysis = json.loads(response.text)
        return analysis

    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON from Gemini: {e}[/red]")
        console.print(f"[dim]Raw output: {response.text[:200]}...[/dim]")
        return {
            "error": f"JSON parse error: {str(e)}",
            "business_impact_score": 0,
            "impact_summary": "Analysis failed - invalid JSON response"
        }
    except Exception as e:
        console.print(f"[red]Error analyzing with Gemini: {e}[/red]")
        return {
            "error": str(e),
            "business_impact_score": 0,
            "impact_summary": "Analysis failed"
        }


# -----------------------------------------------------------------------------
# LLM PROVIDER ROUTING
# -----------------------------------------------------------------------------
# Central dispatcher that routes analysis requests to the configured LLM
# This allows seamless switching between providers via config.yaml
# -----------------------------------------------------------------------------

def analyze_with_llm(item, nrg_context):
    """
    Route legislative analysis to the appropriate LLM provider.

    This is the primary entry point for LLM analysis. It reads the provider
    setting from config.yaml and dispatches to the correct function.

    Args:
        item (dict): Legislative item to analyze (see analyze_with_openai docstring)
        nrg_context (str): NRG business context for analysis

    Returns:
        dict: Structured analysis from the selected LLM provider

    Configuration:
        Set in config.yaml under llm.provider:
        - "gemini": Uses Google Gemini Pro (gemini-3-pro-preview)
        - "openai": Uses OpenAI GPT-5
    """
    provider = config['llm']['provider']

    if provider == 'gemini':
        return analyze_with_gemini(item, nrg_context)
    elif provider == 'openai':
        return analyze_with_openai(item, nrg_context)
    else:
        console.print(f"[red]Unknown LLM provider: {provider}. Defaulting to OpenAI.[/red]")
        return analyze_with_openai(item, nrg_context)


# =============================================================================
# SECTION 3: BILL VERSION TRACKING AND CHANGE ANALYSIS
# =============================================================================
# This section handles the tracking of bill versions through the legislative
# process and provides semantic analysis of changes between versions.
#
# Bill Version Lifecycle:
#   Introduced (IH) → Committee Report (RH) → Passed House (EH) →
#   Senate Committee (RS) → Passed Senate (ES) → Conference → Enrolled (ENR)
#
# Key Functions:
#   - analyze_bill_version(): LLM analysis of individual version
#   - compare_consecutive_versions(): Compute diff between versions
#   - analyze_version_changes_with_llm(): Semantic change analysis
#
# Configuration (config.yaml):
#   version_tracking:
#     enabled: true
#     fetch_all_versions: true
#     analyze_all_versions: true
#     compare_versions: true
# =============================================================================

def analyze_bill_version(version_text, version_type, bill_info, nrg_context):
    """
    Analyze a single bill version independently using configured LLM.

    This function enables version-specific analysis, allowing tracking of how
    a bill's impact on NRG changes as it moves through the legislative process.
    Each version (Introduced, Committee Report, Enrolled, etc.) may have
    significantly different provisions.

    Args:
        version_text (str): Full extracted text of this bill version
        version_type (str): Version stage, e.g.:
            - "Introduced" (IH) - Original filed version
            - "House Committee Report" (RH) - After committee amendments
            - "Engrossed" (EH) - Passed chamber
            - "Senate Committee Report" (RS) - After senate committee
            - "Enrolled" (ENR) - Final passed version
        bill_info (dict): Bill metadata including:
            - number: Bill identifier (e.g., "HB4238")
            - title: Official bill title
            - source: API source ("OpenStates", "Congress.gov")
            - type: "Federal Bill" or "State Bill"
        nrg_context (str): NRG business context for analysis

    Returns:
        dict: Full LLM analysis with:
            - business_impact_score: 0-10 rating for this version
            - impact_summary: Version-specific impact description
            - recommended_action: ignore|monitor|engage|urgent
            - All other standard analysis fields

    Use Case:
        Track how amendments change bill impact. An "Introduced" version
        might score 8/10 but the "Enrolled" version scores 3/10 due to
        amendments that removed problematic provisions.
    """
    provider = config['llm']['provider']

    # Prepare version-specific prompt
    prompt_text = f"""You are a legislative analyst for NRG Energy's Government Affairs team analyzing a specific version of a bill.

**BILL INFORMATION:**
- Bill Number: {bill_info.get('number', 'Unknown')}
- Bill Title: {bill_info.get('title', 'Unknown')}
- **Version Being Analyzed: {version_type}**  <-- THIS IS THE VERSION YOU ARE ANALYZING
- Source: {bill_info.get('source', 'Unknown')}

**VERSION TEXT:**
{version_text[:20000]}

**NRG BUSINESS CONTEXT:**
{nrg_context}

**CRITICAL INSTRUCTIONS:**
Analyze THIS VERSION ({version_type}) as if it represents the current state of the legislation.
- What does THIS VERSION do?
- How does THIS VERSION impact NRG Energy's business?
- What is the business impact score for THIS VERSION specifically?
- What action should NRG take at THIS STAGE of the legislative process?

Provide your analysis in the same JSON format as always, but remember you are analyzing the {version_type} version specifically.
"""

    # Create a temporary item for the LLM analysis
    temp_item = {
        "source": bill_info.get('source', 'Unknown'),
        "number": bill_info.get('number', 'Unknown'),
        "title": f"{bill_info.get('title', 'Unknown')} ({version_type})",
        "summary": version_text[:20000],  # First 20K chars
        "type": bill_info.get('type', 'Bill'),
        "status": version_type,
        "url": bill_info.get('url', '')
    }

    # Use default prompt with enforced JSON schema
    if provider == 'gemini':
        return analyze_with_gemini(temp_item, nrg_context)
    elif provider == 'openai':
        return analyze_with_openai(temp_item, nrg_context)
    else:
        return analyze_with_openai(temp_item, nrg_context)


def compare_consecutive_versions(old_version, new_version):
    """
    Generate diff between two consecutive bill versions

    Parameters:
    - old_version: Dictionary with version_type, full_text, etc.
    - new_version: Dictionary with version_type, full_text, etc.

    Returns:
    - Dictionary with diff statistics and summary
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


def analyze_version_changes_with_llm(old_version, new_version, old_analysis, new_analysis, bill_info, nrg_context):
    """
    Use LLM to analyze substantive changes between two consecutive bill versions

    Parameters:
    - old_version: Previous version dict with full_text, version_type, etc.
    - new_version: Current version dict with full_text, version_type, etc.
    - old_analysis: LLM analysis of old version
    - new_analysis: LLM analysis of new version
    - bill_info: Bill metadata (number, title, etc.)
    - nrg_context: NRG business context

    Returns:
    - Dictionary with semantic change analysis
    """
    provider = config['llm']['provider']

    # Get abbreviated versions of each text (first 15K chars to stay within context limits)
    old_text_sample = old_version.get('full_text', '')[:15000]
    new_text_sample = new_version.get('full_text', '')[:15000]

    # Build prompt for semantic change analysis
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
Analyze the substantive changes between these two versions from NRG's perspective. Focus on:

1. **Key Provisions Changed**: What major legal/operational provisions were added, removed, or modified?
2. **Impact Evolution**: How did NRG's risk/opportunity change? Why did the impact score change from {old_analysis.get('business_impact_score', 'N/A')} to {new_analysis.get('business_impact_score', 'N/A')}?
3. **Compliance Changes**: What new requirements or exemptions appeared/disappeared?
4. **Strategic Significance**: What do these changes tell us about legislative intent?

Provide a concise analysis (200-400 words) focusing on SUBSTANTIVE changes, not formatting/clerical edits.

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

    # Call LLM
    try:
        # Create a temporary item dict with all required fields for the analysis functions
        temp_item = {
            "title": "Version Change Analysis",
            "summary": prompt,
            "number": bill_info.get('number'),
            "type": bill_info.get('type', 'Unknown'),
            "source": bill_info.get('source', 'Unknown'),
            "status": f"Comparing {old_version.get('version_type')} to {new_version.get('version_type')}"
        }

        if provider == 'gemini':
            response = analyze_with_gemini(temp_item, nrg_context)
        else:
            response = analyze_with_openai(temp_item, nrg_context)

        # The response should already be a dict from the LLM analysis functions
        # Extract the change analysis fields if they exist
        if isinstance(response, dict):
            return {
                "key_provisions_added": response.get("key_provisions_added", []),
                "key_provisions_removed": response.get("key_provisions_removed", []),
                "key_provisions_modified": response.get("key_provisions_modified", []),
                "impact_evolution": response.get("impact_evolution", "No analysis available"),
                "compliance_changes": response.get("compliance_changes", "No analysis available"),
                "strategic_significance": response.get("strategic_significance", "No analysis available"),
                "summary": response.get("summary", response.get("impact_summary", "No summary available"))
            }
        else:
            return {
                "summary": "Error: LLM returned unexpected format",
                "key_provisions_added": [],
                "key_provisions_removed": [],
                "key_provisions_modified": [],
                "impact_evolution": "Unable to analyze",
                "compliance_changes": "Unable to analyze",
                "strategic_significance": "Unable to analyze"
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


# =============================================================================
# SECTION 4: OUTPUT AND REPORT GENERATION
# =============================================================================
# This section handles all output formatting and report generation:
#   - Console display with color-coded impact panels (Rich library)
#   - JSON report generation (structured data export)
#   - Markdown report generation (human-readable format)
#   - Word document generation (via pandoc conversion)
#
# Output Files Generated:
#   - nrg_analysis_YYYYMMDD_HHMMSS.json  - Structured data for downstream systems
#   - nrg_analysis_YYYYMMDD_HHMMSS.md    - Human-readable markdown report
#   - nrg_analysis_YYYYMMDD_HHMMSS.docx  - Word document (if pandoc installed)
#
# Impact Color Coding:
#   - RED: High Impact (score 7-10) - Immediate attention required
#   - YELLOW: Medium Impact (score 4-6) - Monitor closely
#   - GREEN: Low Impact (score 0-3) - Routine monitoring
# =============================================================================

def display_analysis(item, analysis):
    """
    Display legislative analysis in the console with color-coded formatting.

    Uses Rich library to create visually organized panels showing:
    - Bill/regulation metadata (source, number, sponsor, dates)
    - Impact score with color coding (red/yellow/green)
    - Business impact summary and affected verticals
    - Legal code changes (amendments, additions, deletions)
    - Recommended action and stakeholders

    Args:
        item (dict): Legislative item with metadata (title, number, source, etc.)
        analysis (dict): LLM analysis results with impact scoring

    Color Coding:
        - Score 7-10: RED panel border (HIGH IMPACT)
        - Score 4-6: YELLOW panel border (MEDIUM IMPACT)
        - Score 0-3: GREEN panel border (LOW IMPACT)

    Console Output:
        Renders a Rich Panel with formatted content including:
        - Title and metadata block
        - Impact score and type
        - Business summary
        - NRG business verticals affected
        - Legal changes summary
        - Recommended action
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

    # Build metadata section based on item type
    metadata_lines = [
        f"[cyan]Source:[/cyan] {item['source']}",
        f"[cyan]Number:[/cyan] {item['number']}",
        f"[cyan]Type:[/cyan] {item['type']}",
    ]

    # Add sponsor/agency
    if 'sponsor' in item and item['sponsor'] != 'Unknown':
        metadata_lines.append(f"[cyan]Sponsor:[/cyan] {item['sponsor']}")
    elif 'agency' in item and item['agency'] != 'Unknown':
        metadata_lines.append(f"[cyan]Agency:[/cyan] {item['agency'].upper()}")

    # Add policy area for bills
    if 'policy_area' in item and item['policy_area'] != 'Unknown':
        metadata_lines.append(f"[cyan]Policy Area:[/cyan] {item['policy_area']}")

    metadata_lines.append(f"[cyan]Status:[/cyan] {item['status']}")

    # Add dates
    if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
        metadata_lines.append(f"[cyan]Introduced:[/cyan] {item['introduced_date']}")
    if 'posted_date' in item and item['posted_date'] != 'Unknown':
        metadata_lines.append(f"[cyan]Posted:[/cyan] {item['posted_date']}")
    if 'comment_end_date' in item and item['comment_end_date'] not in ['N/A', 'Unknown']:
        metadata_lines.append(f"[cyan]Comment Deadline:[/cyan] {item['comment_end_date']}")
    if 'effective_date' in item and item['effective_date'] not in ['N/A', 'Unknown']:
        metadata_lines.append(f"[cyan]Effective:[/cyan] {item['effective_date']}")

    metadata_lines.append(f"[cyan]Link:[/cyan] {item['url']}")

    # Add bill version to metadata
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
        content += f"\n\n[bold cyan]Legal Code Changes:[/bold cyan]"
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
    if effective_dates and len(effective_dates) > 0:
        content += f"\n\n[bold]Effective Dates:[/bold]"
        for ed in effective_dates[:2]:  # Show first 2
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


# -----------------------------------------------------------------------------
# MARKDOWN REPORT GENERATION
# -----------------------------------------------------------------------------

def generate_markdown_report(results, timestamp, output_dir=None):
    """
    Generate comprehensive Markdown report for legislative analysis results.

    Creates a professionally formatted report suitable for Government Affairs
    team consumption, organized by impact priority with full version tracking.

    Report Structure:
        1. Executive Summary - Item counts by impact level
        2. High Impact Section (Score 7-10) - Immediate action items
           - Version timeline with semantic change analysis
           - Key provisions added/removed/modified
        3. Medium Impact Section (Score 4-6) - Monitor items
        4. Low Impact Section (Score 0-3) - Awareness items
        5. Version History Summary (if change tracking enabled)

    Args:
        results (list): List of analysis results, each containing:
            - item: Original legislative item dict
            - analysis: LLM analysis results
            - version_analyses: List of per-version analyses (optional)
            - version_diffs: List of version comparison diffs (optional)
        timestamp (str): Timestamp for filename (YYYYMMDD_HHMMSS format)

    Output:
        Creates file: nrg_analysis_{timestamp}.md

    Version Timeline Features:
        - Tracks bill evolution through legislative stages
        - Shows impact score changes between versions
        - Includes LLM semantic analysis of substantive changes
        - Displays text statistics (lines added/removed, word count delta)
    """

    # Group by impact level
    high_impact = [r for r in results if r["analysis"].get("business_impact_score", 0) >= 7]
    medium_impact = [r for r in results if 4 <= r["analysis"].get("business_impact_score", 0) < 7]
    low_impact = [r for r in results if r["analysis"].get("business_impact_score", 0) < 4]

    # Create file path
    if output_dir:
        md_file = os.path.join(output_dir, f"nrg_analysis_{timestamp}.md")
    else:
        md_file = f"nrg_analysis_{timestamp}.md"

    with open(md_file, "w") as f:
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

                # Impact Assessment as bullet list
                f.write(f"**Impact Assessment:**\n\n")
                f.write(f"- **Score:** {analysis.get('business_impact_score', 0)}/10 ⚠️\n")
                f.write(f"- **Type:** {analysis.get('impact_type', 'N/A').replace('_', ' ').title()}\n")
                f.write(f"- **Risk Level:** {analysis.get('risk_or_opportunity', 'N/A').upper()}\n\n")

                # Bill Information as bullet list
                f.write(f"**Bill Information:**\n\n")
                f.write(f"- **Source:** {item['source']}\n")
                f.write(f"- **Number:** {item['number']}\n")

                # Add sponsor/agency
                if 'sponsor' in item and item['sponsor'] != 'Unknown':
                    f.write(f"- **Sponsor:** {item['sponsor']}\n")
                elif 'agency' in item and item['agency'] != 'Unknown':
                    f.write(f"- **Agency:** {item['agency'].upper()}\n")

                # Add policy area for bills
                if 'policy_area' in item and item['policy_area'] != 'Unknown':
                    f.write(f"- **Policy Area:** {item['policy_area']}\n")

                f.write(f"- **Status:** {item['status']}\n")

                # Add dates
                if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
                    f.write(f"- **Introduced:** {item['introduced_date']}\n")
                if 'posted_date' in item and item['posted_date'] != 'Unknown':
                    f.write(f"- **Posted:** {item['posted_date']}\n")
                if 'comment_end_date' in item and item['comment_end_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Comment Deadline:** {item['comment_end_date']}\n")
                if 'effective_date' in item and item['effective_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Effective Date:** {item['effective_date']}\n")

                f.write(f"- **Bill Version:** {analysis.get('bill_version', 'unknown')}\n")
                f.write(f"- **Link:** {item['url']}\n\n")

                # Version Timeline (if available)
                version_analyses = result.get('version_analyses', [])
                version_diffs = result.get('version_diffs', [])

                if version_analyses and len(version_analyses) > 1:
                    f.write(f"**📚 VERSION TIMELINE ({len(version_analyses)} versions analyzed)**\n\n")
                    f.write(f"This bill has evolved through {len(version_analyses)} legislative versions. Each version was analyzed independently to track how NRG's risk profile changed:\n\n")

                    # Summary table of versions
                    for idx, va in enumerate(version_analyses, 1):
                        version = va['version']
                        v_analysis = va['analysis']
                        v_type = version.get('version_type', 'Unknown')
                        v_date = version.get('version_date', 'N/A')
                        impact_score = v_analysis.get('business_impact_score', 0)

                        f.write(f"{idx}. **{v_type}** ({v_date}) - Impact Score: {impact_score}/10\n")

                    f.write(f"\n")

                    # Detailed version-by-version analysis
                    f.write(f"**Detailed Version Analysis:**\n\n")

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

                        # Show version-to-version diff if available
                        if idx > 1 and len(version_diffs) >= idx - 1:
                            diff = version_diffs[idx - 2]
                            if diff.get('changed'):
                                # Show semantic analysis from LLM
                                semantic = diff.get('semantic_analysis', {})

                                f.write(f"- **Changes from {diff.get('from_version')}:**\n\n")

                                # Summary
                                if semantic.get('summary'):
                                    f.write(f"  *{semantic['summary']}*\n\n")

                                # Key provisions changed
                                if semantic.get('key_provisions_added'):
                                    f.write(f"  **Provisions Added:**\n")
                                    for prov in semantic['key_provisions_added'][:3]:  # Limit to top 3
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_removed'):
                                    f.write(f"  **Provisions Removed:**\n")
                                    for prov in semantic['key_provisions_removed'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_modified'):
                                    f.write(f"  **Provisions Modified:**\n")
                                    for prov in semantic['key_provisions_modified'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                # Impact evolution
                                if semantic.get('impact_evolution'):
                                    f.write(f"  **Impact Evolution:** {semantic['impact_evolution']}\n\n")

                                # Text statistics
                                f.write(f"  *Text changes: {diff.get('lines_added', 0)} lines added, {diff.get('lines_removed', 0)} removed, {diff.get('word_count_change', 0):+d} word delta*\n")
                            else:
                                f.write(f"- **No substantive changes from previous version**\n")

                        f.write(f"\n")

                    f.write(f"---\n\n")

                # Change Information (if available)
                change_data = result.get('change_data')
                change_impact = result.get('change_impact')

                if change_data and change_data.get('has_changes'):
                    if change_data.get('is_new'):
                        f.write(f"**📍 NEW BILL - First Time Analyzed**\n\n")
                    else:
                        f.write(f"**⚠️ CHANGES DETECTED**\n\n")
                        for change in change_data.get('changes', []):
                            f.write(f"- **{change['type'].replace('_', ' ').title()}:** {change.get('summary', 'Change detected')}\n")

                        if change_impact:
                            f.write(f"\n**Change Impact Analysis:**\n\n")
                            f.write(f"- **Change Impact Score:** {change_impact.get('change_impact_score', 'N/A')}/10\n")
                            f.write(f"- **Impact Trend:** {'INCREASED ⬆️' if change_impact.get('impact_increased') else 'DECREASED ⬇️'}\n")
                            f.write(f"- **Summary:** {change_impact.get('change_summary', 'N/A')}\n")
                            f.write(f"- **Recommended Action:** {change_impact.get('recommended_action', 'N/A').upper()}\n")

                        f.write("\n")

                # NRG Business Verticals
                verticals = analysis.get('nrg_business_verticals', [])
                if verticals:
                    f.write(f"**NRG Business Verticals:**\n\n")
                    for vertical in verticals:
                        f.write(f"- {vertical}\n")
                    f.write("\n")

                f.write(f"**Why This Matters to NRG:**\n")
                f.write(f"{analysis.get('impact_summary', 'No summary available')}\n\n")

                # Legal Code Changes
                legal_changes = analysis.get('legal_code_changes', {})
                if legal_changes and any(legal_changes.values()):
                    f.write(f"**Legal Code Changes:**\n\n")
                    if legal_changes.get('sections_amended'):
                        f.write(f"- **Amended:**\n\n")
                        for section in legal_changes['sections_amended']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_added'):
                        f.write(f"\n- **Added:**\n\n")
                        for section in legal_changes['sections_added']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_deleted'):
                        f.write(f"\n- **Deleted:**\n\n")
                        for section in legal_changes['sections_deleted']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('chapters_repealed'):
                        f.write(f"\n- **Repealed:**\n\n")
                        for chapter in legal_changes['chapters_repealed']:
                            f.write(f"    - {chapter}\n")
                    if legal_changes.get('substance_of_changes'):
                        f.write(f"\n- **Substance:** {legal_changes['substance_of_changes']}\n")
                    f.write("\n")

                # Application Scope
                app_scope = analysis.get('application_scope', {})
                if app_scope and any(app_scope.values()):
                    f.write(f"**Application Scope:**\n\n")
                    if app_scope.get('applies_to'):
                        f.write(f"- **Applies To:**\n\n")
                        for entity in app_scope['applies_to']:
                            f.write(f"    - {entity}\n")
                    if app_scope.get('exclusions'):
                        f.write(f"\n- **Exclusions:**\n\n")
                        for exclusion in app_scope['exclusions']:
                            f.write(f"    - {exclusion}\n")
                    if app_scope.get('geographic_scope'):
                        f.write(f"\n- **Geographic:**\n\n")
                        for geo in app_scope['geographic_scope']:
                            f.write(f"    - {geo}\n")
                    f.write("\n")

                # Effective Dates
                effective_dates = analysis.get('effective_dates', [])
                if effective_dates:
                    f.write(f"**Effective Dates:**\n\n")
                    for ed in effective_dates:
                        f.write(f"- {ed.get('date', 'Unknown')}: {ed.get('applies_to', 'All provisions')}\n")
                    f.write("\n")

                # Mandatory vs Permissive
                mvp = analysis.get('mandatory_vs_permissive', {})
                if mvp and (mvp.get('mandatory_provisions') or mvp.get('permissive_provisions')):
                    f.write(f"**Provision Types:**\n\n")
                    if mvp.get('mandatory_provisions'):
                        f.write(f"- **Mandatory ({len(mvp['mandatory_provisions'])}):** {mvp['mandatory_provisions'][0] if mvp['mandatory_provisions'] else 'N/A'}\n")
                    if mvp.get('permissive_provisions'):
                        f.write(f"- **Permissive ({len(mvp['permissive_provisions'])}):** {mvp['permissive_provisions'][0] if mvp['permissive_provisions'] else 'N/A'}\n")
                    f.write("\n")

                # Exceptions and Exemptions
                exc_exempt = analysis.get('exceptions_and_exemptions', {})
                if exc_exempt and (exc_exempt.get('exceptions') or exc_exempt.get('exemptions')):
                    f.write(f"**Exceptions & Exemptions:**\n\n")
                    if exc_exempt.get('exceptions'):
                        f.write(f"- **Exceptions:**\n\n")
                        for exception in exc_exempt['exceptions']:
                            f.write(f"    - {exception}\n")
                    if exc_exempt.get('exemptions'):
                        f.write(f"\n- **Exemptions:**\n\n")
                        for exemption in exc_exempt['exemptions']:
                            f.write(f"    - {exemption}\n")
                    f.write("\n")

                # Affected NRG Assets (enhanced format)
                affected_assets = analysis.get('affected_nrg_assets', {})
                if affected_assets and any(affected_assets.values()):
                    f.write(f"**Affected NRG Assets:**\n\n")
                    if affected_assets.get('generation_facilities'):
                        f.write(f"- **Generation:**\n\n")
                        for facility in affected_assets['generation_facilities']:
                            f.write(f"    - {facility}\n")
                    if affected_assets.get('geographic_exposure'):
                        f.write(f"\n- **Markets:**\n\n")
                        for market in affected_assets['geographic_exposure']:
                            f.write(f"    - {market}\n")
                    if affected_assets.get('business_units'):
                        f.write(f"\n- **Business Units:**\n\n")
                        for unit in affected_assets['business_units']:
                            f.write(f"    - {unit}\n")
                    f.write("\n")
                # Legacy fallback for old format
                elif analysis.get('affected_assets'):
                    f.write(f"**Assets Affected:**\n")
                    for asset in analysis.get('affected_assets', []):
                        f.write(f"- {asset}\n")
                    f.write("\n")

                # NRG Relevant Excerpts
                excerpts = analysis.get('nrg_relevant_excerpts', [])
                if excerpts:
                    f.write(f"**Key Provisions Relevant to NRG:**\n\n")
                    for excerpt in excerpts[:3]:  # Limit to first 3
                        f.write(f"- {excerpt}\n")
                    f.write("\n")

                f.write(f"**Financial Estimate:** {analysis.get('financial_impact', 'Unknown')}\n\n")
                f.write(f"**Timeline:** {analysis.get('timeline', 'Unknown')}\n\n")

                f.write(f"**Recommended Actions:**\n\n")
                f.write(f"- ✅ **{analysis.get('recommended_action', 'N/A').upper()}** - Immediate attention required\n")
                f.write(f"- Track legislative progress closely\n")
                f.write(f"- Coordinate with stakeholders (see below)\n\n")

                # Internal Stakeholders (check both new and old field names)
                stakeholders = analysis.get('internal_stakeholders', analysis.get('stakeholders', []))
                if stakeholders:
                    f.write(f"**Internal Stakeholders:**\n\n")
                    for stakeholder in stakeholders:
                        f.write(f"- {stakeholder}\n")
                    f.write("\n")

                # Vertical Impact Details
                vertical_details = analysis.get('nrg_vertical_impact_details', {})
                if vertical_details:
                    f.write(f"**Impact by Business Vertical:**\n\n")
                    for vertical, impact_desc in vertical_details.items():
                        f.write(f"- **{vertical}:** {impact_desc}\n")
                    f.write("\n")

                f.write("---\n\n")

        # Medium Impact Section
        if medium_impact:
            f.write("## 🟡 MEDIUM IMPACT ITEMS (Score: 4-6)\n\n")
            f.write("*Monitor these items and prepare response strategies.*\n\n")

            for i, result in enumerate(medium_impact, 1):
                item = result["item"]
                analysis = result["analysis"]

                f.write(f"### {i}. {item['number']} - {item['title']}\n\n")

                # Impact Assessment as bullet list
                f.write(f"**Impact Assessment:**\n\n")
                f.write(f"- **Score:** {analysis.get('business_impact_score', 0)}/10\n")
                f.write(f"- **Type:** {analysis.get('impact_type', 'N/A').replace('_', ' ').title()}\n")
                f.write(f"- **Risk Level:** {analysis.get('risk_or_opportunity', 'N/A').upper()}\n\n")

                # Bill Information as bullet list
                f.write(f"**Bill Information:**\n\n")
                f.write(f"- **Source:** {item['source']}\n")
                f.write(f"- **Number:** {item['number']}\n")

                # Add sponsor/agency
                if 'sponsor' in item and item['sponsor'] != 'Unknown':
                    f.write(f"- **Sponsor:** {item['sponsor']}\n")
                elif 'agency' in item and item['agency'] != 'Unknown':
                    f.write(f"- **Agency:** {item['agency'].upper()}\n")

                # Add policy area for bills
                if 'policy_area' in item and item['policy_area'] != 'Unknown':
                    f.write(f"- **Policy Area:** {item['policy_area']}\n")

                f.write(f"- **Status:** {item['status']}\n")

                # Add dates
                if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
                    f.write(f"- **Introduced:** {item['introduced_date']}\n")
                if 'posted_date' in item and item['posted_date'] != 'Unknown':
                    f.write(f"- **Posted:** {item['posted_date']}\n")
                if 'comment_end_date' in item and item['comment_end_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Comment Deadline:** {item['comment_end_date']}\n")
                if 'effective_date' in item and item['effective_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Effective Date:** {item['effective_date']}\n")

                f.write(f"- **Bill Version:** {analysis.get('bill_version', 'unknown')}\n")
                f.write(f"- **Link:** {item['url']}\n\n")

                # Version Timeline (if available)
                version_analyses = result.get('version_analyses', [])
                version_diffs = result.get('version_diffs', [])

                if version_analyses and len(version_analyses) > 1:
                    f.write(f"**📚 VERSION TIMELINE ({len(version_analyses)} versions analyzed)**\n\n")
                    f.write(f"This bill has evolved through {len(version_analyses)} legislative versions. Each version was analyzed independently to track how NRG's risk profile changed:\n\n")

                    # Summary table of versions
                    for idx, va in enumerate(version_analyses, 1):
                        version = va['version']
                        v_analysis = va['analysis']
                        v_type = version.get('version_type', 'Unknown')
                        v_date = version.get('version_date', 'N/A')
                        impact_score = v_analysis.get('business_impact_score', 0)

                        f.write(f"{idx}. **{v_type}** ({v_date}) - Impact Score: {impact_score}/10\n")

                    f.write(f"\n")

                    # Detailed version-by-version analysis
                    f.write(f"**Detailed Version Analysis:**\n\n")

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

                        # Show version-to-version diff if available
                        if idx > 1 and len(version_diffs) >= idx - 1:
                            diff = version_diffs[idx - 2]
                            if diff.get('changed'):
                                # Show semantic analysis from LLM
                                semantic = diff.get('semantic_analysis', {})

                                f.write(f"- **Changes from {diff.get('from_version')}:**\n\n")

                                # Summary
                                if semantic.get('summary'):
                                    f.write(f"  *{semantic['summary']}*\n\n")

                                # Key provisions changed
                                if semantic.get('key_provisions_added'):
                                    f.write(f"  **Provisions Added:**\n")
                                    for prov in semantic['key_provisions_added'][:3]:  # Limit to top 3
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_removed'):
                                    f.write(f"  **Provisions Removed:**\n")
                                    for prov in semantic['key_provisions_removed'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_modified'):
                                    f.write(f"  **Provisions Modified:**\n")
                                    for prov in semantic['key_provisions_modified'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                # Impact evolution
                                if semantic.get('impact_evolution'):
                                    f.write(f"  **Impact Evolution:** {semantic['impact_evolution']}\n\n")

                                # Text statistics
                                f.write(f"  *Text changes: {diff.get('lines_added', 0)} lines added, {diff.get('lines_removed', 0)} removed, {diff.get('word_count_change', 0):+d} word delta*\n")
                            else:
                                f.write(f"- **No substantive changes from previous version**\n")

                        f.write(f"\n")

                    f.write(f"---\n\n")

                # Change Information (if available)
                change_data = result.get('change_data')
                change_impact = result.get('change_impact')

                if change_data and change_data.get('has_changes'):
                    if change_data.get('is_new'):
                        f.write(f"**📍 NEW BILL - First Time Analyzed**\n\n")
                    else:
                        f.write(f"**⚠️ CHANGES DETECTED**\n\n")
                        for change in change_data.get('changes', []):
                            f.write(f"- **{change['type'].replace('_', ' ').title()}:** {change.get('summary', 'Change detected')}\n")

                        if change_impact:
                            f.write(f"\n**Change Impact Analysis:**\n\n")
                            f.write(f"- **Change Impact Score:** {change_impact.get('change_impact_score', 'N/A')}/10\n")
                            f.write(f"- **Impact Trend:** {'INCREASED ⬆️' if change_impact.get('impact_increased') else 'DECREASED ⬇️'}\n")
                            f.write(f"- **Summary:** {change_impact.get('change_summary', 'N/A')}\n")
                            f.write(f"- **Recommended Action:** {change_impact.get('recommended_action', 'N/A').upper()}\n")

                        f.write("\n")

                # NRG Business Verticals
                verticals = analysis.get('nrg_business_verticals', [])
                if verticals:
                    f.write(f"**NRG Business Verticals:**\n\n")
                    for vertical in verticals:
                        f.write(f"- {vertical}\n")
                    f.write("\n")

                f.write(f"**Why This Matters to NRG:**\n")
                f.write(f"{analysis.get('impact_summary', 'No summary available')}\n\n")

                # Legal Code Changes
                legal_changes = analysis.get('legal_code_changes', {})
                if legal_changes and any(legal_changes.values()):
                    f.write(f"**Legal Code Changes:**\n\n")
                    if legal_changes.get('sections_amended'):
                        f.write(f"- **Amended:**\n\n")
                        for section in legal_changes['sections_amended']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_added'):
                        f.write(f"\n- **Added:**\n\n")
                        for section in legal_changes['sections_added']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_deleted'):
                        f.write(f"\n- **Deleted:**\n\n")
                        for section in legal_changes['sections_deleted']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('chapters_repealed'):
                        f.write(f"\n- **Repealed:**\n\n")
                        for chapter in legal_changes['chapters_repealed']:
                            f.write(f"    - {chapter}\n")
                    if legal_changes.get('substance_of_changes'):
                        f.write(f"\n- **Substance:** {legal_changes['substance_of_changes']}\n")
                    f.write("\n")

                # Application Scope
                app_scope = analysis.get('application_scope', {})
                if app_scope and any(app_scope.values()):
                    f.write(f"**Application Scope:**\n\n")
                    if app_scope.get('applies_to'):
                        f.write(f"- **Applies To:**\n\n")
                        for entity in app_scope['applies_to']:
                            f.write(f"    - {entity}\n")
                    if app_scope.get('exclusions'):
                        f.write(f"\n- **Exclusions:**\n\n")
                        for exclusion in app_scope['exclusions']:
                            f.write(f"    - {exclusion}\n")
                    if app_scope.get('geographic_scope'):
                        f.write(f"\n- **Geographic:**\n\n")
                        for geo in app_scope['geographic_scope']:
                            f.write(f"    - {geo}\n")
                    f.write("\n")

                # Effective Dates
                effective_dates = analysis.get('effective_dates', [])
                if effective_dates:
                    f.write(f"**Effective Dates:**\n\n")
                    for ed in effective_dates:
                        f.write(f"- {ed.get('date', 'Unknown')}: {ed.get('applies_to', 'All provisions')}\n")
                    f.write("\n")

                # Mandatory vs Permissive
                mvp = analysis.get('mandatory_vs_permissive', {})
                if mvp and (mvp.get('mandatory_provisions') or mvp.get('permissive_provisions')):
                    f.write(f"**Provision Types:**\n\n")
                    if mvp.get('mandatory_provisions'):
                        f.write(f"- **Mandatory ({len(mvp['mandatory_provisions'])}):** {mvp['mandatory_provisions'][0] if mvp['mandatory_provisions'] else 'N/A'}\n")
                    if mvp.get('permissive_provisions'):
                        f.write(f"- **Permissive ({len(mvp['permissive_provisions'])}):** {mvp['permissive_provisions'][0] if mvp['permissive_provisions'] else 'N/A'}\n")
                    f.write("\n")

                # Exceptions and Exemptions
                exc_exempt = analysis.get('exceptions_and_exemptions', {})
                if exc_exempt and (exc_exempt.get('exceptions') or exc_exempt.get('exemptions')):
                    f.write(f"**Exceptions & Exemptions:**\n\n")
                    if exc_exempt.get('exceptions'):
                        f.write(f"- **Exceptions:**\n\n")
                        for exception in exc_exempt['exceptions']:
                            f.write(f"    - {exception}\n")
                    if exc_exempt.get('exemptions'):
                        f.write(f"\n- **Exemptions:**\n\n")
                        for exemption in exc_exempt['exemptions']:
                            f.write(f"    - {exemption}\n")
                    f.write("\n")

                # Affected NRG Assets (enhanced format)
                affected_assets = analysis.get('affected_nrg_assets', {})
                if affected_assets and any(affected_assets.values()):
                    f.write(f"**Affected NRG Assets:**\n\n")
                    if affected_assets.get('generation_facilities'):
                        f.write(f"- **Generation:**\n\n")
                        for facility in affected_assets['generation_facilities']:
                            f.write(f"    - {facility}\n")
                    if affected_assets.get('geographic_exposure'):
                        f.write(f"\n- **Markets:**\n\n")
                        for market in affected_assets['geographic_exposure']:
                            f.write(f"    - {market}\n")
                    if affected_assets.get('business_units'):
                        f.write(f"\n- **Business Units:**\n\n")
                        for unit in affected_assets['business_units']:
                            f.write(f"    - {unit}\n")
                    f.write("\n")
                # Legacy fallback for old format
                elif analysis.get('affected_assets'):
                    f.write(f"**Assets Affected:**\n")
                    for asset in analysis.get('affected_assets', []):
                        f.write(f"- {asset}\n")
                    f.write("\n")

                # NRG Relevant Excerpts
                excerpts = analysis.get('nrg_relevant_excerpts', [])
                if excerpts:
                    f.write(f"**Key Provisions Relevant to NRG:**\n")
                    for excerpt in excerpts[:3]:  # Limit to first 3
                        f.write(f"- {excerpt}\n")
                    f.write("\n")

                f.write(f"**Financial Estimate:** {analysis.get('financial_impact', 'Unknown')}\n\n")
                f.write(f"**Timeline:** {analysis.get('timeline', 'Unknown')}\n\n")

                f.write(f"**Recommended Actions:**\n")
                f.write(f"- 🔔 **{analysis.get('recommended_action', 'N/A').upper()}** - Monitor and prepare response\n")
                f.write(f"- Track legislative progress\n")
                f.write(f"- Coordinate with stakeholders (see below)\n\n")

                # Internal Stakeholders (check both new and old field names)
                stakeholders = analysis.get('internal_stakeholders', analysis.get('stakeholders', []))
                if stakeholders:
                    f.write(f"**Internal Stakeholders:**\n")
                    for stakeholder in stakeholders:
                        f.write(f"- {stakeholder}\n")
                    f.write("\n")

                # Vertical Impact Details
                vertical_details = analysis.get('nrg_vertical_impact_details', {})
                if vertical_details:
                    f.write(f"**Impact by Business Vertical:**\n")
                    for vertical, impact_desc in vertical_details.items():
                        f.write(f"- **{vertical}:** {impact_desc}\n")
                    f.write("\n")

                f.write("---\n\n")

        # Low Impact Section
        if low_impact:
            f.write("## 🟢 LOW IMPACT ITEMS (Score: 0-3)\n\n")
            f.write("*For awareness only - no immediate action required.*\n\n")

            for i, result in enumerate(low_impact, 1):
                item = result["item"]
                analysis = result["analysis"]

                f.write(f"### {i}. {item['number']} - {item['title']}\n\n")

                # Impact Assessment as bullet list
                f.write(f"**Impact Assessment:**\n\n")
                f.write(f"- **Score:** {analysis.get('business_impact_score', 0)}/10\n")
                f.write(f"- **Type:** {analysis.get('impact_type', 'N/A').replace('_', ' ').title()}\n")
                f.write(f"- **Risk Level:** {analysis.get('risk_or_opportunity', 'N/A').upper()}\n\n")

                # Bill Information as bullet list
                f.write(f"**Bill Information:**\n\n")
                f.write(f"- **Source:** {item['source']}\n")
                f.write(f"- **Number:** {item['number']}\n")

                # Add sponsor/agency
                if 'sponsor' in item and item['sponsor'] != 'Unknown':
                    f.write(f"- **Sponsor:** {item['sponsor']}\n")
                elif 'agency' in item and item['agency'] != 'Unknown':
                    f.write(f"- **Agency:** {item['agency'].upper()}\n")

                # Add policy area for bills
                if 'policy_area' in item and item['policy_area'] != 'Unknown':
                    f.write(f"- **Policy Area:** {item['policy_area']}\n")

                f.write(f"- **Status:** {item['status']}\n")

                # Add dates
                if 'introduced_date' in item and item['introduced_date'] != 'Unknown':
                    f.write(f"- **Introduced:** {item['introduced_date']}\n")
                if 'posted_date' in item and item['posted_date'] != 'Unknown':
                    f.write(f"- **Posted:** {item['posted_date']}\n")
                if 'comment_end_date' in item and item['comment_end_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Comment Deadline:** {item['comment_end_date']}\n")
                if 'effective_date' in item and item['effective_date'] not in ['N/A', 'Unknown']:
                    f.write(f"- **Effective Date:** {item['effective_date']}\n")

                f.write(f"- **Bill Version:** {analysis.get('bill_version', 'unknown')}\n")
                f.write(f"- **Link:** {item['url']}\n\n")

                # Version Timeline (if available)
                version_analyses = result.get('version_analyses', [])
                version_diffs = result.get('version_diffs', [])

                if version_analyses and len(version_analyses) > 1:
                    f.write(f"**📚 VERSION TIMELINE ({len(version_analyses)} versions analyzed)**\n\n")
                    f.write(f"This bill has evolved through {len(version_analyses)} legislative versions. Each version was analyzed independently to track how NRG's risk profile changed:\n\n")

                    # Summary table of versions
                    for idx, va in enumerate(version_analyses, 1):
                        version = va['version']
                        v_analysis = va['analysis']
                        v_type = version.get('version_type', 'Unknown')
                        v_date = version.get('version_date', 'N/A')
                        impact_score = v_analysis.get('business_impact_score', 0)

                        f.write(f"{idx}. **{v_type}** ({v_date}) - Impact Score: {impact_score}/10\n")

                    f.write(f"\n")

                    # Detailed version-by-version analysis
                    f.write(f"**Detailed Version Analysis:**\n\n")

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

                        # Show version-to-version diff if available
                        if idx > 1 and len(version_diffs) >= idx - 1:
                            diff = version_diffs[idx - 2]
                            if diff.get('changed'):
                                # Show semantic analysis from LLM
                                semantic = diff.get('semantic_analysis', {})

                                f.write(f"- **Changes from {diff.get('from_version')}:**\n\n")

                                # Summary
                                if semantic.get('summary'):
                                    f.write(f"  *{semantic['summary']}*\n\n")

                                # Key provisions changed
                                if semantic.get('key_provisions_added'):
                                    f.write(f"  **Provisions Added:**\n")
                                    for prov in semantic['key_provisions_added'][:3]:  # Limit to top 3
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_removed'):
                                    f.write(f"  **Provisions Removed:**\n")
                                    for prov in semantic['key_provisions_removed'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                if semantic.get('key_provisions_modified'):
                                    f.write(f"  **Provisions Modified:**\n")
                                    for prov in semantic['key_provisions_modified'][:3]:
                                        f.write(f"  - {prov}\n")
                                    f.write(f"\n")

                                # Impact evolution
                                if semantic.get('impact_evolution'):
                                    f.write(f"  **Impact Evolution:** {semantic['impact_evolution']}\n\n")

                                # Text statistics
                                f.write(f"  *Text changes: {diff.get('lines_added', 0)} lines added, {diff.get('lines_removed', 0)} removed, {diff.get('word_count_change', 0):+d} word delta*\n")
                            else:
                                f.write(f"- **No substantive changes from previous version**\n")

                        f.write(f"\n")

                    f.write(f"---\n\n")

                # Change Information (if available)
                change_data = result.get('change_data')
                change_impact = result.get('change_impact')

                if change_data and change_data.get('has_changes'):
                    if change_data.get('is_new'):
                        f.write(f"**📍 NEW BILL - First Time Analyzed**\n\n")
                    else:
                        f.write(f"**⚠️ CHANGES DETECTED**\n\n")
                        for change in change_data.get('changes', []):
                            f.write(f"- **{change['type'].replace('_', ' ').title()}:** {change.get('summary', 'Change detected')}\n")

                        if change_impact:
                            f.write(f"\n**Change Impact Analysis:**\n\n")
                            f.write(f"- **Change Impact Score:** {change_impact.get('change_impact_score', 'N/A')}/10\n")
                            f.write(f"- **Impact Trend:** {'INCREASED ⬆️' if change_impact.get('impact_increased') else 'DECREASED ⬇️'}\n")
                            f.write(f"- **Summary:** {change_impact.get('change_summary', 'N/A')}\n")
                            f.write(f"- **Recommended Action:** {change_impact.get('recommended_action', 'N/A').upper()}\n")

                        f.write("\n")

                # NRG Business Verticals
                verticals = analysis.get('nrg_business_verticals', [])
                if verticals:
                    f.write(f"**NRG Business Verticals:**\n\n")
                    for vertical in verticals:
                        f.write(f"- {vertical}\n")
                    f.write("\n")

                f.write(f"**Why This Matters to NRG:**\n")
                f.write(f"{analysis.get('impact_summary', 'No summary available')}\n\n")

                # Legal Code Changes
                legal_changes = analysis.get('legal_code_changes', {})
                if legal_changes and any(legal_changes.values()):
                    f.write(f"**Legal Code Changes:**\n\n")
                    if legal_changes.get('sections_amended'):
                        f.write(f"- **Amended:**\n\n")
                        for section in legal_changes['sections_amended']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_added'):
                        f.write(f"\n- **Added:**\n\n")
                        for section in legal_changes['sections_added']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('sections_deleted'):
                        f.write(f"\n- **Deleted:**\n\n")
                        for section in legal_changes['sections_deleted']:
                            f.write(f"    - {section}\n")
                    if legal_changes.get('chapters_repealed'):
                        f.write(f"\n- **Repealed:**\n\n")
                        for chapter in legal_changes['chapters_repealed']:
                            f.write(f"    - {chapter}\n")
                    if legal_changes.get('substance_of_changes'):
                        f.write(f"\n- **Substance:** {legal_changes['substance_of_changes']}\n")
                    f.write("\n")

                # Application Scope
                app_scope = analysis.get('application_scope', {})
                if app_scope and any(app_scope.values()):
                    f.write(f"**Application Scope:**\n\n")
                    if app_scope.get('applies_to'):
                        f.write(f"- **Applies To:**\n\n")
                        for entity in app_scope['applies_to']:
                            f.write(f"    - {entity}\n")
                    if app_scope.get('exclusions'):
                        f.write(f"\n- **Exclusions:**\n\n")
                        for exclusion in app_scope['exclusions']:
                            f.write(f"    - {exclusion}\n")
                    if app_scope.get('geographic_scope'):
                        f.write(f"\n- **Geographic:**\n\n")
                        for geo in app_scope['geographic_scope']:
                            f.write(f"    - {geo}\n")
                    f.write("\n")

                # Effective Dates
                effective_dates = analysis.get('effective_dates', [])
                if effective_dates:
                    f.write(f"**Effective Dates:**\n\n")
                    for ed in effective_dates:
                        f.write(f"- {ed.get('date', 'Unknown')}: {ed.get('applies_to', 'All provisions')}\n")
                    f.write("\n")

                # Mandatory vs Permissive
                mvp = analysis.get('mandatory_vs_permissive', {})
                if mvp and (mvp.get('mandatory_provisions') or mvp.get('permissive_provisions')):
                    f.write(f"**Provision Types:**\n\n")
                    if mvp.get('mandatory_provisions'):
                        f.write(f"- **Mandatory ({len(mvp['mandatory_provisions'])}):** {mvp['mandatory_provisions'][0] if mvp['mandatory_provisions'] else 'N/A'}\n")
                    if mvp.get('permissive_provisions'):
                        f.write(f"- **Permissive ({len(mvp['permissive_provisions'])}):** {mvp['permissive_provisions'][0] if mvp['permissive_provisions'] else 'N/A'}\n")
                    f.write("\n")

                # Exceptions and Exemptions
                exc_exempt = analysis.get('exceptions_and_exemptions', {})
                if exc_exempt and (exc_exempt.get('exceptions') or exc_exempt.get('exemptions')):
                    f.write(f"**Exceptions & Exemptions:**\n\n")
                    if exc_exempt.get('exceptions'):
                        f.write(f"- **Exceptions:**\n\n")
                        for exception in exc_exempt['exceptions']:
                            f.write(f"    - {exception}\n")
                    if exc_exempt.get('exemptions'):
                        f.write(f"\n- **Exemptions:**\n\n")
                        for exemption in exc_exempt['exemptions']:
                            f.write(f"    - {exemption}\n")
                    f.write("\n")

                # Affected NRG Assets (enhanced format)
                affected_assets = analysis.get('affected_nrg_assets', {})
                if affected_assets and any(affected_assets.values()):
                    f.write(f"**Affected NRG Assets:**\n\n")
                    if affected_assets.get('generation_facilities'):
                        f.write(f"- **Generation:**\n\n")
                        for facility in affected_assets['generation_facilities']:
                            f.write(f"    - {facility}\n")
                    if affected_assets.get('geographic_exposure'):
                        f.write(f"\n- **Markets:**\n\n")
                        for market in affected_assets['geographic_exposure']:
                            f.write(f"    - {market}\n")
                    if affected_assets.get('business_units'):
                        f.write(f"\n- **Business Units:**\n\n")
                        for unit in affected_assets['business_units']:
                            f.write(f"    - {unit}\n")
                    f.write("\n")
                # Legacy fallback for old format
                elif analysis.get('affected_assets'):
                    f.write(f"**Assets Affected:**\n")
                    for asset in analysis.get('affected_assets', []):
                        f.write(f"- {asset}\n")
                    f.write("\n")

                # NRG Relevant Excerpts
                excerpts = analysis.get('nrg_relevant_excerpts', [])
                if excerpts:
                    f.write(f"**Key Provisions Relevant to NRG:**\n")
                    for excerpt in excerpts[:3]:  # Limit to first 3
                        f.write(f"- {excerpt}\n")
                    f.write("\n")

                f.write(f"**Financial Estimate:** {analysis.get('financial_impact', 'Unknown')}\n\n")
                f.write(f"**Timeline:** {analysis.get('timeline', 'Unknown')}\n\n")

                f.write(f"**Recommended Actions:**\n")
                f.write(f"- ℹ️ **{analysis.get('recommended_action', 'N/A').upper()}** - For awareness only\n")
                f.write(f"- No immediate action required\n\n")

                # Internal Stakeholders (check both new and old field names)
                stakeholders = analysis.get('internal_stakeholders', analysis.get('stakeholders', []))
                if stakeholders:
                    f.write(f"**Internal Stakeholders:**\n")
                    for stakeholder in stakeholders:
                        f.write(f"- {stakeholder}\n")
                    f.write("\n")

                # Vertical Impact Details
                vertical_details = analysis.get('nrg_vertical_impact_details', {})
                if vertical_details:
                    f.write(f"**Impact by Business Vertical:**\n")
                    for vertical, impact_desc in vertical_details.items():
                        f.write(f"- **{vertical}:** {impact_desc}\n")
                    f.write("\n")

                f.write("---\n\n")

        # Footer
        f.write("---\n\n")
        f.write("## Next Steps\n\n")

        if high_impact:
            f.write("### Immediate (This Week):\n")
            f.write("- Government Affairs: Track high-impact bills committee assignments\n")
            f.write("- Schedule cross-functional meeting to assess regulatory risks\n")
            f.write("- Prepare initial cost impact assessments\n\n")

        f.write("### Short-term (This Month):\n")
        f.write("- Environmental Compliance: Assess compliance scenarios\n")
        f.write("- Finance: Model financial impact ranges\n")
        f.write("- Legal: Review regulatory exposure\n\n")

        f.write("### Ongoing:\n")
        f.write("- Continue daily monitoring via this tracker\n")
        f.write("- Update analysis as bills progress through legislative process\n")
        f.write("- Brief executive leadership monthly\n\n")

        f.write("---\n\n")
        f.write("**Report Details:**\n")
        f.write("- **APIs Used:** Congress.gov, Regulations.gov, LegiScan\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("*This report is generated automatically. For questions or to update tracking parameters, contact your team.*\n")

    return md_file


# -----------------------------------------------------------------------------
# WORD DOCUMENT GENERATION (via Pandoc)
# -----------------------------------------------------------------------------

def convert_markdown_to_word(md_file):
    """
    Convert Markdown report to Word document using Pandoc.

    Provides a Microsoft Word (.docx) version of the analysis report for
    stakeholders who prefer document formats over Markdown.

    Prerequisites:
        - Pandoc must be installed:
          macOS: brew install pandoc
          Linux: apt-get install pandoc
          Windows: Download from pandoc.org

    Args:
        md_file (str): Path to source Markdown file

    Returns:
        str | None: Path to generated .docx file, or None if:
            - Pandoc is not installed
            - Conversion failed

    Output:
        Creates file: nrg_analysis_{timestamp}.docx (same name as .md)

    Configuration:
        Enabled/disabled via config.yaml:
            output:
              generate_docx: true
    """
    try:
        # Generate Word filename
        docx_file = md_file.replace('.md', '.docx')

        # Check if pandoc is available
        result = subprocess.run(
            ['which', 'pandoc'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            console.print("[yellow]⚠ Pandoc not found - skipping Word document generation[/yellow]")
            console.print("[dim]  Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)[/dim]")
            return None

        # Convert markdown to Word
        result = subprocess.run(
            ['pandoc', md_file, '-o', docx_file],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return docx_file
        else:
            console.print(f"[yellow]⚠ Pandoc conversion failed: {result.stderr}[/yellow]")
            return None

    except Exception as e:
        console.print(f"[yellow]⚠ Error converting to Word: {e}[/yellow]")
        return None


# =============================================================================
# SECTION 5: CHANGE TRACKING AND BILL CACHING
# =============================================================================
# This section provides persistent storage for bill data and change detection.
# Uses SQLite to track bills over time and identify when text, status, or
# amendments change between monitoring runs.
#
# Database Schema (bill_cache.db):
#   bills           - Core bill metadata and text hash
#   bill_changes    - Detected changes with diff summaries
#   amendments      - Amendment tracking per bill
#   bill_versions   - All versions of each bill (Introduced → Enrolled)
#   version_analyses - LLM analysis results per version
#
# Key Functions:
#   - init_database(): Create/migrate database schema
#   - compute_bill_hash(): SHA-256 hash for change detection
#   - get_cached_bill() / save_bill_to_cache(): Bill CRUD operations
#   - save_bill_version(): Store version with text and metadata
#   - compute_text_diff(): Generate unified diff between versions
#
# Configuration (config.yaml):
#   change_tracking:
#     enabled: true
#     database: "bill_cache.db"
#     track_text_changes: true
#     track_amendments: true
#     track_status_changes: true
#     analyze_changes_with_llm: true
# =============================================================================

def init_database(db_path):
    """
    Initialize SQLite database for bill caching and change tracking.

    Creates the database file and all required tables if they don't exist.
    Safe to call on existing database (uses CREATE TABLE IF NOT EXISTS).

    Args:
        db_path (str): Path to SQLite database file (e.g., "bill_cache.db")

    Returns:
        sqlite3.Connection: Active database connection for subsequent operations

    Tables Created:
        bills: Core bill metadata with text hash for change detection
        bill_changes: Log of detected changes with diffs
        amendments: Amendment tracking per bill
        bill_versions: All versions of each bill with full text
        version_analyses: LLM analysis results per version

    Indexes:
        - idx_versions_bill: Speed up version lookups by bill_id
        - idx_versions_type: Speed up version type queries
        - idx_analyses_version: Speed up analysis lookups by version
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create bills table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bills (
            bill_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            bill_number TEXT,
            title TEXT,
            text_hash TEXT,
            status TEXT,
            full_data_json TEXT,
            last_checked TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create bill_changes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bill_changes (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id TEXT,
            change_type TEXT,
            change_detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            old_value TEXT,
            new_value TEXT,
            diff_summary TEXT,
            impact_analysis TEXT,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    # Create amendments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS amendments (
            amendment_id TEXT PRIMARY KEY,
            bill_id TEXT,
            amendment_number TEXT,
            amendment_text TEXT,
            introduced_date TEXT,
            status TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    # Create bill_versions table for version tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bill_versions (
            version_id TEXT PRIMARY KEY,
            bill_id TEXT NOT NULL,
            version_type TEXT,
            version_date TEXT,
            version_number INTEGER,
            pdf_url TEXT,
            full_text TEXT,
            text_hash TEXT,
            word_count INTEGER,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    # Create version_analyses table for storing LLM analysis of each version
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS version_analyses (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL,
            impact_score INTEGER,
            impact_type TEXT,
            impact_summary TEXT,
            affected_assets TEXT,
            recommended_action TEXT,
            stakeholders TEXT,
            full_analysis_json TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (version_id) REFERENCES bill_versions(version_id)
        )
    ''')

    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_bill ON bill_versions(bill_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_type ON bill_versions(version_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_version ON version_analyses(version_id)')

    conn.commit()
    return conn


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS - Hashing, PDF extraction, Text diffing
# -----------------------------------------------------------------------------

def compute_bill_hash(bill_text):
    """
    Compute SHA-256 hash of bill text for change detection.

    Used to quickly identify when bill text has changed between runs
    without storing full text in memory for comparison.

    Args:
        bill_text (str): Full text of bill to hash

    Returns:
        str: 64-character hexadecimal SHA-256 hash, or empty string if no text
    """
    if not bill_text:
        return ""
    return hashlib.sha256(bill_text.encode('utf-8')).hexdigest()


def extract_pdf_text(pdf_url):
    """
    Extract text from legislative PDF using pdfplumber

    Parameters:
    - pdf_url: URL to PDF file (e.g., Texas Legislature website)

    Returns:
    - Extracted text as string, or None if extraction fails
    """
    try:
        import pdfplumber
        import io

        console.print(f"[dim]    Downloading PDF from {pdf_url[:60]}...[/dim]")

        # Download PDF with timeout
        with httpx.Client(timeout=30.0) as http:
            response = http.get(pdf_url, follow_redirects=True)
            response.raise_for_status()

            # Load PDF into memory
            pdf_bytes = io.BytesIO(response.content)

        # Extract text from all pages
        text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            console.print(f"[dim]    Extracting text from {len(pdf.pages)} pages...[/dim]")
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            console.print("[yellow]    ⚠ PDF appears empty or text extraction failed[/yellow]")
            return None

        console.print(f"[green]    ✓ Extracted {len(text)} characters ({len(text.split())} words)[/green]")
        return text

    except httpx.HTTPStatusError as e:
        console.print(f"[red]    HTTP Error downloading PDF: {e.response.status_code}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]    Error extracting PDF text: {e}[/red]")
        return None


def fetch_bill_versions_from_openstates(openstates_id, bill_number):
    """
    Fetch all versions of a bill from Open States API

    Parameters:
    - openstates_id: Open States bill ID (e.g., "ocd-bill/...")
    - bill_number: Bill number for display (e.g., "HB 4238")

    Returns:
        list[dict]: List of bill version dictionaries, each containing:
            - version_id: Open States version identifier
            - name: Version name (e.g., "Introduced", "Enrolled")
            - date: Version release date
            - url: Direct URL to PDF document
            - text: Extracted full text from PDF (empty if extraction failed)
            - word_count: Number of words in extracted text
            - hash: SHA-256 hash of text for change detection
    - List of version dictionaries with extracted text
    """
    if not os.getenv("OPENSTATES_API_KEY"):
        return []

    base_url = "https://v3.openstates.org"
    headers = {
        "X-API-KEY": os.getenv("OPENSTATES_API_KEY"),
        "User-Agent": "NRG-Energy-Legislative-Tracker/1.0"
    }

    try:
        console.print(f"[dim]  Fetching versions for {bill_number}...[/dim]")

        # Fetch bill with versions included
        with httpx.Client(timeout=30.0) as http:
            bill_url = f"{base_url}/bills/{openstates_id}"
            params = {"include": "versions"}  # Request versions inline

            response = http.get(bill_url, headers=headers, params=params)
            response.raise_for_status()
            bill_data = response.json()
        versions_raw = bill_data.get("versions", [])

        if not versions_raw:
            console.print(f"[yellow]  ⚠ No versions found for {bill_number}[/yellow]")
            return []
        console.print(f"[green]  ✓ Found {len(versions_raw)} versions[/green]")
        
        # DEBUG: Log API version order if enabled
        if config.get('debug', {}).get('enabled', False) and config.get('debug', {}).get('api_order', True):
            console.print(f"[dim]  DEBUG: API version order:[/dim]")
            for i, v in enumerate(versions_raw):
                console.print(f"[dim]    [{i}] {v.get('note', 'Unknown')} ({v.get('date', 'N/A')})[/dim]")
        
        # Process each version
        versions = []
        for idx, version_data in enumerate(versions_raw, 1):
            version_type = version_data.get("note", "Unknown")
            version_date = version_data.get("date", "")
            links = version_data.get("links", [])

            # Find PDF link
            pdf_url = None
            for link in links:
                if link.get("media_type") == "application/pdf":
                    pdf_url = link.get("url")
                    break

            if not pdf_url:
                console.print(f"[yellow]    Version {idx} ({version_type}): No PDF found, skipping[/yellow]")
                continue

            console.print(f"[cyan]  Version {idx}/{len(versions_raw)}: {version_type}[/cyan]")

            # Extract PDF text
            full_text = extract_pdf_text(pdf_url)

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

        console.print(f"[green]  ✓ Extracted text from {len(versions)}/{len(versions_raw)} versions[/green]")
        return versions

    except httpx.HTTPStatusError as e:
        console.print(f"[red]  HTTP Error fetching versions: {e.response.status_code}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]  Error fetching versions: {e}[/red]")
        return []


def get_cached_bill(bill_id, conn):
    """Retrieve bill from cache"""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM bills WHERE bill_id = ?', (bill_id,))
    row = cursor.fetchone()

    if row:
        return {
            "bill_id": row[0],
            "source": row[1],
            "bill_number": row[2],
            "title": row[3],
            "text_hash": row[4],
            "status": row[5],
            "full_data_json": row[6],
            "last_checked": row[7],
            "created_at": row[8]
        }
    return None


def save_bill_to_cache(bill, conn):
    """Save or update bill in cache
    
    Args:
        bill (dict): Bill object with schema defined in fetch_openstates_bills:989-1003 
                    or fetch_congress_bills:289-304
        conn: Database connection
    """
    bill_id = f"{bill['source']}:{bill['number']}"
    bill_text = bill.get('summary', '')
    text_hash = compute_bill_hash(bill_text)

    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO bills
        (bill_id, source, bill_number, title, text_hash, status, full_data_json, last_checked)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        bill_id,
        bill['source'],
        bill['number'],
        bill['title'],
        text_hash,
        bill.get('status', 'Unknown'),
        json.dumps(bill),
        datetime.now().isoformat()
    ))

    # Save amendments if present
    if 'amendments' in bill and bill['amendments']:
        for amendment in bill['amendments']:
            amendment_id = f"{bill_id}:amendment:{amendment.get('amendment_id', len(bill['amendments']))}"
            cursor.execute('''
                INSERT OR IGNORE INTO amendments
                (amendment_id, bill_id, amendment_number, amendment_text, introduced_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                amendment_id,
                bill_id,
                amendment.get('amendment_number', 'Unknown'),
                json.dumps(amendment),
                amendment.get('date', 'Unknown'),
                amendment.get('status', 'Unknown')
            ))

    conn.commit()


def save_bill_version(bill_id, version, conn):
    """Save a bill version to the database"""
    version_id = f"{bill_id}:v{version['version_number']}:{version['version_type'].replace(' ', '_')}"

    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO bill_versions
        (version_id, bill_id, version_type, version_date, version_number, pdf_url, full_text, text_hash, word_count, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        version_id,
        bill_id,
        version['version_type'],
        version.get('version_date', ''),
        version['version_number'],
        version.get('pdf_url', ''),
        version.get('full_text', ''),
        version['text_hash'],
        version.get('word_count', 0),
        datetime.now().isoformat()
    ))

    conn.commit()
    return version_id


def save_version_analysis(version_id, analysis, conn):
    """Save LLM analysis of a bill version"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO version_analyses
        (version_id, impact_score, impact_type, impact_summary, affected_assets, recommended_action, stakeholders, full_analysis_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        version_id,
        analysis.get('business_impact_score', 0),
        analysis.get('impact_type', 'unknown'),
        analysis.get('impact_summary', ''),
        json.dumps(analysis.get('affected_nrg_assets', {})),
        analysis.get('recommended_action', 'monitor'),
        json.dumps(analysis.get('internal_stakeholders', [])),
        json.dumps(analysis),
        datetime.now().isoformat()
    ))

    conn.commit()


def get_bill_versions(bill_id, conn):
    """Retrieve all versions for a bill from database"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT version_id, version_type, version_date, version_number, pdf_url, full_text, text_hash, word_count
        FROM bill_versions
        WHERE bill_id = ?
        ORDER BY version_number
    ''', (bill_id,))

    rows = cursor.fetchall()
    versions = []
    for row in rows:
        versions.append({
            "version_id": row[0],
            "version_type": row[1],
            "version_date": row[2],
            "version_number": row[3],
            "pdf_url": row[4],
            "full_text": row[5],
            "text_hash": row[6],
            "word_count": row[7]
        })

    return versions


def get_version_analysis(version_id, conn):
    """Retrieve LLM analysis for a specific version"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT full_analysis_json
        FROM version_analyses
        WHERE version_id = ?
    ''', (version_id,))

    row = cursor.fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return None


def compute_text_diff(old_text, new_text):
    """Generate unified diff between two texts"""
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


def detect_bill_changes(cached_bill, current_bill):
    """
    Compare cached and current bill data to detect changes in:
    1. hashes of bill text
    2. status
    3. new amendments (count)
    Then generate unified diff for text changes.

    Args:
        cached_bill (dict): Previously cached bill data with keys:
            - text_hash: SHA-256 hash of bill summary text
            - status: Current bill status
            - title: Bill title
            - full_data_json: Complete bill nested JSON data including summary, amendments, etc.
                          (See schema definition at fetch_openstates_bills:989-1003 or fetch_congress_bills:289-304)
        current_bill (dict): Current bill data with keys:
            - summary: Bill summary text content
            - status: Current bill status
            - title: Bill title
            - amendments: List of bill amendments (if any)
            - Other bill metadata fields

    Returns:
        dict: Change detection results containing:
            - has_changes (bool): True if any changes detected, False otherwise
            - is_new (bool): True if this is a new bill (no cached version exists)
            - change_type (str): Type of change: "new_bill", "modified", or "unchanged"
            - changes (list): Detailed list of changes found, each containing:
                - type (str): Change type ("text_change", "status_change", "new_amendments")
                - summary (str): Human-readable description of the change
                - For text_change: diff (str) - unified diff of text changes
                - For status_change: old_value (str), new_value (str) - status values
                - For new_amendments: count (int), amendments (list) - new amendment details
    """
    if not cached_bill:
        return {
            "has_changes": False,
            "is_new": True,
            "change_type": "new_bill",
            "changes": []
        }

    changes = []

    # Parse cached data
    cached_data = json.loads(cached_bill['full_data_json'])
    current_text = current_bill.get('summary', '')
    cached_text = cached_data.get('summary', '')

    # Check text changes
    current_hash = compute_bill_hash(current_text)
    if current_hash != cached_bill['text_hash']:
        diff = compute_text_diff(cached_text, current_text)
        changes.append({
            "type": "text_change",
            "diff": diff,
            "summary": "Bill text has been modified"
        })

    # Check status changes
    if current_bill.get('status') != cached_bill['status']:
        changes.append({
            "type": "status_change",
            "old_value": cached_bill['status'],
            "new_value": current_bill.get('status'),
            "summary": f"Status changed from '{cached_bill['status']}' to '{current_bill.get('status')}'"
        })

    # Check for new amendments
    current_amendments = current_bill.get('amendments', [])
    cached_amendments = cached_data.get('amendments', [])

    if len(current_amendments) > len(cached_amendments):
        new_amendments = current_amendments[len(cached_amendments):]
        changes.append({
            "type": "new_amendments",
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


def analyze_changes_with_llm(bill, change_data, nrg_context):
    """Use LLM to analyze impact of bill changes"""
    if not change_data.get("has_changes"):
        return None

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
        # Use configured LLM provider
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
            # Use OpenAI - simpler format without reasoning
            response = openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for change analysis (faster than GPT-5)
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

    except Exception as e:
        console.print(f"[yellow]⚠ Error analyzing changes with LLM: {e}[/yellow]")
        return {
            "change_impact_score": 5,
            "change_summary": "Changes detected but analysis failed",
            "recommended_action": "review"
        }


# =============================================================================
# SECTION 6: MAIN EXECUTION FLOW
# =============================================================================
# This is the primary entry point that orchestrates the entire analysis pipeline.
#
# Execution Flow:
#   1. Load configuration from config.yaml
#   2. Initialize SQLite database (if change tracking enabled)
#   3. Load NRG business context from nrg_business_context.txt
#   4. Fetch bills from configured sources:
#      - Congress.gov (federal bills)
#      - Regulations.gov (federal regulations)
#      - Open States (Texas state bills - PRIMARY)
#      - LegiScan (backup for state bills)
#   5. Fetch all versions for each bill (if version tracking enabled)
#   6. Detect changes from previous run (if change tracking enabled)
#   7. Analyze each item (or each version) with configured LLM
#   8. Generate version comparison diffs with semantic analysis
#   9. Display results in console with color-coded panels
#  10. Save reports: JSON, Markdown, Word (optional)
#  11. Update cache database with new bill data
#
# Configuration File: config.yaml
# Context File: nrg_business_context.txt
# Output Files: nrg_analysis_YYYYMMDD_HHMMSS.{json,md,docx}
# =============================================================================

def main():
    """
    Main POC script - orchestrates the full analysis pipeline.

    This function is the entry point for the NRG Legislative Intelligence POC.
    It coordinates all phases of the analysis workflow:

    Phase 1 - Configuration:
        - Reads config.yaml for API sources, LLM provider, and feature flags
        - Initializes SQLite database for change tracking (if enabled)
        - Loads NRG business context for LLM analysis

    Phase 2 - Data Collection:
        - Fetches bills from Congress.gov (federal)
        - Fetches regulations from Regulations.gov (federal)
        - Fetches state bills from Open States (Texas)
        - Falls back to LegiScan if Open States fails

    Phase 3 - Version Tracking (if enabled):
        - Fetches all versions of each bill (Introduced → Enrolled)
        - Extracts full text from PDFs for each version

    Phase 4 - Change Detection (if enabled):
        - Compares current bill data to cached versions
        - Identifies text changes, status changes, new amendments

    Phase 5 - LLM Analysis:
        - Sends each item (or each version) to configured LLM
        - Receives structured JSON analysis with impact scores
        - For multi-version bills, analyzes each version independently

    Phase 6 - Version Comparison (if multi-version):
        - Generates diffs between consecutive versions
        - Uses LLM for semantic change analysis
        - Tracks impact score evolution through legislative process

    Phase 7 - Output Generation:
        - Displays color-coded panels in console (Rich library)
        - Saves JSON report for programmatic consumption
        - Saves Markdown report for human reading
        - Converts to Word document (if pandoc installed)

    Phase 8 - Cache Update:
        - Saves current bill data to SQLite for next run's change detection

    Command Line:
        uv run poc.py

    Exit Codes:
        Returns normally on success, prints error messages to console on failure.
    """
    console.print("\n[bold magenta]═══════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  NRG Energy Bill Tracker - POC  [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/bold magenta]\n")

    # Show configuration summary
    llm_provider = config['llm']['provider']
    llm_model = config['llm'][llm_provider]['model']
    change_tracking_enabled = config.get('change_tracking', {}).get('enabled', False)

    # Check Open States configuration
    openstates_enabled = config.get('sources', {}).get('openstates', {}).get('enabled', False)
    texas_config = config.get('sources', {}).get('openstates', {}).get('texas_bills', {})
    texas_bills_enabled = texas_config.get('enabled', False) if openstates_enabled else False
    texas_bill_numbers = texas_config.get('bills', [])

    console.print(f"[bold cyan]🔧 Configuration:[/bold cyan]")
    console.print(f"  LLM Provider: {llm_provider.upper()} ({llm_model})")
    console.print(f"  Change Tracking: {'ENABLED' if change_tracking_enabled else 'DISABLED'}")
    console.print(f"  Open States API: {'ENABLED' if openstates_enabled else 'DISABLED'}")
    console.print(f"  Texas Bills: {'ENABLED' if texas_bills_enabled else 'DISABLED'} ({len(texas_bill_numbers)} bills)")
    console.print()

    # Initialize database for change tracking
    db_conn = None
    if change_tracking_enabled:
        console.print("[cyan]Initializing change tracking database...[/cyan]")
        
        # Use Azure Functions TEMP directory for writable storage
        # 
        # Azure Functions Linux Consumption plan has read-only filesystem
        # The deployment directory (/home/site/wwwroot) is read-only, causing
        # SQLite "unable to open database file" errors when trying to create/write DB files
        # 
        # Only the TEMP directory is writable on Azure Functions Linux
        # TEMP environment variable points to /tmp which is fully writable
        # 
        # - https://learn.microsoft.com/en-us/answers/questions/1002268/azure-functions-directory-read-only-on-linux
        # - https://stackoverflow.com/questions/63961284/using-the-temp-directory-for-azure-functions
        # - https://github.com/Azure/azure-functions-host/issues/3626
        # 
        # LIMITATIONS:
        # - TEMP directory is ephemeral (resets on cold starts)
        # - Not shared across function instances
        # - Storage limited to ~500MB on Consumption plan
        # 
        # For MVP: Acceptable since change tracking only needed during execution
        # For production: Will migrate to client's data lake as per business requirements
        
        temp_dir = os.environ.get('TEMP', tempfile.gettempdir())
        db_filename = config.get('change_tracking', {}).get('database', 'bill_cache.db')
        db_path = os.path.join(temp_dir, db_filename)
        
        db_conn = init_database(db_path)
        console.print(f"[green]✓ Database ready: {db_path}[/green]")

    # Load NRG business context
    console.print("[cyan]Loading NRG business context...[/cyan]")
    nrg_context = load_nrg_context()
    console.print("[green]✓ Context loaded[/green]")

    ###########################################
    ### Fetch from all APIs based on config ###
    ###########################################
    all_items = []

    if config.get('sources', {}).get('congress', {}).get('enabled', True):
        limit = config['sources']['congress'].get('limit', 3)
        congress_bills = fetch_congress_bills(limit=limit)
        all_items.extend(congress_bills)

    if config.get('sources', {}).get('regulations', {}).get('enabled', True):
        limit = config['sources']['regulations'].get('limit', 3)
        regulations = fetch_regulations(limit=limit)
        all_items.extend(regulations)

    if config.get('sources', {}).get('legiscan_federal', {}).get('enabled', True):
        limit = config['sources']['legiscan_federal'].get('limit', 3)
        legiscan_bills = fetch_legiscan_bills(limit=limit)
        all_items.extend(legiscan_bills)
    
    # Fetch Texas bills via Open States if enabled
    # Note: only searching for specific configured bills due to API tier (Free) limitation
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

                # Fallback to LegiScan if Open States fails
                if not texas_bills and config.get('sources', {}).get('legiscan_federal', {}).get('enabled', False):
                    console.print("[yellow]⚠ Open States returned no results, trying LegiScan fallback...[/yellow]")
                    state = config['sources'].get('texas_bills', {}).get('state', 'TX')
                    fallback_bills = fetch_specific_texas_bills(bill_numbers, state)
                    all_items.extend(fallback_bills)

    console.print(f"\n[bold green]Total items collected: {len(all_items)}[/bold green]\n")

    if not all_items:
        console.print("[red]No items found. Please check API keys and try again.[/red]")
        if db_conn:
            db_conn.close()
        return

    ###########################################################
    # Fetch versions for bills if version tracking is enabled #
    ###########################################################
    # Performs two phase bill tracking:
    # 1. Version Collection: Fetch all historical versions of each bill from APIs
    #    - Open States: fetch_bill_versions_from_openstates() for state bills
    #    - Congress.gov: fetch_bill_versions_from_congress() for federal bills
    #    - Each version includes full text, metadata, and computed hash
    #
    # 2. Change Detection & Analysis (see section below):
    #    - Compare hash of current vs cached bill summary for quick change detection
    #    - If hash differs, proceed to LLM analysis of all versions
    #    - Sliding window analysis compares consecutive versions (v1 -> v2, v2 -> v3, etc.)
    #    - Generates semantic change analysis using both bill text and prior LLM analyses
    
    version_tracking_enabled = config.get('version_tracking', {}).get('enabled', False)
    version_tracking_openstates = config.get('version_tracking', {}).get('sources', {}).get('openstates', True)
    version_tracking_congress = config.get('version_tracking', {}).get('sources', {}).get('congress', False)

    if version_tracking_enabled and (version_tracking_openstates or version_tracking_congress):
        console.print("\n[bold cyan]📚 Fetching bill versions...[/bold cyan]\n")

        for item in all_items:
            # Fetch versions for Open States bills
            if version_tracking_openstates and item['source'] == 'Open States' and 'openstates_id' in item:
                bill_number = item['number']
                openstates_id = item['openstates_id']

                console.print(f"[cyan]Processing {bill_number}...[/cyan]")

                # Fetch all versions with full text for a specific Bill
                versions = fetch_bill_versions_from_openstates(openstates_id, bill_number)

                if versions:
                    # Attach versions to the item
                    item['versions'] = versions
                    console.print(f"[green]✓ Loaded {len(versions)} versions for {bill_number}[/green]\n")
                else:
                    console.print(f"[yellow]⚠ No versions found for {bill_number}[/yellow]\n")
                    item['versions'] = []

            # Fetch versions for Congress.gov bills
            elif version_tracking_congress and item['source'] == 'Congress.gov':
                # Verify required metadata exists
                if 'congress_num' in item and 'bill_type' in item and 'bill_number' in item:
                    bill_number = item['number']
                    congress_num = item['congress_num']
                    bill_type = item['bill_type']
                    raw_bill_num = item['bill_number']

                    console.print(f"[cyan]Processing {bill_number}...[/cyan]")

                    # Fetch all versions with full text
                    versions = fetch_bill_versions_from_congress(
                        congress_num,
                        bill_type,
                        raw_bill_num,
                        bill_number
                    )

                    if versions:
                        # Attach versions to the item
                        item['versions'] = versions
                        console.print(f"[green]✓ Loaded {len(versions)} versions for {bill_number}[/green]\n")
                    else:
                        console.print(f"[yellow]⚠ No versions found for {bill_number}[/yellow]\n")
                        item['versions'] = []
                else:
                    console.print(f"[yellow]⚠ Skipping {item['number']}: Missing version tracking metadata[/yellow]\n")

    
    # Phase 2: Change Detection & Analysis - Hash comparison and LLM analysis
    if change_tracking_enabled and db_conn:
        console.print("\n[bold cyan]🔍 Checking for changes...[/bold cyan]\n")
    console.print(f"\n[bold cyan]🤖 Analyzing with {llm_provider.upper()} ({llm_model})...[/bold cyan]\n")

    results = []
    for i, item in enumerate(all_items, 1):
        bill_id = f"{item['source']}:{item['number']}"

        # Check for changes if tracking enabled
        # Note that there are two different levels of change being tracked:
        # 1. Bill-Level Changes (change_data) Detected by comparing cached vs current bill metadata:
        #       Summary text updates (API refreshes description)
        #       Status changes: "In Committee" → "Passed House" → "Sent to Senate"
        #       New amendments added to the current version
        #       Title changes
        # Example: Bill stays as "Introduced" version but status updates to "Passed Committee"

        # 2. Version-Level Changes (version_diffs) New legislative text versions with different full bill text:
        #       Introduced → Committee Substitute → Engrossed → Enrolled
        #       Each has complete rewritten/amended bill text
        
        # 1. Hash check changes to current bill summary/status since last run
        #    - Triggers when bill status, title, or summary text changes
        #    - Optimizes performance by avoiding unnecessary LLM analysis of unchanged bills
        # 2. Version check, analyzes all historical versions (Introduced -> Engrossed -> Enacted)
        #    - Provides complete legislative evolution regardless of current changes
        #    - Uses sliding window comparison between consecutive versions
        # These are independent: version analysis runs even when hash check shows no changes
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

        # Check if this bill has versions
        has_versions = 'versions' in item and len(item.get('versions', [])) > 0
        analyze_all_versions = config.get('version_tracking', {}).get('analyze_all_versions', True)

        if has_versions and analyze_all_versions:
            # Version-based analysis
            console.print(f"[cyan]  Analyzing {item['number']} ({len(item['versions'])} versions) with {llm_provider.upper()}...[/cyan]")

            version_analyses = []
            version_diffs = []

            # Analyze each version independently
            for v_idx, version in enumerate(item['versions'], 1):
                console.print(f"[dim]    Version {v_idx}/{len(item['versions'])}: {version['version_type']}[/dim]")

                # Independent version analysis
                # TODO: OPTIMIZATION - Implement cached version analysis retrieval
                # Current: Always runs LLM analysis on every execution
                # Problem: Re-analyzing unchanged versions wastes LLM costs and API limits
                # Solution: Check get_version_analysis() first, only call analyze_bill_version() if no cached result
                
                # NOTE: LLM Error Handling
                # analyze_bill_version() -> analyze_with_gemini/openai() handle API errors via try-except
                # On error: Returns dict with {"error": "...", "business_impact_score": 0, "impact_summary": "Analysis failed"}
                # This means analysis continues with degraded data rather than stopping execution
                # Risk: If all versions fail, version_analyses = [] and final analysis = {} (see line 4004)
                
                # TODO: CRITICAL BUG - Missing bill-level change analysis in Path A
                # Problem: when a bill already has versions but only metadata changes, 
                # it stays in Path A which doesn't analyze the metadata change.
                # occurs when: 
                #    has_versions=True AND change_data['has_changes']=True AND no new versions added
                # This might results in wrong cost assumptions, 
                # misses critical intermediate signals (committee passage, amendments filed, etc.)
                # Example: 
                #   Run 1 (Day 1):
                #       - Fetch bill, has version: ["Introduced"]
                #       - has_versions = True → Path A
                #       - Analyzes "Introduced" version

                #   Run 2 (Day 5):
                #       - Bill status changed: "In Committee" → "Passed Committee"
                #       - Fetch versions: Still ["Introduced"] (no new version yet)
                #       - has_versions = True → Path A again
                #       - change_data detects status change
                #       - Path A re-analyzes old "Introduced" version
                #       - Path A sets change_impact = None (doesn't analyze status change)
                version_analysis = analyze_bill_version(
                    version['full_text'],
                    version['version_type'],
                    item,
                    nrg_context
                )

                version_analyses.append({
                    'version': version,
                    'analysis': version_analysis
                })

                # Save version to database
                if change_tracking_enabled and db_conn:
                    version_id = save_bill_version(bill_id, version, db_conn)
                    save_version_analysis(version_id, version_analysis, db_conn)

                # Compare with previous version
                if v_idx > 1:
                    # NOTE: IndexError Risk
                    # Assumes version_analyses has same length as versions processed so far
                    # If previous version analysis failed silently (returned error dict but didn't append):
                    #   - version_analyses shorter than expected
                    #   - Accessing [v_idx - 2] raises IndexError
                    # Currently safe because analyze_bill_version() always returns a dict (even on error)
                    # and we always append to version_analyses regardless of success/failure
                    prev_version = item['versions'][v_idx - 2]
                    prev_analysis = version_analyses[v_idx - 2]['analysis']

                    # Text-based diff
                    diff = compare_consecutive_versions(prev_version, version)

                    # LLM-based semantic analysis of changes
                    console.print(f"[dim]      Analyzing substantive changes from {prev_version.get('version_type')} to {version.get('version_type')}...[/dim]")
                    semantic_changes = analyze_version_changes_with_llm(
                        prev_version,
                        version,
                        prev_analysis,
                        version_analysis,
                        item,
                        nrg_context
                    )

                    # Combine text diff with semantic analysis
                    diff['semantic_analysis'] = semantic_changes
                    version_diffs.append(diff)

            # DEBUG: Log version_analyses order if enabled
            if config.get('debug', {}).get('enabled', False) and config.get('debug', {}).get('analysis_order', True):
                console.print(f"[dim]  DEBUG: version_analyses order:[/dim]")
                for i, va in enumerate(version_analyses):
                    v_type = va['version'].get('version_type', 'Unknown')
                    score = va['analysis'].get('business_impact_score', 0)
                    console.print(f"[dim]    [{i}] {v_type} (score: {score})[/dim]")
            
            analysis = version_analyses[0]['analysis'] if version_analyses else {}

            # Store result with version info
            result = {
                "item": item,
                "analysis": analysis,
                "version_analyses": version_analyses,
                "version_diffs": version_diffs,
                "change_data": change_data,
                "change_impact": None
            }
        else:
            # Regular analysis if change_tracking enabled (no versions or version tracking disabled)
            console.print(f"[dim]  Analyzing {i}/{len(all_items)} with {llm_provider.upper()}...[/dim]")
            analysis = analyze_with_llm(item, nrg_context)

            # Analyze changes if detected
            change_impact = None
            if change_data and change_data['has_changes'] and config.get('change_tracking', {}).get('analyze_changes_with_llm', True):
                console.print(f"[dim]    Analyzing change impact...[/dim]")
                change_impact = analyze_changes_with_llm(item, change_data, nrg_context)

            # Store result with change info
            result = {
                "item": item,
                "analysis": analysis,
                "change_data": change_data,
                "change_impact": change_impact
            }

        results.append(result)

        # Save to cache if tracking enabled
        if change_tracking_enabled and db_conn:
            save_bill_to_cache(item, db_conn)

    # Display results sorted by impact score
    console.print("\n[bold magenta]═══════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  ANALYSIS RESULTS  [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/bold magenta]\n")

    # Sort by impact score (highest first)
    # Handle both dict and list cases (in case of malformed analysis)
    results.sort(key=lambda x: x["analysis"].get("business_impact_score", 0) if isinstance(x["analysis"], dict) else 0, reverse=True)

    for result in results:
        display_analysis(result["item"], result["analysis"])

    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = f"nrg_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[cyan]Created output directory: {output_dir}/[/cyan]")

    # Save to JSON file
    json_file = os.path.join(output_dir, f"nrg_analysis_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]✓ JSON results saved to {json_file}[/green]")

    # Generate markdown report
    console.print("[cyan]Generating markdown report...[/cyan]")
    md_file = generate_markdown_report(results, timestamp, output_dir)
    console.print(f"[green]✓ Markdown report saved to {md_file}[/green]")

    # Generate Word document
    console.print("[cyan]Generating Word document...[/cyan]")
    docx_file = convert_markdown_to_word(md_file)
    if docx_file:
        console.print(f"[green]✓ Word document saved to {docx_file}[/green]\n")
    else:
        console.print("[dim]  (Word document generation skipped)[/dim]\n")

    # Summary statistics
    high_impact = sum(1 for r in results if r["analysis"].get("business_impact_score", 0) >= 7)
    medium_impact = sum(1 for r in results if 4 <= r["analysis"].get("business_impact_score", 0) < 7)
    low_impact = sum(1 for r in results if r["analysis"].get("business_impact_score", 0) < 4)

    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  High Impact (7-10):   {high_impact}")
    console.print(f"  Medium Impact (4-6):  {medium_impact}")
    console.print(f"  Low Impact (0-3):     {low_impact}")

    # Show change tracking summary if enabled
    if change_tracking_enabled:
        new_bills = sum(1 for r in results if r.get('change_data', {}).get('is_new', False))
        modified_bills = sum(1 for r in results if r.get('change_data', {}).get('has_changes', False) and not r.get('change_data', {}).get('is_new', False))
        unchanged_bills = sum(1 for r in results if not r.get('change_data', {}).get('has_changes', False))
        console.print(f"\n[bold]Change Tracking:[/bold]")
        console.print(f"  New Bills:        {new_bills}")
        console.print(f"  Modified Bills:   {modified_bills}")
        console.print(f"  Unchanged Bills:  {unchanged_bills}")

    console.print()

    # Close database connection
    if db_conn:
        db_conn.close()


# =============================================================================
# ENTRY POINT
# =============================================================================
# Execute with: uv run poc.py
# =============================================================================

if __name__ == "__main__":
    main()