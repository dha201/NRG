"""
NRG Legislative Intelligence - V2 Full Pipeline

End-to-end pipeline that:
1. Fetches bills from configured sources (Open States, Congress.gov, etc.)
2. Runs Sequential Evolution analysis on all versions
3. Applies Two-Tier Validation (Judge + Multi-sample + Fallback)
4. Scores findings using rubric-based assessment
5. Generates reports in multiple formats (JSON, Markdown, DOCX)

Usage:
    python run_v2_full_pipeline.py [options]

Options:
    --debug             Debug mode (show all traces and detailed debugging)
    --config FILE       Path to config file (default: config.yaml)
    --output DIR        Output directory for reports (default: ./reports)

Examples:
    python run_v2_full_pipeline.py                    # Normal mode (with traces)
    python run_v2_full_pipeline.py --debug            # Debug mode (all details)
    python run_v2_full_pipeline.py --output ./out     # Custom output dir
"""
import os
import sys
import json
import time
import tempfile
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import yaml
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

# Import V2 analysis components
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.v2.sequential_evolution import SequentialEvolutionAgent, BillVersion
from nrg_core.config import load_nrg_context

# Import change detection and cache components
from nrg_core.analysis.changes import detect_bill_changes, CHANGE_TYPE_TEXT, CHANGE_TYPE_STATUS, CHANGE_TYPE_AMENDMENTS
from nrg_core.db.cache import init_database, get_cached_bill, save_bill_to_cache, compute_bill_hash

# Import display function for rich CLI output
from nrg_core.reports.display import display_analysis

load_dotenv()

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class LogLevel(Enum):
    NORMAL = 1
    DEBUG = 2


@dataclass
class PipelineLogger:
    """Centralized logger with trace support for pipeline visibility."""
    console: Console = field(default_factory=Console)
    level: LogLevel = LogLevel.NORMAL
    trace_config: dict = field(default_factory=dict)
    show_timestamps: bool = True
    show_token_counts: bool = True
    show_cost_estimates: bool = True
    log_file: Optional[Path] = None
    _log_lines: list = field(default_factory=list)

    def _format_time(self) -> str:
        if self.show_timestamps and self.level.value >= LogLevel.NORMAL.value:
            return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
        return ""

    def _format_time_plain(self) -> str:
        """Plain text timestamp for log file."""
        if self.show_timestamps:
            return f"{datetime.now().strftime('%H:%M:%S')} "
        return ""

    def _log_to_file(self, message: str):
        """Append message to log buffer."""
        self._log_lines.append(message)

    def save_log(self, log_path: Path):
        """Save accumulated log to file."""
        log_path.write_text("\n".join(self._log_lines))

    def info(self, message: str, emoji: str = ""):
        """Standard info message."""
        if self.level.value >= LogLevel.NORMAL.value:
            prefix = f"{emoji} " if emoji else ""
            self.console.print(f"{self._format_time()}{prefix}{message}")
            self._log_to_file(f"{self._format_time_plain()}{prefix}{message}")

    def verbose(self, message: str, emoji: str = ""):
        """Verbose message (traces - shown in normal mode and above)."""
        if self.level.value >= LogLevel.NORMAL.value:
            prefix = f"{emoji} " if emoji else ""
            self.console.print(f"{self._format_time()}[dim]{prefix}{message}[/dim]")
            self._log_to_file(f"{self._format_time_plain()}  {prefix}{message}")

    def debug(self, message: str):
        """Debug message (only in debug mode)."""
        if self.level.value >= LogLevel.DEBUG.value:
            self.console.print(f"{self._format_time()}[dim cyan]DEBUG: {message}[/dim cyan]")
            self._log_to_file(f"{self._format_time_plain()}DEBUG: {message}")

    def success(self, message: str):
        """Success message."""
        if self.level.value >= LogLevel.NORMAL.value:
            self.console.print(f"{self._format_time()}[green]✓[/green] {message}")
            self._log_to_file(f"{self._format_time_plain()}✓ {message}")

    def warning(self, message: str):
        """Warning message."""
        self.console.print(f"{self._format_time()}[yellow]⚠[/yellow] {message}")
        self._log_to_file(f"{self._format_time_plain()}⚠ WARNING: {message}")

    def error(self, message: str):
        """Error message."""
        self.console.print(f"{self._format_time()}[red]✗[/red] {message}")
        self._log_to_file(f"{self._format_time_plain()}✗ ERROR: {message}")

    def stage_header(self, stage_num: int, title: str, description: str = ""):
        """Log a pipeline stage header."""
        if self.level.value >= LogLevel.NORMAL.value:
            self.console.print()
            self.console.rule(f"[bold cyan]Stage {stage_num}: {title}[/bold cyan]", style="cyan")
            self._log_to_file(f"\n{'='*60}")
            self._log_to_file(f"Stage {stage_num}: {title}")
            self._log_to_file(f"{'='*60}")
            if description and self.level.value >= LogLevel.DEBUG.value:
                self.console.print(f"[dim]{description}[/dim]")
                self._log_to_file(f"  {description}")

    def trace_decision(self, component: str, decision: str, rationale: str, details: dict = None):
        """Log a decision with rationale - core for traceability."""
        trace_enabled = self.trace_config.get(component, False)
        if self.level.value >= LogLevel.NORMAL.value and trace_enabled:
            self.console.print(f"  [bold yellow]→ Decision:[/bold yellow] {decision}")
            self.console.print(f"    [dim]Rationale: {rationale}[/dim]")
            if details and self.level.value >= LogLevel.DEBUG.value:
                for k, v in details.items():
                    self.console.print(f"    [dim]{k}: {v}[/dim]")

    def trace_finding(self, finding_num: int, statement: str, quotes: list, confidence: float, impact: int):
        """Log finding extraction details."""
        if self.level.value >= LogLevel.NORMAL.value and self.trace_config.get("sequential_evolution", False):
            tree = Tree(f"[bold]Finding F{finding_num}[/bold]")
            tree.add(f"Statement: {statement[:100]}{'...' if len(statement) > 100 else ''}")
            if quotes:
                quotes_branch = tree.add("Supporting Quotes")
                for q in quotes[:2]:
                    text = q.get("text", "") if isinstance(q, dict) else str(q)
                    section = q.get("section", "?") if isinstance(q, dict) else "?"
                    quotes_branch.add(f'[dim]"{text[:60]}..." (§{section})[/dim]')
            tree.add(f"Confidence: [{'green' if confidence >= 0.8 else 'yellow' if confidence >= 0.6 else 'red'}]{confidence:.2f}[/]")
            tree.add(f"Impact Estimate: [{'red' if impact >= 7 else 'yellow' if impact >= 4 else 'green'}]{impact}/10[/]")
            self.console.print(tree)

    def trace_validation(self, finding_num: int, quote_verified: bool, hallucination: bool,
                         evidence_quality: float, judge_confidence: float, reason: str = ""):
        """Log validation result with explanation."""
        if self.level.value >= LogLevel.NORMAL.value and self.trace_config.get("two_tier_validation", False):
            status_icon = "[green]✓[/green]" if quote_verified and not hallucination else "[red]✗[/red]"
            status_text = "Verified" if quote_verified and not hallucination else "Hallucination" if hallucination else "Unverified"

            self.console.print(f"  F{finding_num}: {status_icon} {status_text}")
            self.console.print(f"    [dim]Quote Verified: {quote_verified} | Evidence Quality: {evidence_quality:.2f} | Judge Confidence: {judge_confidence:.2f}[/dim]")

            if hallucination and self.trace_config.get("hallucination_detection", False):
                self.console.print(f"    [yellow]→ Hallucination Reason: {reason or 'Finding claims impact not supported by bill text'}[/yellow]")

    def trace_rubric_score(self, finding_id: str, dimension: str, score: int, rationale: str, anchor: str):
        """Log rubric scoring details."""
        if self.level.value >= LogLevel.NORMAL.value and self.trace_config.get("rubric_scoring", False):
            color = "green" if score <= 2 else "yellow" if score <= 5 else "red" if score <= 8 else "bold red"
            self.console.print(f"  {dimension}: [{color}]{score}/10[/{color}] → {anchor}")
            if self.level.value >= LogLevel.DEBUG.value and rationale:
                self.console.print(f"    [dim]Rationale: {rationale[:150]}...[/dim]")

    def trace_stability(self, finding_id: str, score: float, origin_version: int, mod_count: int, prediction: str):
        """Log stability analysis."""
        if self.level.value >= LogLevel.NORMAL.value and self.trace_config.get("stability_analysis", False):
            color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
            self.console.print(f"  {finding_id}: Stability [{color}]{score:.2f}[/{color}]")
            self.console.print(f"    [dim]Origin: v{origin_version} | Modifications: {mod_count} | Prediction: {prediction}[/dim]")

    def trace_complexity(self, bill_id: str, score: int, route: str, reasons: list):
        """Log complexity assessment."""
        if self.level.value >= LogLevel.NORMAL.value:
            self.console.print(f"  [bold]Complexity Assessment:[/bold] {bill_id}")
            self.console.print(f"    Score: {score} → Route: [cyan]{route}[/cyan]")
            for reason in reasons:
                self.console.print(f"    [dim]• {reason}[/dim]")

    def show_table(self, title: str, columns: list, rows: list):
        """Display a formatted table."""
        if self.level.value >= LogLevel.NORMAL.value:
            table = Table(title=title, box=box.ROUNDED)
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*[str(x) for x in row])
            self.console.print(table)

    def log_llm_call(self, component: str, model: str, prompt: str, prompt_tokens: int = 0):
        """Log LLM API call with prompt and token count."""
        if self.level.value >= LogLevel.DEBUG.value:
            self.console.print()
            self.console.print(f"[bold cyan]→ LLM Call:[/bold cyan] {component} ({model})")
            self.console.print(f"  [dim]Tokens: {prompt_tokens}[/dim]")
            self.console.print(f"  [dim]Prompt:[/dim]")
            # Show first 500 chars of prompt
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            self.console.print(f"  [dim]{prompt_preview}[/dim]")

    def log_llm_response(self, component: str, response: str, response_tokens: int = 0, cost: float = 0.0):
        """Log LLM API response with tokens and cost."""
        if self.level.value >= LogLevel.DEBUG.value:
            self.console.print(f"  [dim]Response tokens: {response_tokens}[/dim]")
            if cost > 0:
                self.console.print(f"  [dim]Estimated cost: ${cost:.6f}[/dim]")
            self.console.print(f"  [dim]Response:[/dim]")
            # Show first 500 chars of response
            response_preview = response[:500] + "..." if len(response) > 500 else response
            self.console.print(f"  [dim][yellow]{response_preview}[/yellow][/dim]")

    def log_token_usage(self, component: str, prompt_tokens: int, completion_tokens: int,
                       total_tokens: int, cost: float = 0.0, model: str = ""):
        """Log token usage and cost for an LLM interaction."""
        if self.level.value >= LogLevel.DEBUG.value:
            self.console.print(f"  [cyan]Token Usage ({component})[/cyan]")
            self.console.print(f"    Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
            if model:
                self.console.print(f"    Model: {model}")
            if cost > 0:
                self.console.print(f"    Estimated cost: ${cost:.6f}")

    def log_llm_interaction(self, component: str, model: str, prompt: str, response: str,
                           prompt_tokens: int = 0, completion_tokens: int = 0, cost: float = 0.0):
        """Log complete LLM interaction for full audit trail."""
        if self.level.value >= LogLevel.DEBUG.value:
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]{component}[/bold cyan] ({model})",
                title="LLM Interaction",
                border_style="cyan"
            ))

            total_tokens = prompt_tokens + completion_tokens
            self.console.print(f"[yellow]Prompt ({prompt_tokens} tokens):[/yellow]")
            prompt_preview = prompt[:800] + "\n..." if len(prompt) > 800 else prompt
            self.console.print(f"[dim]{prompt_preview}[/dim]")

            self.console.print(f"\n[yellow]Response ({completion_tokens} tokens):[/yellow]")
            response_preview = response[:800] + "\n..." if len(response) > 800 else response
            self.console.print(f"[dim]{response_preview}[/dim]")

            self.console.print(f"\n[yellow]Stats:[/yellow]")
            self.console.print(f"  Total Tokens: {total_tokens} | Cost: ${cost:.6f}")
            self.console.print()


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found")
        sys.exit(1)


def get_logger_from_config(config: dict, cli_level: Optional[str] = None) -> PipelineLogger:
    """Create logger from config."""
    log_config = config.get("logging", {})

    # CLI overrides config
    if cli_level == "debug":
        level = LogLevel.DEBUG
    else:
        level_str = log_config.get("level", "normal")
        level = LogLevel[level_str.upper()] if level_str.upper() in LogLevel.__members__ else LogLevel.NORMAL

    trace_config = log_config.get("trace", {
        "sequential_evolution": True,
        "two_tier_validation": True,
        "rubric_scoring": True,
        "hallucination_detection": True,
        "stability_analysis": True,
    })
    format_config = log_config.get("format", {})

    return PipelineLogger(
        level=level,
        trace_config=trace_config,
        show_timestamps=format_config.get("show_timestamps", True),
        show_token_counts=format_config.get("show_token_counts", True),
        show_cost_estimates=format_config.get("show_cost_estimates", True),
    )


# ============================================================================
# PDF TEXT EXTRACTION
# ============================================================================

def extract_pdf_text(pdf_url: str, logger: PipelineLogger) -> str:
    """Extract text from PDF URL using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed - skipping PDF extraction")
        return ""

    try:
        logger.verbose(f"Downloading PDF from {pdf_url[:60]}...")
        with httpx.Client(timeout=30.0, follow_redirects=True) as http:
            response = http.get(pdf_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            text_parts = []
            with pdfplumber.open(tmp_path) as pdf:
                logger.verbose(f"Extracting text from {len(pdf.pages)} pages...")
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            os.unlink(tmp_path)
            full_text = "\n".join(text_parts)
            logger.verbose(f"Extracted {len(full_text)} characters ({len(full_text.split())} words)")
            return full_text

    except Exception as e:
        logger.debug(f"PDF extraction failed: {e}")
        return ""


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_openstates_bills(jurisdiction: str, bill_numbers: List[str], logger: PipelineLogger) -> List[Dict]:
    """Fetch bills from Open States API."""
    api_key = os.getenv("OPENSTATES_API_KEY")
    if not api_key:
        logger.warning("OPENSTATES_API_KEY not set")
        return []

    logger.info(f"Fetching {len(bill_numbers)} bills from Open States ({jurisdiction})...")

    base_url = "https://v3.openstates.org"
    headers = {
        "X-API-KEY": api_key,
        "User-Agent": "NRG-Energy-Legislative-Tracker/2.0"
    }
    bills = []

    try:
        with httpx.Client(timeout=60.0) as http:
            for bill_num in bill_numbers:
                logger.verbose(f"Fetching {jurisdiction} {bill_num}...")

                # Search for bill
                search_url = f"{base_url}/bills"
                params = {
                    "jurisdiction": jurisdiction.lower(),
                    "q": bill_num,
                    "per_page": 5
                }

                response = http.get(search_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                for result in data.get("results", []):
                    identifier = result.get("identifier", "")
                    if identifier.replace(" ", "").upper() == bill_num.replace(" ", "").upper():
                        openstates_id = result.get("id", "")

                        # Fetch versions
                        versions = []
                        if openstates_id:
                            detail_url = f"{base_url}/bills/{openstates_id}"
                            detail_params = {"include": "versions"}
                            detail_resp = http.get(detail_url, headers=headers, params=detail_params)
                            if detail_resp.status_code == 200:
                                detail_data = detail_resp.json()
                                raw_versions = detail_data.get("versions", [])

                                for idx, v in enumerate(raw_versions, 1):
                                    version_text = ""
                                    links = v.get("links", [])

                                    # Try to get PDF text
                                    for link in links:
                                        if link.get("media_type") == "application/pdf":
                                            pdf_url = link.get("url")
                                            version_text = extract_pdf_text(pdf_url, logger)
                                            if version_text:
                                                break

                                    if version_text:
                                        versions.append({
                                            "version_number": idx,
                                            "version_type": v.get("note", f"Version {idx}"),
                                            "version_date": v.get("date", ""),
                                            "full_text": version_text,
                                            "word_count": len(version_text.split())
                                        })

                        # Get latest version text for primary analysis
                        bill_text = ""
                        if versions:
                            bill_text = versions[-1].get("full_text", "")

                        # Extract sponsor info
                        sponsorships = result.get("sponsorships", [])
                        primary_sponsor = ""
                        if sponsorships:
                            for sp in sponsorships:
                                if sp.get("primary", False) or sp.get("classification") == "primary":
                                    primary_sponsor = sp.get("name", "")
                                    break
                            if not primary_sponsor and sponsorships:
                                primary_sponsor = sponsorships[0].get("name", "")

                        # Extract sources/URLs
                        sources = result.get("sources", [])
                        bill_url = ""
                        if sources:
                            bill_url = sources[0].get("url", "")
                        # Fallback to Texas Legislature URL
                        if not bill_url and jurisdiction.upper() == "TX":
                            session = result.get("session", {})
                            session_id = session.get("identifier", "89R") if isinstance(session, dict) else "89R"
                            bill_url = f"https://capitol.texas.gov/BillLookup/History.aspx?LegSess={session_id}&Bill={identifier.replace(' ', '')}"

                        # Extract introduced date
                        introduced_date = result.get("created_at", "")[:10] if result.get("created_at") else ""

                        if bill_text:
                            bills.append({
                                "source": "Open States",
                                "type": f"{jurisdiction} State Bill",
                                "bill_id": identifier.replace(" ", ""),
                                "number": identifier,
                                "title": result.get("title", "No title"),
                                "status": result.get("actions", [{}])[-1].get("description", "Unknown") if result.get("actions") else "Unknown",
                                "sponsor": primary_sponsor,
                                "introduced_date": introduced_date,
                                "url": bill_url,
                                "bill_text": bill_text,
                                "versions": versions,
                                "openstates_id": openstates_id,
                                "jurisdiction": jurisdiction
                            })
                            logger.success(f"Found {identifier} ({len(versions)} versions)")
                        break

                time.sleep(0.3)  # Rate limiting

    except Exception as e:
        logger.error(f"Error fetching from Open States: {e}")

    logger.success(f"Fetched {len(bills)} bills from Open States")
    return bills


def fetch_congress_bills(limit: int = 3, logger: Optional[PipelineLogger] = None) -> List[Dict]:
    """
    Fetch federal bills from Congress.gov API with ALL text versions.
        - Searches 118th Congress House bills
        - Energy keyword filtering (oil, gas, pipeline, renewable, etc.)
        - Fetches ALL bill versions (IH, EH, RFS, ENR, etc.) for version tracking
        - Extracts title, sponsor, status, dates
        - Rate limited to config value (default 0.5s) between requests
    """
    api_key = os.getenv("CONGRESS_API_KEY")
    if not api_key:
        if logger:
            logger.warning("CONGRESS_API_KEY not set, skipping Congress.gov")
        return []

    if logger:
        logger.info(f"Fetching federal bills from Congress.gov (limit: {limit})...")

    bills = []
    congress_num = "118"  # Current Congress
    bill_type = "hr"  # House bills

    try:
        with httpx.Client(timeout=30.0) as http:
            # Fetch recent House bills
            url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}"
            params = {"api_key": api_key, "limit": 20, "sort": "updateDate desc"}

            response = http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            energy_keywords = {
                "oil", "gas", "petroleum", "natural gas", "fossil fuel", "energy",
                "drilling", "fracking", "pipeline", "lng", "renewable", "solar",
                "wind", "hydro", "battery", "electric", "fossil"
            }

            for bill in data.get("bills", [])[:20]:
                bill_num = bill.get("number", "").replace("H.R. ", "")
                title = bill.get("title", "").lower()

                # Check title for energy keywords
                if not any(kw in title for kw in energy_keywords):
                    continue

                # Fetch detailed info
                detail_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_num}"
                detail_resp = http.get(detail_url, params={"api_key": api_key})

                if detail_resp.status_code == 200:
                    bill_info = detail_resp.json().get("bill", {})

                    # Fetch ALL bill text versions from /text endpoint
                    text_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_num}/text"
                    text_resp = http.get(text_url, params={"api_key": api_key, "format": "json"})

                    versions = []
                    bill_text = ""  # Will hold the latest version text

                    if text_resp.status_code == 200:
                        text_data = text_resp.json()
                        text_versions_raw = text_data.get("textVersions", [])

                        if text_versions_raw:
                            if logger:
                                logger.verbose(f"  Found {len(text_versions_raw)} versions for H.R. {bill_num}")

                            # Process ALL versions (like POC 2)
                            for idx, version_data in enumerate(text_versions_raw, 1):
                                version_type = version_data.get("type", "Unknown")
                                version_date = version_data.get("date", "")

                                # Find Formatted Text or TXT format
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
                                    if logger:
                                        logger.debug(f"    Version {idx} ({version_type}): No text format, skipping")
                                    continue

                                # Fetch the actual bill text for this version
                                try:
                                    text_content_resp = http.get(txt_url, params={"api_key": api_key})
                                    if text_content_resp.status_code == 200:
                                        full_text = text_content_resp.text
                                        word_count = len(full_text.split()) if full_text else 0

                                        versions.append({
                                            "version_number": idx,
                                            "version_type": version_type,
                                            "version_date": version_date,
                                            "full_text": full_text,
                                            "word_count": word_count,
                                            "text_url": txt_url
                                        })

                                        if logger:
                                            logger.verbose(f"    Version {idx}: {version_type} ({word_count} words)")

                                        # Keep track of latest version text (first in list is most recent)
                                        if not bill_text:
                                            bill_text = full_text

                                except Exception as e:
                                    if logger:
                                        logger.debug(f"    Version {idx}: Error fetching text: {e}")
                                    continue

                            if logger and versions:
                                logger.verbose(f"  Extracted {len(versions)}/{len(text_versions_raw)} versions")

                    # Fallback to summary if no full text available
                    if not bill_text:
                        bill_text = bill_info.get("summary", {}).get("text", "")
                        if bill_text and logger:
                            logger.verbose(f"  Using summary text ({len(bill_text.split())} words)")

                    # If no versions were extracted, create a single "Current" version
                    if not versions and bill_text:
                        versions = [{
                            "version_number": 1,
                            "version_type": "Current",
                            "version_date": bill_info.get("introducedDate", ""),
                            "full_text": bill_text,
                            "word_count": len(bill_text.split()) if bill_text else 0
                        }]

                    # Fetch amendments
                    amendments = []
                    amendments_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_num}/amendments"
                    try:
                        amendments_resp = http.get(amendments_url, params={"api_key": api_key})
                        if amendments_resp.status_code == 200:
                            amendments_data = amendments_resp.json()
                            raw_amendments = amendments_data.get("amendments", [])

                            for amend in raw_amendments:
                                amendment_entry = {
                                    "amendment_id": amend.get("number", ""),
                                    "amendment_number": amend.get("number", ""),
                                    "type": amend.get("type", ""),
                                    "date": amend.get("latestAction", {}).get("actionDate", ""),
                                    "status": amend.get("latestAction", {}).get("text", "Unknown"),
                                    "purpose": amend.get("purpose", ""),
                                    "congress": amend.get("congress", congress_num),
                                }
                                amendments.append(amendment_entry)

                            if logger and amendments:
                                logger.verbose(f"  Found {len(amendments)} amendments for H.R. {bill_num}")
                    except Exception as amend_err:
                        if logger:
                            logger.debug(f"  Could not fetch amendments: {amend_err}")

                    # Fetch subjects
                    subjects_url = f"https://api.congress.gov/v3/bill/{congress_num}/{bill_type}/{bill_num}/subjects"
                    subjects_resp = http.get(subjects_url, params={"api_key": api_key})
                    subjects_data = {}
                    if subjects_resp.status_code == 200:
                        subjects_data = subjects_resp.json().get("subjects", {})

                    if bill_text or subjects_data:
                        bills.append({
                            "source": "Congress.gov",
                            "type": "Federal Bill",
                            "bill_id": f"HR{bill_num}",
                            "number": f"H.R. {bill_num}",
                            "title": bill_info.get("title", title),  # Use full title from detail
                            "status": bill_info.get("latestAction", {}).get("text", "Unknown"),
                            "sponsor": bill_info.get("sponsor", {}).get("fullName", "Unknown"),
                            "introduced_date": bill_info.get("introducedDate", ""),
                            "url": f"https://www.congress.gov/bill/{congress_num}th-congress/house-bill/{bill_num}",
                            "jurisdiction": "Federal",
                            "bill_text": bill_text or "",
                            "versions": versions,
                            "amendments": amendments,  # Store amendments from Congress.gov
                            "congress_num": congress_num,
                            "bill_type": bill_type,
                            "bill_number": bill_num  # Raw number for version tracking
                        })
                        if logger:
                            logger.success(f"Found H.R. {bill_num} ({len(versions)} versions)")

                if len(bills) >= limit:
                    break

                time.sleep(0.5)  # Rate limiting

    except Exception as e:
        if logger:
            logger.error(f"Error fetching Congress bills: {e}")

    if logger:
        logger.success(f"Fetched {len(bills)} bills from Congress.gov")
    return bills


def fetch_regulations(limit: int = 3, logger: Optional[PipelineLogger] = None) -> List[Dict]:
    """
    Fetch federal regulations from Regulations.gov API.
        - Energy keyword filtering across title and summary
        - Extracts agency, document type, dates, comment periods    
        - Rate limited to config value (default 0.5s) between requests
    """
    api_key = os.getenv("CONGRESS_API_KEY")  # Regulations.gov uses same API key
    if not api_key:
        if logger:
            logger.warning("CONGRESS_API_KEY not set, skipping Regulations.gov")
        return []

    if logger:
        logger.info(f"Fetching federal regulations (limit: {limit})...")

    regulations = []

    try:
        with httpx.Client(timeout=30.0) as http:
            url = "https://api.regulations.gov/v4/documents"
            # Note: Regulations.gov API v4 uses page[size] not per_page
            params = {
                "api_key": api_key,
                "page[size]": 20,  # API v4 syntax
                "sort": "-postedDate"
            }

            response = http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            energy_keywords = {
                "oil", "gas", "petroleum", "natural gas", "fossil fuel", "energy",
                "pipeline", "drilling", "lng", "emissions", "renewable", "solar",
                "wind", "hydro", "battery", "electric"
            }

            for doc in data.get("data", [])[:20]:
                attributes = doc.get("attributes", {})
                title = attributes.get("title", "").lower()
                summary = attributes.get("summary", "").lower()
                doc_type = attributes.get("documentType", "")

                # Check title and summary for energy keywords
                if not (any(kw in title for kw in energy_keywords) or
                        any(kw in summary for kw in energy_keywords)):
                    continue

                doc_id = doc.get("id", "")
                if doc_id:
                    regulations.append({
                        "source": "Regulations.gov",
                        "type": "Regulation",
                        "bill_id": doc_id,
                        "number": doc_id,
                        "title": attributes.get("title", ""),
                        "status": doc_type,
                        "agency": attributes.get("agencyId", ""),
                        "posted_date": attributes.get("postedDate", ""),
                        "comment_end_date": attributes.get("commentEndDate", "N/A"),
                        "effective_date": attributes.get("effectiveDate", "N/A"),
                        "bill_text": attributes.get("summary", ""),
                        "versions": [{
                            "version_number": 1,
                            "version_type": "Current",
                            "full_text": attributes.get("summary", ""),
                            "word_count": len(attributes.get("summary", "").split())
                        }],
                        "url": f"https://www.regulations.gov/document/{doc_id}"
                    })
                    if logger:
                        logger.success(f"Found regulation: {doc_id}")

                if len(regulations) >= limit:
                    break

                time.sleep(0.5)  # Rate limiting

    except Exception as e:
        if logger:
            logger.error(f"Error fetching regulations: {e}")

    if logger:
        logger.success(f"Fetched {len(regulations)} regulations from Regulations.gov")
    return regulations


def fetch_bills_from_config(config: dict, logger: PipelineLogger) -> List[Dict]:
    """
    Fetch all bills configured in sources:
        - aggregate from all sources
        - Congress.gov (configurable limit, default 3)
        - Regulations.gov (configurable limit, default 3)
        - Open States (existing, Texas bills)
        - Provides summary of items fetched by source
    """
    bills = []

    sources_config = config.get("sources", {})

    # Congress.gov federal bills
    congress_config = sources_config.get("congress", {})
    if congress_config.get("enabled"):
        limit = congress_config.get("limit", 3)
        congress_bills = fetch_congress_bills(limit, logger)
        bills.extend(congress_bills)

    # Regulations.gov federal regulations
    regulations_config = sources_config.get("regulations", {})
    if regulations_config.get("enabled"):
        limit = regulations_config.get("limit", 3)
        regs = fetch_regulations(limit, logger)
        bills.extend(regs)

    # Open States state bills (TX)
    openstates_config = sources_config.get("openstates", {})
    if openstates_config.get("enabled"):
        texas_config = openstates_config.get("texas_bills", {})
        if texas_config.get("enabled"):
            jurisdiction = texas_config.get("jurisdiction", "TX")
            bill_numbers = texas_config.get("bills", [])
            if bill_numbers:
                texas_bills = fetch_openstates_bills(jurisdiction, bill_numbers, logger)
                bills.extend(texas_bills)

    logger.info(f"Total items fetched: {len(bills)} (Congress: {sum(1 for b in bills if b['source']=='Congress.gov')}, "
                f"Regulations: {sum(1 for b in bills if b['source']=='Regulations.gov')}, "
                f"Open States: {sum(1 for b in bills if b['source']=='Open States')})")

    return bills


# ============================================================================
# COMPLEXITY ASSESSMENT
# ============================================================================

def assess_complexity(bill: dict, config: dict, logger: PipelineLogger) -> tuple:
    """Assess bill complexity and determine routing.

    NOTE: enhanced_min threshold temporarily lowered to 2 for testing ENHANCED route.
    Calibrate after full testing cycle with real-world results.
    """
    v2_config = config.get("v2", {}).get("orchestration", {})
    scoring = v2_config.get("scoring", {
        "pages_20_50": 1,
        "pages_over_50": 2,
        "versions_2_5": 1,
        "versions_over_5": 2,
        "domain_environmental": 1,
        "domain_energy_tax": 2,
    })
    thresholds = v2_config.get("complexity_thresholds", {"standard_max": 2})

    score = 0
    reasons = []

    # Version count
    version_count = len(bill.get("versions", []))
    if version_count > 5:
        score += scoring.get("versions_over_5", 2)
        reasons.append(f"Many versions ({version_count}) +{scoring.get('versions_over_5', 2)}")
    elif version_count >= 2:
        score += scoring.get("versions_2_5", 1)
        reasons.append(f"Multiple versions ({version_count}) +{scoring.get('versions_2_5', 1)}")

    # Page count estimate
    total_text = sum(len(v.get("full_text", "")) for v in bill.get("versions", []))
    page_estimate = total_text // 3000
    if page_estimate > 50:
        score += scoring.get("pages_over_50", 2)
        reasons.append(f"Long bill (~{page_estimate} pages) +{scoring.get('pages_over_50', 2)}")
    elif page_estimate >= 20:
        score += scoring.get("pages_20_50", 1)
        reasons.append(f"Medium length (~{page_estimate} pages) +{scoring.get('pages_20_50', 1)}")

    # Domain detection
    title_lower = bill.get("title", "").lower()
    if any(kw in title_lower for kw in ["energy", "tax", "oil", "gas", "power"]):
        score += scoring.get("domain_energy_tax", 2)
        reasons.append(f"Energy/Tax domain +{scoring.get('domain_energy_tax', 2)}")
    elif any(kw in title_lower for kw in ["environment", "emission", "climate", "pollution"]):
        score += scoring.get("domain_environmental", 1)
        reasons.append(f"Environmental domain +{scoring.get('domain_environmental', 1)}")

    # Determine route
    standard_max = thresholds.get("standard_max", 2)
    route = "STANDARD" if score <= standard_max else "ENHANCED"

    logger.trace_complexity(bill.get("bill_id", "Unknown"), score, route, reasons)

    return route, score, reasons


# ============================================================================
# REPORT GENERATION
# ============================================================================

def calculate_overall_impact_score(bill_result: dict) -> int:
    """Calculate overall impact score from rubric scores or findings."""
    rubric_scores = bill_result.get("rubric_scores", [])
    if rubric_scores:
        # Average of unique dimension scores
        seen = {}
        for score in rubric_scores:
            dim = score.get("dimension", "")
            if dim and dim not in seen:
                seen[dim] = score.get("score", 0)
        if seen:
            return round(sum(seen.values()) / len(seen))

    # Fallback: average of finding impact estimates
    findings = bill_result.get("findings", [])
    if findings:
        impacts = [f.get("impact_estimate", 0) for f in findings]
        return round(sum(impacts) / len(impacts)) if impacts else 0
    return 0


def get_impact_category(score: int) -> tuple:
    """Return (category, emoji, color, action) for impact score."""
    if score >= 7:
        return ("HIGH", "🔴", "red", "ENGAGE")
    elif score >= 4:
        return ("MEDIUM", "🟡", "yellow", "MONITOR")
    else:
        return ("LOW", "🟢", "green", "AWARENESS")


def get_risk_type(bill_result: dict) -> str:
    """Determine risk type from rubric scores."""
    rubric_scores = bill_result.get("rubric_scores", [])
    if not rubric_scores:
        return "operational"

    # Find highest scoring dimension
    max_score = 0
    max_dim = "operational"
    seen = {}
    for score in rubric_scores:
        dim = score.get("dimension", "")
        s = score.get("score", 0)
        if dim and dim not in seen:
            seen[dim] = s
            if s > max_score:
                max_score = s
                max_dim = dim

    dim_map = {
        "legal_risk": "regulatory_compliance",
        "financial_impact": "financial",
        "operational_disruption": "operational",
        "ambiguity_risk": "strategic"
    }
    return dim_map.get(max_dim, "operational")


def generate_markdown_report(results: dict, output_path: Path, logger: PipelineLogger):
    """Generate detailed Markdown report matching POC 2 format with V2 enhancements."""
    logger.verbose("Generating Markdown report...")

    all_results = results.get("results", [])

    # Calculate impact scores and categorize bills
    for bill_result in all_results:
        bill_result["_impact_score"] = calculate_overall_impact_score(bill_result)
        cat, emoji, color, action = get_impact_category(bill_result["_impact_score"])
        bill_result["_impact_category"] = cat
        bill_result["_impact_emoji"] = emoji
        bill_result["_recommended_action"] = action

    # Group by impact level
    high_impact = [b for b in all_results if b["_impact_category"] == "HIGH"]
    medium_impact = [b for b in all_results if b["_impact_category"] == "MEDIUM"]
    low_impact = [b for b in all_results if b["_impact_category"] == "LOW"]

    # Sort each group by score descending
    high_impact.sort(key=lambda x: x["_impact_score"], reverse=True)
    medium_impact.sort(key=lambda x: x["_impact_score"], reverse=True)
    low_impact.sort(key=lambda x: x["_impact_score"], reverse=True)

    # Get change detection summary
    change_summary = results.get("change_detection_summary", {})

    lines = [
        "# NRG Energy Legislative Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## How to Read This Report",
        "",
        "This report uses a structured analysis approach to assess legislative impact on NRG:",
        "",
        "1. **Findings** are specific provisions extracted from the bill text with supporting evidence (direct quotes)",
        "2. **Rubric Assessment** scores the bill across four dimensions - each score is justified by the findings",
        "3. **Stability Analysis** tracks which findings persisted across bill versions (showing only findings with version history)",
        "4. **Rubric Anchor** indicates the scoring guideline used (e.g., '6-8: Significant obligations' means the score falls in that range)",
        "",
        "---",
        "",
        "## Change Detection Summary",
        "",
        f"- **New Bills:** {change_summary.get('new_bills', 0)} (first time seen)",
        f"- **Modified Bills:** {change_summary.get('modified_bills', 0)} (text/status/amendments changed)",
        f"- **Unchanged Bills:** {change_summary.get('unchanged_bills', 0)}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total Items Analyzed:** {results.get('bills_analyzed', 0)}",
        f"- **High Impact (7-10):** {len(high_impact)} items - IMMEDIATE ACTION REQUIRED" if high_impact else f"- **High Impact (7-10):** 0 items",
        f"- **Medium Impact (4-6):** {len(medium_impact)} items - MONITOR CLOSELY" if medium_impact else f"- **Medium Impact (4-6):** 0 items",
        f"- **Low Impact (0-3):** {len(low_impact)} items - AWARENESS ONLY" if low_impact else f"- **Low Impact (0-3):** 0 items",
        "",
        "**Pipeline Details:**",
        f"- Architecture: V2 (Sequential Evolution + Two-Tier Validation)",
        f"- Provider: {results.get('provider', 'Unknown')}",
        f"- Total Time: {results.get('elapsed_seconds', 0):.1f}s",
        "",
        "---",
        "",
    ]

    def generate_bill_section(bill_result: dict, section_num: int) -> list:
        """Generate markdown for a single bill."""
        section_lines = []

        bill_id = bill_result.get("bill_id", "Unknown")
        title = bill_result.get("bill_title", "No title")
        impact_score = bill_result["_impact_score"]
        impact_emoji = bill_result["_impact_emoji"]
        recommended_action = bill_result["_recommended_action"]
        risk_type = get_risk_type(bill_result)

        # Get change status from change_data
        change_data = bill_result.get("change_data", {})
        if change_data.get("is_new"):
            change_badge = "NEW"
            change_emoji = "🆕"
        elif change_data.get("has_changes"):
            change_badge = "MODIFIED"
            change_emoji = "📝"
        else:
            change_badge = "UNCHANGED"
            change_emoji = "✓"

        # Determine risk/opportunity
        risk_or_opportunity = "risk"  # Default, can be enhanced with LLM analysis later

        section_lines.extend([
            f"### {section_num}. {bill_id} - {title}",
            "",
            f"**Change Status:** {change_emoji} {change_badge}",
            "",
        ])

        # If modified, show what changed
        if change_data.get("has_changes") and change_data.get("changes"):
            changes = change_data.get("changes", [])
            change_details = []
            for change in changes:
                change_type = change.get("type", "unknown")
                if change_type == CHANGE_TYPE_TEXT:
                    change_details.append("text")
                elif change_type == CHANGE_TYPE_STATUS:
                    old_val = change.get("old_value", "?")
                    new_val = change.get("new_value", "?")
                    change_details.append(f"status ({old_val} -> {new_val})")
                elif change_type == CHANGE_TYPE_AMENDMENTS:
                    count = change.get("count", 0)
                    change_details.append(f"amendments (+{count})")
                else:
                    change_details.append(change_type)
            section_lines.append(f"*Changes detected: {', '.join(change_details)}*")
            section_lines.append("")

        section_lines.extend([
            "**Impact Assessment:**",
            "",
            f"- **Score:** {impact_score}/10 {'⚠️' if impact_score >= 7 else '🔔' if impact_score >= 4 else 'ℹ️'}",
            f"- **Type:** {risk_type.replace('_', ' ').title()}",
            f"- **Risk Level:** {risk_or_opportunity.upper()}",
            "",
            "**Bill Information:**",
            "",
            f"- **Source:** {bill_result.get('source', 'Unknown')}",
            f"- **Number:** {bill_id}",
        ])

        # Add optional metadata if available
        if bill_result.get("sponsor"):
            section_lines.append(f"- **Sponsor:** {bill_result.get('sponsor')}")
        if bill_result.get("status"):
            section_lines.append(f"- **Status:** {bill_result.get('status')}")
        if bill_result.get("introduced_date"):
            section_lines.append(f"- **Introduced:** {bill_result.get('introduced_date')}")
        if bill_result.get("url"):
            section_lines.append(f"- **Link:** {bill_result.get('url')}")

        # Add bill version being analyzed
        bill_version = bill_result.get("bill_version", "")
        if bill_version:
            section_lines.append(f"- **Version Analyzed:** {bill_version}")

        section_lines.extend([
            f"- **Route:** {bill_result.get('route', 'STANDARD')} (Complexity: {bill_result.get('complexity_score', 0)})",
            "",
        ])

        # NRG Business Verticals Section (prominent placement after Bill Info)
        nrg_verticals = bill_result.get("nrg_business_verticals", [])
        vertical_details = bill_result.get("nrg_vertical_impact_details", {})

        if nrg_verticals:
            section_lines.extend([
                "**NRG Business Verticals Affected:**",
                "",
            ])
            for vertical in nrg_verticals:
                section_lines.append(f"- {vertical}")
            section_lines.append("")

            # Impact by Business Vertical (detailed breakdown)
            if vertical_details:
                section_lines.extend([
                    "**Impact by Business Vertical:**",
                    "",
                ])
                for vertical, impact_desc in vertical_details.items():
                    section_lines.extend([
                        f"*{vertical}:*",
                        f"> {impact_desc}",
                        "",
                    ])

        # Version Timeline (if multiple versions)
        versions_processed = bill_result.get("versions_processed", 1)
        version_analyses = bill_result.get("version_analyses", [])

        # Always show version timeline with actual version info
        section_lines.extend([
            f"**📚 VERSION TIMELINE ({versions_processed} version{'s' if versions_processed != 1 else ''} analyzed)**",
            "",
        ])

        if version_analyses:
            section_lines.append(f"This bill has evolved through {versions_processed} legislative version{'s' if versions_processed != 1 else ''}. Each version was analyzed to track how NRG's risk profile changed:")
            section_lines.append("")
            for i, va in enumerate(version_analyses, 1):
                v_type = va.get("version_type", f"Version {i}")
                v_date = va.get("version_date", "")[:10] if va.get("version_date") else ""
                v_score = va.get("impact_score")
                date_str = f" ({v_date})" if v_date else ""
                score_str = f"{v_score}/10" if v_score is not None else "Pending"
                section_lines.append(f"{i}. **{v_type}**{date_str} - Impact Score: {score_str}")
            section_lines.append("")
        else:
            # Show bill version from metadata if available
            bill_versions = bill_result.get("versions", [])
            if bill_versions:
                section_lines.append("Legislative versions available:")
                section_lines.append("")
                for i, v in enumerate(bill_versions, 1):
                    v_type = v.get("version_type", f"Version {i}")
                    v_date = v.get("version_date", "")[:10] if v.get("version_date") else ""
                    word_count = v.get("word_count", 0)
                    date_str = f" ({v_date})" if v_date else ""
                    section_lines.append(f"{i}. **{v_type}**{date_str} - {word_count:,} words")
                section_lines.append("")
            else:
                section_lines.append("*Single version analyzed*")
                section_lines.append("")

        # Validation Summary
        verified = bill_result.get("verified_findings", 0)
        halluc = bill_result.get("hallucinations_detected", 0)
        total_findings = bill_result.get("findings_count", 0)

        section_lines.extend([
            "**Validation Summary:**",
            "",
            f"- Total Findings: {total_findings}",
            f"- ✓ Verified: {verified}",
            f"- ✗ Hallucinations Filtered: {halluc}",
            "",
        ])

        # Findings Section
        findings = bill_result.get("findings", [])
        validations = bill_result.get("validations", [])

        if findings:
            section_lines.extend([
                "**Findings:**",
                "",
                "*Findings are specific provisions extracted from bill text. They serve as evidence for the Rubric Assessment scores below.*",
                "",
            ])

            for i, finding in enumerate(findings, 1):
                validation = validations[i - 1] if i <= len(validations) else {}
                is_verified = validation.get("quote_verified") and not validation.get("hallucination_detected")
                is_halluc = validation.get("hallucination_detected", False)

                status_icon = "✓" if is_verified else "✗"
                status_text = " [Hallucination - Filtered]" if is_halluc else ""

                section_lines.extend([
                    f"**Finding {i}** {status_icon}{status_text}",
                    "",
                    f"> {finding.get('statement', 'No statement')}",
                    "",
                    f"- Confidence: {finding.get('confidence', 0):.2f}",
                    f"- Impact Estimate: {finding.get('impact_estimate', 0)}/10",
                ])

                # Add impact type if available
                impact_type = finding.get("impact_type", "")
                if impact_type:
                    section_lines.append(f"- Impact Type: {impact_type.replace('_', ' ').title()}")

                # Add affected verticals if available
                finding_verticals = finding.get("affected_verticals", [])
                if finding_verticals:
                    section_lines.append(f"- Affected Verticals: {', '.join(finding_verticals)}")

                # Evidence quality from validation
                if validation.get('evidence_quality'):
                    section_lines.append(f"- Evidence Quality: {validation.get('evidence_quality', 0):.2f}")

                # Supporting Quotes
                quotes = finding.get("quotes", [])
                if quotes:
                    section_lines.append("")
                    section_lines.append("*Supporting Evidence:*")
                    for quote in quotes[:3]:  # Limit to 3 quotes
                        text = quote.get("text", "")[:200]
                        section = quote.get("section", "?")
                        section_lines.append(f'> "{text}{"..." if len(quote.get("text", "")) > 200 else ""}" — §{section}')

                section_lines.append("")

        # Rubric Scores
        rubric_scores = bill_result.get("rubric_scores", [])
        if rubric_scores:
            section_lines.extend([
                "**Rubric Assessment (Four-Dimension Scoring):**",
                "",
                "| Dimension | Score | Rubric Anchor |",
                "|-----------|-------|---------------|",
            ])

            # Deduplicate and show each dimension once
            seen = {}
            for score in rubric_scores:
                dim = score.get("dimension", "")
                if dim and dim not in seen:
                    seen[dim] = score

            dim_display = {
                "legal_risk": "Legal Risk",
                "financial_impact": "Financial Impact",
                "operational_disruption": "Operational Disruption",
                "ambiguity_risk": "Ambiguity/Interpretive Risk"
            }

            for dim, score in seen.items():
                display_name = dim_display.get(dim, dim.replace("_", " ").title())
                s = score.get("score", 0)
                anchor = score.get("rubric_anchor", "")
                section_lines.append(f"| {display_name} | {s}/10 | {anchor} |")

            # Calculate and show average
            if seen:
                avg = round(sum(s.get("score", 0) for s in seen.values()) / len(seen), 1)
                section_lines.append(f"| **Overall Average** | **{avg}/10** | |")

            section_lines.append("")
            section_lines.append("*Note: Rubric Anchor shows the scoring guideline that applies (e.g., \"6-8: Significant obligations\" means the assessed score falls within that defined range).*")
            section_lines.append("")

            # Show rationales as "Why This Matters to NRG"
            section_lines.append("**Why This Matters to NRG:**")
            section_lines.append("")
            for dim, score in seen.items():
                display_name = dim_display.get(dim, dim.replace("_", " ").title())
                rationale = score.get("rationale", "No rationale provided")
                section_lines.append(f"- **{display_name}:** {rationale}")
            section_lines.append("")

        # Stability Analysis (if available)
        stability_data = bill_result.get("stability_analysis", {})
        if stability_data:
            # Only show findings that have stability scores
            total_findings = len(findings) if findings else 0
            tracked_findings = len(stability_data)

            section_lines.extend([
                "**Stability Analysis:**",
                "",
                f"*Tracking {tracked_findings} of {total_findings} findings across bill versions. Stability score indicates likelihood the provision will remain in final bill (0.20 = volatile/last-minute, 0.95 = stable across all versions).*",
                "",
            ])

            for finding_id, stability in stability_data.items():
                if isinstance(stability, dict):
                    score = stability.get("score", 0)
                    prediction = stability.get("prediction", "Unknown")
                    # Add interpretation
                    if score >= 0.85:
                        interpretation = "Stable - likely to persist"
                    elif score >= 0.60:
                        interpretation = "Moderately stable"
                    elif score >= 0.40:
                        interpretation = "Contentious - subject to change"
                    else:
                        interpretation = "Volatile - recently added"
                    section_lines.append(f"- {finding_id}: {score:.2f} → {interpretation}")
                else:
                    # Simple float value
                    if stability >= 0.85:
                        interpretation = "Stable"
                    elif stability >= 0.60:
                        interpretation = "Moderately stable"
                    elif stability >= 0.40:
                        interpretation = "Contentious"
                    else:
                        interpretation = "Volatile"
                    section_lines.append(f"- {finding_id}: {stability:.2f} → {interpretation}")
            section_lines.append("")

        # --- POC 2 Style Detailed Analysis Sections ---

        # Legal Code Changes
        legal_changes = bill_result.get("legal_code_changes", {})
        if legal_changes and any(legal_changes.values()):
            section_lines.extend([
                "**Legal Code Changes:**",
                "",
            ])
            if legal_changes.get("sections_added"):
                section_lines.append("- **Added:**")
                for s in legal_changes["sections_added"]:
                    section_lines.append(f"    - {s}")
            if legal_changes.get("sections_amended"):
                section_lines.append("- **Amended:**")
                for s in legal_changes["sections_amended"]:
                    section_lines.append(f"    - {s}")
            if legal_changes.get("sections_repealed"):
                section_lines.append("- **Repealed:**")
                for s in legal_changes["sections_repealed"]:
                    section_lines.append(f"    - {s}")
            if legal_changes.get("substance"):
                section_lines.extend([
                    "",
                    f"- **Substance:** {legal_changes['substance']}",
                ])
            section_lines.append("")

        # Application Scope
        app_scope = bill_result.get("application_scope", {})
        if app_scope and any(app_scope.values()):
            section_lines.extend([
                "**Application Scope:**",
                "",
            ])
            if app_scope.get("applies_to"):
                section_lines.append("- **Applies To:**")
                for item in app_scope["applies_to"]:
                    section_lines.append(f"    - {item}")
            if app_scope.get("exclusions"):
                section_lines.append("- **Exclusions:**")
                for item in app_scope["exclusions"]:
                    section_lines.append(f"    - {item}")
            if app_scope.get("geographic_scope"):
                section_lines.append(f"- **Geographic:** {app_scope['geographic_scope']}")
            section_lines.append("")

        # Effective Dates
        eff_dates = bill_result.get("effective_dates", [])
        if eff_dates:
            section_lines.extend([
                "**Effective Dates:**",
                "",
            ])
            for ed in eff_dates:
                date = ed.get("date", "Unknown")
                applies = ed.get("applies_to", "")
                section_lines.append(f"- {date}: {applies}")
            section_lines.append("")

        # Provision Types
        prov_types = bill_result.get("provision_types", {})
        if prov_types and any(prov_types.values()):
            section_lines.extend([
                "**Provision Types:**",
                "",
            ])
            if prov_types.get("mandatory"):
                section_lines.append(f"- **Mandatory ({len(prov_types['mandatory'])}):**")
                for p in prov_types["mandatory"][:3]:  # Limit to first 3
                    section_lines.append(f"    - {p[:150]}{'...' if len(p) > 150 else ''}")
            if prov_types.get("permissive"):
                section_lines.append(f"- **Permissive ({len(prov_types['permissive'])}):**")
                for p in prov_types["permissive"][:3]:
                    section_lines.append(f"    - {p[:150]}{'...' if len(p) > 150 else ''}")
            section_lines.append("")

        # Exceptions & Exemptions
        exc_exempt = bill_result.get("exceptions_and_exemptions", {})
        if exc_exempt and any(exc_exempt.values()):
            section_lines.extend([
                "**Exceptions & Exemptions:**",
                "",
            ])
            if exc_exempt.get("exceptions"):
                section_lines.append("- **Exceptions:**")
                for e in exc_exempt["exceptions"]:
                    section_lines.append(f"    - {e}")
            if exc_exempt.get("exemptions"):
                section_lines.append("- **Exemptions:**")
                for e in exc_exempt["exemptions"]:
                    section_lines.append(f"    - {e}")
            section_lines.append("")

        # Affected NRG Assets
        nrg_assets = bill_result.get("affected_nrg_assets", {})
        if nrg_assets and any(nrg_assets.values()):
            section_lines.extend([
                "**Affected NRG Assets:**",
                "",
            ])
            if nrg_assets.get("facilities"):
                section_lines.append("- **Facilities:**")
                for f in nrg_assets["facilities"]:
                    section_lines.append(f"    - {f}")
            if nrg_assets.get("markets"):
                section_lines.append("- **Markets:**")
                for m in nrg_assets["markets"]:
                    section_lines.append(f"    - {m}")
            if nrg_assets.get("business_units"):
                section_lines.append("- **Business Units:**")
                for bu in nrg_assets["business_units"]:
                    section_lines.append(f"    - {bu}")
            section_lines.append("")

        # Key Provisions
        key_prov = bill_result.get("key_provisions", [])
        if key_prov:
            section_lines.extend([
                "**Key Provisions Relevant to NRG:**",
                "",
            ])
            for kp in key_prov[:5]:  # Limit to 5
                section_lines.append(f"- {kp}")
            section_lines.append("")

        # Financial Estimate
        fin_est = bill_result.get("financial_estimate", "")
        if fin_est:
            section_lines.extend([
                f"**Financial Estimate:** {fin_est}",
                "",
            ])

        # Internal Stakeholders
        stakeholders = bill_result.get("internal_stakeholders", [])
        if stakeholders:
            section_lines.extend([
                "**Internal Stakeholders:**",
                "",
            ])
            for s in stakeholders:
                section_lines.append(f"- {s}")
            section_lines.append("")

        # --- End POC 2 Style Sections ---

        # Recommended Actions
        # Use LLM's recommended action if available, otherwise derive from score
        llm_action = bill_result.get("llm_recommended_action", "").upper()
        if llm_action in ["URGENT", "ENGAGE", "SUPPORT"]:
            recommended_action = "ENGAGE"
        elif llm_action == "IGNORE":
            recommended_action = "AWARENESS"

        action_emoji = "✅" if recommended_action == "ENGAGE" else "🔔" if recommended_action == "MONITOR" else "ℹ️"
        action_text = {
            "ENGAGE": "Immediate attention required - engage Government Affairs",
            "MONITOR": "Monitor closely and prepare response strategies",
            "AWARENESS": "For awareness only - no immediate action required"
        }

        section_lines.extend([
            "**Recommended Actions:**",
            "",
            f"- {action_emoji} **{recommended_action}** - {action_text.get(recommended_action, '')}",
        ])

        if recommended_action == "ENGAGE":
            section_lines.extend([
                "- Track legislative progress closely",
                "- Coordinate with internal stakeholders",
                "- Prepare impact assessment for leadership",
            ])
        elif recommended_action == "MONITOR":
            section_lines.extend([
                "- Add to weekly monitoring dashboard",
                "- Prepare contingency response if elevated",
            ])

        section_lines.extend(["", "---", ""])

        return section_lines

    # Generate High Impact Section
    if high_impact:
        lines.extend([
            "## 🔴 HIGH IMPACT ITEMS (Score: 7-10)",
            "",
            "*These items require immediate engagement with Government Affairs.*",
            "",
        ])
        for i, bill in enumerate(high_impact, 1):
            lines.extend(generate_bill_section(bill, i))

    # Generate Medium Impact Section
    if medium_impact:
        lines.extend([
            "## 🟡 MEDIUM IMPACT ITEMS (Score: 4-6)",
            "",
            "*Monitor these items and prepare response strategies.*",
            "",
        ])
        for i, bill in enumerate(medium_impact, 1):
            lines.extend(generate_bill_section(bill, i))

    # Generate Low Impact Section
    if low_impact:
        lines.extend([
            "## 🟢 LOW IMPACT ITEMS (Score: 0-3)",
            "",
            "*For awareness only - no immediate action required.*",
            "",
        ])
        for i, bill in enumerate(low_impact, 1):
            lines.extend(generate_bill_section(bill, i))

    # Footer / Next Steps
    lines.extend([
        "---",
        "",
        "## Next Steps",
        "",
        "### Immediate (This Week):",
        "- Government Affairs: Track high-impact bills committee assignments",
        "- Schedule cross-functional meeting to assess regulatory risks",
        "- Prepare initial cost impact assessments for high-impact items",
        "",
        "### Short-term (This Month):",
        "- Environmental Compliance: Assess compliance scenarios",
        "- Finance: Model financial impact ranges",
        "- Legal: Review regulatory exposure",
        "",
        "### Ongoing:",
        "- Continue monitoring via this tracker",
        "- Update analysis as bills progress through legislative process",
        "- Brief executive leadership monthly",
        "",
        "---",
        "",
        "**Report Details:**",
        f"- **Pipeline:** V2 (Sequential Evolution + Two-Tier Validation)",
        f"- **Provider:** {results.get('provider', 'Unknown')}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "*This report is generated automatically. For questions or to update tracking parameters, contact your Government Affairs team.*",
    ])

    # Clean up empty lines
    cleaned_lines = []
    for line in lines:
        if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(line)

    output_path.write_text("\n".join(cleaned_lines))
    logger.success(f"Markdown report: {output_path}")


def markdown_to_docx(md_path: Path, docx_path: Path, logger: PipelineLogger) -> bool:
    """Convert Markdown report to DOCX using pandoc."""
    # Check if pandoc is available
    try:
        result = subprocess.run(
            ['which', 'pandoc'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            logger.warning(f"pandoc not found. Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)")
            return False
    except Exception as e:
        logger.warning(f"Could not check for pandoc: {e}")
        return False

    # Convert Markdown to DOCX
    try:
        result = subprocess.run(
            ['pandoc', str(md_path), '-o', str(docx_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"pandoc conversion failed: {result.stderr}")
            return False
        logger.success(f"DOCX report: {docx_path}")
        return True
    except Exception as e:
        logger.warning(f"DOCX conversion error: {e}")
        return False


def generate_reports(results: dict, output_dir: Path, config: dict, logger: PipelineLogger) -> Path:
    """Generate all configured report formats in timestamped subdirectory.

    Creates folder: output_dir/nrg_analysis_YYYYMMDD_HHMMSS/
    With files:
      - nrg_analysis_YYYYMMDD_HHMMSS.json
      - nrg_analysis_YYYYMMDD_HHMMSS.md
      - nrg_analysis_YYYYMMDD_HHMMSS.docx
    """
    # Create timestamped subdirectory (matching POC 2 structure)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"nrg_analysis_{timestamp}"
    report_dir = output_dir / base_name
    report_dir.mkdir(parents=True, exist_ok=True)

    formats = config.get("output", {}).get("formats", {"json": True, "markdown": True, "docx": True})

    # Enhance results with executive summary (POC 2 format)
    all_results = results.get("results", [])

    # Calculate impact scores and categorize
    high_impact_count = 0
    medium_impact_count = 0
    low_impact_count = 0
    total_findings = 0
    total_verified = 0
    total_hallucinations = 0

    for bill_result in all_results:
        # Calculate overall impact score
        impact_score = calculate_overall_impact_score(bill_result)
        bill_result["overall_impact_score"] = impact_score

        # Categorize
        cat, emoji, color, action = get_impact_category(impact_score)
        bill_result["impact_category"] = cat
        bill_result["recommended_action"] = action

        if cat == "HIGH":
            high_impact_count += 1
        elif cat == "MEDIUM":
            medium_impact_count += 1
        else:
            low_impact_count += 1

        # Aggregate metrics
        total_findings += bill_result.get("findings_count", 0)
        total_verified += bill_result.get("verified_findings", 0)
        total_hallucinations += bill_result.get("hallucinations_detected", 0)

    # Add executive summary to results (POC 2 format)
    results["executive_summary"] = {
        "total_items_analyzed": results.get("bills_analyzed", 0),
        "high_impact_count": high_impact_count,
        "medium_impact_count": medium_impact_count,
        "low_impact_count": low_impact_count,
        "total_findings": total_findings,
        "verified_findings": total_verified,
        "hallucinations_filtered": total_hallucinations,
        "high_impact_action": "IMMEDIATE ACTION REQUIRED" if high_impact_count > 0 else None,
        "medium_impact_action": "MONITOR CLOSELY" if medium_impact_count > 0 else None,
        "low_impact_action": "AWARENESS ONLY" if low_impact_count > 0 else None,
    }

    # Display rich CLI results (POC 2 style)
    console = Console()
    console.print()
    console.print("═" * 47)
    console.print("  ANALYSIS RESULTS  ")
    console.print("═" * 47)
    console.print()

    # Sort by impact score (high first)
    sorted_results = sorted(all_results, key=lambda x: x.get("overall_impact_score", 0), reverse=True)

    for bill_result in sorted_results:
        # Build item dict for display_analysis
        item = {
            "source": bill_result.get("source", "Unknown"),
            "number": bill_result.get("bill_id", ""),
            "type": bill_result.get("type", "Bill"),
            "title": bill_result.get("bill_title", "No title"),
            "url": bill_result.get("url", ""),
            "status": bill_result.get("status", "Unknown"),
            "sponsor": bill_result.get("sponsor", "Unknown"),
            "introduced_date": bill_result.get("introduced_date", ""),
            "agency": bill_result.get("agency", ""),
            "policy_area": bill_result.get("policy_area", ""),
        }

        # Build analysis dict for display_analysis
        # Determine impact type from highest rubric dimension
        rubric_scores = bill_result.get("rubric_scores", [])
        impact_type = "operational"
        if rubric_scores:
            max_score = max(rubric_scores, key=lambda x: x.get("score", 0))
            dim = max_score.get("dimension", "")
            if "legal" in dim.lower():
                impact_type = "regulatory_compliance"
            elif "financial" in dim.lower():
                impact_type = "financial"
            elif "operational" in dim.lower():
                impact_type = "operational"
            elif "ambiguity" in dim.lower():
                impact_type = "strategic"

        # Generate impact summary from verticals and findings
        verticals = bill_result.get("nrg_business_verticals", [])
        impact_details = bill_result.get("nrg_vertical_impact_details", {})
        impact_summary = ""
        if impact_details:
            # Use first vertical's impact detail as summary
            for v, detail in impact_details.items():
                impact_summary = detail
                break
        if not impact_summary:
            findings = bill_result.get("findings", [])
            if findings:
                impact_summary = findings[0].get("statement", "")

        analysis = {
            "business_impact_score": bill_result.get("overall_impact_score", 0),
            "bill_version": bill_result.get("bill_version", "unknown"),
            "impact_type": impact_type,
            "risk_or_opportunity": "risk" if bill_result.get("overall_impact_score", 0) > 5 else "neutral",
            "impact_summary": impact_summary or "See findings for details.",
            "nrg_business_verticals": verticals,
            "legal_code_changes": bill_result.get("legal_code_changes", {}),
            "application_scope": bill_result.get("application_scope", {}),
            "effective_dates": bill_result.get("effective_dates", []),
            "affected_nrg_assets": bill_result.get("affected_nrg_assets", {}),
            "financial_impact": bill_result.get("financial_estimate", "Unknown"),
            "timeline": "See effective dates",
            "recommended_action": bill_result.get("llm_recommended_action", bill_result.get("recommended_action", "monitor")),
            "internal_stakeholders": bill_result.get("internal_stakeholders", []),
        }

        display_analysis(item, analysis)

    # Add report metadata
    results["report_metadata"] = {
        "report_version": "2.0",
        "format": "POC 2 Compatible",
        "generated_at": datetime.now().isoformat(),
        "report_id": base_name,
        "output_directory": str(report_dir),
    }

    # JSON (always)
    json_path = report_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"JSON report: {json_path}")

    # Markdown
    md_path = report_dir / f"{base_name}.md"
    if formats.get("markdown", True):
        generate_markdown_report(results, md_path, logger)

    # DOCX (convert from Markdown via pandoc)
    if formats.get("docx", True):
        docx_path = report_dir / f"{base_name}.docx"
        markdown_to_docx(md_path, docx_path, logger)

    # Save CLI log file
    log_path = report_dir / f"{base_name}.log"
    logger.save_log(log_path)
    logger.success(f"CLI log: {log_path}")

    logger.success(f"All reports saved to: {report_dir}/")
    return report_dir


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_bill_v2(bill: Dict[str, Any], config: dict, nrg_context: str, api_key: str, logger: PipelineLogger) -> Dict[str, Any]:
    """Run full V2 analysis pipeline on a single bill."""
    bill_id = bill.get("bill_id", bill.get("number", "Unknown"))
    bill_text = bill.get("bill_text", "")

    if not bill_text:
        logger.error(f"No bill text available for {bill_id}")
        return {"error": "No bill text available", "bill_id": bill_id}

    # Stage 0: Complexity Assessment
    route, complexity_score, complexity_reasons = assess_complexity(bill, config, logger)

    # Stage 1: Sequential Evolution
    logger.stage_header(1, "Sequential Evolution", "Walk versions chronologically, extract findings, track modifications")

    # Convert versions to BillVersion objects
    versions = []
    raw_versions = bill.get("versions", [])
    if raw_versions:
        for v in raw_versions:
            if v.get("full_text"):
                versions.append(BillVersion(
                    version_number=v["version_number"],
                    text=v["full_text"],
                    name=v.get("version_type", f"Version {v['version_number']}")
                ))
    else:
        versions = [BillVersion(version_number=1, text=bill_text, name="Current")]

    logger.verbose(f"Processing {len(versions)} versions sequentially...")

    try:
        evolution_agent = SequentialEvolutionAgent(api_key=api_key)
        logger.debug(f"Sequential Evolution Agent initialized for {bill_id}")

        evolution_result = evolution_agent.walk_versions(bill_id=bill_id, versions=versions)

        findings_count = len(evolution_result.findings_registry)
        logger.success(f"Extracted {findings_count} findings from {len(versions)} versions")

        # Debug: Log detailed LLM interactions if available
        if logger.level.value >= LogLevel.DEBUG.value:
            if hasattr(evolution_result, 'llm_interactions'):
                for interaction in evolution_result.llm_interactions:
                    logger.log_llm_interaction(
                        component="Sequential Evolution",
                        model=interaction.get("model", "unknown"),
                        prompt=interaction.get("prompt", ""),
                        response=interaction.get("response", ""),
                        prompt_tokens=interaction.get("prompt_tokens", 0),
                        completion_tokens=interaction.get("completion_tokens", 0),
                        cost=interaction.get("cost", 0.0)
                    )

            if hasattr(evolution_result, 'token_usage'):
                logger.log_token_usage(
                    component="Sequential Evolution",
                    prompt_tokens=evolution_result.token_usage.get("prompt_tokens", 0),
                    completion_tokens=evolution_result.token_usage.get("completion_tokens", 0),
                    total_tokens=evolution_result.token_usage.get("total_tokens", 0),
                    cost=evolution_result.token_usage.get("cost", 0.0),
                    model=evolution_result.token_usage.get("model", "")
                )

        # Log each finding with details
        # findings_registry is a dict: {"F1": {...}, "F2": {...}}
        for i, (finding_id, finding) in enumerate(evolution_result.findings_registry.items(), 1):
            # finding is a dict with keys: id, statement, quotes, confidence, impact_estimate, etc.
            quotes = finding.get("quotes", [])
            logger.trace_finding(
                i,
                finding.get("statement", str(finding_id)),
                quotes,
                finding.get("confidence", 0.5),
                finding.get("impact_estimate", 0)
            )

            # Log stability for this finding
            origin_version = finding.get("origin_version")
            mod_count = finding.get("modification_count", 0)
            if origin_version is not None:
                stability = evolution_result.stability_scores.get(finding_id, 0.5)
                prediction = "Very stable" if stability >= 0.9 else "Stable" if stability >= 0.7 else "May change" if stability >= 0.5 else "Volatile"
                logger.trace_stability(
                    finding_id,
                    stability,
                    origin_version,
                    mod_count,
                    prediction
                )

    except Exception as e:
        logger.error(f"Sequential Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "bill_id": bill_id}

    # Stage 2: Two-Tier Validation
    logger.stage_header(2, "Two-Tier Validation", "Validate findings, detect hallucinations, assign confidence")

    try:
        orchestrator = TwoTierOrchestrator(
            judge_api_key=api_key,
            enable_multi_sample=config.get("v2", {}).get("two_tier", {}).get("multi_sample", {}).get("enabled", True),
            enable_fallback=config.get("v2", {}).get("two_tier", {}).get("fallback", {}).get("enabled", True)
        )

        logger.debug(f"Two-Tier Orchestrator initialized for {bill_id}")

        result = orchestrator.validate(
            bill_id=bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context,
            findings_registry=evolution_result.findings_registry,
            stability_scores=evolution_result.stability_scores
        )

        # Debug: Log detailed LLM interactions if available
        if logger.level.value >= LogLevel.DEBUG.value:
            if hasattr(result, 'llm_interactions'):
                for interaction in result.llm_interactions:
                    logger.log_llm_interaction(
                        component=interaction.get("component", "Two-Tier Validation"),
                        model=interaction.get("model", "unknown"),
                        prompt=interaction.get("prompt", ""),
                        response=interaction.get("response", ""),
                        prompt_tokens=interaction.get("prompt_tokens", 0),
                        completion_tokens=interaction.get("completion_tokens", 0),
                        cost=interaction.get("cost", 0.0)
                    )

            if hasattr(result, 'token_usage'):
                logger.log_token_usage(
                    component="Two-Tier Validation (Judge)",
                    prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                    completion_tokens=result.token_usage.get("completion_tokens", 0),
                    total_tokens=result.token_usage.get("total_tokens", 0),
                    cost=result.token_usage.get("cost", 0.0),
                    model=result.token_usage.get("model", "")
                )

            if hasattr(result, 'multi_sample_interactions') and result.multi_sample_interactions:
                logger.log_token_usage(
                    component="Two-Tier Validation (Multi-Sample)",
                    prompt_tokens=result.multi_sample_interactions.get("prompt_tokens", 0),
                    completion_tokens=result.multi_sample_interactions.get("completion_tokens", 0),
                    total_tokens=result.multi_sample_interactions.get("total_tokens", 0),
                    cost=result.multi_sample_interactions.get("cost", 0.0),
                    model=result.multi_sample_interactions.get("model", "")
                )

            if hasattr(result, 'fallback_interactions') and result.fallback_interactions:
                logger.log_token_usage(
                    component="Two-Tier Validation (Fallback)",
                    prompt_tokens=result.fallback_interactions.get("prompt_tokens", 0),
                    completion_tokens=result.fallback_interactions.get("completion_tokens", 0),
                    total_tokens=result.fallback_interactions.get("total_tokens", 0),
                    cost=result.fallback_interactions.get("cost", 0.0),
                    model=result.fallback_interactions.get("model", "")
                )

        # Log validation for each finding
        for i, validation in enumerate(result.judge_validations, 1):
            reason = ""
            if validation.hallucination_detected:
                reason = "Claim about NRG impact not grounded in bill text"

            logger.trace_validation(
                i,
                validation.quote_verified,
                validation.hallucination_detected,
                validation.evidence_quality,
                validation.judge_confidence,
                reason
            )

        verified_count = sum(1 for v in result.judge_validations if v.quote_verified and not v.hallucination_detected)
        halluc_count = sum(1 for v in result.judge_validations if v.hallucination_detected)
        logger.success(f"Validated: {verified_count} verified, {halluc_count} hallucinations filtered")

        # Log decision about multi-sample or fallback
        if result.multi_sample_agreement is not None:
            logger.trace_decision(
                "two_tier_validation",
                f"Multi-sample check: {result.multi_sample_agreement:.1%} agreement",
                f"Ran {config.get('v2', {}).get('two_tier', {}).get('multi_sample', {}).get('samples', 3)} samples to verify consistency"
            )

        if result.second_model_reviewed:
            logger.trace_decision(
                "two_tier_validation",
                "Fallback model invoked",
                "Judge confidence was uncertain (0.6-0.8) and impact >= 7"
            )

    except Exception as e:
        logger.error(f"Two-Tier Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "bill_id": bill_id}

    # Stage 3: Rubric Scoring
    logger.stage_header(3, "Rubric Scoring", "Score validated findings on 4 dimensions: legal, financial, operational, ambiguity")

    # Debug: Log rubric scoring LLM interactions if available
    if logger.level.value >= LogLevel.DEBUG.value:
        if hasattr(result, 'rubric_llm_interactions'):
            for interaction in result.rubric_llm_interactions:
                logger.log_llm_interaction(
                    component=f"Rubric Scoring - {interaction.get('dimension', 'unknown')}",
                    model=interaction.get("model", "unknown"),
                    prompt=interaction.get("prompt", ""),
                    response=interaction.get("response", ""),
                    prompt_tokens=interaction.get("prompt_tokens", 0),
                    completion_tokens=interaction.get("completion_tokens", 0),
                    cost=interaction.get("cost", 0.0)
                )

        if hasattr(result, 'rubric_token_usage'):
            logger.log_token_usage(
                component="Rubric Scoring",
                prompt_tokens=result.rubric_token_usage.get("prompt_tokens", 0),
                completion_tokens=result.rubric_token_usage.get("completion_tokens", 0),
                total_tokens=result.rubric_token_usage.get("total_tokens", 0),
                cost=result.rubric_token_usage.get("cost", 0.0),
                model=result.rubric_token_usage.get("model", "")
            )

    rubric_scores = result.rubric_scores
    if rubric_scores:
        logger.verbose(f"Scoring {len(rubric_scores)} dimension assessments...")

        # Group and log by finding
        current_finding = None
        for score in rubric_scores:
            logger.trace_rubric_score(
                "",
                score.dimension,
                score.score,
                score.rationale,
                score.rubric_anchor
            )

        logger.success(f"Generated {len(rubric_scores)} rubric scores")

    # Build version analyses array from raw versions data
    version_analyses = []
    raw_versions = bill.get("versions", [])
    for v in raw_versions:
        version_analyses.append({
            "version_number": v.get("version_number", 0),
            "version_type": v.get("version_type", f"Version {v.get('version_number', 0)}"),
            "version_date": v.get("version_date", ""),
            "word_count": v.get("word_count", 0),
            "impact_score": None,  # Would be filled by per-version analysis if enabled
        })

    # Build stability analysis from evolution result
    stability_analysis = {}
    if hasattr(evolution_result, 'stability_scores') and evolution_result.stability_scores:
        for finding_id, score in evolution_result.stability_scores.items():
            prediction = "Very stable" if score >= 0.9 else "Stable" if score >= 0.7 else "May change" if score >= 0.5 else "Volatile"
            stability_analysis[finding_id] = {
                "score": score,
                "prediction": prediction
            }

    # Determine current bill version being analyzed
    latest_version = versions[-1] if versions else None
    bill_version_name = latest_version.name if latest_version else "Unknown"

    # Compile result with enhanced metadata (matching POC 2 format)
    analysis_result = {
        # Bill Metadata
        "bill_id": bill_id,
        "bill_title": bill.get("title", ""),
        "source": bill.get("source", ""),
        "type": bill.get("type", "State Bill"),
        "status": bill.get("status", "Unknown"),
        "sponsor": bill.get("sponsor", ""),
        "introduced_date": bill.get("introduced_date", ""),
        "url": bill.get("url", ""),
        "jurisdiction": bill.get("jurisdiction", ""),
        "bill_version": bill_version_name,  # Version being analyzed

        # Original bill item data for JSON output (POC 2 format)
        "item": {
            "source": bill.get("source", ""),
            "type": bill.get("type", ""),
            "number": bill.get("number", bill_id),
            "title": bill.get("title", ""),
            "url": bill.get("url", ""),
            "status": bill.get("status", ""),
            "sponsor": bill.get("sponsor", ""),
            "introduced_date": bill.get("introduced_date", ""),
            "bill_text": bill.get("bill_text", "")[:500] + "..." if len(bill.get("bill_text", "")) > 500 else bill.get("bill_text", ""),
            "versions": [
                {
                    "version_number": v.get("version_number", i + 1),
                    "version_type": v.get("version_type", f"Version {i + 1}"),
                    "version_date": v.get("version_date", ""),
                    "word_count": v.get("word_count", 0),
                    "text_hash": v.get("text_hash", ""),
                }
                for i, v in enumerate(bill.get("versions", []))
            ]
        },

        # Pipeline Metadata
        "route": route,
        "complexity_score": complexity_score,
        "complexity_reasons": complexity_reasons,
        "versions_processed": len(versions),

        # Version Tracking (POC 2 feature) - also store raw versions
        "versions": bill.get("versions", []),
        "version_analyses": version_analyses,

        # NRG Business Verticals (extracted from Sequential Evolution)
        "nrg_business_verticals": evolution_result.nrg_business_verticals,
        "nrg_vertical_impact_details": evolution_result.nrg_vertical_impact_details,

        # POC 2 style detailed analysis fields
        "legal_code_changes": evolution_result.legal_code_changes,
        "application_scope": evolution_result.application_scope,
        "effective_dates": evolution_result.effective_dates,
        "provision_types": evolution_result.provision_types,
        "exceptions_and_exemptions": evolution_result.exceptions_and_exemptions,
        "affected_nrg_assets": evolution_result.affected_nrg_assets,
        "key_provisions": evolution_result.key_provisions,
        "financial_estimate": evolution_result.financial_estimate,
        "llm_recommended_action": evolution_result.recommended_action,
        "internal_stakeholders": evolution_result.internal_stakeholders,

        # Findings
        "findings_count": len(result.primary_analysis.findings),
        "findings": [
            {
                "finding_id": f"F{i+1}",
                "statement": f.statement,
                "quotes": [{"text": q.text, "section": q.section} for q in f.quotes],
                "confidence": f.confidence,
                "impact_estimate": f.impact_estimate,
                "origin_version": getattr(f, 'origin_version', 1),
                "modification_count": getattr(f, 'modification_count', 0),
            }
            for i, f in enumerate(result.primary_analysis.findings)
        ],

        # Validations (Two-Tier)
        "validations": [
            {
                "finding_id": v.finding_id,
                "quote_verified": v.quote_verified,
                "hallucination_detected": v.hallucination_detected,
                "evidence_quality": v.evidence_quality,
                "judge_confidence": v.judge_confidence
            }
            for v in result.judge_validations
        ],

        # Rubric Scores (Four-Dimension)
        "rubric_scores": [
            {
                "dimension": s.dimension,
                "score": s.score,
                "rationale": s.rationale,  # Full rationale for JSON
                "rubric_anchor": s.rubric_anchor
            }
            for s in rubric_scores
        ],

        # Stability Analysis (Sequential Evolution feature)
        "stability_analysis": stability_analysis,

        # Validation Metadata
        "multi_sample_agreement": result.multi_sample_agreement,
        "second_model_reviewed": result.second_model_reviewed,

        # Summary Metrics
        "hallucinations_detected": halluc_count,
        "verified_findings": verified_count,

        # Change Detection (from pre-analysis phase)
        "change_data": bill.get("change_data", {}),
    }

    # Display summary panel
    logger.console.print(Panel(
        f"[bold]{bill_id}[/bold]: {bill.get('title', '')[:60]}...\n\n"
        f"Route: {route} (complexity: {complexity_score})\n"
        f"Versions: {len(versions)}\n\n"
        f"[bold]Findings:[/bold] {len(result.primary_analysis.findings)} total\n"
        f"  [green]Verified:[/green] {verified_count}\n"
        f"  [red]Hallucinations:[/red] {halluc_count}\n"
        f"  Rubric Scores: {len(rubric_scores)}",
        title=f"Analysis Complete: {bill_id}",
        border_style="green" if halluc_count == 0 else "yellow" if halluc_count < 3 else "red",
    ))

    return analysis_result


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for V2 pipeline."""
    parser = argparse.ArgumentParser(
        description="NRG Legislative Intelligence V2 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode (verbose output)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", help="Output directory for reports")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logger
    cli_level = "debug" if args.debug else None
    logger = get_logger_from_config(config, cli_level)

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path(config.get("output", {}).get("directory", "./reports"))

    # Initialize SQLite cache database
    db_path = config.get("cache", {}).get("database", "./data/nrg_cache.db")
    db_conn = None
    try:
        # Ensure data directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        db_conn = init_database(db_path)
        logger.info(f"Initialized cache database: {db_path}")
    except Exception as e:
        logger.warning(f"Could not initialize cache database: {e} - continuing without caching")
        db_conn = None

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Load NRG context
    nrg_context = load_nrg_context()

    # Show startup banner
    provider = config.get("llm", {}).get("provider", "openai")
    logger.console.print(Panel(
        f"[bold cyan]NRG Legislative Intelligence v2.0[/bold cyan]\n"
        f"Provider: {provider}\n"
        f"Architecture: Sequential Evolution + Two-Tier Validation\n"
        f"Logging: {logger.level.name}\n"
        f"Output: {output_dir}",
        title="Pipeline Starting",
        border_style="cyan",
    ))

    # Log startup banner to file
    logger._log_to_file("=" * 60)
    logger._log_to_file("NRG Legislative Intelligence v2.0")
    logger._log_to_file("=" * 60)
    logger._log_to_file(f"Provider: {provider}")
    logger._log_to_file(f"Architecture: Sequential Evolution + Two-Tier Validation")
    logger._log_to_file(f"Logging: {logger.level.name}")
    logger._log_to_file(f"Output: {output_dir}")
    logger._log_to_file("")

    start_time = time.time()

    # Fetch bills
    bills = fetch_bills_from_config(config, logger)
    if not bills:
        logger.error("No bills fetched - check config.yaml and API keys")
        sys.exit(1)

    # =========================================================================
    # CHANGE DETECTION: Compare fetched bills against cached versions
    # =========================================================================
    logger.console.print()
    logger.console.rule("[bold cyan]Change Detection[/bold cyan]")

    new_bills_count = 0
    modified_bills_count = 0
    unchanged_bills_count = 0

    for bill in bills:
        bill_id_for_cache = f"{bill['source']}:{bill['number']}"

        # Get cached version if database is available
        cached_bill = None
        if db_conn:
            cached_bill = get_cached_bill(bill_id_for_cache, db_conn)

        # Detect changes using existing change detection module
        change_data = detect_bill_changes(cached_bill, bill)

        # Store change data in the bill for later use in reports
        bill["change_data"] = change_data

        # Track counts
        if change_data.get("is_new"):
            new_bills_count += 1
            logger.verbose(f"  [green]NEW[/green] {bill['number']}: First time seen")
        elif change_data.get("has_changes"):
            modified_bills_count += 1
            # Log what changed
            changes = change_data.get("changes", [])
            change_types = [c.get("type", "unknown") for c in changes]
            logger.verbose(f"  [yellow]MODIFIED[/yellow] {bill['number']}: {', '.join(change_types)}")
        else:
            unchanged_bills_count += 1
            logger.verbose(f"  [dim]UNCHANGED[/dim] {bill['number']}: No changes detected")

    # Show change detection summary
    logger.console.print()
    logger.console.print(Panel(
        f"[bold]Change Detection Summary[/bold]\n\n"
        f"  [green]New Bills:[/green] {new_bills_count}\n"
        f"  [yellow]Modified Bills:[/yellow] {modified_bills_count}\n"
        f"  [dim]Unchanged Bills:[/dim] {unchanged_bills_count}",
        title="Change Detection",
        border_style="cyan",
    ))

    # Log to file
    logger._log_to_file("")
    logger._log_to_file("Change Detection Summary:")
    logger._log_to_file(f"  New Bills: {new_bills_count}")
    logger._log_to_file(f"  Modified Bills: {modified_bills_count}")
    logger._log_to_file(f"  Unchanged Bills: {unchanged_bills_count}")
    logger._log_to_file("")

    # Store change detection summary for reports
    change_detection_summary = {
        "new_bills": new_bills_count,
        "modified_bills": modified_bills_count,
        "unchanged_bills": unchanged_bills_count,
    }

    # Analyze each bill (still analyze ALL - optimization for later)
    logger.info(f"Analyzing {len(bills)} bills with V2 pipeline...")
    results = []
    cache_hits = 0
    cache_misses = 0

    for idx, bill in enumerate(bills, 1):
        bill_id = bill.get("bill_id", "Unknown")
        logger.console.print()
        logger.console.rule(f"[bold]({idx}/{len(bills)}) {bill_id}[/bold]")
        logger._log_to_file(f"\n{'='*60}")
        logger._log_to_file(f"({idx}/{len(bills)}) {bill_id}")
        logger._log_to_file(f"{'='*60}")

        # Check cache for existing bill data
        cached_bill = None
        if db_conn:
            try:
                cache_key = f"{bill.get('source', 'Unknown')}:{bill.get('number', bill_id)}"
                cached_bill = get_cached_bill(cache_key, db_conn)
                if cached_bill:
                    cache_hits += 1
                    logger.verbose(f"Found bill in cache (last checked: {cached_bill.get('last_checked', 'unknown')})")
                    # Note: For now we still analyze - change detection is a separate task
                    # Future: compare text_hash to detect changes
                else:
                    cache_misses += 1
                    logger.verbose("Bill not in cache - will cache after analysis")
            except Exception as cache_err:
                logger.debug(f"Cache lookup error: {cache_err}")

        analysis = analyze_bill_v2(bill, config, nrg_context, api_key, logger)
        results.append(analysis)

        # Save bill and analysis to cache
        if db_conn and not analysis.get("error"):
            try:
                # Prepare bill data for caching (include analysis summary)
                bill_for_cache = {
                    **bill,
                    "summary": bill.get("bill_text", "")[:5000],  # Store first 5000 chars for hashing
                    "analysis_summary": {
                        "findings_count": analysis.get("findings_count", 0),
                        "verified_findings": analysis.get("verified_findings", 0),
                        "hallucinations_detected": analysis.get("hallucinations_detected", 0),
                        "route": analysis.get("route", "STANDARD"),
                        "complexity_score": analysis.get("complexity_score", 0),
                    }
                }
                save_bill_to_cache(bill_for_cache, db_conn)
                logger.verbose(f"Saved {bill_id} to cache (with {len(bill.get('amendments', []))} amendments)")
            except Exception as save_err:
                logger.warning(f"Could not save to cache: {save_err}")

    elapsed = time.time() - start_time

    # Log cache statistics
    if db_conn:
        logger.info(f"Cache statistics: {cache_hits} hits, {cache_misses} misses")

    # Compile final output
    output = {
        "pipeline": "v2",
        "provider": provider,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": elapsed,
        "bills_analyzed": len(results),
        "change_detection_summary": change_detection_summary,
        "results": results,
    }

    # Generate reports
    logger.console.print()
    logger.console.rule("[bold]Generating Reports[/bold]")
    report_path = generate_reports(output, output_dir, config, logger)

    # Show final summary with impact categorization (POC 2 format)
    total_findings = sum(r.get("findings_count", 0) for r in results)
    total_verified = sum(r.get("verified_findings", 0) for r in results)
    total_halluc = sum(r.get("hallucinations_detected", 0) for r in results)

    # Calculate impact categories
    high_count = sum(1 for r in results if r.get("impact_category") == "HIGH")
    medium_count = sum(1 for r in results if r.get("impact_category") == "MEDIUM")
    low_count = sum(1 for r in results if r.get("impact_category") == "LOW")

    logger.console.print()
    logger.console.print(Panel(
        f"[bold green]Analysis Complete[/bold green]\n\n"
        f"[bold]Bills Analyzed:[/bold] {len(results)}\n"
        f"  🔴 High Impact (7-10): {high_count}\n"
        f"  🟡 Medium Impact (4-6): {medium_count}\n"
        f"  🟢 Low Impact (0-3): {low_count}\n\n"
        f"[bold]Findings:[/bold]\n"
        f"  Total: {total_findings}\n"
        f"  ✓ Verified: {total_verified}\n"
        f"  ✗ Hallucinations Filtered: {total_halluc}\n\n"
        f"[bold]Total Time:[/bold] {elapsed:.1f}s\n\n"
        f"[bold]Reports saved to:[/bold] {report_path}",
        title="Summary",
        border_style="green" if high_count == 0 else "yellow" if high_count < 2 else "red",
    ))

    # Log final summary to file
    logger._log_to_file("")
    logger._log_to_file("=" * 60)
    logger._log_to_file("ANALYSIS COMPLETE")
    logger._log_to_file("=" * 60)
    logger._log_to_file(f"Bills Analyzed: {len(results)}")
    logger._log_to_file(f"  🔴 High Impact (7-10): {high_count}")
    logger._log_to_file(f"  🟡 Medium Impact (4-6): {medium_count}")
    logger._log_to_file(f"  🟢 Low Impact (0-3): {low_count}")
    logger._log_to_file(f"Findings:")
    logger._log_to_file(f"  Total: {total_findings}")
    logger._log_to_file(f"  ✓ Verified: {total_verified}")
    logger._log_to_file(f"  ✗ Hallucinations Filtered: {total_halluc}")
    logger._log_to_file(f"Total Time: {elapsed:.1f}s")
    logger._log_to_file(f"Reports saved to: {report_path}")

    # Close database connection
    if db_conn:
        try:
            db_conn.close()
            logger.debug("Closed cache database connection")
        except Exception as e:
            logger.debug(f"Error closing database: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
