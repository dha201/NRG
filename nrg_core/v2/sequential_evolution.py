"""
Sequential Evolution Agent.

Single agent walks bill versions with structured memory.
Tracks finding origins, modifications, and stability over time.

Design:
- Process versions chronologically
- Maintain findings registry with IDs (F1, F2, etc.)
- Track origin version and modification count
- Compute stability scores based on survival and changes

Why:
- Bills evolve through multiple versions
- Need to track which findings survive changes
- Stability indicates reliability of findings
- Memory enables consistent tracking across versions

Stability Formula:
- origin=1, mods=0 → 0.95 (survived all, stable)
- origin=1, mods=1 → 0.85 (one refinement)
- origin=1, mods=3+ → 0.40 (contentious)
- origin=N (last version) → 0.20 (last-minute, risky)
"""
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class BillVersion:
    """Single version of a bill.
    
    Attributes:
        version_number: Sequential version number (1, 2, 3...)
        text: Full text of this version
        name: Version name ("Introduced", "Engrossed", "Enrolled", etc.)
    """
    version_number: int
    text: str
    name: str  # "Introduced", "Engrossed", etc.


@dataclass
class EvolutionResult:
    """Result of sequential version walk.

    Attributes:
        bill_id: Bill identifier
        findings_registry: Dictionary of findings with metadata
        stability_scores: Stability score for each finding (0-1)
        nrg_business_verticals: List of NRG business verticals impacted by the bill
        nrg_vertical_impact_details: Dict mapping vertical name to impact description
    """
    bill_id: str
    findings_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    nrg_business_verticals: List[str] = field(default_factory=list)
    nrg_vertical_impact_details: Dict[str, str] = field(default_factory=dict)
    # POC 2 style detailed analysis fields
    legal_code_changes: Dict[str, Any] = field(default_factory=dict)
    application_scope: Dict[str, Any] = field(default_factory=dict)
    effective_dates: List[Dict[str, str]] = field(default_factory=list)
    provision_types: Dict[str, List[str]] = field(default_factory=dict)
    exceptions_and_exemptions: Dict[str, List[str]] = field(default_factory=dict)
    affected_nrg_assets: Dict[str, List[str]] = field(default_factory=dict)
    key_provisions: List[str] = field(default_factory=list)
    financial_estimate: str = ""
    recommended_action: str = "monitor"
    internal_stakeholders: List[str] = field(default_factory=list)


NRG_BUSINESS_VERTICALS = [
    "Upstream Oil & Gas Production",
    "Offshore Drilling & Production",
    "Refining Operations",
    "Retail Marketing & Fuel Distribution",
    "Pipelines & Midstream Infrastructure",
    "Natural Gas Markets & Trading",
    "Environmental Compliance & Air Quality",
    "Climate Policy & Carbon Management",
    "Carbon Capture & Storage (CCS)",
    "Hydrogen & Clean Fuels",
    "Renewable Natural Gas (RNG) & Biogas",
    "Biofuels & Sustainable Aviation Fuel (SAF)",
    "EV Charging Infrastructure",
    "Renewable Energy & Power Generation",
    "Tax Policy",
    "Trade & Tariffs",
    "Environmental Remediation & Liability",
    "Workforce & Labor Relations",
    "General Business & Corporate Governance",
    "Cyber & Critical Infrastructure Security",
]

EVOLUTION_PROMPT_V1 = """Analyze this bill (Version 1: {version_name}).

BILL TEXT:
{bill_text}

Extract findings that could impact NRG Energy's business.
Assign IDs (F1, F2, etc.) and include supporting quotes from the bill.

REQUIREMENTS:
1. Each finding MUST have at least one verbatim quote from the bill
2. Include the section reference for each quote
3. Estimate impact 0-10 (0=no impact, 10=existential threat)
4. Identify which NRG business verticals are impacted by this bill
5. Extract legal analysis details (code changes, application scope, effective dates)
6. Classify provisions as mandatory (SHALL/MUST) or permissive (MAY)

NRG BUSINESS VERTICALS (select from this exact list):
- Upstream Oil & Gas Production
- Offshore Drilling & Production
- Refining Operations
- Retail Marketing & Fuel Distribution
- Pipelines & Midstream Infrastructure
- Natural Gas Markets & Trading
- Environmental Compliance & Air Quality
- Climate Policy & Carbon Management
- Carbon Capture & Storage (CCS)
- Hydrogen & Clean Fuels
- Renewable Natural Gas (RNG) & Biogas
- Biofuels & Sustainable Aviation Fuel (SAF)
- EV Charging Infrastructure
- Renewable Energy & Power Generation
- Tax Policy
- Trade & Tariffs
- Environmental Remediation & Liability
- Workforce & Labor Relations
- General Business & Corporate Governance
- Cyber & Critical Infrastructure Security

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",
      "statement": "Clear, specific finding statement",
      "quotes": [
        {{"text": "Exact verbatim quote from bill", "section": "2.1"}}
      ],
      "origin_version": 1,
      "affected_sections": ["2.1"],
      "modification_count": 0,
      "impact_estimate": 7,
      "impact_type": "regulatory_compliance|financial|operational|market|strategic",
      "affected_verticals": ["<verticals this finding specifically impacts>"]
    }}
  ],
  "nrg_business_verticals": ["<select from the 20 verticals above>"],
  "nrg_vertical_impact_details": {{
    "<vertical_name>": "<specific impact on this vertical>"
  }},
  "legal_code_changes": {{
    "sections_added": ["<new sections created by this bill>"],
    "sections_amended": ["<existing code sections modified>"],
    "sections_repealed": ["<sections removed or repealed>"],
    "substance": "<summary of what the legal changes actually do>"
  }},
  "application_scope": {{
    "applies_to": ["<who this bill applies to: companies, agencies, individuals>"],
    "exclusions": ["<who is explicitly excluded>"],
    "geographic_scope": "<federal/state/specific regions>"
  }},
  "effective_dates": [
    {{"date": "<effective date or 'upon enactment'>", "applies_to": "<what becomes effective>"}}
  ],
  "provision_types": {{
    "mandatory": ["<provisions using SHALL/MUST language>"],
    "permissive": ["<provisions using MAY language or non-binding>"]
  }},
  "exceptions_and_exemptions": {{
    "exceptions": ["<exceptions where bill applies but person can prove exception>"],
    "exemptions": ["<exemptions where person is categorically exempt>"]
  }},
  "affected_nrg_assets": {{
    "facilities": ["<specific NRG facilities potentially affected>"],
    "markets": ["<geographic markets affected: Texas, Gulf of Mexico, etc.>"],
    "business_units": ["<NRG business units: NRGx energy, Archaea, etc.>"]
  }},
  "key_provisions": ["<direct quotes of the most NRG-relevant provisions with section refs>"],
  "financial_estimate": "<estimated financial impact to NRG if known>",
  "recommended_action": "ignore|monitor|engage|urgent|support",
  "internal_stakeholders": ["<NRG departments that should be involved>"]
}}

CRITICAL: Every statement must be supported by a direct quote from the bill."""

EVOLUTION_PROMPT_VN = """Analyze Version {version_number}: {version_name}.

PREVIOUS FINDINGS (from memory):
{previous_findings}

NEW VERSION TEXT:
{bill_text}

TASK:
1. Compare to previous findings
2. Mark findings as: STABLE (unchanged), MODIFIED (changed), or NEW
3. Update modification counts
4. Update quotes if text changed
5. Update NRG business verticals if impacted areas have changed

NRG BUSINESS VERTICALS (select from this exact list):
- Upstream Oil & Gas Production
- Offshore Drilling & Production
- Refining Operations
- Retail Marketing & Fuel Distribution
- Pipelines & Midstream Infrastructure
- Natural Gas Markets & Trading
- Environmental Compliance & Air Quality
- Climate Policy & Carbon Management
- Carbon Capture & Storage (CCS)
- Hydrogen & Clean Fuels
- Renewable Natural Gas (RNG) & Biogas
- Biofuels & Sustainable Aviation Fuel (SAF)
- EV Charging Infrastructure
- Renewable Energy & Power Generation
- Tax Policy
- Trade & Tariffs
- Environmental Remediation & Liability
- Workforce & Labor Relations
- General Business & Corporate Governance
- Cyber & Critical Infrastructure Security

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",
      "statement": "Clear, specific finding statement",
      "quotes": [
        {{"text": "Exact verbatim quote from bill", "section": "2.1"}}
      ],
      "origin_version": 1,
      "modification_count": 1,
      "status": "MODIFIED",
      "impact_estimate": 7
    }}
  ],
  "nrg_business_verticals": ["<select from the 20 verticals above>"],
  "nrg_vertical_impact_details": {{
    "<vertical_name>": "<specific impact on this vertical>"
  }}
}}

CRITICAL: Every statement must be supported by a direct quote from the bill."""


class SequentialEvolutionAgent:
    """
    Walk bill versions sequentially, maintaining structured memory.
    
    Process:
    1. Version 1: Extract initial findings with IDs
    2. Version N+: Compare to memory, update/track changes
    3. Compute stability scores based on survival and modifications
    
    Memory Structure:
    - findings_registry: {id: {statement, origin, mods, status}}
    - stability_scores: {id: score} based on formula
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """
        Initialize sequential evolution agent.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.memory: Dict[str, Any] = {}
    
    def walk_versions(
        self,
        bill_id: str,
        versions: List[BillVersion]
    ) -> EvolutionResult:
        """
        Walk versions in order, maintaining findings registry.

        Algorithm:
        1. For each version in chronological order
        2. If first version: extract initial findings
        3. If subsequent: compare to memory and update
        4. After all versions: compute stability scores
        5. Extract NRG business verticals from final analysis

        Args:
            bill_id: Bill identifier
            versions: List of BillVersion objects in chronological order

        Returns:
            EvolutionResult with findings registry, stability scores, and NRG verticals
        """
        findings_registry = {}
        nrg_business_verticals = []
        nrg_vertical_impact_details = {}
        # POC 2 style detailed analysis fields (captured from latest version)
        legal_code_changes = {}
        application_scope = {}
        effective_dates = []
        provision_types = {}
        exceptions_and_exemptions = {}
        affected_nrg_assets = {}
        key_provisions = []
        financial_estimate = ""
        recommended_action = "monitor"
        internal_stakeholders = []

        for idx, version in enumerate(versions):
            if idx == 0:
                # First version: extract initial findings
                analysis = self._analyze_version(version, is_first=True)
            else:
                # Subsequent versions: compare to memory
                analysis = self._analyze_version(version, is_first=False, memory=findings_registry)

            # Update registry
            for finding in analysis.get("findings", []):
                finding_id = finding["id"]
                findings_registry[finding_id] = finding

            # Capture NRG business verticals (use latest version's assessment)
            if "nrg_business_verticals" in analysis:
                nrg_business_verticals = analysis["nrg_business_verticals"]
            if "nrg_vertical_impact_details" in analysis:
                nrg_vertical_impact_details = analysis["nrg_vertical_impact_details"]

            # Capture POC 2 style detailed fields (use latest version's assessment)
            if "legal_code_changes" in analysis:
                legal_code_changes = analysis["legal_code_changes"]
            if "application_scope" in analysis:
                application_scope = analysis["application_scope"]
            if "effective_dates" in analysis:
                effective_dates = analysis["effective_dates"]
            if "provision_types" in analysis:
                provision_types = analysis["provision_types"]
            if "exceptions_and_exemptions" in analysis:
                exceptions_and_exemptions = analysis["exceptions_and_exemptions"]
            if "affected_nrg_assets" in analysis:
                affected_nrg_assets = analysis["affected_nrg_assets"]
            if "key_provisions" in analysis:
                key_provisions = analysis["key_provisions"]
            if "financial_estimate" in analysis:
                financial_estimate = analysis["financial_estimate"]
            if "recommended_action" in analysis:
                recommended_action = analysis["recommended_action"]
            if "internal_stakeholders" in analysis:
                internal_stakeholders = analysis["internal_stakeholders"]

        # Compute stability scores
        stability_scores = self._compute_stability(findings_registry, num_versions=len(versions))

        return EvolutionResult(
            bill_id=bill_id,
            findings_registry=findings_registry,
            stability_scores=stability_scores,
            nrg_business_verticals=nrg_business_verticals,
            nrg_vertical_impact_details=nrg_vertical_impact_details,
            legal_code_changes=legal_code_changes,
            application_scope=application_scope,
            effective_dates=effective_dates,
            provision_types=provision_types,
            exceptions_and_exemptions=exceptions_and_exemptions,
            affected_nrg_assets=affected_nrg_assets,
            key_provisions=key_provisions,
            financial_estimate=financial_estimate,
            recommended_action=recommended_action,
            internal_stakeholders=internal_stakeholders
        )
    
    def _analyze_version(
        self,
        version: BillVersion,
        is_first: bool,
        memory: Dict = None
    ) -> Dict[str, Any]:
        """
        Analyze single version with or without memory context.
        
        For first version: Use V1 prompt to extract initial findings
        For subsequent: Use VN prompt with previous findings context
        
        Args:
            version: Bill version to analyze
            is_first: Whether this is the first version
            memory: Previous findings registry (for subsequent versions)
        
        Returns:
            Parsed JSON response with findings
        """
        if is_first:
            prompt = EVOLUTION_PROMPT_V1.format(
                version_name=version.name,
                bill_text=version.text
            )
        else:
            previous_findings_str = json.dumps(list(memory.values()), indent=2)
            prompt = EVOLUTION_PROMPT_VN.format(
                version_number=version.version_number,
                version_name=version.name,
                previous_findings=previous_findings_str,
                bill_text=version.text
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse version analysis response as JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON for version analysis: {e}") from e
    
    def _compute_stability(self, registry: Dict, num_versions: int) -> Dict[str, float]:
        """
        Compute stability score for each finding.
        
        Stability formula reflects legislative reality:
        - Findings that survive unchanged are most reliable
        - Findings with few refinements are still reliable
        - Findings with many changes are contentious
        - Last-minute additions are risky
        
        Args:
            registry: Findings registry with origin and modification counts
            num_versions: Total number of versions processed
        
        Returns:
            Dictionary of stability scores (0-1)
        """
        scores = {}
        for finding_id, finding in registry.items():
            origin = finding.get("origin_version", 1)
            mods = finding.get("modification_count", 0)
            
            if origin == num_versions:
                # Last-minute addition
                score = 0.20
            elif mods == 0:
                # Never modified, very stable
                score = 0.95
            elif mods == 1:
                # One refinement
                score = 0.85
            elif mods == 2:
                # Two changes
                score = 0.70
            else:
                # 3+ modifications, contentious
                score = 0.40
            
            scores[finding_id] = score
        
        return scores
