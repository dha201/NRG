from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# =============================================================================
# VERSION NORMALIZATION
# =============================================================================
# Maps source-specific version labels to normalized terms

VERSION_NORMALIZATION = {
    # Open States (Texas) → Normalized
    "Introduced": "introduced",
    "Engrossed": "passed_originating_chamber",
    "Enrolled": "enrolled",
    "House Committee Report": "committee_report",
    "Senate Committee Report": "committee_report",
    "Committee Report": "committee_report",
    
    # Congress.gov → Normalized
    "IH": "introduced",      # Introduced in House
    "IS": "introduced",      # Introduced in Senate
    "RH": "committee_report",  # Reported in House
    "RS": "committee_report",  # Reported in Senate
    "EH": "passed_originating_chamber",  # Engrossed in House
    "ES": "passed_originating_chamber",  # Engrossed in Senate
    "ENR": "enrolled",
}


def normalize_version_type(raw_type: str) -> str:
    return VERSION_NORMALIZATION.get(raw_type, raw_type.lower().replace(" ", "_"))


# =============================================================================
# CORE DATA MODELS
# =============================================================================

@dataclass
class BillVersion:
    """Single bill version (Introduced, Enrolled, etc).
    
    KEY DIFFERENCES vs Bill.summary:
    - full_text: Version-specific legislative text (CHANGES between versions)
    - Bill.summary: Static analysis text (SAME across all versions)
    
    Use Case: Track how bill text evolves through legislative process."""
    version_number: int
    version_type_raw: str          # Original: "Engrossed"
    version_type_normalized: str   # Normalized: "passed_originating_chamber"
    version_date: str
    full_text: str                  # Complete legislative text for THIS VERSION ONLY
    text_hash: str
    word_count: int
    pdf_url: Optional[str] = None
    text_url: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "BillVersion":
        raw_type = data.get('version_type', data.get('note', 'Unknown'))
        return cls(
            version_number=data.get('version_number', 0),
            version_type_raw=raw_type,
            version_type_normalized=normalize_version_type(raw_type),
            version_date=data.get('version_date', data.get('date', '')),
            full_text=data.get('full_text', ''),
            text_hash=data.get('text_hash', ''),
            word_count=data.get('word_count', 0),
            pdf_url=data.get('pdf_url'),
            text_url=data.get('text_url'),
        )


@dataclass
class Bill:
    """Main bill metadata with analysis input.
    
    KEY DIFFERENCES vs BillVersion.full_text:
    - summary: Static text for LLM analysis (SAME across all versions)
    - BillVersion.full_text: Version-specific text (CHANGES between versions)
    
    Use Case: Primary input for business impact analysis."""
    source: str                    # "Congress.gov", "OpenStates", etc.
    bill_type: str                 # "Federal Bill", "State Bill", "Regulation"
    number: str                    # "H.R. 1234", "HB 4238"
    title: str
    status: str
    url: str
    summary: str                   # Bill text/summary for analysis (STATIC across versions)
    
    # Optional metadata
    sponsor: Optional[str] = None   # the legislator who introduces the bill
                                    # Congress.gov bills: Uses sponsor.fullName (line 287)
                                    # OpenStates bills: Uses first item from sponsorships array (lines 936, 1057)
                                    # Falls back to "Unknown" if no sponsor data
    introduced_date: Optional[str] = None
    policy_area: Optional[str] = None
    state: Optional[str] = None
    
    # Version tracking
    versions: list = field(default_factory=list)
    openstates_id: Optional[str] = None
    congress_num: Optional[str] = None
    bill_type_code: Optional[str] = None
    bill_number: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "Bill":
        versions = [
            BillVersion.from_dict(v) if isinstance(v, dict) else v 
            for v in data.get('versions', [])
        ]
        return cls(
            source=data.get('source', 'Unknown'),
            bill_type=data.get('type', 'Bill'),
            number=data.get('number', 'Unknown'),
            title=data.get('title', 'No title'),
            status=data.get('status', 'Unknown'),
            url=data.get('url', ''),
            summary=data.get('summary', ''),
            sponsor=data.get('sponsor'),
            introduced_date=data.get('introduced_date'),
            policy_area=data.get('policy_area'),
            state=data.get('state'),
            versions=versions,
            openstates_id=data.get('openstates_id'),
            congress_num=data.get('congress_num'),
            bill_type_code=data.get('bill_type'),
            bill_number=data.get('bill_number'),
        )
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'type': self.bill_type,
            'number': self.number,
            'title': self.title,
            'status': self.status,
            'url': self.url,
            'summary': self.summary,
            'sponsor': self.sponsor,
            'introduced_date': self.introduced_date,
            'policy_area': self.policy_area,
            'state': self.state,
            'versions': [v.__dict__ if hasattr(v, '__dict__') else v for v in self.versions],
            'openstates_id': self.openstates_id,
            'congress_num': self.congress_num,
            'bill_type': self.bill_type_code,
            'bill_number': self.bill_number,
        }


@dataclass
class Analysis:
    """ LLM analysis result for a bill or version."""
    business_impact_score: int     # 0-10
    impact_type: str               # regulatory_compliance, financial, operational, etc.
    impact_summary: str
    recommended_action: str        # ignore, monitor, engage, urgent
    
    # Detailed analysis fields
    bill_version: Optional[str] = None
    legal_code_changes: Optional[dict] = None
    application_scope: Optional[dict] = None
    effective_dates: Optional[list] = None
    mandatory_vs_permissive: Optional[dict] = None
    exceptions_and_exemptions: Optional[dict] = None
    nrg_business_verticals: list = field(default_factory=list)
    nrg_vertical_impact_details: Optional[list] = None
    nrg_relevant_excerpts: list = field(default_factory=list)
    affected_nrg_assets: Optional[dict] = None
    financial_impact: Optional[str] = None
    timeline: Optional[str] = None
    risk_or_opportunity: Optional[str] = None
    internal_stakeholders: list = field(default_factory=list)
    
    # Error tracking
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "Analysis":
        return cls(
            business_impact_score=data.get('business_impact_score', 0),
            impact_type=data.get('impact_type', 'unknown'),
            impact_summary=data.get('impact_summary', ''),
            recommended_action=data.get('recommended_action', 'monitor'),
            bill_version=data.get('bill_version'),
            legal_code_changes=data.get('legal_code_changes'),
            application_scope=data.get('application_scope'),
            effective_dates=data.get('effective_dates'),
            mandatory_vs_permissive=data.get('mandatory_vs_permissive'),
            exceptions_and_exemptions=data.get('exceptions_and_exemptions'),
            nrg_business_verticals=data.get('nrg_business_verticals', []),
            nrg_vertical_impact_details=data.get('nrg_vertical_impact_details'),
            nrg_relevant_excerpts=data.get('nrg_relevant_excerpts', []),
            affected_nrg_assets=data.get('affected_nrg_assets'),
            financial_impact=data.get('financial_impact'),
            timeline=data.get('timeline'),
            risk_or_opportunity=data.get('risk_or_opportunity'),
            internal_stakeholders=data.get('internal_stakeholders', []),
            error=data.get('error'),
        )
    
    def to_dict(self) -> dict:
        return {
            'business_impact_score': self.business_impact_score,
            'impact_type': self.impact_type,
            'impact_summary': self.impact_summary,
            'recommended_action': self.recommended_action,
            'bill_version': self.bill_version,
            'legal_code_changes': self.legal_code_changes,
            'application_scope': self.application_scope,
            'effective_dates': self.effective_dates,
            'mandatory_vs_permissive': self.mandatory_vs_permissive,
            'exceptions_and_exemptions': self.exceptions_and_exemptions,
            'nrg_business_verticals': self.nrg_business_verticals,
            'nrg_vertical_impact_details': self.nrg_vertical_impact_details,
            'nrg_relevant_excerpts': self.nrg_relevant_excerpts,
            'affected_nrg_assets': self.affected_nrg_assets,
            'financial_impact': self.financial_impact,
            'timeline': self.timeline,
            'risk_or_opportunity': self.risk_or_opportunity,
            'internal_stakeholders': self.internal_stakeholders,
            'error': self.error,
        }


@dataclass
class ChangeData:
    """ Changes between bill versions."""
    has_changes: bool
    is_new: bool
    change_type: str              # "new_bill", "modified", "unchanged"
    changes: list = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChangeData":
        if data is None:
            return cls(has_changes=False, is_new=False, change_type="unchanged")
        return cls(
            has_changes=data.get('has_changes', False),
            is_new=data.get('is_new', False),
            change_type=data.get('change_type', 'unchanged'),
            changes=data.get('changes', []),
        )


@dataclass
class AnalysisResult:
    """Complete analysis result for a bill, including version analyses."""
    item: dict                              # Original bill dict (for backward compatibility)
    analysis: dict                          # Primary analysis result
    change_data: Optional[dict] = None
    change_impact: Optional[dict] = None
    version_analyses: list = field(default_factory=list)
    version_diffs: list = field(default_factory=list)
