"""
Rubric Definitions for Phase 1 - Standardized scoring scales.

Design Decisions:
- Four dimensions: legal_risk, financial_impact, operational_disruption, ambiguity_risk
- Anchors defined as ranges (0-2, 3-5, 6-8, 9-10) for consistent scoring
- Each anchor has clear, objective criteria to reduce inter-rater variability

Why Rubrics Matter:
- Consistency: Same finding should get same score across runs
- Auditability: Rationale explains why this score vs another
- Calibration: Can tune thresholds based on historical accuracy
- Prioritization: High scores surface most important findings
"""

# Legal Risk Rubric - Severity of new obligations and penalties
# Design: Focuses on what the bill REQUIRES, not what it might cost
LEGAL_RISK_RUBRIC = {
    "0-2": "No new obligations - cosmetic language changes only",
    "3-5": "Minor compliance - reporting, training, disclosure requirements",
    "6-8": "Significant obligations + penalties - new taxes, operational requirements, financial consequences",
    "9-10": "Existential threats - bans, license revocations, structural prohibitions"
}


# Financial Impact Rubric - Estimated dollar impact to NRG
# Design: Uses NRG's scale (38 GW portfolio) to calibrate thresholds
FINANCIAL_IMPACT_RUBRIC = {
    "0-2": "<$100K annual - negligible impact",
    "3-5": "$100K-$500K - minor but trackable cost",
    "6-8": "$500K-$5M - material to P&L",
    "9-10": ">$5M or revenue at risk - major financial exposure"
}


# Operational Disruption Rubric - Degree of operational changes required
# Design: Scales from no change to business model transformation
# Why separate from financial: Some changes are costly in effort/time, not just dollars
OPERATIONAL_DISRUPTION_RUBRIC = {
    "0-2": "No operational changes required",
    "3-5": "Process adjustments - training, reporting procedures",
    "6-8": "System changes - hiring, infrastructure, restructuring",
    "9-10": "Business model changes - asset divestitures, strategic pivots"
}


# Ambiguity Risk Rubric - Clarity of legislative language
# Design: Measures interpretation uncertainty and regulatory guidance needs
# Why this matters: Ambiguous bills create compliance risk even if text seems benign
AMBIGUITY_RISK_RUBRIC = {
    "0-2": "Clear, explicit language - no interpretation needed",
    "3-5": "Some ambiguity - interpretations clear from context",
    "6-8": "Significant ambiguity - regulatory guidance TBD",
    "9-10": "Vague, contradictory - novel legal concepts, high uncertainty"
}


# All rubrics consolidated for easy iteration
# Design: Dict allows dynamic scoring loop without hardcoding dimension names
ALL_RUBRICS = {
    "legal_risk": LEGAL_RISK_RUBRIC,
    "financial_impact": FINANCIAL_IMPACT_RUBRIC,
    "operational_disruption": OPERATIONAL_DISRUPTION_RUBRIC,
    "ambiguity_risk": AMBIGUITY_RISK_RUBRIC
}


# Prompt template for rubric scoring
# Design: Explicit requirements prevent common scoring errors:
# - "using the rubric scale above" anchors to defined thresholds
# - "minimum 50 chars" ensures substantive rationale
# - "cite the rubric anchor" creates audit trail
RUBRIC_SCORING_PROMPT = """You are scoring a legislative finding on the {dimension} dimension.

RUBRIC SCALE:
{rubric_scale}

NRG BUSINESS CONTEXT:
{nrg_context}

BILL TEXT:
{bill_text}

FINDING:
Statement: {statement}
Quotes: {quotes}

TASK:
Score this finding 0-10 on {dimension} using the rubric scale above.

REQUIREMENTS:
1. Select score that best matches rubric anchor
2. Provide detailed rationale (minimum 50 chars) explaining the score
3. Reference specific quotes as evidence
4. Cite the rubric anchor used

OUTPUT (JSON):
{{
  "dimension": "{dimension}",
  "score": 0-10,
  "rationale": "Detailed explanation with business impact calculation",
  "evidence": [{{"text": "quote", "section": "X.X", "page": null}}],
  "rubric_anchor": "X-Y: anchor text from rubric"
}}
"""


def format_rubric_scale(rubric: dict[str, str]) -> str:
    """
    Format rubric dict into readable scale for prompt insertion.
    
    Args:
        rubric: Dict mapping score ranges to descriptions
        
    Returns:
        Formatted string with one anchor per line
    """
    return "\n".join([f"{k}: {v}" for k, v in rubric.items()])
