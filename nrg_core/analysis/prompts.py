
# Gemini API response schema - enforces object output (not array)
# Using OpenAPI 3.0 schema subset supported by Gemini
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "bill_version": {"type": "string"},
        "business_impact_score": {"type": "integer"},
        "impact_type": {"type": "string"},
        "impact_summary": {"type": "string"},
        "legal_code_changes": {
            "type": "object",
            "properties": {
                "sections_amended": {"type": "array", "items": {"type": "string"}},
                "sections_added": {"type": "array", "items": {"type": "string"}},
                "sections_deleted": {"type": "array", "items": {"type": "string"}},
                "chapters_repealed": {"type": "array", "items": {"type": "string"}},
                "substance_of_changes": {"type": "string"}
            }
        },
        "application_scope": {
            "type": "object",
            "properties": {
                "applies_to": {"type": "array", "items": {"type": "string"}},
                "exclusions": {"type": "array", "items": {"type": "string"}},
                "geographic_scope": {"type": "array", "items": {"type": "string"}}
            }
        },
        "effective_dates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "applies_to": {"type": "string"}
                }
            }
        },
        "mandatory_vs_permissive": {
            "type": "object",
            "properties": {
                "mandatory_provisions": {"type": "array", "items": {"type": "string"}},
                "permissive_provisions": {"type": "array", "items": {"type": "string"}}
            }
        },
        "exceptions_and_exemptions": {
            "type": "object",
            "properties": {
                "exceptions": {"type": "array", "items": {"type": "string"}},
                "exemptions": {"type": "array", "items": {"type": "string"}}
            }
        },
        "nrg_business_verticals": {"type": "array", "items": {"type": "string"}},
        "nrg_vertical_impact_details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vertical": {"type": "string"},
                    "impact": {"type": "string"}
                }
            }
        },
        "nrg_relevant_excerpts": {"type": "array", "items": {"type": "string"}},
        "affected_nrg_assets": {
            "type": "object",
            "properties": {
                "generation_facilities": {"type": "array", "items": {"type": "string"}},
                "geographic_exposure": {"type": "array", "items": {"type": "string"}},
                "business_units": {"type": "array", "items": {"type": "string"}}
            }
        },
        "financial_impact": {"type": "string"},
        "timeline": {"type": "string"},
        "risk_or_opportunity": {"type": "string"},
        "recommended_action": {"type": "string"},
        "internal_stakeholders": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "business_impact_score",
        "impact_type",
        "impact_summary",
        "recommended_action"
    ]
}

# Human-readable JSON schema for prompts (non-Gemini LLMs)
ANALYSIS_JSON_SCHEMA = """{
  "bill_version": "as_filed | house_committee | passed_house | senate_committee | passed_senate | conference_committee | enrolled | unknown",
  "business_impact_score": <0-10 integer>,
  "impact_type": "regulatory_compliance | financial | operational | market | strategic | minimal",
  "impact_summary": "<2-3 sentences on NRG business impact>",

  "legal_code_changes": {
    "sections_amended": ["<Code Name> §<section>", ...],
    "sections_added": ["New Chapter X of <Code Name>", ...],
    "sections_deleted": ["<Code Name> §<section>", ...],
    "chapters_repealed": ["Chapter X of <Code Name>", ...],
    "substance_of_changes": "<Detailed explanation of what's changing from current law>"
  },

  "application_scope": {
    "applies_to": ["<who this bill applies to>", ...],
    "exclusions": ["<who/what is excluded>", ...],
    "geographic_scope": ["<states/regions>", ...]
  },

  "effective_dates": [
    {
      "date": "<date or 'upon passage' or 'unknown'>",
      "applies_to": "<which provisions>"
    }
  ],

  "mandatory_vs_permissive": {
    "mandatory_provisions": ["<SHALL/MUST/REQUIRED provisions>", ...],
    "permissive_provisions": ["<MAY/CAN/OPTIONAL provisions>", ...]
  },

  "exceptions_and_exemptions": {
    "exceptions": ["<exceptions - bill applies, person must prove exception applies>", ...],
    "exemptions": ["<exemptions - person is exempt, must be proven otherwise>", ...]
  },

  "nrg_business_verticals": ["<select from: Cyber and Grid Security, Data Privacy/Public Information Act, Disaster/Business Continuity, Electric Vehicles, Environmental/Water/Sustainability, Natural Gas, General Business, General Government, Renewables/Distributed Generation/Demand Response, Retail Commodity, Retail Non-commodity, Services, Tax, Transmission-Distribution, Wholesale Market Reforms, Artificial Intelligence, Economic Development/Workforce Development, Electric Generation, Public Utility Commission of Texas>"],

  "nrg_vertical_impact_details": [
    {
      "vertical": "<vertical_name>",
      "impact": "<specific impact on this vertical>"
    }
  ],

  "nrg_relevant_excerpts": ["<Section X: Direct quote of NRG-relevant provision>", ...],

  "affected_nrg_assets": {
    "generation_facilities": ["<specific facilities or types affected>", ...],
    "geographic_exposure": ["<markets/states>", ...],
    "business_units": ["<NRG business units>", ...]
  },

  "financial_impact": "<estimated cost/revenue impact or 'unknown'>",
  "timeline": "<when this matters>",
  "risk_or_opportunity": "risk | opportunity | mixed | neutral",
  "recommended_action": "ignore | monitor | engage | urgent",

  "internal_stakeholders": ["<NRG departments/teams to involve>", ...]
}"""


def build_analysis_prompt(item: dict, nrg_context: str) -> str:
    """
    Args:
        item: Legislative item dict with type, source, number, title, status, summary
        nrg_context: NRG business context string
        
    Returns:
        Complete prompt string for LLM analysis
    """
    return f"""You are a legislative analyst for NRG Energy's Government Affairs team. Your role is to provide deep, professional-grade legislative analysis on par of a law firm.

NRG BUSINESS CONTEXT:
{nrg_context}

CRITICAL INSTRUCTIONS:
1. Analyze this legislation with LEGAL PRECISION - identify specific code sections, mandatory vs permissive language, exceptions vs exemptions
2. Focus ONLY on portions relevant to NRG Energy - ignore unrelated sections
3. Map to NRG's business verticals (may be multiple)
4. Extract the substance of legal changes, not just summaries
5. Respond ONLY with valid JSON - no explanatory text before or after

REQUIRED JSON STRUCTURE:
{ANALYSIS_JSON_SCHEMA}

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
