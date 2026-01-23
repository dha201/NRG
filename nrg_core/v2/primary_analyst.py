# nrg_core/v2/primary_analyst.py
"""
Primary Analyst Agent (Tier 1) - Extracts findings from bill text with supporting quotes.

Design Decisions:
- Uses GPT-4o for high-quality extraction (could swap to cheaper model for STANDARD route)
- Structured JSON output enforces consistent response format
- Low temperature (0.2) for deterministic, reproducible analysis
- requires_multi_sample flag triggers Tier 1.5 when findings are high-stakes or uncertain

Why separate from Judge:
- Single Responsibility: Analyst extracts, Judge validates
- Enables different models for each role (e.g., analyst=GPT-4o, judge=Claude)
- Clearer audit trail of who made what claim
"""
import json
from typing import Dict, Any
from openai import OpenAI
from nrg_core.models_v2 import PrimaryAnalysis, Finding, Quote


# Prompt template for primary analyst
# Design: Explicit requirements prevent common LLM errors:
# - "verbatim" prevents paraphrasing quotes
# - "at least one quote" prevents unsupported claims
# - Impact scale 0-10 enables consistent ranking
PRIMARY_ANALYST_PROMPT = """You are a legislative analyst for NRG Energy.

BUSINESS CONTEXT:
{nrg_context}

TASK:
Analyze the following bill and identify ALL provisions that could impact NRG Energy's business.

BILL TEXT:
{bill_text}

REQUIREMENTS:
1. Extract specific findings (not general observations)
2. Each finding MUST have at least one supporting quote from the bill
3. Quote the exact text (verbatim) with section reference
4. Estimate impact 0-10 (0=no impact, 10=existential threat)
5. Provide confidence 0-1 for each finding

OUTPUT FORMAT (JSON):
{{
  "findings": [
    {{
      "statement": "Clear, specific finding",
      "quotes": [
        {{"text": "Exact quote from bill", "section": "2.1", "page": null}}
      ],
      "confidence": 0.85,
      "impact_estimate": 7
    }}
  ],
  "overall_confidence": 0.80
}}

CRITICAL:
- Every statement must be supported by a direct quote
- Do not infer provisions not explicitly stated
- Mark low confidence (<0.7) if interpretation is uncertain
"""


class PrimaryAnalyst:
    """
    Tier 1: Primary Analyst - Single-pass analysis extracting findings with evidence.
    
    Responsibilities:
    - Extract findings with mandatory supporting quotes
    - Estimate impact (0-10) and confidence (0-1)
    - Flag when multi-sample validation needed
    
    Usage:
        analyst = PrimaryAnalyst(api_key=os.getenv("OPENAI_API_KEY"))
        result = analyst.analyze(bill_id, bill_text, nrg_context)
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        """
        Initialize primary analyst with LLM configuration.
        
        Args:
            model: OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key (required for actual calls, optional for testing)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def analyze(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> PrimaryAnalysis:
        """
        Analyze bill and extract findings with supporting evidence.
        
        Process:
        1. Call LLM with structured prompt
        2. Parse findings into validated Pydantic models
        3. Determine if multi-sample needed based on impact/confidence
        
        Args:
            bill_id: Bill identifier (e.g., "HB123")
            bill_text: Full bill text
            nrg_context: NRG business context for relevance assessment
        
        Returns:
            PrimaryAnalysis with findings and metadata
        """
        # Call LLM for extraction
        llm_output = self._call_llm(bill_text, nrg_context)
        
        # Parse findings into validated models
        findings = [
            Finding(
                statement=f["statement"],
                quotes=[Quote(**q) for q in f["quotes"]],
                confidence=f["confidence"],
                impact_estimate=f["impact_estimate"]
            )
            for f in llm_output["findings"]
        ]
        
        # Determine if multi-sample needed
        # Trigger conditions: high impact (>=6) OR low confidence (<0.7)
        # Why: High-stakes or uncertain findings warrant additional validation
        requires_multi_sample = any(
            f.impact_estimate >= 6 or f.confidence < 0.7
            for f in findings
        )
        
        return PrimaryAnalysis(
            bill_id=bill_id,
            findings=findings,
            overall_confidence=llm_output["overall_confidence"],
            requires_multi_sample=requires_multi_sample
        )
    
    def _call_llm(self, bill_text: str, nrg_context: str) -> Dict[str, Any]:
        """
        Call OpenAI API with structured JSON output.
        
        Args:
            bill_text: Full bill text
            nrg_context: NRG business context
            
        Returns:
            Parsed JSON response with findings
            
        Raises:
            ValueError: If client not initialized
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        prompt = PRIMARY_ANALYST_PROMPT.format(
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2  # Low temperature for consistent extraction
        )
        
        return json.loads(response.choices[0].message.content)
