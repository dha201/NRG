# nrg_core/v2/judge.py
"""
Judge Model (Tier 2) - Validates findings, detects hallucinations, scores evidence quality.

Design Decisions:
- Separate from Primary Analyst to enable independent validation
- Lower temperature (0.1) than analyst for more deterministic validation
- Explicit hallucination detection to catch unsupported claims
- Evidence quality and ambiguity scores enable filtering/prioritization

Validation Checks:
1. quote_verified: Do all quoted texts exist VERBATIM in bill?
2. hallucination_detected: Does statement claim things not in quotes?
3. evidence_quality: How well do quotes support the statement?
4. ambiguity: How much interpretation is required?

Why This Matters:
- Primary analyst may hallucinate (claim things not in bill)
- Quotes may be paraphrased or slightly modified
- Statements may extrapolate beyond what quotes actually say
- Judge catches these issues before findings reach downstream consumers
"""
from typing import Dict, Any
from openai import OpenAI
from nrg_core.models_v2 import Finding, JudgeValidation


# Prompt for validation - emphasizes verbatim matching and hallucination detection
JUDGE_VALIDATION_PROMPT = """You are a validation judge for legislative analysis.

BILL TEXT:
{bill_text}

FINDING TO VALIDATE:
Statement: {statement}
Quotes: {quotes}

TASK:
Validate this finding by checking:
1. Do all quoted texts exist VERBATIM in the bill? (exact match required)
2. Does the statement make claims NOT supported by the quotes?
3. How strong is the evidence (0-1)?
4. How ambiguous is the interpretation (0-1)?

OUTPUT (JSON):
{{
  "quote_verified": true/false,
  "hallucination_detected": true/false,
  "evidence_quality": 0.0-1.0,
  "ambiguity": 0.0-1.0,
  "judge_confidence": 0.0-1.0
}}

CRITICAL:
- quote_verified = false if ANY quote is not exact match
- hallucination_detected = true if statement claims things not in quotes
- evidence_quality = how well quotes support the statement
- ambiguity = how much interpretation is required
"""


class JudgeModel:
    """
    Tier 2: Judge - Validates findings from primary analyst.
    
    Responsibilities:
    - Verify quotes exist verbatim in bill text
    - Detect hallucinations (unsupported claims)
    - Score evidence quality and ambiguity
    - Score findings on rubric dimensions (Task 5)
    
    Usage:
        judge = JudgeModel(api_key=os.getenv("OPENAI_API_KEY"))
        validation = judge.validate(finding_id, finding, bill_text)
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        """
        Initialize judge with LLM configuration.
        
        Args:
            model: OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key (required for actual calls, optional for testing)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def validate(
        self,
        finding_id: int,
        finding: Finding,
        bill_text: str
    ) -> JudgeValidation:
        """
        Validate a single finding from primary analyst.
        
        Checks:
        - Do quotes exist verbatim in bill?
        - Does statement add claims not in quotes (hallucination)?
        - How strong is the evidence?
        - How much interpretation required?
        
        Args:
            finding_id: Index of finding in primary analysis
            finding: Finding to validate
            bill_text: Full bill text for verification
        
        Returns:
            JudgeValidation with verification results
        """
        # Call LLM for validation
        validation_output = self._call_llm(finding, bill_text)
        
        return JudgeValidation(
            finding_id=finding_id,
            quote_verified=validation_output["quote_verified"],
            hallucination_detected=validation_output["hallucination_detected"],
            evidence_quality=validation_output["evidence_quality"],
            ambiguity=validation_output["ambiguity"],
            judge_confidence=validation_output["judge_confidence"]
        )
    
    def score_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Score a finding on a rubric dimension.
        
        Placeholder - implemented in Task 5 (Rubric Scoring).
        
        Args:
            dimension: "legal_risk" or "financial_impact"
            finding: Finding to score
            bill_text: Full bill text
            nrg_context: NRG business context
            rubric_anchors: Scale definitions
        
        Returns:
            Dict with score, rationale, evidence, rubric_anchor
        """
        # Will implement in Task 5
        pass
    
    def _call_llm(self, finding: Finding, bill_text: str) -> Dict[str, Any]:
        """
        Call OpenAI for validation.
        
        Args:
            finding: Finding to validate
            bill_text: Full bill text
            
        Returns:
            Parsed JSON with validation results
            
        Raises:
            ValueError: If client not initialized
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Format quotes for prompt
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        
        prompt = JUDGE_VALIDATION_PROMPT.format(
            bill_text=bill_text,
            statement=finding.statement,
            quotes=quotes_text
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for validation (more deterministic)
        )
        
        import json
        return json.loads(response.choices[0].message.content)
