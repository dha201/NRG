
class ConsensusPrompts:
    """
    Prompt templates for LLM bill analysis.

    Prompt design philosophy:
    - Explicit instructions prevent hallucinations (be precise, quote exactly)
    - Structured JSON format prevents parsing errors
    - NRG context slot filters for business-relevant findings
    - Task decomposition (analysis vs verification) improves accuracy

    See: nrg_core/analysis/prompts.py for existing NRG context patterns
    """

    @staticmethod
    def get_consensus_analysis_prompt() -> str:
        """
        Initial bill analysis prompt (run 3x in parallel).

        Design:
        - "Extract key findings" sets expectation scope (not full bill summary)
        - "Impact a business" filters for relevant provisions (not all text)
        - Structured JSON enforces machine-parseable output
        - "Exact quote" requirement combats hallucination
        - Confidence score enables downstream voting

        Weakness: Some models conflate "confidence in finding" with "confidence you have data"
        (i.e., may give high confidence to hallucinated findings they believe strongly in).
        Mitigated by: Quote verification task (Task 6) asks for evidence.
        """
        return """You are a legislative analyst specializing in business impact assessment.

Analyze the following bill and extract key findings that would impact businesses.

For EACH finding, provide:
1. statement: A clear, concise description of what the bill does
2. quote: An EXACT quote from the bill text supporting this finding
3. confidence: Your confidence this is accurate (0.0 to 1.0 scale)

CRITICAL: Only include findings with supporting quotes from the bill.
Do NOT infer, speculate, or include implied requirements.
Quote exact bill language - paraphrasing introduces error.

Return your response as valid JSON in this exact format:
{
  "findings": [
    {
      "statement": "Clear, specific description (1-2 sentences)",
      "quote": "Exact text from the bill (quote marks included if in original)",
      "confidence": 0.85
    }
  ]
}

Focus on:
- Tax implications and rates
- Regulatory requirements and deadlines
- Compliance obligations and reporting
- Financial impacts and costs
- Operational changes required
- Effective dates and transition periods
- Exemptions and exceptions
- Penalties and enforcement

Example finding:
{
  "statement": "Tax applies to energy generation exceeding 50 megawatts capacity",
  "quote": "Section 2.1(b): exceeding fifty megawatts capacity",
  "confidence": 0.95
}

Be conservative with confidence:
- 0.95+: Direct, unambiguous bill language
- 0.80-0.94: Clear but requires minor interpretation
- 0.60-0.79: Implied from multiple sections, some ambiguity
- <0.60: Speculative - generally exclude these"""

    @staticmethod
    def get_quote_verification_prompt(finding_statement: str) -> str:
        """
        Follow-up verification for disputed findings (vote 1/3, need evidence).

        Design:
        - Re-asks for exact quote (forces model to re-read, catches hallucinations)
        - "EXACT quote" + "NO_QUOTE_FOUND" forces binary (no wishy-washy answers)
        - JSON format enables automatic vote counting
        - Used only for disputed/low-confidence findings (saves tokens on unanimous)

        Example flow:
        - Gemini-A claims: "All energy companies affected"
        - Prompt: "Give me the exact bill quote supporting this"
        - Gemini-A response: "NO_QUOTE_FOUND" OR paraphrased text (not actual quote)
        - Verdict: Hallucination detected, lower confidence or exclude finding
        """
        return f"""You previously identified this finding:

"{finding_statement}"

Please provide the EXACT quote from the bill text that supports this finding.

Rules:
- Quote must be WORD-FOR-WORD from the bill
- Include section references if mentioned in quote
- If the exact quote doesn't exist in the bill, respond with "NO_QUOTE_FOUND"
- Do NOT paraphrase or infer - only exact text

Return your response as JSON:
{{
  "finding": "{finding_statement}",
  "exact_quote": "The exact text from the bill (or 'NO_QUOTE_FOUND')",
  "section_reference": "Section number if available",
  "confidence_in_quote": 0.95
}}"""
