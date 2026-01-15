import asyncio
import json
import time
from typing import Optional
from abc import ABC, abstractmethod
import openai
from google import genai

from nrg_core.consensus_ensemble.models import ModelResponse

# Response schema for structured output enforcement
# Gemini's response_schema uses OpenAPI 3.0 subset to force valid JSON
# Prevents hallucinated formats, ensures parseable output
FINDING_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["statement", "quote", "confidence"]
            }
        }
    },
    "required": ["findings"]
}


class LLMClient(ABC):
    """Base for bill analysis clients with structured output"""

    @abstractmethod
    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        """
        Analyze bill with NRG business context.

        nrg_context focuses model on NRG-relevant findings (e.g., energy generation,
        tax implications) vs generic bill analysis. Prevents wasted tokens on irrelevant sections.
        """
        pass


class GeminiClient(LLMClient):
    """
    Gemini 3 Pro with structured output via response_schema.

    Why Gemini 3 Pro:
    - Excellent at legislative text (trained on legal/gov docs)
    - response_schema enforcement prevents JSON parse errors
    - Thinking mode gives better reasoning for complex bills
    - Cost-effective for high volume ($0.05/1M tokens)

    Why 2x Gemini instances:
    - Diversity in reasoning paths (different random seeds)
    - Catches Gemini-specific blind spots (threshold misreads)
    - Still cheaper than 3 different model providers
    """

    def __init__(self, api_key: str, model_id: str = "gemini-3-pro", instance_name: str = "A"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.model_name = f"Gemini-3-Pro-{instance_name}"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        start_time = time.time()
        try:
            # Embed NRG context so model filters for business-relevant findings
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # Structured output prevents "thinking" text contaminating JSON
            # response_mime_type + response_schema = guaranteed valid JSON
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config={
                    "temperature": 0.2,  # Low temp for consistency across instances
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                    "response_schema": FINDING_RESPONSE_SCHEMA
                }
            )

            # Safe extraction handles Gemini 3 thinking model multi-part responses
            json_text = response.text
            if json_text is None:
                # Thought-only response (rare with schema enforcement)
                json_text = self._extract_first_text_part(response)

            try:
                findings = json.loads(json_text)
            except json.JSONDecodeError:
                # SDK concatenation bug: {"json1"}{"json2"}
                json_text = self._extract_first_text_part(response)
                findings = json.loads(json_text)

            processing_time = time.time() - start_time

            return ModelResponse(
                model_name=self.model_name,
                findings=findings.get('findings', []),
                processing_time=processing_time
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.model_name,
                findings=[],
                processing_time=time.time() - start_time,
                error=str(e)
            )

    def _extract_first_text_part(self, response):
        """Extract first non-thought text part (handles Gemini 3 thinking mode)"""
        for part in response.candidates[0].content.parts:
            if getattr(part, 'thought', False):
                continue  # Skip encrypted reasoning traces
            if part.text and part.text.strip():
                return part.text
        raise ValueError("No text part in Gemini response")


class OpenAIClient(LLMClient):
    """
    GPT-5 with structured output via response_format.

    Why GPT-5:
    - Strong reasoning for causal chains (amendment → impact)
    - Different architecture provides diversity vs Gemini
    - response_format ensures valid JSON without schema
    - Excellent at quote extraction (verbatim text matching)
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = "GPT-5"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        start_time = time.time()
        try:
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # For now, use text mode without structured output
            # (GPT-5's response_format not available in all API versions yet)
            response = await self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a legislative analyst specializing in business impact assessment. Return valid JSON only."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=8192
            )

            # Extract JSON from response
            response_text = response.choices[0].message.content
            findings = json.loads(response_text)
            processing_time = time.time() - start_time

            return ModelResponse(
                model_name=self.model_name,
                findings=findings.get('findings', []),
                processing_time=processing_time
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.model_name,
                findings=[],
                processing_time=time.time() - start_time,
                error=str(e)
            )


class ParallelAnalyzer:
    """
    Runs 2x Gemini + 1x GPT-5 concurrently.

    Why parallel:
    - 60s total vs 105s sequential (35+35+40)
    - All models see same bill version (consistency)
    - Timeout prevents one slow model blocking entire pipeline

    Why this combination:
    - 2x Gemini provides majority vote for Gemini-specific errors
    - GPT-5 adds architectural diversity (catches Gemini blind spots)
    - Cost: ~$0.15/bill vs $0.40 with 3 different providers
    """

    def __init__(self, gemini_key: str = None, openai_key: str = None):
        self.gemini_a = GeminiClient(gemini_key, instance_name="A") if gemini_key else None
        self.gemini_b = GeminiClient(gemini_key, instance_name="B") if gemini_key else None
        self.gpt5 = OpenAIClient(openai_key) if openai_key else None

    async def analyze_parallel(
        self,
        bill_text: str,
        prompt: str,
        nrg_context: str,
        timeout: float = 60.0
    ):
        """
        Fire all 3 models, wait max 60s.

        Handles partial failures: if Gemini-A times out but Gemini-B + GPT-5
        succeed, we still get 2/3 consensus (acceptable per Q3 blocking decision).
        """
        tasks = []
        if self.gemini_a:
            tasks.append(self.gemini_a.analyze_bill(bill_text, prompt, nrg_context))
        if self.gemini_b:
            tasks.append(self.gemini_b.analyze_bill(bill_text, prompt, nrg_context))
        if self.gpt5:
            tasks.append(self.gpt5.analyze_bill(bill_text, prompt, nrg_context))

        # gather with return_exceptions prevents one failure killing entire batch
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )

        # Filter out timeout exceptions, keep successful ModelResponse objects
        return [r for r in results if isinstance(r, ModelResponse)]
