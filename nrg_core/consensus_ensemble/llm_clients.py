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

    @abstractmethod
    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        pass


class GeminiClient(LLMClient):
    
    def __init__(self, api_key: str, model_id: str = "gemini-3-pro", instance_name: str = "A"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.model_name = f"Gemini-3-Pro-{instance_name}"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        """
        Args:
            bill_text: Full text of the bill to analyze
            prompt: Analysis instructions with JSON schema definition
                   (from ConsensusPrompts.get_consensus_analysis_prompt() )
                   Should include: task description, output format, focus areas, examples
            nrg_context: NRG business context to filter for relevant findings
        """
        start_time = time.time()
        try:
            # Build complete prompt: instructions + context + bill text
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # Structured output prevents "thinking" text contaminating JSON
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config={
                    "temperature": 0.2,
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
        """ First non-thought text part, second part is the encrypted thinking part"""
        for part in response.candidates[0].content.parts:
            if getattr(part, 'thought', False):
                continue  # Skip encrypted reasoning traces
            if part.text and part.text.strip():
                return part.text
        raise ValueError("No text part in Gemini response")


class OpenAIClient(LLMClient):
    """
    Note: GPT-5 doesn't support schema-based structured output like Gemini's response_schema.
    We use response_format={"type": "json_object"} to ensure valid JSON, but the schema
    is enforced via prompt instructions rather than API parameters.
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = "GPT-5"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        """
        Analyze bill text and extract findings.

        Args:
            bill_text: Full text of the bill to analyze
            prompt: Analysis instructions with JSON schema definition
                   (from ConsensusPrompts.get_consensus_analysis_prompt()
                   Should include: task description, output format, focus areas, examples
            nrg_context: NRG business context to filter for relevant findings
        """
        start_time = time.time()
        try:
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # Use JSON mode to ensure valid JSON output
            # Schema adherence comes from prompt instructions (GPT-5 follows well per POC tests)
            response = await self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=8192,
                response_format={"type": "json_object"}  # Ensures valid JSON, not schema
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

    NOTE: Replace one Gemini instance w/ Claude or other model for diversity. 
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
        timeout: float = 120.0
    ):
        """
        Fire all 3 models, wait max 2 minutes.

        Handles partial failures: if Gemini-A times out but Gemini-B + GPT-5
        succeed, we still get 2/3 consensus (acceptable per Q3 blocking decision).
        
        TODO: consider retry and exponential backoff
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
