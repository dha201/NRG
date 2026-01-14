import json
import time
from typing import Any, Optional

from rich.console import Console

from nrg_core.analysis.prompts import build_analysis_prompt, GEMINI_RESPONSE_SCHEMA
from nrg_core.utils import log_llm_cost
from nrg_core.config import load_config

console = Console()
_config = load_config()
_debug_llm_responses = _config.get('debug', {}).get('llm_responses', False)

# Error response structure
ERROR_RESPONSE: dict[str, Any] = {
    "business_impact_score": 0,
    "impact_summary": "Analysis failed"
}


# =============================================================================
# GEMINI RESPONSE PARSING FIX
# =============================================================================
# Gemini 3 "thinking" models return multi-part responses:
#   - thought_signature: encrypted reasoning traces (for multi-turn context)
#   - text: actual JSON content
#
# The SDK's response.text accessor does two things that break JSON parsing:
#   1. Returns None when only thought_signature exists (no text parts)
#      → json.loads(None) crashes: "must be str, bytes or bytearray, not NoneType"
#
#   2. Concatenates ALL text parts into one string
#      → You get: {"json1"}{"json2"} instead of just {"json1"}
#      → json.loads() unable to process this: "Extra data: line X column Y"
#
# Under the hood, the SDK does the following (types.py:6014-6036):
#   text = ''
#   for part in self.candidates[0].content.parts:
#       if isinstance(part.text, str):
#           text += part.text  # Oops, concatenation
#   return text if any_text_part_text else None  # Oops, None
#
# To resolve, we need to grab the FIRST text part manually, skip thought parts entirely.
# Works with tool calling too - you can still access part.function_call separately.
# - GitHub #196: https://github.com/google/generative-ai-python/issues/196
# - GitHub #515: https://github.com/google-gemini/deprecated-generative-ai-python/issues/515
# - Thought Signatures: https://ai.google.dev/gemini-api/docs/thought-signatures
#   What those encrypted blobs actually are
# =============================================================================

def extract_json_from_gemini_response(response: Any) -> str:
    """
    Extract JSON string from Gemini response, handling thought_signature parts.
    
    Gemini 3 models return responses with multiple parts including thought_signature
    (reasoning traces). The SDK's response.text accessor concatenates all text parts,
    causing "Extra data" JSON parse errors. This function extracts ONLY the first
    non-thought text part to get clean JSON.
    
    Args:
        response: Gemini API response object with candidates[0].content.parts
        
    Returns:
        str: JSON string from first text part
        
    Raises:
        ValueError: If response has no candidates, no parts, or no text parts
        
    Example:
        >>> response = gemini_client.models.generate_content(...)
        >>> json_str = extract_json_from_gemini_response(response)
        >>> analysis = json.loads(json_str)  # Safe - no NoneType or concatenation
    """
    # Debug: response structure
    if _debug_llm_responses:
        console.print(f"[dim]DEBUG extract_json: response type = {type(response)}[/dim]")
        console.print(f"[dim]DEBUG extract_json: response.text = {response.text[:200] if response.text else None}...[/dim]")
        console.print(f"[dim]DEBUG extract_json: has candidates = {hasattr(response, 'candidates')}[/dim]")
    
    if not response.candidates:
        if _debug_llm_responses:
            console.print(f"[red]DEBUG extract_json: response.candidates is empty or None[/red]")
        raise ValueError("Gemini response has no candidates")
    
    if _debug_llm_responses:
        console.print(f"[dim]DEBUG extract_json: candidates count = {len(response.candidates)}[/dim]")
        console.print(f"[dim]DEBUG extract_json: candidate[0] type = {type(response.candidates[0])}[/dim]")
        console.print(f"[dim]DEBUG extract_json: candidate[0].content = {response.candidates[0].content}[/dim]")
    
    if not response.candidates[0].content:
        if _debug_llm_responses:
            console.print(f"[red]DEBUG extract_json: candidate[0].content is None[/red]")
        raise ValueError("Gemini response candidate has no content")
    
    if _debug_llm_responses:
        console.print(f"[dim]DEBUG extract_json: content type = {type(response.candidates[0].content)}[/dim]")
        console.print(f"[dim]DEBUG extract_json: content.parts = {response.candidates[0].content.parts}[/dim]")
    
    if not response.candidates[0].content.parts:
        if _debug_llm_responses:
            console.print(f"[red]DEBUG extract_json: content.parts is empty or None[/red]")
            console.print(f"[red]DEBUG extract_json: Full response dump = {response}[/red]")
        raise ValueError("Gemini response content has no parts")
    
    # Iterate through parts to find first text part (skip thought parts)
    if _debug_llm_responses:
        console.print(f"[dim]DEBUG extract_json: Iterating through {len(response.candidates[0].content.parts)} parts[/dim]")
    
    for i, part in enumerate(response.candidates[0].content.parts):
        if _debug_llm_responses:
            console.print(f"[dim]DEBUG extract_json: Part {i} - type={type(part)}, thought={getattr(part, 'thought', False)}, text_len={len(part.text) if part.text else 0}[/dim]")
        
        # Skip thought_signature parts (encrypted reasoning state)
        # Note: thought attribute can be True or None (not always False)
        thought_val = getattr(part, 'thought', False)
        if thought_val is True or (thought_val is not False and hasattr(part, 'thought_signature')):
            if _debug_llm_responses:
                console.print(f"[dim]DEBUG extract_json: Part {i} - SKIPPED (thought={thought_val} or has thought_signature)[/dim]")
            continue
        
        # Return first non-empty text part only (prevents concatenation)
        # Skip None, empty strings, and whitespace-only parts
        if part.text and part.text.strip():
            if _debug_llm_responses:
                console.print(f"[dim]DEBUG extract_json: Part {i} - RETURNING (text found, len={len(part.text)})[/dim]")
                console.print(f"[dim]DEBUG extract_json: First 200 chars = {part.text[:200]}...[/dim]")
            return part.text
        else:
            if _debug_llm_responses:
                console.print(f"[dim]DEBUG extract_json: Part {i} - SKIPPED (empty or whitespace)[/dim]")
    
    # No text part found (only thought_signature or empty parts)
    if _debug_llm_responses:
        console.print(f"[red]DEBUG extract_json: NO TEXT PARTS FOUND after checking all {len(response.candidates[0].content.parts)} parts[/red]")
    raise ValueError(
        "No text part found in Gemini response. "
        "Response may contain only thought_signature parts or be empty."
    )


def analyze_with_openai(
    item: dict[str, Any],
    nrg_context: str,
    custom_prompt: Optional[str] = None,
    openai_client: Optional[Any] = None,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict[str, Any]:
    """
    Analyze bill/regulation using OpenAI GPT-5 with retry logic.
    
    Retries handle transient failures (rate limits, timeouts) with exponential backoff.
    
    Args:
        item: Bill/regulation dictionary
        nrg_context: NRG business context
        custom_prompt: Optional custom prompt override
        openai_client: OpenAI client instance
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        
    Returns:
        Structured analysis dictionary or error response
    """
    if not openai_client:
        return {**ERROR_RESPONSE, "error": "OpenAI client not provided"}
    
    combined_input = custom_prompt if custom_prompt else build_analysis_prompt(item, nrg_context)
    last_error: Optional[str] = None

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            response = openai_client.responses.create(
                model="gpt-5",
                input=combined_input,
                reasoning={"effort": "medium"},
                text={"verbosity": "high"}
            )

            analysis = json.loads(response.output_text)
            log_llm_cost("gpt-5", combined_input, response.output_text)
            return analysis

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {str(e)}"
            if attempt == max_retries - 1:
                console.print(f"[red]Error parsing JSON from GPT-5: {e}[/red]")
                console.print(f"[dim]Raw output: {response.output_text[:200]}...[/dim]")
                return {
                    **ERROR_RESPONSE,
                    "error": last_error,
                    "impact_summary": "Analysis failed - invalid JSON response"
                }
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries - 1:
                console.print(f"[red]GPT-5 call failed after {max_retries} attempts: {e}[/red]")
                return {
                    **ERROR_RESPONSE,
                    "error": last_error,
                    "impact_summary": f"Analysis failed after {max_retries} retries"
                }
        
        # Exponential backoff before retry
        delay = base_delay * (2 ** attempt)
        console.print(
            f"[yellow]GPT-5 call failed ({last_error}), retrying in {delay}s... "
            f"(attempt {attempt + 1}/{max_retries})[/yellow]"
        )
        time.sleep(delay)
    
    return {**ERROR_RESPONSE, "error": str(last_error)}


def analyze_with_gemini(
    item: dict[str, Any],
    nrg_context: str,
    custom_prompt: Optional[str] = None,
    gemini_client: Optional[Any] = None,
    config: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict[str, Any]:
    """
    Analyze legislative item using Google Gemini with retry logic.
    
    Retries handle transient failures (rate limits, timeouts) with exponential backoff.
    
    Args:
        item: Legislative item dictionary
        nrg_context: NRG business context
        custom_prompt: Optional custom prompt override
        gemini_client: Gemini client instance
        config: Configuration dictionary
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        
    Returns:
        Structured analysis dictionary or error response
    """
    if not gemini_client:
        return {**ERROR_RESPONSE, "error": "Gemini client not provided"}
    if not config:
        return {**ERROR_RESPONSE, "error": "Configuration not provided"}
    
    combined_input = custom_prompt if custom_prompt else build_analysis_prompt(item, nrg_context)
    model_name = config['llm']['gemini']['model']
    last_error: Optional[str] = None

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=combined_input,
                config={
                    "temperature": config['llm']['gemini'].get('temperature', 0.2),
                    "max_output_tokens": config['llm']['gemini'].get('max_output_tokens', 8192),
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_RESPONSE_SCHEMA
                }
            )

            # Try SDK's response.text first (preserves all content)
            # Fall back to extract_json_from_gemini_response for edge cases:
            #   - response.text is None (thought-only responses)
            #   - response.text has concatenated JSON (multi-part responses)
            json_text = response.text
            
            if json_text is None:
                # Gemini 3 thinking models may return None for thought-only responses
                json_text = extract_json_from_gemini_response(response)
            
            try:
                analysis = json.loads(json_text)
            except (json.JSONDecodeError, TypeError):
                # SDK concatenation bug: {"json1"}{"json2"} - try first part only
                json_text = extract_json_from_gemini_response(response)
                analysis = json.loads(json_text)
            
            if not isinstance(analysis, dict):
                raise ValueError(f"Schema enforcement failed: got {type(analysis)} instead of dict")

            log_llm_cost(model_name, combined_input, json_text)
            return analysis

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {str(e)}"
            if attempt == max_retries - 1:
                console.print(f"[red]Error parsing JSON from Gemini: {e}[/red]")
                # Extract text safely for error logging
                if _debug_llm_responses:
                    try:
                        debug_text = extract_json_from_gemini_response(response)
                        console.print(f"[dim]Raw output: {debug_text[:200]}...[/dim]")
                    except Exception:
                        console.print("[dim]Could not extract text for debugging[/dim]")
                return {
                    **ERROR_RESPONSE,
                    "error": last_error,
                    "impact_summary": "Analysis failed - invalid JSON response"
                }
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries - 1:
                console.print(f"[red]Gemini call failed after {max_retries} attempts: {e}[/red]")
                return {
                    **ERROR_RESPONSE,
                    "error": last_error,
                    "impact_summary": f"Analysis failed after {max_retries} retries"
                }
        
        # Exponential backoff before retry
        delay = base_delay * (2 ** attempt)
        console.print(
            f"[yellow]Gemini call failed ({last_error}), retrying in {delay}s... "
            f"(attempt {attempt + 1}/{max_retries})[/yellow]"
        )
        time.sleep(delay)
    
    return {**ERROR_RESPONSE, "error": str(last_error)}




def analyze_bill_version(
    version_text: str,
    version_type: str,
    bill_info: dict[str, Any],
    nrg_context: str,
    config: dict[str, Any],
    openai_client: Optional[Any] = None,
    gemini_client: Optional[Any] = None
) -> dict[str, Any]:
    """
    Analyze a single bill version using configured LLM provider.
    
    Args:
        version_text: Full extracted text of this bill version
        version_type: Version stage (e.g., Introduced, Enrolled)
        bill_info: Bill metadata dictionary
        nrg_context: NRG business context
        config: Configuration dictionary
        openai_client: OpenAI client instance
        gemini_client: Gemini client instance
        
    Returns:
        Full LLM analysis for this version
    """
    provider = config['llm']['provider']
    retry_config = config.get('llm', {}).get('retry', {})
    max_retries = retry_config.get('max_retries', 3)
    base_delay = retry_config.get('base_delay', 1.0)

    prompt_text = f"""You are a legislative analyst for NRG Energy's Government Affairs team analyzing a specific version of a bill.

**BILL INFORMATION:**
- Bill Number: {bill_info.get('number', 'Unknown')}
- Bill Title: {bill_info.get('title', 'Unknown')}
- **Version Being Analyzed: {version_type}**  <-- THIS IS THE VERSION YOU ARE ANALYZING
- Source: {bill_info.get('source', 'Unknown')}

**VERSION TEXT:**
{version_text[:20000]}

**NRG BUSINESS CONTEXT:**
{nrg_context}

Analyze this specific version and return JSON with the standard analysis schema."""

    temp_item = {
        "source": bill_info.get('source', 'Unknown'),
        "number": bill_info.get('number', 'Unknown'),
        "title": f"{bill_info.get('title', 'Unknown')} ({version_type})",
        "summary": version_text[:20000],
        "type": bill_info.get('type', 'Bill'),
        "status": version_type,
        "url": bill_info.get('url', '')
    }

    # Route to appropriate LLM provider
    if provider == 'gemini':
        return analyze_with_gemini(
            temp_item, nrg_context,
            custom_prompt=prompt_text,
            gemini_client=gemini_client,
            config=config,
            max_retries=max_retries,
            base_delay=base_delay
        )
    elif provider == 'openai':
        return analyze_with_openai(
            temp_item, nrg_context,
            custom_prompt=prompt_text,
            openai_client=openai_client,
            max_retries=max_retries,
            base_delay=base_delay
        )
    else:
        console.print(f"[yellow]Unknown provider '{provider}', defaulting to OpenAI[/yellow]")
        return analyze_with_openai(
            temp_item, nrg_context,
            custom_prompt=prompt_text,
            openai_client=openai_client,
            max_retries=max_retries,
            base_delay=base_delay
        )
