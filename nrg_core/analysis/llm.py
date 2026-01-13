import json
import time
from typing import Any, Optional

from rich.console import Console

from nrg_core.analysis.prompts import build_analysis_prompt
from nrg_core.utils import log_llm_cost

console = Console()

# Error response structure
ERROR_RESPONSE: dict[str, Any] = {
    "business_impact_score": 0,
    "impact_summary": "Analysis failed"
}


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
                    "response_mime_type": "application/json"
                }
            )

            analysis = json.loads(response.text)
            log_llm_cost(model_name, combined_input, response.text)
            return analysis

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {str(e)}"
            if attempt == max_retries - 1:
                console.print(f"[red]Error parsing JSON from Gemini: {e}[/red]")
                console.print(f"[dim]Raw output: {response.text[:200]}...[/dim]")
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
