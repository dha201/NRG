import pytest
from nrg_core.consensus_ensemble.prompts import ConsensusPrompts


def test_consensus_prompt_generation():
    """Prompt must enforce structured JSON output"""
    prompts = ConsensusPrompts()
    prompt = prompts.get_consensus_analysis_prompt()

    # Verify structured output instructions present
    assert "JSON" in prompt
    assert "findings" in prompt
    assert "quote" in prompt
    assert "confidence" in prompt


def test_quote_verification_prompt():
    """Quote verification should force exact text extraction"""
    prompts = ConsensusPrompts()
    prompt = prompts.get_quote_verification_prompt("Tax applies to >50MW")

    assert "Tax applies to >50MW" in prompt
    assert "EXACT" in prompt.upper() and "QUOTE" in prompt.upper()
