import pytest
from nrg_core.consensus_ensemble.ensemble import ConsensusEnsemble


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ensemble_pipeline():
    """Integration test with real bill text (requires API keys)"""
    bill_text = """
    Section 2.1: Tax on Energy Generation

    This bill imposes a tax on energy generation exceeding fifty megawatts capacity.

    Section 3.2: Exemptions

    Renewable energy facilities are exempt as defined in Section 5.2.

    Section 4.1: Effective Date

    This act shall take effect on January 1, 2026.
    """

    ensemble = ConsensusEnsemble()
    result = await ensemble.analyze(bill_text, nrg_context="NRG energy portfolio", timeout=90.0)

    assert len(result.findings) >= 2
    assert result.overall_confidence > 0.5

    statements = [f.statement for f in result.findings]
    assert any("50" in s or "fifty" in s.lower() for s in statements)
    assert any("renewable" in s.lower() for s in statements)
