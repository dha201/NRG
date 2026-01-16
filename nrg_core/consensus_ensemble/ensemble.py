import os
from typing import Optional
from nrg_core.consensus_ensemble.llm_clients import ParallelAnalyzer
from nrg_core.consensus_ensemble.consensus import ConsensusEngine
from nrg_core.consensus_ensemble.prompts import ConsensusPrompts
from nrg_core.consensus_ensemble.models import ConsensusAnalysis


class ConsensusEnsemble:
    """
    Main orchestrator for consensus ensemble analysis
    
    Coordinates parallel model invocation, semantic clustering, and voting
    to achieve <1% false positive rate through multi-model agreement.
    """

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        similarity_threshold: float = 0.85
    ):
        self.gemini_key = gemini_key or os.getenv('GEMINI_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')

        self.analyzer = ParallelAnalyzer(
            gemini_key=self.gemini_key,
            openai_key=self.openai_key
        )
        self.consensus_engine = ConsensusEngine(similarity_threshold=similarity_threshold)
        self.prompts = ConsensusPrompts()

    async def analyze(
        self, 
        bill_text: str, 
        nrg_context: str = "",
        timeout: float = 60.0
    ) -> ConsensusAnalysis:
        """
        Run consensus analysis on bill text
        
        Args:
            bill_text: Full text of the bill to analyze
            nrg_context: NRG business context for filtering relevant findings
            timeout: Maximum time in seconds for parallel analysis
            
        Returns:
            ConsensusAnalysis with findings grouped by agreement level
        """
        prompt = self.prompts.get_consensus_analysis_prompt()

        model_responses = await self.analyzer.analyze_parallel(
            bill_text=bill_text,
            prompt=prompt,
            nrg_context=nrg_context,
            timeout=timeout
        )

        consensus = self.consensus_engine.build_consensus(
            responses=model_responses,
            bill_text=bill_text
        )

        return consensus
