from typing import List
import re
from nrg_core.consensus_ensemble.models import ModelResponse, Finding, ConsensusAnalysis, ConsensusLevel
from nrg_core.consensus_ensemble.clustering import SemanticClusterer


class ConsensusEngine:
    """
    Reduces hallucinations by voting on findings across 3 LLM models

    1. Group similar findings by meaning
    2. Vote: 3/3 unanimous, 2/3 majority, 1/3 disputed
    3. Verify quotes exist in bill text (catches hallucinations)
    4. Assign confidence based on agreement level

    This catches hallucinations like:
    - Model claims "50MW tax" but bill says "all energy"
    - Model provides quote that doesn't exist in bill
    - Result: Finding included but flagged for review

    Confidence scores (TODO: calibrate on validation set):
    - 0.95: All 3 models agree, quotes verified
    - 0.70: 2/3 agree, quotes from majority verified  
    - 0.50: 1/3 or conflicting, needs escalation
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.clusterer = SemanticClusterer(similarity_threshold=similarity_threshold)
        self.threshold = similarity_threshold

    def build_consensus(self, responses: List[ModelResponse], bill_text: str) -> ConsensusAnalysis:
        """
        Combines findings from 3 models into a consensus analysis.
        
        Args:
            responses: List of model responses with findings
            bill_text: Full bill text for quote verification
            
        Returns:
            ConsensusAnalysis with findings and overall confidence
        """
        # Flatten findings from all models for clustering and voting
        all_findings = []
        for response in responses:
            for finding in response.findings:
                all_findings.append({
                    "statement": finding.get("statement", ""),
                    "quote": finding.get("quote", ""),
                    "confidence": finding.get("confidence", 0.0),
                    "model": response.model_name
                })

        clusters = self.clusterer.cluster_findings(all_findings)

        # Convert each cluster into a consensus finding with vote results
        consensus_findings = []
        for cluster in clusters:
            finding = self._build_finding_from_cluster(cluster, bill_text)
            consensus_findings.append(finding)

        # Overall confidence = average of all finding confidences
        # (could use weighted average later)
        if consensus_findings:
            overall_confidence = sum(f.confidence for f in consensus_findings) / len(consensus_findings)
        else:
            overall_confidence = 0.0

        return ConsensusAnalysis(
            findings=consensus_findings,
            model_responses=responses,
            overall_confidence=overall_confidence
        )

    def _build_finding_from_cluster(self, cluster: List[dict], bill_text: str) -> Finding:
        """
        Converts a cluster of similar findings into one consensus finding.
        
        Args:
            cluster: List of paraphrases from different models
                [{"statement": "Tax >50MW", "model": "Gemini-A"}, ...]
            bill_text: Full bill text for quote verification
            
        Returns:
            Finding with consensus level, confidence, and verification status
            Ex: Finding(statement="Tax on >50MW energy", confidence=0.70, 
                      consensus_level="MAJORITY", verification_status="verified")
        """

        # Count unique models that agree on this finding
        models_found = list(set(finding["model"] for finding in cluster))
        num_models = len(models_found)
        if num_models == 3:
            consensus_level = ConsensusLevel.UNANIMOUS.value
            confidence = 0.95
        elif num_models == 2:
            consensus_level = ConsensusLevel.MAJORITY.value
            confidence = 0.70
        else:
            consensus_level = ConsensusLevel.DISPUTED.value
            confidence = 0.50

        # Use first statement as representative (could improve with most common)
        statement = cluster[0]["statement"]

        supporting_quotes = [finding["quote"] for finding in cluster if finding.get("quote")]

        verification_status = "verified" if self._verify_quotes(supporting_quotes, bill_text) else "unverified"

        # This will be used downstream for confidence aggregation and routing decisions
        # High confidence + verified quotes = auto-publish, low confidence = expert review
        return Finding(
            statement=statement,
            confidence=confidence,
            supporting_quotes=supporting_quotes,
            found_by=models_found,
            consensus_level=consensus_level,
            verification_status=verification_status
        )

    def _verify_quotes(self, quotes: List[str], bill_text: str) -> bool:
        """
        Checks if quotes actually exist in bill text (hallucination detection).
        TODO: Consider fuzzy matching for better tolerance of minor variations.
        """
        if not quotes:
            return False

        lower_bill = bill_text.lower()
        
        for quote in quotes:
            if quote:
                pattern = r'\b' + re.escape(quote.lower()) + r'\b'
                if re.search(pattern, lower_bill):
                    return True

        return False
