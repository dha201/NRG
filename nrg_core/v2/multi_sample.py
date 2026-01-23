"""
Multi-Sample Consistency Checker.

Resamples analysis multiple times and extracts consensus findings.
Uses cosine similarity to measure consistency across samples.

Design:
- Run LLM analysis 2-3 times with different seeds
- Compute pairwise cosine similarity between findings
- Extract consensus findings (appear in 2+ samples)
- Flag low consistency for human review

Why:
- LLMs can have variability in responses
- Multiple samples increase reliability
- Consistency scoring helps identify uncertain findings
"""
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nrg_core.models_v2 import Finding, Quote


@dataclass
class ConsensusResult:
    """Result of multi-sample consistency check.
    
    Attributes:
        num_samples: Number of LLM samples run
        consistency_score: Average pairwise similarity (0-1)
        consensus_findings: Findings that appear in multiple samples
        requires_human_review: Flag for low consistency findings
    """
    num_samples: int
    consistency_score: float  # 0-1
    consensus_findings: List[Finding]
    requires_human_review: bool


class MultiSampleChecker:
    """
    Run analysis multiple times with different seeds.
    Cluster results by semantic similarity.
    Extract consensus elements.
    
    Uses TF-IDF and cosine similarity to measure semantic consistency
    between different LLM responses to the same bill.
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, num_samples: int = 3):
        """
        Initialize multi-sample checker.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key
            num_samples: Number of times to resample (default: 3)
        """
        self.model = model
        self.num_samples = num_samples
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def check_consistency(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> ConsensusResult:
        """
        Run analysis num_samples times and check consistency.
        
        Process:
        1. Call LLM multiple times with different seeds
        2. Compute pairwise cosine similarity between findings
        3. Extract consensus findings (appear in 2+ samples)
        4. Flag for human review if consistency < 0.6
        
        Args:
            bill_id: Bill identifier
            bill_text: Full bill text
            nrg_context: NRG business context
        
        Returns:
            ConsensusResult with consensus findings and consistency score
        """
        # Resample multiple times
        samples = []
        for i in range(self.num_samples):
            sample = self._call_llm(bill_text, nrg_context, seed=i * 100)
            samples.append(sample)
        
        # Compute consistency score
        consistency_score = self._compute_consistency(samples)
        
        # Extract consensus findings (appear in 2+ samples)
        consensus_findings = self._extract_consensus(samples)
        
        # Flag for human review if low consistency
        requires_review = bool(consistency_score < 0.6)
        
        return ConsensusResult(
            num_samples=self.num_samples,
            consistency_score=consistency_score,
            consensus_findings=consensus_findings,
            requires_human_review=requires_review
        )
    
    def _compute_consistency(self, samples: list[dict[str, Any]]) -> float:
        """
        Compute consistency score across samples using cosine similarity.
        
        Method:
        1. Extract all finding statements from each sample
        2. Join statements with " | " separator
        3. Compute TF-IDF vectors
        4. Calculate pairwise cosine similarities
        5. Return average similarity (excluding diagonal)
        
        Args:
            samples: List of LLM outputs with findings
        
        Returns:
            Average pairwise cosine similarity (0-1)
        """
        # Extract statements from all samples
        all_statements = []
        for sample in samples:
            statements = [f["statement"] for f in sample.get("findings", [])]
            all_statements.append(" | ".join(statements))
        
        if len(all_statements) < 2:
            return 1.0
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_statements)
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(vectors)
        
        # Return average similarity (exclude diagonal)
        n = len(samples)
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarities[i][j]
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    def _extract_consensus(self, samples: list[dict[str, Any]]) -> list[Finding]:
        """
        Extract findings that appear in 2+ samples.
        
        Simple heuristic: statements with >0.85 cosine similarity are same finding.
        For Phase 1, return all findings as consensus.
        
        TODO: Implement proper clustering in production using:
        - Hierarchical clustering on similarity matrix
        - Threshold-based grouping
        - Vote counting for each cluster
        
        Args:
            samples: List of LLM outputs with findings
        
        Returns:
            List of consensus findings
        """
        all_findings = []
        for sample in samples:
            for f in sample.get("findings", []):
                finding = Finding(
                    statement=f["statement"],
                    quotes=[Quote(**q) for q in f["quotes"]],
                    confidence=f["confidence"],
                    impact_estimate=f["impact_estimate"]
                )
                all_findings.append(finding)
        
        # Cluster by similarity (simplified: just return all for now)
        # TODO: Implement proper clustering in production
        return all_findings
    
    def _call_llm(self, bill_text: str, nrg_context: str, seed: int) -> dict[str, Any]:
        """
        Call LLM with specific seed for reproducibility.
        
        Uses same prompt as primary analyst for consistency.
        Temperature set to 0.3 for slight diversity while maintaining focus.
        
        Args:
            bill_text: Bill text to analyze
            nrg_context: NRG business context
            seed: Random seed for reproducibility
        
        Returns:
            Parsed JSON response from LLM
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Use same prompt as primary analyst
        from nrg_core.v2.primary_analyst import PRIMARY_ANALYST_PROMPT
        
        prompt = PRIMARY_ANALYST_PROMPT.format(
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,  # Slightly higher for diversity
            seed=seed
        )
        
        return json.loads(response.choices[0].message.content)
