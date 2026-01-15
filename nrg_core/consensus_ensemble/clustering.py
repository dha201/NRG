from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticClusterer:
    """
    Group model findings by semantic meaning for concensus

    Why all-MiniLM-L6-v2 vs all-mpnet-base-v2:
    - MiniLM: 5x faster (real-time response needed), good enough quality
    - MPNet: Higher accuracy (85% vs 80% on MTEB), but 6x slower
    - MVP choice: Speed matters, MiniLM acceptable for 0.85 threshold
    - Post-MVP: Evaluate upgrading to MPNet if clustering FP/FN rates are high

    Why NOT use FAISS/vector DB for 3-model consensus:
    - Consensus ensemble only needs to compare 3 findings. 
        Instead of "Find 5 similar docs from 1M"
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Args:
            similarity_threshold: Cosine similarity >= this value counts as same finding.
                0.85 is calibrated for legislative text (good balance of false positives vs false negatives).
                Higher (0.95): Only catch exact paraphrases, miss legitimate variations.
                Lower (0.70): Cluster "tax on energy" with "solar incentive" (wrong).
                Post-MVP: Validate on labeled corpus, adjust per finding type.
                TODO: Create threshold tuning script once we have 100-bill validation set
            model_name: Sentence transformer model from Hugging Face.
        """
        self.threshold = similarity_threshold
        self.embedder = SentenceTransformer(model_name)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Cosine similarity between two bill statements.

        Returns value in [0, 1] for sentence-transformers embeddings.
        Note: cosine_similarity mathematically returns [-1, 1], but sentence-transformers
        produce embeddings that cluster in positive similarity space.

        - 1.0: Identical meaning (vectors point in same direction)
        - 0.85: Paraphrase ("exceeding 50MW" vs "over 50 megawatts")
        - 0.60: Related but different concepts
        - 0.0: Completely unrelated (orthogonal vectors)

        Source:
        - "In practical text-embedding scenarios embeddings are trained or normalized 
          so that their cosine similarities typically stay between 0 and 1"
          https://codesignal.com/learn/courses/text-representation-techniques-for-rag-systems/lessons/generating-and-comparing-sentence-embeddings
        """
        embeddings = self.embedder.encode([text1, text2])   # converts each text into a high-dimensional vector
                                                            # Each dimension captures some semantic feature
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # Measures the angle between two vectors
        return float(similarity)

    def cluster_findings(self, findings: List[dict]) -> List[List[dict]]:
        """
        Group findings by semantic similarity (greedy clustering):
        - Pick first unclustered finding
        - Add all similar findings to its cluster
        - Move to next unclustered finding
        Limitation: Order-dependent (might split clusters if middle item dissimilar)
        
        Note: 
        Confidence = model's self-assessment of how certain it is about the finding (0.0 to 1.0)
        Not important for this operation, can be omitted.(see schema at docs/plans/2026-01-15-consensus-ensemble.md:351-368)

        Called by: ConsensusEnsemble.analyze_consensus() after parallel model responses are received

        Args:
            findings: List of finding dictionaries from model responses.
                Example: [
                    {"statement": "Tax applies to solar", "confidence": 0.9},
                    {"statement": "Solar energy is taxed", "confidence": 0.85},
                    {"statement": "Wind exemption exists", "confidence": 0.8}
                ]

        Returns:
            List of clusters, where each cluster is a list of similar findings.
            Example: [
                [
                    {"statement": "Tax applies to solar", "confidence": 0.9},
                    {"statement": "Solar energy is taxed", "confidence": 0.85}
                ],
                [
                    {"statement": "Wind exemption exists", "confidence": 0.8}
                ]
            ]

        Post-MVP improvement: Use hierarchical clustering if grouping quality issues
        TODO: Benchmark greedy vs hierarchical on validation set
        """
        if not findings:
            return []

        # Embed all statement texts once
        statements = [f["statement"] for f in findings]
        embeddings = self.embedder.encode(statements)

        # Compute all-pairs cosine similarity matrix (3x3 for consensus ensemble)
        similarity_matrix = cosine_similarity(embeddings)

        # Greedy clustering: iterate unassigned, build clusters
        clusters = []
        assigned = set()

        for i, finding in enumerate(findings):
            if i in assigned:
                continue

            # Start new cluster with this finding
            cluster = [finding]
            assigned.add(i)

            # Find all similar unassigned findings
            for j in range(i + 1, len(findings)):
                if j in assigned:
                    continue

                # Similarity threshold is magic number calibrated for bill text
                # 0.85 chosen because:
                # - Catches "50MW" vs "fifty megawatts" (threshold variation paraphrases)
                # - Avoids "tax applies" vs "exemption applies" (semantic opposites)
                if similarity_matrix[i][j] >= self.threshold:
                    cluster.append(findings[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters
