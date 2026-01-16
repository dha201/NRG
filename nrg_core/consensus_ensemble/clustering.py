from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticClusterer:
    """
    Group model findings by semantic meaning for consensus

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

        - "In practical text-embedding scenarios embeddings are trained or normalized 
          so that their cosine similarities typically stay between 0 and 1"
          https://codesignal.com/learn/courses/text-representation-techniques-for-rag-systems/lessons/generating-and-comparing-sentence-embeddings
        """
        embeddings = self.embedder.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def cluster_findings(self, findings: List[dict]) -> List[List[dict]]:
        """
        Args:
            findings: List of finding dictionaries from model responses.
                Example: [{"statement": "Tax applies to solar", "confidence": 0.9}]

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

        statements = [f["statement"] for f in findings]
        embeddings = self.embedder.encode(statements)
        
        # similarity_matrix[i][j] = how similar finding i is to finding j
        # similarity_matrix = [
        #     [1.0, 0.92, 0.45],  # Gemini vs [Gemini, GPT-4o, Claude]
        #     [0.92, 1.0, 0.40],  # GPT-4o vs [Gemini, GPT-4o, Claude]  
        #     [0.45, 0.40, 1.0]   # Claude vs [Gemini, GPT-4o, Claude]
        # ]
        # similarity_matrix[0][1] = 0.92 (Gemini vs GPT similarity)
        similarity_matrix = cosine_similarity(embeddings)

        # Group similar findings to detect model agreement (cluster size = consensus level)
        # Group findings by semantic similarity (greedy clustering):
        # - Pick first unclustered finding
        # - Add all similar findings to its cluster
        # - Move to next unclustered finding
        clusters = []
        assigned_indices = set()

        for current_idx, finding in enumerate(findings):
            if current_idx in assigned_indices:
                continue

            cluster = [finding]
            assigned_indices.add(current_idx)

            for other_idx in range(current_idx + 1, len(findings)):
                if other_idx in assigned_indices:
                    continue

                if similarity_matrix[current_idx][other_idx] >= self.threshold:
                    cluster.append(findings[other_idx])
                    assigned_indices.add(other_idx)

            clusters.append(cluster)

        return clusters
