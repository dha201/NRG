"""
Precision/Recall/F1 Metrics for Silver Set Evaluation.

This module provides evaluation metrics for comparing system predictions against
expert labels in the silver set, using semantic similarity to match findings.

Design:
- Uses TF-IDF vectorization and cosine similarity for semantic matching
- Configurable similarity threshold (default: 0.75) for finding matches
- Greedy matching algorithm to avoid duplicate matches
- Supports precision, recall, F1 calculation and rubric score MAE

Why semantic matching:
- Exact string matching is too strict for legislative analysis
- Different phrasings can represent the same finding
- Cosine similarity captures semantic similarity effectively
- TF-IDF is computationally efficient and well-established
"""
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PrecisionRecallEvaluator:
    """
    Evaluate system predictions against expert labels.
    
    Uses semantic similarity (cosine) to match predicted findings to expert labels.
    Similarity threshold: 0.75 for a match (configurable).
    
    Matching strategy:
    - Greedy matching: each predicted finding matches at most one expert finding
    - Highest similarity matches first
    - Prevents duplicate matches for fair evaluation
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize evaluator with similarity threshold.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match (0-1)
        """
        self.similarity_threshold = similarity_threshold
    
    def compute(
        self,
        predicted: List[Dict[str, Any]],
        expert: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 between predictions and expert labels.
        
        Method:
        1. Match findings using semantic similarity
        2. Count true positives, false positives, false negatives
        3. Compute precision, recall, F1
        
        Args:
            predicted: List of predicted findings with 'statement' field
            expert: List of expert-labeled findings with 'statement' field
        
        Returns:
            Dict with precision, recall, f1, and raw counts
        """
        # Match findings using semantic similarity
        matches = self._match_findings(predicted, expert)
        
        true_positives = len(matches)
        false_positives = len(predicted) - true_positives
        false_negatives = len(expert) - true_positives
        
        # Compute metrics with edge case handling
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def rubric_mae(
        self,
        predicted_scores: Dict[str, int],
        expert_scores: Dict[str, int]
    ) -> float:
        """
        Compute mean absolute error for rubric scores.
        
        MAE measures the average absolute difference between predicted and expert
        rubric scores across all dimensions.
        
        Args:
            predicted_scores: Dict of dimension -> predicted score
            expert_scores: Dict of dimension -> expert score
        
        Returns:
            Mean absolute error across dimensions (0-10 scale)
        """
        errors = []
        for dimension in expert_scores.keys():
            if dimension in predicted_scores:
                error = abs(predicted_scores[dimension] - expert_scores[dimension])
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def _match_findings(
        self,
        predicted: List[Dict],
        expert: List[Dict]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted findings to expert findings using semantic similarity.
        
        Algorithm:
        1. Compute TF-IDF vectors for all statements
        2. Calculate cosine similarity matrix
        3. Greedy matching: for each predicted, find best unmatched expert
        
        Args:
            predicted: List of predicted findings
            expert: List of expert findings
        
        Returns:
            List of (predicted_idx, expert_idx) tuples for matches
        """
        if not predicted or not expert:
            return []
        
        # Extract statements for vectorization
        pred_statements = [f["statement"] for f in predicted]
        expert_statements = [f["statement"] for f in expert]
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better semantic matching
            min_df=1,  # Include all terms (small datasets)
            max_df=0.9  # Exclude very common terms
        )
        all_statements = pred_statements + expert_statements
        vectors = vectorizer.fit_transform(all_statements)
        
        pred_vectors = vectors[:len(predicted)]
        expert_vectors = vectors[len(predicted):]
        
        # Compute similarity matrix
        similarities = cosine_similarity(pred_vectors, expert_vectors)
        
        # Greedy matching: for each predicted, find best expert match
        matches = []
        matched_experts = set()
        
        for pred_idx in range(len(predicted)):
            best_expert_idx = None
            best_sim = self.similarity_threshold
            
            for expert_idx in range(len(expert)):
                if expert_idx in matched_experts:
                    continue
                
                sim = similarities[pred_idx, expert_idx]
                if sim > best_sim:
                    best_sim = sim
                    best_expert_idx = expert_idx
            
            if best_expert_idx is not None:
                matches.append((pred_idx, best_expert_idx))
                matched_experts.add(best_expert_idx)
        
        return matches
