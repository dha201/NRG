# nrg_core/v2/audit_trail.py
"""
Audit Trail Generator - Creates compliance-ready documentation of analysis.

Design Decisions:
- Standalone class for Single Responsibility - only handles trail generation
- Returns dict (not Pydantic model) for JSON serialization flexibility
- Includes ISO timestamps for chronological ordering
- Trail version field enables schema evolution without breaking consumers

Why Audit Trails Matter:
- Compliance: Regulators may require evidence of analysis methodology
- Transparency: Stakeholders can trace how conclusions were reached
- Debugging: Engineers can identify where pipeline made errors
- Reproducibility: Complete input/output capture enables re-analysis
"""
from typing import Dict, Any, List
from datetime import datetime, timezone
from nrg_core.models_v2 import Finding, JudgeValidation, RubricScore


class AuditTrailGenerator:
    """
    Generate comprehensive audit trails for legislative analysis findings.
    
    Each trail captures:
    - Finding details (statement, confidence, impact)
    - All supporting quotes with source locations
    - Validation results from judge
    - All rubric dimension scores with rationales
    - Timestamps for temporal tracking
    
    Usage:
        generator = AuditTrailGenerator()
        trail = generator.generate(finding, validation, rubric_scores)
    """
    
    # Schema version for audit trail format
    # Increment when structure changes to enable backward compatibility
    TRAIL_VERSION = "v2.0"
    
    def generate(
        self,
        finding: Finding,
        validation: JudgeValidation,
        rubric_scores: List[RubricScore]
    ) -> Dict[str, Any]:
        """
        Generate complete audit trail for a single finding.
        
        Combines all analysis artifacts into a single auditable record:
        - Primary analyst's finding and confidence
        - Judge's validation checks (quote verification, hallucination)
        - All rubric dimension scores with rationales and evidence
        
        Args:
            finding: Primary analyst's finding to document
            validation: Judge's validation result for this finding
            rubric_scores: All rubric scores for this finding (typically 4)
        
        Returns:
            Dict containing complete audit trail, JSON-serializable
        """
        trail = {
            # Finding metadata
            "finding": {
                "statement": finding.statement,
                "confidence": finding.confidence,
                "impact_estimate": finding.impact_estimate
            },
            
            # Supporting evidence - all quotes used to justify finding
            "quotes_used": [
                {
                    "text": q.text,
                    "section": q.section,
                    "page": q.page
                }
                for q in finding.quotes
            ],
            
            # Validation checks from judge
            "validation_result": {
                "quote_verified": validation.quote_verified,
                "hallucination_detected": validation.hallucination_detected,
                "evidence_quality": validation.evidence_quality,
                "ambiguity": validation.ambiguity,
                "judge_confidence": validation.judge_confidence
            },
            
            # Rubric scores with full rationale trail
            "rubric_scores": [
                {
                    "dimension": s.dimension,
                    "score": s.score,
                    "rationale": s.rationale,
                    "rubric_anchor": s.rubric_anchor,
                    # Include evidence quotes for each score
                    "evidence": [
                        {"text": e.text, "section": e.section, "page": e.page}
                        for e in s.evidence
                    ]
                }
                for s in rubric_scores
            ],
            
            # Temporal metadata
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trail_version": self.TRAIL_VERSION
        }
        
        return trail
    
    def generate_batch(
        self,
        findings: List[Finding],
        validations: List[JudgeValidation],
        all_rubric_scores: List[RubricScore],
        dimensions_per_finding: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate audit trails for multiple findings.
        
        Convenience method for processing entire analysis results.
        Assumes rubric_scores are ordered by finding index, with
        `dimensions_per_finding` scores per finding.
        
        Args:
            findings: List of findings from primary analyst
            validations: List of judge validations (1:1 with findings)
            all_rubric_scores: All rubric scores (4 per finding typically)
            dimensions_per_finding: Number of rubric dimensions per finding
        
        Returns:
            List of audit trails, one per finding
        """
        trails = []
        
        for idx, finding in enumerate(findings):
            validation = validations[idx]
            
            # Extract scores for this finding
            start_idx = idx * dimensions_per_finding
            end_idx = start_idx + dimensions_per_finding
            finding_scores = all_rubric_scores[start_idx:end_idx]
            
            trail = self.generate(finding, validation, finding_scores)
            trails.append(trail)
        
        return trails
