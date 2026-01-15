from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ConsensusLevel(Enum):
    """Agreement level across models"""
    UNANIMOUS = "unanimous"      # All 3 agree
    MAJORITY = "majority"        # 2 of 3 agree
    DISPUTED = "disputed"        # 1 or conflicting
    VERIFIED = "verified"        # Quote verified but not consensus


@dataclass
class Finding:
    """A single finding from bill analysis"""
    statement: str
    confidence: float
    supporting_quotes: List[str] = field(default_factory=list)
    found_by: List[str] = field(default_factory=list)
    consensus_level: str = "unknown"
    verification_status: str = "unverified"

    def to_dict(self) -> dict:
        return {
            'statement': self.statement,
            'confidence': self.confidence,
            'supporting_quotes': self.supporting_quotes,
            'found_by': self.found_by,
            'consensus_level': self.consensus_level,
            'verification_status': self.verification_status
        }


@dataclass
class ModelResponse:
    """Response from a single LLM model"""
    model_name: str
    findings: List[dict]
    processing_time: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, model_name: str) -> "ModelResponse":
        return cls(
            model_name=model_name,
            findings=data.get('findings', []),
            processing_time=0.0
        )


@dataclass
class ConsensusAnalysis:
    """Consensus result from all models"""
    findings: List[Finding]
    model_responses: List[ModelResponse] = field(default_factory=list)
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            'findings': [f.to_dict() for f in self.findings],
            'model_responses': [
                {
                    'model_name': r.model_name,
                    'findings': r.findings,
                    'error': r.error
                }
                for r in self.model_responses
            ],
            'overall_confidence': self.overall_confidence
        }
