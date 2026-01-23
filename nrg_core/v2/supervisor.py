# nrg_core/v2/supervisor.py
"""
Supervisor Router - Code-based complexity assessment and routing.

Architecture v2.0 - Phase 1

Design Decision: Use deterministic rules instead of LLM for routing.
Why: Routing is a classification task with clear criteria. Using code:
- Reduces cost ($0.00 vs $0.02 per bill for LLM)
- Ensures consistent, explainable routing decisions
- Enables easy threshold tuning without prompt engineering

Complexity Scoring System:
- Length: >50 pages = +2, 20-50 pages = +1
- Versions: >5 = +2, 2-5 = +1
- Domain: Energy/Tax = +2, Environmental = +1

Route Decision:
- 0-2 points: STANDARD (80% of bills) - Simple single-pass analysis
- 3+ points: ENHANCED (20% of bills) - Full two-tier pipeline
"""
from enum import Enum
from typing import Dict, Any


class Route(str, Enum):
    """
    Analysis route selection.
    
    STANDARD: Simple bills, single-pass analysis, lower cost
    ENHANCED: Complex bills, full two-tier validation pipeline
    """
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"


class SupervisorRouter:
    """
    Code-based router for bill analysis complexity assessment.
    
    Usage:
        router = SupervisorRouter()
        route = router.assess_complexity(bill_metadata)
        budget = router.get_budget_constraints(route)
    
    The router scores bills on three dimensions (length, versions, domain)
    and routes to STANDARD or ENHANCED based on total score.
    """
    
    def __init__(self):
        """Initialize router with zero complexity score."""
        self.complexity_score = 0
        self.score_breakdown = {}
    
    def assess_complexity(self, bill_metadata: Dict[str, Any]) -> Route:
        """
        Assess bill complexity using deterministic rules.
        
        Scoring:
        - Length: >50 pages (+2), 20-50 pages (+1), <20 pages (+0)
        - Versions: >5 versions (+2), 2-5 versions (+1), 1 version (+0)
        - Domain: energy/tax (+2), environmental (+1), general (+0)
        
        Args:
            bill_metadata: Dict with keys: page_count, version_count, domain
        
        Returns:
            Route.STANDARD (0-2 points) or Route.ENHANCED (3+ points)
        """
        score = 0
        breakdown = {}
        
        # Length scoring - longer bills need more thorough analysis
        page_count = bill_metadata.get("page_count", 0)
        if page_count > 50:
            score += 2
            breakdown["length"] = 2
        elif page_count >= 20:
            score += 1
            breakdown["length"] = 1
        else:
            breakdown["length"] = 0
        
        # Version count scoring - more versions = more complex legislative history
        version_count = bill_metadata.get("version_count", 1)
        if version_count > 5:
            score += 2
            breakdown["versions"] = 2
        elif version_count >= 2:
            score += 1
            breakdown["versions"] = 1
        else:
            breakdown["versions"] = 0
        
        # Domain scoring - energy/tax bills are core business, need deeper analysis
        domain = bill_metadata.get("domain", "general").lower()
        if domain in ["energy", "tax"]:
            score += 2
            breakdown["domain"] = 2
        elif domain == "environmental":
            score += 1
            breakdown["domain"] = 1
        else:
            breakdown["domain"] = 0
        
        self.complexity_score = score
        self.score_breakdown = breakdown
        
        # Route decision: threshold at 3 points
        if score >= 3:
            return Route.ENHANCED
        else:
            return Route.STANDARD
    
    def get_budget_constraints(self, route: Route) -> Dict[str, int]:
        """
        Get token and time budgets for the selected route.
        
        STANDARD: Lower budgets for simple bills (faster, cheaper)
        ENHANCED: Higher budgets for complex bills (thorough, accurate)
        
        Args:
            route: Route.STANDARD or Route.ENHANCED
            
        Returns:
            Dict with token_budget, time_budget_seconds, min_evidence_count
        """
        if route == Route.STANDARD:
            return {
                "token_budget": 50000,
                "time_budget_seconds": 30,
                "min_evidence_count": 1
            }
        else:  # ENHANCED
            return {
                "token_budget": 100000,
                "time_budget_seconds": 300,
                "min_evidence_count": 2
            }
