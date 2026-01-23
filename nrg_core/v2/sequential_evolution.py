"""
Sequential Evolution Agent.

Single agent walks bill versions with structured memory.
Tracks finding origins, modifications, and stability over time.

Design:
- Process versions chronologically
- Maintain findings registry with IDs (F1, F2, etc.)
- Track origin version and modification count
- Compute stability scores based on survival and changes

Why:
- Bills evolve through multiple versions
- Need to track which findings survive changes
- Stability indicates reliability of findings
- Memory enables consistent tracking across versions

Stability Formula:
- origin=1, mods=0 → 0.95 (survived all, stable)
- origin=1, mods=1 → 0.85 (one refinement)
- origin=1, mods=3+ → 0.40 (contentious)
- origin=N (last version) → 0.20 (last-minute, risky)
"""
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class BillVersion:
    """Single version of a bill.
    
    Attributes:
        version_number: Sequential version number (1, 2, 3...)
        text: Full text of this version
        name: Version name ("Introduced", "Engrossed", "Enrolled", etc.)
    """
    version_number: int
    text: str
    name: str  # "Introduced", "Engrossed", etc.


@dataclass
class EvolutionResult:
    """Result of sequential version walk.
    
    Attributes:
        bill_id: Bill identifier
        findings_registry: Dictionary of findings with metadata
        stability_scores: Stability score for each finding (0-1)
    """
    bill_id: str
    findings_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)


EVOLUTION_PROMPT_V1 = """Analyze this bill (Version 1: {version_name}).

BILL TEXT:
{bill_text}

Extract findings that could impact NRG Energy's business.
Assign IDs (F1, F2, etc.) and include supporting quotes from the bill.

REQUIREMENTS:
1. Each finding MUST have at least one verbatim quote from the bill
2. Include the section reference for each quote
3. Estimate impact 0-10 (0=no impact, 10=existential threat)

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",
      "statement": "Clear, specific finding statement",
      "quotes": [
        {{"text": "Exact verbatim quote from bill", "section": "2.1"}}
      ],
      "origin_version": 1,
      "affected_sections": ["2.1"],
      "modification_count": 0,
      "impact_estimate": 7
    }}
  ]
}}

CRITICAL: Every statement must be supported by a direct quote from the bill."""

EVOLUTION_PROMPT_VN = """Analyze Version {version_number}: {version_name}.

PREVIOUS FINDINGS (from memory):
{previous_findings}

NEW VERSION TEXT:
{bill_text}

TASK:
1. Compare to previous findings
2. Mark findings as: STABLE (unchanged), MODIFIED (changed), or NEW
3. Update modification counts
4. Update quotes if text changed

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",
      "statement": "Clear, specific finding statement",
      "quotes": [
        {{"text": "Exact verbatim quote from bill", "section": "2.1"}}
      ],
      "origin_version": 1,
      "modification_count": 1,
      "status": "MODIFIED",
      "impact_estimate": 7
    }}
  ]
}}

CRITICAL: Every statement must be supported by a direct quote from the bill."""


class SequentialEvolutionAgent:
    """
    Walk bill versions sequentially, maintaining structured memory.
    
    Process:
    1. Version 1: Extract initial findings with IDs
    2. Version N+: Compare to memory, update/track changes
    3. Compute stability scores based on survival and modifications
    
    Memory Structure:
    - findings_registry: {id: {statement, origin, mods, status}}
    - stability_scores: {id: score} based on formula
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """
        Initialize sequential evolution agent.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.memory: Dict[str, Any] = {}
    
    def walk_versions(
        self,
        bill_id: str,
        versions: List[BillVersion]
    ) -> EvolutionResult:
        """
        Walk versions in order, maintaining findings registry.
        
        Algorithm:
        1. For each version in chronological order
        2. If first version: extract initial findings
        3. If subsequent: compare to memory and update
        4. After all versions: compute stability scores
        
        Args:
            bill_id: Bill identifier
            versions: List of BillVersion objects in chronological order
        
        Returns:
            EvolutionResult with findings registry and stability scores
        """
        findings_registry = {}
        
        for idx, version in enumerate(versions):
            if idx == 0:
                # First version: extract initial findings
                analysis = self._analyze_version(version, is_first=True)
            else:
                # Subsequent versions: compare to memory
                analysis = self._analyze_version(version, is_first=False, memory=findings_registry)
            
            # Update registry
            for finding in analysis["findings"]:
                finding_id = finding["id"]
                findings_registry[finding_id] = finding
        
        # Compute stability scores
        stability_scores = self._compute_stability(findings_registry, num_versions=len(versions))
        
        return EvolutionResult(
            bill_id=bill_id,
            findings_registry=findings_registry,
            stability_scores=stability_scores
        )
    
    def _analyze_version(
        self,
        version: BillVersion,
        is_first: bool,
        memory: Dict = None
    ) -> Dict[str, Any]:
        """
        Analyze single version with or without memory context.
        
        For first version: Use V1 prompt to extract initial findings
        For subsequent: Use VN prompt with previous findings context
        
        Args:
            version: Bill version to analyze
            is_first: Whether this is the first version
            memory: Previous findings registry (for subsequent versions)
        
        Returns:
            Parsed JSON response with findings
        """
        if is_first:
            prompt = EVOLUTION_PROMPT_V1.format(
                version_name=version.name,
                bill_text=version.text
            )
        else:
            previous_findings_str = json.dumps(list(memory.values()), indent=2)
            prompt = EVOLUTION_PROMPT_VN.format(
                version_number=version.version_number,
                version_name=version.name,
                previous_findings=previous_findings_str,
                bill_text=version.text
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse version analysis response as JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON for version analysis: {e}") from e
    
    def _compute_stability(self, registry: Dict, num_versions: int) -> Dict[str, float]:
        """
        Compute stability score for each finding.
        
        Stability formula reflects legislative reality:
        - Findings that survive unchanged are most reliable
        - Findings with few refinements are still reliable
        - Findings with many changes are contentious
        - Last-minute additions are risky
        
        Args:
            registry: Findings registry with origin and modification counts
            num_versions: Total number of versions processed
        
        Returns:
            Dictionary of stability scores (0-1)
        """
        scores = {}
        for finding_id, finding in registry.items():
            origin = finding.get("origin_version", 1)
            mods = finding.get("modification_count", 0)
            
            if origin == num_versions:
                # Last-minute addition
                score = 0.20
            elif mods == 0:
                # Never modified, very stable
                score = 0.95
            elif mods == 1:
                # One refinement
                score = 0.85
            elif mods == 2:
                # Two changes
                score = 0.70
            else:
                # 3+ modifications, contentious
                score = 0.40
            
            scores[finding_id] = score
        
        return scores
