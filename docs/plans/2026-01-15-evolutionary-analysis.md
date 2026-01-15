# Evolutionary Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Track bill evolution across versions to measure stability and identify contentious provisions, enhancing confidence assessment

**Architecture:** Version diff computation, finding lineage tracking, complexity metrics calculation, and stability scoring based on modification frequency

**Tech Stack:** Python 3.9+, difflib for text comparison, hashlib for content hashing, dataclasses for models

**Business Context:** Stability signal helps identify provisions that survived legislative scrutiny (high stability = less likely to change) vs. contentious provisions (modified frequently = riskier for business planning).

---

## Component 2: Evolutionary Analysis - Visual Flow

```
INPUT: List of bill versions (V1, V2, ..., VN)
│
└─→ Version Diffing (for each consecutive pair)
    │
    ├─ V1 vs V2: difflib.SequenceMatcher
    │  ├─ Additions: New sections not in V1
    │  ├─ Deletions: Sections removed from V1
    │  └─ Modifications: Sections with changed text
    │
    ├─ V2 vs V3: difflib.SequenceMatcher
    │  └─ ... (same pattern)
    │
    └─ Result: Version diff list with all changes

Finding Lineage Tracking (per finding statement)
│
└─→ For each finding from Consensus Ensemble:
    │
    ├─ Search all versions for finding statement
    │  ├─ V1: NOT FOUND | V2: FOUND | V3: MODIFIED
    │  └─ Result: origin_version=2, modification_count=1
    │
    ├─ Stability Scoring (0.0-1.0 scale)
    │  ├─ From V1, 0 mods: 0.95 (survived entire legislative process)
    │  ├─ From V1, 1 mod: 0.85 (minor tweak, mostly stable)
    │  ├─ From V1, 3+ mods: 0.40 (contentious, changed many times)
    │  ├─ Introduced at V3 (final): 0.20 (last-minute addition, unproven)
    │  └─ Rationale: More modifications = less confidence (may change again)
    │
    └─ Output: FindingLineage with stability_score + version_history

Complexity Metrics (per version)
│
└─→ Calculate section-level complexity
    │
    ├─ Measure 1: Number of sections modified
    │  └─ High = complex, fast-changing bill
    │
    ├─ Measure 2: Average modification frequency
    │  └─ Tracks which sections change most often
    │
    └─ Output: ComplexityMetrics per version

Final Output: EvolutionAnalysis
│
└─ Findings with stability scores:
   ├─ {
   │   statement: "Tax applies to >50MW",
   │   origin_version: 1,
   │   modification_count: 1,
   │   stability_score: 0.85,
   │   version_history: [v1: "introduced", v2: "modified", v3: "stable"]
   │  }
   │
   └─ {
       statement: "Effective January 1, 2026",
       origin_version: 1,
       modification_count: 0,
       stability_score: 0.95,
       version_history: [v1: "introduced", v2: "stable", v3: "stable"]
      }
```

---

## Task 1: Version Diff Models and Data Structures

**Files:**
- Create: `nrg_core/sota/evolution/models.py`
- Test: `tests/test_sota/test_evolution_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_evolution_models.py
import pytest
from nrg_core.sota.evolution.models import VersionDiff, FindingLineage, EvolutionAnalysis

def test_version_diff_creation():
    diff = VersionDiff(
        from_version=1,
        to_version=2,
        additions=["Tax on energy generation exceeding 50MW"],
        deletions=["Tax on energy companies"],
        modifications=[{
            "section": "Section 2.1",
            "old": "energy companies",
            "new": "energy generation exceeding 50MW"
        }]
    )

    assert diff.from_version == 1
    assert diff.to_version == 2
    assert len(diff.additions) == 1
    assert len(diff.deletions) == 1
    assert len(diff.modifications) == 1

def test_finding_lineage():
    lineage = FindingLineage(
        finding_statement="Tax applies to >50MW",
        origin_version=1,
        modification_count=1,
        stability_score=0.85,
        version_history=[
            {"version": 1, "status": "introduced"},
            {"version": 2, "status": "modified"},
            {"version": 3, "status": "stable"}
        ]
    )

    assert lineage.origin_version == 1
    assert lineage.modification_count == 1
    assert lineage.stability_score == 0.85
    assert len(lineage.version_history) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_evolution_models.py::test_version_diff_creation -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/evolution/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class VersionDiff:
    """Diff between two consecutive bill versions"""
    from_version: int
    to_version: int
    additions: List[str] = field(default_factory=list)
    deletions: List[str] = field(default_factory=list)
    modifications: List[Dict[str, str]] = field(default_factory=list)
    affected_sections: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'from_version': self.from_version,
            'to_version': self.to_version,
            'additions': self.additions,
            'deletions': self.deletions,
            'modifications': self.modifications,
            'affected_sections': self.affected_sections
        }

@dataclass
class FindingLineage:
    """Tracks a finding's evolution across versions"""
    finding_statement: str
    origin_version: int
    modification_count: int
    stability_score: float
    version_history: List[Dict] = field(default_factory=list)
    contentious: bool = False

    def to_dict(self) -> dict:
        return {
            'finding_statement': self.finding_statement,
            'origin_version': self.origin_version,
            'modification_count': self.modification_count,
            'stability_score': self.stability_score,
            'version_history': self.version_history,
            'contentious': self.contentious
        }

@dataclass
class ComplexityMetrics:
    """Complexity metrics across versions"""
    section_counts: List[int] = field(default_factory=list)
    definition_counts: List[int] = field(default_factory=list)
    vocabulary_sizes: List[int] = field(default_factory=list)
    growth_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            'section_counts': self.section_counts,
            'definition_counts': self.definition_counts,
            'vocabulary_sizes': self.vocabulary_sizes,
            'growth_rate': self.growth_rate
        }

@dataclass
class EvolutionAnalysis:
    """Complete evolutionary analysis result"""
    version_diffs: List[VersionDiff] = field(default_factory=list)
    finding_lineages: List[FindingLineage] = field(default_factory=list)
    complexity_metrics: Optional[ComplexityMetrics] = None
    overall_assessment: str = ""

    def to_dict(self) -> dict:
        return {
            'version_diffs': [v.to_dict() for v in self.version_diffs],
            'finding_lineages': [f.to_dict() for f in self.finding_lineages],
            'complexity_metrics': self.complexity_metrics.to_dict() if self.complexity_metrics else None,
            'overall_assessment': self.overall_assessment
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_evolution_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/evolution/models.py tests/test_sota/test_evolution_models.py
git commit -m "feat(sota): add evolutionary analysis data models

- Add VersionDiff for tracking version changes
- Add FindingLineage for tracking finding evolution
- Add ComplexityMetrics for bill complexity measurement
- Add EvolutionAnalysis result container
- Add tests for model creation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Version Diff Computation Engine

**Files:**
- Create: `nrg_core/sota/evolution/differ.py`
- Test: `tests/test_sota/test_evolution_differ.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_evolution_differ.py
import pytest
from nrg_core.sota.evolution.differ import VersionDiffer

def test_text_diff_detection():
    version1_text = """
    Section 1: Tax on energy companies
    Section 2: Reporting requirements
    """

    version2_text = """
    Section 1: Tax on energy generation exceeding 50 megawatts
    Section 2: Reporting requirements
    Section 3: Exemptions for renewable energy
    """

    differ = VersionDiffer()
    diff = differ.compute_diff(
        version1_text=version1_text,
        version2_text=version2_text,
        version1_num=1,
        version2_num=2
    )

    assert diff.from_version == 1
    assert diff.to_version == 2
    assert len(diff.additions) > 0
    assert any("renewable" in add.lower() for add in diff.additions)
    assert len(diff.modifications) > 0

def test_section_extraction():
    differ = VersionDiffer()
    text = """
    Section 1: Title
    Some content here.

    Section 2: Another Title
    More content.
    """

    sections = differ.extract_sections(text)
    assert len(sections) >= 2
    assert "Section 1" in sections[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_evolution_differ.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/evolution/differ.py
import difflib
import re
from typing import List, Dict, Tuple
from nrg_core.sota.evolution.models import VersionDiff

class VersionDiffer:
    """Computes diffs between bill versions"""

    def compute_diff(
        self,
        version1_text: str,
        version2_text: str,
        version1_num: int,
        version2_num: int
    ) -> VersionDiff:
        """Compute diff between two version texts"""
        # Split into lines for diffing
        lines1 = version1_text.split('\n')
        lines2 = version2_text.split('\n')

        # Compute unified diff
        diff = difflib.unified_diff(lines1, lines2, lineterm='')

        additions = []
        deletions = []
        modifications = []

        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                additions.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                deletions.append(line[1:].strip())

        # Detect modifications (lines that appear in both additions and deletions with similar content)
        for deletion in deletions[:]:
            for addition in additions[:]:
                # Check if similar (simple heuristic: share keywords)
                similarity = self._calculate_similarity(deletion, addition)
                if similarity > 0.5:
                    modifications.append({
                        'old': deletion,
                        'new': addition,
                        'similarity': similarity
                    })
                    # Remove from lists to avoid duplicates
                    if deletion in deletions:
                        deletions.remove(deletion)
                    if addition in additions:
                        additions.remove(addition)
                    break

        # Extract affected sections
        affected_sections = self._extract_affected_sections(additions + deletions)

        return VersionDiff(
            from_version=version1_num,
            to_version=version2_num,
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            affected_sections=affected_sections
        )

    def extract_sections(self, text: str) -> List[str]:
        """Extract section text from bill"""
        # Simple regex to find sections
        section_pattern = r'Section\s+\d+[:\.].*?(?=Section\s+\d+[:\.]|$)'
        sections = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)
        return sections

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _extract_affected_sections(self, changed_lines: List[str]) -> List[str]:
        """Extract section identifiers from changed lines"""
        sections = set()
        for line in changed_lines:
            # Look for "Section X" patterns
            match = re.search(r'Section\s+(\d+\.?\d*)', line, re.IGNORECASE)
            if match:
                sections.add(match.group(0))
        return list(sections)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_evolution_differ.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/evolution/differ.py tests/test_sota/test_evolution_differ.py
git commit -m "feat(sota): add version diff computation engine

- Add VersionDiffer for computing text diffs
- Implement additions, deletions, modifications detection
- Add section extraction and affected section tracking
- Add similarity-based modification pairing
- Add tests for diff computation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 2

**Q1: Modification Detection Granularity**
- **Gap**: Currently tracks statement-level modifications
- **Question**: Should we track sub-clause modifications (e.g., threshold change 50MW→25MW)?
- **Priority**: Low | **Blocker**: No - statement-level sufficient for MVP

**Q2: Section Numbering Changes** ⚠️ BLOCKING
- **Gap**: If section numbers change (renumbering), lineage tracking may break
- **Question**: How to handle section renumbering between versions?
- **Priority**: Medium | **Blocker**: YES
- **Recommended Answer**: Use content-based matching instead of section numbers. Match sections by cosine similarity of content, not section number.

---

## Task 3: Finding Lineage Tracker

**Files:**
- Create: `nrg_core/sota/evolution/lineage.py`
- Test: `tests/test_sota/test_evolution_lineage.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_evolution_lineage.py
import pytest
from nrg_core.sota.evolution.lineage import LineageTracker
from nrg_core.sota.models import Finding, ConsensusAnalysis

def test_lineage_tracking():
    # Version 1 findings
    v1_findings = [
        Finding(
            statement="Tax on energy companies",
            confidence=0.9,
            supporting_quotes=["Section 1"],
            found_by=["GPT-4o", "Claude"],
            consensus_level="majority"
        )
    ]
    v1_analysis = ConsensusAnalysis(findings=v1_findings, overall_confidence=0.9)

    # Version 2 findings (modified)
    v2_findings = [
        Finding(
            statement="Tax on energy generation exceeding 50MW",
            confidence=0.95,
            supporting_quotes=["Section 1"],
            found_by=["Gemini", "GPT-4o", "Claude"],
            consensus_level="unanimous"
        )
    ]
    v2_analysis = ConsensusAnalysis(findings=v2_findings, overall_confidence=0.95)

    # Version 3 findings (stable)
    v3_findings = v2_findings  # Same as v2
    v3_analysis = ConsensusAnalysis(findings=v3_findings, overall_confidence=0.95)

    # Track lineage
    tracker = LineageTracker(similarity_threshold=0.6)
    lineages = tracker.track_lineages([
        (1, v1_analysis),
        (2, v2_analysis),
        (3, v3_analysis)
    ])

    assert len(lineages) >= 1
    lineage = lineages[0]
    assert lineage.origin_version == 1
    assert lineage.modification_count == 1  # Modified v1→v2
    assert lineage.stability_score > 0.7  # Stable v2→v3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_evolution_lineage.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/evolution/lineage.py
from typing import List, Tuple, Dict
from nrg_core.sota.models import ConsensusAnalysis
from nrg_core.sota.evolution.models import FindingLineage
from nrg_core.sota.clustering import SemanticClusterer

class LineageTracker:
    """Track finding lineages across versions"""

    def __init__(self, similarity_threshold: float = 0.75):
        self.clusterer = SemanticClusterer(similarity_threshold=similarity_threshold)
        self.threshold = similarity_threshold

    def track_lineages(
        self,
        version_analyses: List[Tuple[int, ConsensusAnalysis]]
    ) -> List[FindingLineage]:
        """
        Track how findings evolve across versions

        Args:
            version_analyses: List of (version_number, ConsensusAnalysis) tuples

        Returns:
            List of FindingLineage objects tracking evolution
        """
        if not version_analyses:
            return []

        # Sort by version number
        version_analyses = sorted(version_analyses, key=lambda x: x[0])

        # Get final version findings
        final_version, final_analysis = version_analyses[-1]
        final_findings = final_analysis.findings

        # Track lineage for each final finding
        lineages = []
        for finding in final_findings:
            lineage = self._trace_lineage(finding, version_analyses)
            lineages.append(lineage)

        return lineages

    def _trace_lineage(
        self,
        final_finding,
        version_analyses: List[Tuple[int, ConsensusAnalysis]]
    ) -> FindingLineage:
        """Trace a finding back through versions"""
        version_history = []
        modification_count = 0
        origin_version = version_analyses[-1][0]  # Default to final version

        # Walk backwards through versions
        current_statement = final_finding.statement
        for i in range(len(version_analyses) - 1, -1, -1):
            version_num, analysis = version_analyses[i]

            # Find matching finding in this version
            match = self._find_matching_finding(current_statement, analysis.findings)

            if match:
                status = "stable"
                if i > 0:
                    # Check if modified from previous version
                    prev_version, prev_analysis = version_analyses[i - 1]
                    prev_match = self._find_matching_finding(current_statement, prev_analysis.findings)

                    if prev_match and prev_match.statement != match.statement:
                        status = "modified"
                        modification_count += 1
                        current_statement = prev_match.statement
                    elif not prev_match:
                        status = "introduced"
                        origin_version = version_num
                else:
                    status = "introduced"
                    origin_version = version_num

                version_history.insert(0, {
                    'version': version_num,
                    'status': status,
                    'statement': match.statement,
                    'confidence': match.confidence
                })
            else:
                # Not found in this version
                if i == 0:
                    origin_version = version_analyses[i + 1][0] if i + 1 < len(version_analyses) else version_num
                break

        # Calculate stability score
        stability_score = self._calculate_stability(origin_version, modification_count, len(version_analyses))

        # Determine if contentious (3+ modifications)
        contentious = modification_count >= 3

        return FindingLineage(
            finding_statement=final_finding.statement,
            origin_version=origin_version,
            modification_count=modification_count,
            stability_score=stability_score,
            version_history=version_history,
            contentious=contentious
        )

    def _find_matching_finding(self, statement: str, findings: List) -> any:
        """Find matching finding by semantic similarity"""
        for finding in findings:
            similarity = self.clusterer.calculate_similarity(statement, finding.statement)
            if similarity >= self.threshold:
                return finding
        return None

    def _calculate_stability(
        self,
        origin_version: int,
        modification_count: int,
        total_versions: int
    ) -> float:
        """
        Calculate stability score (0-1 scale).

        Hypothesis: Provisions that survived legislative scrutiny (unchanged or minimally modified)
        are less likely to be challenged or overturned. Last-minute additions are risky.

        Scoring methodology:
        - 0.95: Survived from V1 with zero modifications (legislative consensus)
        - 0.85: Survived from V1 with 1 modification (minor tweaks, core concept stable)
        - 0.70: Introduced later but stable (good enough if not contentious)
        - 0.60: Introduced later, 2 modifications (moderate change risk)
        - 0.40: 3+ modifications across versions (contentious, frequent changes = risky)
        - 0.20: Added in final version only (no time for legislative refinement)

        Business rationale:
        - High stability (>0.80) = provision likely to survive legal/legislative challenges
        - Low stability (<0.40) = provision may be repealed or modified in future amendments
        - Stability feeds into Confidence Aggregation (10% weight) to lower overall confidence
          for risky (unstable) findings

        Post-MVP: Calibrate thresholds on 100-bill dataset to validate stability-risk correlation
        TODO: Build validation that stability scores correlate with actual future amendments
        """
        # Check if provision was in bill from the start
        from_first = (origin_version == 1)

        # Check if provision was added very late (only in final version)
        recently_added = (origin_version == total_versions)

        # Decision tree (order matters: most specific cases first)
        if recently_added:
            # Last-minute additions are unproven and risky
            return 0.20
        elif modification_count >= 3:
            # Contentious provisions that keep changing = instability signal
            return 0.40
        elif from_first and modification_count == 0:
            # Original provision, unchanged = strongest legislative consensus
            return 0.95
        elif from_first and modification_count == 1:
            # Original provision, minor tweak = mostly settled
            return 0.85
        elif modification_count <= 1:
            # Introduced later but not heavily modified
            return 0.70
        else:
            # Introduced later, multiple modifications = moderate instability
            return 0.60
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_evolution_lineage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/evolution/lineage.py tests/test_sota/test_evolution_lineage.py
git commit -m "feat(sota): add finding lineage tracking

- Add LineageTracker for tracing findings across versions
- Implement semantic matching of findings across versions
- Add stability score calculation based on origin and modifications
- Add contentious flag for heavily modified findings (3+)
- Add version history tracking per finding
- Add tests for lineage tracking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 3

**Q3: Similarity Threshold for Lineage Matching**
- **Gap**: 0.75 threshold for matching findings across versions not validated
- **Question**: Should we A/B test thresholds? What's optimal for different bill types?
- **Priority**: Medium | **Blocker**: No - can validate with test set

**Q4: Contentious Threshold**
- **Gap**: 3+ modifications marked as contentious, but not validated
- **Question**: Is 3 the right threshold? Vary by bill type?
- **Priority**: Low | **Blocker**: No - reasonable heuristic for MVP

**Q5: Performance with Many Versions**
- **Gap**: Lineage tracking is O(n²) in number of versions
- **Question**: Optimize for bills with 20+ versions?
- **Priority**: Low | **Blocker**: No - most bills have <10 versions

---

## Task 4: Complexity Metrics Calculator

**Files:**
- Create: `nrg_core/sota/evolution/complexity.py`
- Test: `tests/test_sota/test_evolution_complexity.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_evolution_complexity.py
import pytest
from nrg_core.sota.evolution.complexity import ComplexityCalculator

def test_section_counting():
    v1_text = """
    Section 1: Introduction
    Section 2: Definitions
    """

    v2_text = """
    Section 1: Introduction
    Section 2: Definitions
    Section 3: Requirements
    Section 4: Penalties
    """

    calculator = ComplexityCalculator()
    v1_metrics = calculator.calculate_metrics(v1_text)
    v2_metrics = calculator.calculate_metrics(v2_text)

    assert v1_metrics['section_count'] == 2
    assert v2_metrics['section_count'] == 4

def test_definition_counting():
    text = """
    As used in this Act:
    (1) "Energy generation" means commercial production
    (2) "Renewable energy" means solar or wind
    (3) "Capacity" means maximum output
    """

    calculator = ComplexityCalculator()
    metrics = calculator.calculate_metrics(text)

    assert metrics['definition_count'] >= 3

def test_vocabulary_size():
    text = "Tax tax tax applies applies bill bill bill"
    calculator = ComplexityCalculator()
    metrics = calculator.calculate_metrics(text)

    # 3 unique words: tax, applies, bill
    assert metrics['vocabulary_size'] >= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_evolution_complexity.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/evolution/complexity.py
import re
from typing import Dict, List
from nrg_core.sota.evolution.models import ComplexityMetrics

class ComplexityCalculator:
    """Calculate complexity metrics for bill texts"""

    def calculate_metrics(self, text: str) -> Dict:
        """Calculate all complexity metrics for a single version"""
        return {
            'section_count': self._count_sections(text),
            'definition_count': self._count_definitions(text),
            'vocabulary_size': self._count_vocabulary(text)
        }

    def calculate_growth_metrics(
        self,
        version_texts: List[str]
    ) -> ComplexityMetrics:
        """Calculate complexity growth across versions"""
        section_counts = []
        definition_counts = []
        vocabulary_sizes = []

        for text in version_texts:
            metrics = self.calculate_metrics(text)
            section_counts.append(metrics['section_count'])
            definition_counts.append(metrics['definition_count'])
            vocabulary_sizes.append(metrics['vocabulary_size'])

        # Calculate growth rate (average per-version growth)
        growth_rate = 0.0
        if len(section_counts) > 1:
            growth_rates = []
            for i in range(1, len(section_counts)):
                if section_counts[i - 1] > 0:
                    rate = (section_counts[i] - section_counts[i - 1]) / section_counts[i - 1]
                    growth_rates.append(rate)
            if growth_rates:
                growth_rate = sum(growth_rates) / len(growth_rates)

        return ComplexityMetrics(
            section_counts=section_counts,
            definition_counts=definition_counts,
            vocabulary_sizes=vocabulary_sizes,
            growth_rate=growth_rate
        )

    def _count_sections(self, text: str) -> int:
        """Count number of sections in bill"""
        # Match "Section X" patterns
        pattern = r'Section\s+\d+'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(set(matches))  # Use set to avoid double-counting

    def _count_definitions(self, text: str) -> int:
        """Count defined terms in bill"""
        # Look for definition patterns:
        # - "term" means
        # - (1) "term"
        # - As used in this Act, "term"

        definition_patterns = [
            r'["""]([^"""]+)[""]\s+means',  # "term" means
            r'\(\d+\)\s+["""]([^"""]+)["""]',  # (1) "term"
            r'["""]([^"""]+)["""].*?shall\s+mean',  # "term" shall mean
        ]

        definitions = set()
        for pattern in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            definitions.update(matches)

        return len(definitions)

    def _count_vocabulary(self, text: str) -> int:
        """Count unique words in text"""
        # Tokenize and normalize
        words = re.findall(r'\b\w+\b', text.lower())
        return len(set(words))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_evolution_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/evolution/complexity.py tests/test_sota/test_evolution_complexity.py
git commit -m "feat(sota): add complexity metrics calculator

- Add ComplexityCalculator for bill text analysis
- Implement section counting
- Implement definition counting with multiple patterns
- Implement vocabulary size calculation
- Add growth rate calculation across versions
- Add tests for complexity metrics

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 4

**Q6: Definition Extraction Accuracy**
- **Gap**: Regex patterns may miss non-standard definition formats
- **Question**: Should we use NER or more sophisticated parsing?
- **Priority**: Low | **Blocker**: No - regex adequate for common patterns

**Q7: Vocabulary Growth Interpretation**
- **Gap**: Vocabulary size increases, but unclear what's meaningful
- **Question**: What vocabulary growth rate indicates problematic complexity?
- **Priority**: Low | **Blocker**: No - informational metric only

---

## Task 5: Main Evolution Analyzer

**Files:**
- Create: `nrg_core/sota/evolution/analyzer.py`
- Test: `tests/test_sota/test_evolution_analyzer.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_evolution_analyzer.py
import pytest
from nrg_core.sota.evolution.analyzer import EvolutionAnalyzer
from nrg_core.models import BillVersion
from nrg_core.sota.models import Finding, ConsensusAnalysis

@pytest.mark.asyncio
async def test_full_evolution_analysis():
    # Create bill versions
    versions = [
        BillVersion(
            version_number=1,
            version_type_raw="Introduced",
            version_type_normalized="introduced",
            version_date="2024-01-01",
            full_text="Section 1: Tax on energy companies",
            text_hash="hash1",
            word_count=10
        ),
        BillVersion(
            version_number=2,
            version_type_raw="Engrossed",
            version_type_normalized="passed_originating_chamber",
            version_date="2024-02-01",
            full_text="Section 1: Tax on energy generation exceeding 50MW",
            text_hash="hash2",
            word_count=15
        ),
        BillVersion(
            version_number=3,
            version_type_raw="Enrolled",
            version_type_normalized="enrolled",
            version_date="2024-03-01",
            full_text="Section 1: Tax on energy generation exceeding 50MW\nSection 2: Renewable exempt",
            text_hash="hash3",
            word_count=20
        )
    ]

    # Create consensus analyses for each version
    consensus_analyses = [
        (1, ConsensusAnalysis(
            findings=[Finding(
                statement="Tax on energy companies",
                confidence=0.9,
                supporting_quotes=["Section 1"],
                found_by=["GPT-4o"],
                consensus_level="majority"
            )],
            overall_confidence=0.9
        )),
        (2, ConsensusAnalysis(
            findings=[Finding(
                statement="Tax on energy generation >50MW",
                confidence=0.95,
                supporting_quotes=["Section 1"],
                found_by=["Gemini", "GPT-4o", "Claude"],
                consensus_level="unanimous"
            )],
            overall_confidence=0.95
        )),
        (3, ConsensusAnalysis(
            findings=[
                Finding(
                    statement="Tax on energy generation >50MW",
                    confidence=0.95,
                    supporting_quotes=["Section 1"],
                    found_by=["Gemini", "GPT-4o", "Claude"],
                    consensus_level="unanimous"
                ),
                Finding(
                    statement="Renewable energy exempt",
                    confidence=0.85,
                    supporting_quotes=["Section 2"],
                    found_by=["GPT-4o", "Claude"],
                    consensus_level="majority"
                )
            ],
            overall_confidence=0.90
        ))
    ]

    # Analyze evolution
    analyzer = EvolutionAnalyzer()
    result = await analyzer.analyze(versions, consensus_analyses)

    assert len(result.version_diffs) == 2  # v1→v2, v2→v3
    assert len(result.finding_lineages) >= 1
    assert result.complexity_metrics is not None
    assert result.complexity_metrics.growth_rate > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_evolution_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/evolution/analyzer.py
from typing import List, Tuple
from nrg_core.models import BillVersion
from nrg_core.sota.models import ConsensusAnalysis
from nrg_core.sota.evolution.models import EvolutionAnalysis
from nrg_core.sota.evolution.differ import VersionDiffer
from nrg_core.sota.evolution.lineage import LineageTracker
from nrg_core.sota.evolution.complexity import ComplexityCalculator

class EvolutionAnalyzer:
    """Main orchestrator for evolutionary analysis"""

    def __init__(self, similarity_threshold: float = 0.75):
        self.differ = VersionDiffer()
        self.lineage_tracker = LineageTracker(similarity_threshold=similarity_threshold)
        self.complexity_calculator = ComplexityCalculator()

    async def analyze(
        self,
        versions: List[BillVersion],
        consensus_analyses: List[Tuple[int, ConsensusAnalysis]]
    ) -> EvolutionAnalysis:
        """
        Perform complete evolutionary analysis

        Args:
            versions: List of BillVersion objects
            consensus_analyses: List of (version_num, ConsensusAnalysis) tuples

        Returns:
            EvolutionAnalysis with diffs, lineages, and complexity metrics
        """
        # Sort versions by version number
        versions = sorted(versions, key=lambda v: v.version_number)

        # 1. Compute version diffs
        version_diffs = []
        for i in range(len(versions) - 1):
            diff = self.differ.compute_diff(
                version1_text=versions[i].full_text,
                version2_text=versions[i + 1].full_text,
                version1_num=versions[i].version_number,
                version2_num=versions[i + 1].version_number
            )
            version_diffs.append(diff)

        # 2. Track finding lineages
        finding_lineages = self.lineage_tracker.track_lineages(consensus_analyses)

        # 3. Calculate complexity metrics
        version_texts = [v.full_text for v in versions]
        complexity_metrics = self.complexity_calculator.calculate_growth_metrics(version_texts)

        # 4. Generate overall assessment
        overall_assessment = self._generate_assessment(
            version_diffs=version_diffs,
            finding_lineages=finding_lineages,
            complexity_metrics=complexity_metrics
        )

        return EvolutionAnalysis(
            version_diffs=version_diffs,
            finding_lineages=finding_lineages,
            complexity_metrics=complexity_metrics,
            overall_assessment=overall_assessment
        )

    def _generate_assessment(
        self,
        version_diffs,
        finding_lineages,
        complexity_metrics
    ) -> str:
        """Generate human-readable assessment"""
        num_versions = len(version_diffs) + 1
        num_contentious = sum(1 for l in finding_lineages if l.contentious)
        avg_stability = sum(l.stability_score for l in finding_lineages) / len(finding_lineages) if finding_lineages else 0

        assessment = f"Bill evolved through {num_versions} versions. "

        if complexity_metrics.growth_rate > 0.3:
            assessment += "Significant complexity growth detected. "
        elif complexity_metrics.growth_rate > 0:
            assessment += "Moderate complexity growth. "
        else:
            assessment += "Complexity remained stable. "

        if num_contentious > 0:
            assessment += f"{num_contentious} contentious provisions (3+ modifications). "

        if avg_stability > 0.8:
            assessment += "Most provisions stable."
        elif avg_stability > 0.6:
            assessment += "Provisions moderately stable."
        else:
            assessment += "Significant changes across versions."

        return assessment
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_evolution_analyzer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/evolution/analyzer.py tests/test_sota/test_evolution_analyzer.py
git commit -m "feat(sota): add evolution analyzer orchestrator

- Add EvolutionAnalyzer main class
- Integrate differ, lineage tracker, and complexity calculator
- Add overall assessment generation
- Add comprehensive tests for full pipeline
- Support async analysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 5

**Q8: Missing Versions** ⚠️ BLOCKING
- **Gap**: No handling for gaps in version sequence (v1, v3, v5)
- **Question**: How should we handle missing intermediate versions?
- **Priority**: Medium | **Blocker**: YES
- **Recommended Answer**: Process available versions only. Mark lineages as "incomplete version history" if gaps detected.

---

## Task 6: Integration and Package Initialization

**Files:**
- Create: `nrg_core/sota/evolution/__init__.py`
- Create: `nrg_core/sota/evolution/README.md`

**Step 1: Create package initialization**

```python
# nrg_core/sota/evolution/__init__.py
"""
Evolutionary Analysis Component

Tracks bill evolution across versions to measure stability and identify
contentious provisions.
"""

from nrg_core.sota.evolution.analyzer import EvolutionAnalyzer
from nrg_core.sota.evolution.models import (
    VersionDiff,
    FindingLineage,
    ComplexityMetrics,
    EvolutionAnalysis
)
from nrg_core.sota.evolution.differ import VersionDiffer
from nrg_core.sota.evolution.lineage import LineageTracker
from nrg_core.sota.evolution.complexity import ComplexityCalculator

__all__ = [
    'EvolutionAnalyzer',
    'VersionDiff',
    'FindingLineage',
    'ComplexityMetrics',
    'EvolutionAnalysis',
    'VersionDiffer',
    'LineageTracker',
    'ComplexityCalculator'
]
```

**Step 2: Create README**

```markdown
# Evolutionary Analysis Component

## Overview

Tracks how bill provisions change across versions to measure stability and identify contentious areas.

## Key Features

- **Version Diffing**: Detects additions, deletions, modifications between versions
- **Lineage Tracking**: Traces findings back through version history
- **Stability Scoring**: Quantifies provision stability (0.20-0.95)
- **Complexity Metrics**: Measures bill growth and complexity

## Stability Scoring

- **0.95**: Present from v1, no modifications
- **0.85**: Present from v1, modified once, then stable
- **0.70**: Introduced later, 1-2 modifications
- **0.40**: 3+ modifications (contentious)
- **0.20**: Recently added (final version only)

## Usage

```python
from nrg_core.sota.evolution import EvolutionAnalyzer

analyzer = EvolutionAnalyzer()
result = await analyzer.analyze(versions, consensus_analyses)

for lineage in result.finding_lineages:
    print(f"{lineage.finding_statement}")
    print(f"Origin: v{lineage.origin_version}")
    print(f"Modifications: {lineage.modification_count}")
    print(f"Stability: {lineage.stability_score:.2f}")
    if lineage.contentious:
        print("⚠️ CONTENTIOUS")
```
```

**Step 3: Commit**

```bash
git add nrg_core/sota/evolution/__init__.py nrg_core/sota/evolution/README.md
git commit -m "docs(sota): add evolutionary analysis documentation

- Add package __init__ with public API
- Add README with stability scoring reference
- Add usage examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary of Implementation Questions

All gaps and clarification questions have been distributed into the relevant tasks above. Look for sections marked "Implementation Questions for Task X" after each task's commit step.

**Blocking Questions** (marked with ⚠️):
- Task 2, Q2: Section Numbering Changes
- Task 5, Q8: Missing Versions

**See** [GAPS_AND_QUESTIONS.md](./GAPS_AND_QUESTIONS.md) for complete cross-component questions and integration issues.

---
