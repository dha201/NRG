# Phase 2: Reference Materials (Generated from Phase 1 Exploration)

This document contains all reference tables, API signatures, and extension guides generated during Phase 1 exploration for integration into Architecture_v2.md during Phase 3.

---

## Table 1: Component Implementation Map

| Architecture Concept | Implementation | File | Key Class | Main Method | Responsibility |
|---|---|---|---|---|---|
| **Sequential Evolution Agent** | Finding extraction & version tracking | `nrg_core/v2/sequential_evolution.py` | `SequentialEvolutionAgent` | `walk_versions(bill_id, versions)` → `EvolutionResult` | Walk bill versions chronologically, extract findings, track origin/modifications |
| **Two-Tier Orchestrator** | Validation coordination & pipeline control | `nrg_core/v2/two_tier.py` | `TwoTierOrchestrator` | `validate(bill_id, bill_text, nrg_context, findings_registry, stability_scores)` → `TwoTierAnalysisResult` | Orchestrate multi-tier validation: multi-sample (1.5) → judge (2) → fallback (2.5) |
| **Judge Model** | Finding validation & rubric scoring | `nrg_core/v2/judge.py` | `JudgeModel` | `validate(finding_id, finding, bill_text)` → `JudgeValidation` | Verify quotes, detect hallucinations, score evidence quality, ambiguity, and 4 rubric dimensions |
| **Multi-Sample Checker** | Tier 1.5 consistency validation | `nrg_core/v2/multi_sample.py` | `MultiSampleChecker` | `check_consistency(bill_id, bill_text, nrg_context)` → `ConsensusResult` | Run 2-3 analysis samples, deduplicate via TF-IDF similarity (0.85 threshold), compute consensus |
| **Fallback Analyst** | Tier 2.5 second opinion | `nrg_core/v2/fallback.py` | `FallbackAnalyst` | `get_second_opinion(finding, bill_text)` → `SecondOpinion` | Get Claude Opus's assessment for uncertain high-impact findings (architectural diversity) |
| **Rubrics Engine** | Configuration module for dimension scoring | `nrg_core/v2/rubrics.py` | `ALL_RUBRICS` (dict) | `format_rubric_scale(rubric)` → str | Define 4 rubric dimensions (legal_risk, financial_impact, operational_disruption, ambiguity_risk) with anchored 0-10 scales |
| **Audit Trail Generator** | Compliance documentation | `nrg_core/v2/audit_trail.py` | `AuditTrailGenerator` | `generate(finding, validation, rubric_scores)` → Dict | Generate compliance-ready audit trail with timestamps, findings, quotes, validations, scores |
| **Deep Research Agent** | Phase 4: External context enrichment | `nrg_core/v2/deep_research/research_agent.py` | `DeepResearchAgent` | `research(finding, bill_text, max_sources=5)` → `ResearchResult` | Query OpenStates, BillTrack50, Congress.gov for relevant legislative precedents |
| **Reference Detector** | Phase 4: Cross-bill citation detection | `nrg_core/v2/cross_bill/reference_detector.py` | `ReferenceDetector` | `detect(bill_text)` → List[BillReference] | Regex-based detection of U.S. Code citations, amendments, definitions, Public Law refs |
| **Base LLM Agent** | Abstract base for LLM-powered agents | `nrg_core/v2/base.py` | `BaseLLMAgent` | `__init__(model="gpt-4o", api_key=None)` | Initialize OpenAI client, manage API keys, error handling |
| **Configuration** | Centralized threshold constants | `nrg_core/v2/config.py` | `ThresholdConfig` (frozen dataclass) | `DEFAULT_CONFIG` (singleton) | Judge confidence: [0.6, 0.8], Fallback impact: 7, Multi-sample: impact >= 6 OR confidence < 0.7 |
| **Exceptions** | Custom exception hierarchy | `nrg_core/v2/exceptions.py` | `NRGAnalysisError`, `LLMResponseError`, `ValidationError`, etc. | — | Error handling and diagnostics across V2 pipeline |

---

## Table 2: Data Model Reference

| Concept | Data Model | Location | Key Fields | Usage | Validation |
|---|---|---|---|---|---|
| **Supporting Evidence** | `Quote` (Pydantic) | `nrg_core/models_v2.py:22` | `text` (str, min 10 chars), `section` (str), `page` (int \| None) | Part of Finding.quotes; used in validation and scoring | min_length=10 enforced on text field |
| **Analytical Finding** | `Finding` (Pydantic) | `nrg_core/models_v2.py:34` | `finding_id` (str), `statement` (str, min 20), `quotes` (List[Quote], min 1), `confidence` (float 0-1), `impact_estimate` (int 0-10) | Output of Sequential Evolution; input to validation; core model throughout | Field validator enforces ≥1 quote (line 50-56) |
| **Tier 1 Output** | `PrimaryAnalysis` (Pydantic) | `nrg_core/models_v2.py:82` | `bill_id`, `findings` (List[Finding]), `overall_confidence` (0-1), `requires_multi_sample` (bool), `timestamp` | Sequential Evolution output; input to Two-Tier Validation | Triggers multi-sample if impact≥6 OR confidence<0.7 |
| **Tier 2 Validation** | `JudgeValidation` (Pydantic) | `nrg_core/models_v2.py:100` | `finding_id`, `quote_verified` (bool), `hallucination_detected` (bool), `evidence_quality` (0-1), `ambiguity` (0-1), `judge_confidence` (0-1) | Judge's validation decision per finding; enables filtering and fallback routing | Triggers fallback if confidence in [0.6, 0.8] AND impact≥7 |
| **Dimension Score** | `RubricScore` (Pydantic) | `nrg_core/models_v2.py:59` | `dimension` (legal_risk, financial_impact, operational_disruption, ambiguity_risk), `score` (0-10), `rationale` (str, min 50), `evidence` (List[Quote]), `rubric_anchor` (str) | Output of Judge scoring; used in reports and filtering | Enforces min 50-char rationale for explainability |
| **Fallback Opinion** | `FallbackResult` (Pydantic) | `nrg_core/models_v2.py:142` | `finding_index`, `agrees` (bool), `alternative_interpretation` (str), `rationale`, `confidence` (0-1) | Tier 2.5 second opinion; optional component | Only generated if triggers met (confidence [0.6, 0.8] AND impact≥7) |
| **Research Insight** | `ResearchInsight` (Pydantic) | `nrg_core/models_v2.py:120` | `claim`, `source_url`, `snippet` (str), `relevance` (high/medium/low), `checker_validated` (bool), `trust` (0-1) | Phase 4 deep research output; optional enrichment | Trust score computed from source count + relevance |
| **Complete Result** | `TwoTierAnalysisResult` (Pydantic) | `nrg_core/models_v2.py:163` | `bill_id`, `primary_analysis` (PrimaryAnalysis), `judge_validations` (List[JudgeValidation]), `rubric_scores` (List[RubricScore]), `audit_trails` (List[Dict]), `multi_sample_agreement` (float \| None), `second_model_reviewed` (bool), `fallback_results` (List[FallbackResult]), `research_insights` (List[ResearchInsight]), `cross_bill_references` (Dict), `route` (STANDARD/ENHANCED), `cost_estimate` (float) | Final API output; aggregates all pipeline results | Aggregate container for complete analysis |

---

## Table 3: Configuration Reference

| Feature | Config Section | Key Options | Default Values | File | Purpose |
|---|---|---|---|---|---|
| **Logging & Traces** | `logging:` | `level`, `trace.*` (seq_evolution, two_tier, rubric, hallucination, stability) | level="normal" | `config.yaml:10-39` | Control verbosity and diagnostic output |
| **LLM Provider** | `llm:` | `provider` (gemini/openai/anthropic), per-provider config | provider="gemini" | `config.yaml:43-62` | Select LLM and configure model/temperature |
| **Complexity Routing** | `v2.orchestration:` | `complexity_thresholds` (standard_max=2, enhanced_min=3), `scoring` weights | — | `config.yaml:69-83` | Route bills to STANDARD or ENHANCED path based on complexity |
| **Sequential Evolution** | `v2.sequential_evolution:` | `enabled`, `max_memory_tokens` (2000), `extract_quotes`, `track_modifications` | enabled=true | `config.yaml:85-90` | Control extraction behavior and memory limits |
| **Tier 1.5: Multi-Sample** | `v2.two_tier.multi_sample:` | `enabled`, `samples` (3), `agreement_threshold` (0.85), `impact_trigger` (6), `confidence_trigger` (0.7) | enabled=true | `config.yaml:92-103` | Configure consistency validation triggers and thresholds |
| **Tier 2: Judge** | `v2.two_tier.judge:` | `enabled`, `verify_quotes`, `detect_hallucinations` | enabled=true | `config.yaml:105-110` | Enable/disable judge validation checks |
| **Tier 2.5: Fallback** | `v2.two_tier.fallback:` | `enabled`, `confidence_range` ([0.6, 0.8]), `impact_threshold` (7), `provider` (anthropic) | enabled=true | `config.yaml:112-117` | Configure fallback model triggers and provider |
| **Rubric Scoring** | `v2.rubric_scoring:` | `enabled`, `dimensions` (list of 4), `anchors` (low/medium/high/critical) | enabled=true | `config.yaml:119-132` | Define dimensions and score bands |
| **Data Sources** | `sources:` | congress, regulations, openstates, legiscan | congress/regulations enabled | `config.yaml:136-168` | Configure which APIs to use |
| **Change Tracking** | `change_tracking:` | `enabled`, `database`, `track_text_changes`, `analyze_changes_with_llm` | enabled=true | `config.yaml:172-180` | Control version tracking and change detection |
| **Output Formats** | `output:` | `directory`, `formats` (json, markdown, docx, pdf) | formats: json, markdown, docx enabled | `config.yaml:198-215` | Configure report output |

---

## Table 4: V2 Trigger Conditions

| Trigger | Condition | Threshold Config | Component | Action |
|---|---|---|---|---|
| **Multi-Sample Check** (Tier 1.5) | `impact >= 6` OR `confidence < 0.7` | `MULTI_SAMPLE_IMPACT_THRESHOLD=6`, `MULTI_SAMPLE_LOW_CONFIDENCE=0.7` | `MultiSampleChecker` | Run 2-3 samples, compute 0.85+ consensus |
| **Fallback Model** (Tier 2.5) | `judge_confidence` in [0.6, 0.8] AND `impact >= 7` | `JUDGE_LOW_CONFIDENCE=0.6`, `JUDGE_HIGH_CONFIDENCE=0.8`, `FALLBACK_IMPACT_THRESHOLD=7` | `FallbackAnalyst` | Invoke Claude Opus for second opinion |
| **Deep Research** (Phase 4) | `enable_deep_research=True` (optional) | — | `DeepResearchAgent` | Query external APIs for context |
| **Cross-Bill References** (Phase 4) | `enable_cross_bill_refs=True` (optional) | — | `ReferenceDetector` | Detect statutory citations in bill |
| **Hallucination Filtering** | `hallucination_detected=True` | — | `AuditTrailGenerator` | Exclude from output, flag in audit trail |
| **Route Selection** (STANDARD vs ENHANCED) | Complexity score 0-2 = STANDARD, 3+ = ENHANCED | Scoring factors in `v2.orchestration.scoring` | `TwoTierOrchestrator` | Determine pipeline depth and time/token budgets |

---

## API Signature Reference

### Primary Entry Points (nrg_core/v2/api.py)

#### **analyze_bill()**
**Location:** `nrg_core/v2/api.py:37`

```python
def analyze_bill(
    bill_id: str,
    bill_text: str,
    nrg_context: str,
    versions: Optional[List[BillVersion]] = None,
    api_key: Optional[str] = None,
    enable_multi_sample: bool = True,
    enable_fallback: bool = True
) -> TwoTierAnalysisResult
```

**Purpose:** Full two-tier pipeline entry point combining Sequential Evolution extraction + validation

**Parameters:**
- `bill_id` (str): Unique bill identifier (e.g., "HR150-118")
- `bill_text` (str): Full bill text for analysis
- `nrg_context` (str): NRG business context (~60KB document)
- `versions` (Optional[List[BillVersion]]): Bill version history; if None, single-version analysis
- `api_key` (Optional[str]): OpenAI API key; uses OPENAI_API_KEY env var if None
- `enable_multi_sample` (bool, default=True): Enable Tier 1.5 consistency checks
- `enable_fallback` (bool, default=True): Enable Tier 2.5 fallback model

**Returns:** `TwoTierAnalysisResult` containing:
- `primary_analysis` - Sequential Evolution findings
- `judge_validations` - Tier 2 validation per finding
- `rubric_scores` - 4-dimension scores (legal_risk, financial_impact, operational_disruption, ambiguity_risk)
- `audit_trails` - Compliance documentation
- `multi_sample_agreement` - Consistency score if Tier 1.5 enabled
- `second_model_reviewed` - True if Tier 2.5 triggered
- `fallback_results` - Alternative interpretations if Tier 2.5 ran

**Example Usage:**
```python
from nrg_core.v2.api import analyze_bill
from nrg_core.config import load_nrg_context

# Load business context
nrg_context = load_nrg_context("nrg_business_context.txt")

# Analyze a multi-version bill
result = analyze_bill(
    bill_id="HR150-118",
    bill_text="...",  # Full bill text
    nrg_context=nrg_context,
    versions=[
        BillVersion(version_type="Introduced", full_text="..."),
        BillVersion(version_type="Amended", full_text="..."),
    ],
    api_key="sk-..."  # Optional, uses env var if omitted
)

# Access results
for rubric_score in result.rubric_scores:
    print(f"{rubric_score.dimension}: {rubric_score.score}/10")
    print(f"  Rationale: {rubric_score.rationale}")
```

---

#### **validate_findings()**
**Location:** `nrg_core/v2/api.py:112`

```python
def validate_findings(
    bill_id: str,
    bill_text: str,
    nrg_context: str,
    findings_registry: Dict[str, Dict[str, Any]],
    stability_scores: Dict[str, float],
    api_key: Optional[str] = None
) -> TwoTierAnalysisResult
```

**Purpose:** Validation-only entry point when extraction is pre-done (Sequential Evolution run separately)

**Parameters:**
- `bill_id`, `bill_text`, `nrg_context`, `api_key`: Same as analyze_bill
- `findings_registry` (Dict[str, Dict]): Pre-extracted findings from Sequential Evolution (keys: F1, F2, etc.)
- `stability_scores` (Dict[str, float]): Stability scores from Sequential Evolution (maps finding_id → 0-1 score)

**Returns:** Same as analyze_bill (validation/scoring only, no extraction)

**Use Case:** When Sequential Evolution is run in a separate process or findings come from another source

**Example Usage:**
```python
# After Sequential Evolution has been run separately
findings_registry = {
    "F1": {"statement": "Tax applies to >50MW", "quotes": [...], "confidence": 0.85},
    "F2": {"statement": "Renewables exempt", "quotes": [...], "confidence": 0.78},
}
stability_scores = {"F1": 0.85, "F2": 0.70}

# Run validation only
result = validate_findings(
    bill_id="HR150-118",
    bill_text="...",
    nrg_context=nrg_context,
    findings_registry=findings_registry,
    stability_scores=stability_scores
)
```

---

### Core Class Methods

#### **SequentialEvolutionAgent.walk_versions()**
**Location:** `nrg_core/v2/sequential_evolution.py:128`

```python
def walk_versions(
    bill_id: str,
    versions: List[BillVersion]
) -> EvolutionResult
```

**Purpose:** Walk bill versions chronologically, extract findings with stability scores

**Parameters:**
- `bill_id` (str): Bill identifier for context
- `versions` (List[BillVersion]): Ordered list of bill versions from earliest to latest

**Returns:** `EvolutionResult` containing:
- `findings_registry` (Dict[str, Dict]): Findings with IDs (F1, F2, etc.), statements, quotes, impact, confidence
- `stability_scores` (Dict[str, float]): Stability score per finding (0-1 scale)
  - 0.95: Survived all versions unchanged
  - 0.85: One modification
  - 0.40: Three or more modifications (contentious)
  - 0.20: Added in last version (risky)

---

#### **TwoTierOrchestrator.validate()**
**Location:** `nrg_core/v2/two_tier.py:40`

```python
def validate(
    bill_id: str,
    bill_text: str,
    nrg_context: str,
    findings_registry: Dict[str, Dict[str, Any]],
    stability_scores: Dict[str, float]
) -> TwoTierAnalysisResult
```

**Purpose:** Orchestrate multi-tier validation pipeline

**Process Flow:**
1. Convert findings_registry to Finding objects
2. Tier 1.5 (Conditional): Multi-sample consistency check if impact≥6 OR confidence<0.7
3. Tier 2: Judge validation (quote verification, hallucination detection, evidence quality)
4. Tier 2.5 (Conditional): Fallback model if judge_confidence in [0.6, 0.8] AND impact≥7
5. Rubric scoring: Score validated findings on 4 dimensions
6. Audit trail: Generate compliance documentation
7. Optional Phase 4: Deep research and cross-bill references

**Returns:** Complete `TwoTierAnalysisResult`

---

#### **JudgeModel.validate()**
**Location:** `nrg_core/v2/judge.py:115`

```python
def validate(
    finding_id: int,
    finding: Finding,
    bill_text: str
) -> JudgeValidation
```

**Purpose:** Validate single finding against bill text

**Validation Checks:**
- `quote_verified`: Do all quotes exist verbatim in bill?
- `hallucination_detected`: Are claims not in quotes?
- `evidence_quality` (0-1): How well do quotes support statement?
- `ambiguity` (0-1): Interpretive uncertainty?
- `judge_confidence` (0-1): Judge's overall confidence?

**Returns:** `JudgeValidation` object with validation results

---

### Configuration Usage

#### **ThresholdConfig**
**Location:** `nrg_core/v2/config.py:9`

```python
from nrg_core.v2.config import DEFAULT_CONFIG

# Access thresholds
if finding.impact_estimate >= DEFAULT_CONFIG.MULTI_SAMPLE_IMPACT_THRESHOLD:
    # Trigger multi-sample check
    pass

if (DEFAULT_CONFIG.JUDGE_LOW_CONFIDENCE <= judge_confidence <= DEFAULT_CONFIG.JUDGE_HIGH_CONFIDENCE) and \
   (finding.impact_estimate >= DEFAULT_CONFIG.FALLBACK_IMPACT_THRESHOLD):
    # Trigger fallback model
    pass
```

**Available Constants:**
- `JUDGE_LOW_CONFIDENCE` = 0.6
- `JUDGE_HIGH_CONFIDENCE` = 0.8
- `FALLBACK_IMPACT_THRESHOLD` = 7
- `MULTI_SAMPLE_IMPACT_THRESHOLD` = 6
- `MULTI_SAMPLE_LOW_CONFIDENCE` = 0.7
- `STABILITY_HIGH` = 0.95
- `STABILITY_MEDIUM` = 0.85
- `STABILITY_LOW` = 0.40
- `STABILITY_LAST_MINUTE` = 0.20

---

## "How to Extend" Guide

### Extending the Architecture

#### **1. Adding a New Validation Tier (Tier 3: Fact Checking)**

**Step 1: Create a new agent class** inheriting from `BaseLLMAgent`
- Location: Create `nrg_core/v2/fact_checker.py`
- Template:
  ```python
  from nrg_core.v2.base import BaseLLMAgent
  from nrg_core.models_v2 import Finding, FactCheckResult

  class FactCheckAgent(BaseLLMAgent):
      def fact_check(self, findings: List[Finding], bill_text: str) -> List[FactCheckResult]:
          """Fact-check findings against external sources."""
          # Implementation
          pass
  ```

**Step 2: Update TwoTierOrchestrator**
- File: `nrg_core/v2/two_tier.py:98`
- Add dependency injection:
  ```python
  def __init__(
      self,
      ...,
      fact_checker: Optional[FactCheckAgent] = None,
      ...
  ):
      self.fact_checker = fact_checker
  ```
- Add stage in validation pipeline (around line 210):
  ```python
  if self.fact_checker:
      fact_check_results = self.fact_checker.fact_check(validated_findings, bill_text)
      # Process results
  ```

**Step 3: Update configuration**
- File: `config.yaml`
- Add section:
  ```yaml
  v2:
    fact_check:
      enabled: false
      sources: [snopes, factcheck.org, pbs]
      min_confidence: 0.7
  ```

**Step 4: Add tests**
- Location: Create `tests/test_v2/test_fact_checker.py`
- Follow pattern from `tests/test_v2/test_judge.py`

**Step 5: Update API if public-facing**
- File: `nrg_core/v2/api.py`
- Add optional parameter:
  ```python
  def analyze_bill(
      ...,
      enable_fact_check: bool = False
  ) -> TwoTierAnalysisResult:
      # Pass to orchestrator
  ```

---

#### **2. Implementing a Custom Judge Model**

**Step 1: Create custom class** implementing Judge interface
```python
from nrg_core.v2.base import BaseLLMAgent
from nrg_core.models_v2 import Finding, JudgeValidation

class CustomJudgeModel(BaseLLMAgent):
    def validate(self, finding_id: int, finding: Finding, bill_text: str) -> JudgeValidation:
        # Custom validation logic
        pass
```

**Step 2: Dependency inject in TwoTierOrchestrator**
```python
from nrg_core.v2.two_tier import TwoTierOrchestrator
from my_module import CustomJudgeModel

judge = CustomJudgeModel(model="my-custom-model")
orchestrator = TwoTierOrchestrator(judge=judge)
result = orchestrator.validate(...)
```

**Validation Interface Requirements:**
- Method: `validate(finding_id: int, finding: Finding, bill_text: str) -> JudgeValidation`
- Must return `JudgeValidation` with fields:
  - `quote_verified` (bool)
  - `hallucination_detected` (bool)
  - `evidence_quality` (0-1)
  - `ambiguity` (0-1)
  - `judge_confidence` (0-1)

---

#### **3. Adding a New Rubric Dimension**

**Step 1: Define dimension in rubrics.py**
- File: `nrg_core/v2/rubrics.py`
- Add to `ALL_RUBRICS`:
  ```python
  ALL_RUBRICS = {
      "legal_risk": {...},
      "financial_impact": {...},
      "operational_disruption": {...},
      "ambiguity_risk": {...},
      "environmental_impact": {  # NEW
          "0-2": "No environmental effects",
          "3-5": "Minor environmental impact",
          "6-8": "Significant environmental impact",
          "9-10": "Critical environmental threat"
      }
  }
  ```

**Step 2: Update data model**
- File: `nrg_core/models_v2.py`
- Modify `RubricScore.dimension` Literal:
  ```python
  dimension: Literal["legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk", "environmental_impact"]
  ```

**Step 3: Update configuration**
- File: `config.yaml`
- Add to dimensions list:
  ```yaml
  v2.rubric_scoring.dimensions:
    - legal_risk
    - financial_impact
    - operational_disruption
    - ambiguity_risk
    - environmental_impact
  ```

**Step 4: Add prompts**
- File: `nrg_core/v2/rubrics.py`
- Update `RUBRIC_SCORING_PROMPT` with new dimension

**Step 5: Update Judge model**
- File: `nrg_core/v2/judge.py`
- Modify `score_rubric()` to handle new dimension

---

#### **4. Integrating a Custom LLM Provider**

**Step 1: Create adapter** inheriting from `BaseLLMAgent`
```python
from nrg_core.v2.base import BaseLLMAgent

class CustomLLMAdapter(BaseLLMAgent):
    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("CUSTOM_API_KEY")

    def _call_llm(self, prompt: str, **kwargs) -> str:
        # Custom LLM API call
        pass
```

**Step 2: Use in components**
```python
judge = JudgeModel(
    model="custom-judge-v1",
    api_key_env="CUSTOM_API_KEY"
)
```

**Step 3: Update config**
```yaml
llm:
  custom:
    api_key_env: CUSTOM_API_KEY
    model: custom-judge-v1
    temperature: 0.2
```

---

### Data Model Extension

#### **Adding a New Finding Attribute**

**Step 1: Extend Finding model** (`nrg_core/models_v2.py`)
```python
class Finding(BaseModel):
    finding_id: Optional[str] = None
    statement: str  # min_length=20
    quotes: List[Quote]  # min_items=1
    confidence: float  # 0.0-1.0
    impact_estimate: int  # 0-10
    category: Optional[str] = None  # NEW
    requires_research: Optional[bool] = False  # NEW
```

**Step 2: Update Sequential Evolution** (`sequential_evolution.py`)
```python
# Extract new attributes from bill during version walk
finding["category"] = extract_category_from_bill(bill_text, statement)
finding["requires_research"] = assess_research_needs(statement)
```

**Step 3: Update prompts** if LLM extraction needed
```python
EVOLUTION_PROMPT_V1 = """
Extract findings with:
- statement
- quotes (with sections)
- confidence
- impact_estimate
- category (tax/environmental/reporting/other)  # NEW
- requires_research (bool)  # NEW
"""
```

---

## Configuration Schema Reference

### Environment Variables Required

```bash
# OpenAI API
export OPENAI_API_KEY="sk-..."

# Google Gemini API
export GOOGLE_API_KEY="..."

# Anthropic Claude API
export ANTHROPIC_API_KEY="sk-ant-..."

# Legislative APIs
export CONGRESS_API_KEY="..."  # Optional
export OPENSTATES_API_KEY="..."  # For state bills
export LEGISCAN_API_KEY="..."  # Legacy

# Optional for Phase 4
export BILLTRACK50_API_KEY="..."  # Commercial tracking
```

### config.yaml Structure

```yaml
# LOGGING [V2 ONLY]
logging:
  level: normal|debug
  trace:
    sequential_evolution: bool
    two_tier_validation: bool
    rubric_scoring: bool

# LLM [SHARED]
llm:
  provider: gemini|openai|anthropic
  openai: {...}
  gemini: {...}
  anthropic: {...}

# V2 PIPELINE [V2 ONLY]
v2:
  orchestration: {...}
  sequential_evolution: {...}
  two_tier:
    multi_sample: {...}
    judge: {...}
    fallback: {...}
  rubric_scoring: {...}

# DATA SOURCES [SHARED]
sources:
  congress: {...}
  regulations: {...}
  openstates: {...}

# CHANGE TRACKING [SHARED]
change_tracking: {...}

# OUTPUT [SHARED]
output:
  directory: ./reports
  formats: {json, markdown, docx}
```

---

This completes Phase 2 reference material generation. All tables, API signatures, and extension guides are now ready for integration into Architecture_v2.md during Phase 3.
