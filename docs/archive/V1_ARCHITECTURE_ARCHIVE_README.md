# v1.0 Architecture Archive

**Archived Date**: 2026-01-20  
**Reason**: Superseded by v2.0 cost-optimized redesign

---

## Archived Documents

The following v1.0 architecture documents have been superseded by v2.0 equivalents:

### 1. Three-Model Consensus Ensemble

**Original Plan**: `/docs/plans/2026-01-15-consensus-ensemble.md`  
**Superseded By**: `/docs/plans/2026-01-20-two-tier-consensus.md`

**Reason for Change**:
- Cost reduction: 57% ($0.30 → $0.13 per bill)
- Replaced 3-model parallel ensemble with 2-tier approach:
  - Tier 1: Single primary analyst
  - Tier 1.5: Self-consistency sampling (conditional)
  - Tier 2: Judge validation
  - Fallback: Second model (conditional, ~18% of findings)
- Self-consistency achieves 80-90% of full ensemble benefits at 1/10th cost
- Judge validation catches errors without full third-model analysis

**Status**: Implementation **NOT STARTED** - superseded before implementation

---

### 2. Independent Per-Version Evolutionary Analysis

**Original Plan**: `/docs/plans/2026-01-15-evolutionary-analysis.md`  
**Superseded By**: `/docs/plans/2026-01-20-sequential-evolution.md`

**Reason for Change**:
- Cost reduction: 54% ($0.24 → $0.11 per bill)
- Replaced independent per-version analysis with sequential agent:
  - Single agent walks versions sequentially with structured memory
  - Context reuse reduces token usage (150K → 70K typical)
  - Judge computes stability scores (not LLM-based)
  - Optional hot spot re-analysis for unstable high-impact findings
- Maintains finding lineage explicitly in memory vs implicit in prompts

**Status**: Implementation **NOT STARTED** - superseded before implementation

---

### 3. Confidence Aggregation System

**Original Plan**: `/docs/plans/2026-01-15-confidence-aggregation.md`  
**Updated By**: `/docs/plans/2026-01-20-rubric-scoring.md`

**Reason for Change**:
- Opaque confidence scores (0-1) replaced with explicit rubrics (0-10 scales)
- 4 dimensions: Legal Risk, Operational Disruption, Financial Impact, Ambiguity
- Each dimension has anchored scales with concrete examples
- Provides explainability: "Why 7/10?" → rubric anchor + evidence
- Enables calibration: Adjust anchor definitions based on feedback
- Business-actionable: "Legal risk 7/10 = significant obligations" vs "confidence 0.83"

**Status**: Implementation **NOT STARTED** - superseded before implementation

---

## Documents Retained from v1.0

The following v1.0 documents remain valid for v2.0:

### Causal Chain Reasoning

**Plan**: `/docs/plans/2026-01-15-causal-chain-reasoning.md`  
**Status**: **RETAINED** - Compatible with v2.0 architecture

**Reason**: Causal chain reasoning logic is independent of consensus/evolution/scoring changes. The researcher-judge pattern in v2.0 is compatible with causal chain construction.

---

## Original Architecture Documents

The following documents describe the original v1.0 architecture (now superseded):

### Core Architecture

1. **`/docs/redesign/Architecture.md`**
   - Status: Superseded by `Architecture_v2.md`
   - Original 3-model ensemble architecture
   - Independent per-version evolution
   - Opaque confidence aggregation
   - Cost: ~$0.67 per bill

2. **`/docs/redesign/Components_Deep_Dive.md`**
   - Status: Superseded by `Components_Deep_Dive_v2.md`
   - Detailed breakdowns of v1.0 components
   - Consensus ensemble (parallel 3-model)
   - Evolutionary analysis (independent)
   - Confidence aggregation components

---

## Why v2.0 Redesign?

### Cost Concerns

**v1.0 Monthly Cost** (at 1,000 bills/month):
- 800 simple bills × $0.15 = $120
- 200 complex bills × $0.67 = $134
- **Total: $254/month**

**v2.0 Monthly Cost** (same volume):
- 800 simple bills × $0.08 = $64
- 200 complex bills × $0.33 = $66
- **Total: $130/month**

**Savings: 49% ($254 → $130)**

### Architectural Improvements

Beyond cost, v2.0 introduces:

1. **Supervisor Pattern**: Explicit orchestration, budget control, quality gates
2. **Judge Validation**: Catches hallucinations, calibrates confidence
3. **Sequential Memory**: Explicit finding lineage vs implicit context
4. **Rubric Explainability**: Actionable multi-dimensional scores vs opaque confidence
5. **Evaluation Strategy**: Silver set + LLM-judge + spot-checks (no full golden dataset needed)

### Research Support

v2.0 design based on:
- Self-consistency (Wang et al., 2023): 10-20% accuracy improvement at 1/10th ensemble cost
- LLM-as-judge (Zheng et al., 2024): 0.85+ correlation with human judgment
- Supervisor pattern (LangGraph 2024): Proven multi-agent orchestration
- Prediction-Powered Inference (Angelopoulos et al., 2023): Statistical evaluation without full labels

---

## Migration Status

**Current Status**: v1.0 plans archived before implementation

**Next Steps**:
1. Implement v2.0 architecture per new plans
2. No v1.0 → v2.0 migration needed (v1.0 never deployed)
3. v2.0 implementation timeline: 12 weeks (per individual plan timelines)

---

## Historical Context

**v1.0 Planning Period**: 2026-01-15  
**v2.0 Redesign Period**: 2026-01-20  
**Reason for Redesign**: Cost optimization analysis revealed 51% savings opportunity with comparable accuracy

**Key Decision**: Redesign before implementation avoids costly refactor later

---

## Document References

**v2.0 Architecture**:
- Main doc: `/docs/redesign/Architecture_v2.md`
- Components: `/docs/redesign/Components_Deep_Dive_v2.md`
- Index: `/docs/ARCHITECTURE_INDEX.md`

**v2.0 Implementation Plans**:
- Two-tier consensus: `/docs/plans/2026-01-20-two-tier-consensus.md`
- Sequential evolution: `/docs/plans/2026-01-20-sequential-evolution.md`
- Rubric scoring: `/docs/plans/2026-01-20-rubric-scoring.md`
- Evaluation strategy: `/docs/plans/2026-01-20-evaluation-strategy.md`

**v1.0 Archived Plans** (this folder):
- `2026-01-15-consensus-ensemble.md` - Superseded, not implemented
- `2026-01-15-evolutionary-analysis.md` - Superseded, not implemented
- `2026-01-15-confidence-aggregation.md` - Superseded, not implemented
- `2026-01-15-causal-chain-reasoning.md` - **Retained**, still valid

---

## Questions?

For questions about why specific v1.0 approaches were changed, see:
- Architecture v2.0 design rationale: `/docs/redesign/Architecture_v2.md`
- Cost comparison tables in each v2.0 implementation plan
- Architecture index: `/docs/ARCHITECTURE_INDEX.md`
