# SOTA Bill Analysis System - Gaps, Ambiguities, and Clarification Questions

This document consolidates all gaps, ambiguities, and open questions identified during implementation planning for the 5 components of the SOTA Bill Analysis System.

---

## Component 1: Consensus Ensemble

### 1.1 API Key Management
**Gap**: How should API keys be stored and rotated in production?
**Question**: Should we use Azure Key Vault or environment variables? What's the rotation policy?
**Priority**: High
**Blocker**: No - can use environment variables initially

### 1.2 Model Version Pinning
**Gap**: Architecture specifies "Gemini 1.5 Pro" but doesn't specify version date
**Question**: Should we pin to specific model versions (e.g., gemini-1.5-pro-001) for consistency?
**Priority**: Medium
**Blocker**: No - can use latest versions initially

### 1.3 Quote Verification Accuracy
**Gap**: Current implementation uses simple substring matching
**Question**: Should we implement fuzzy matching for paraphrased quotes? What threshold?
**Priority**: Medium
**Blocker**: No - substring matching is acceptable for MVP

### 1.4 Handling Model Failures
**Gap**: What happens if one model fails but others succeed?
**Question**: Should we proceed with 2/3 models or fail the entire analysis?
**Priority**: High
**Blocker**: Yes - need decision before implementation

**Recommended Answer**: Proceed with 2/3 models. Mark finding as "partial consensus" and lower confidence score appropriately (e.g., 2/3 majority drops from 0.70 to 0.60).

### 1.5 Semantic Similarity Threshold
**Gap**: 0.85 threshold chosen but not validated
**Question**: Should we A/B test different thresholds (0.80, 0.85, 0.90)? What's the false positive/negative tradeoff?
**Priority**: Medium
**Blocker**: No - can validate post-MVP with test set

### 1.6 Token Limits
**Gap**: No handling for bills exceeding model token limits
**Question**: How should we handle bills >100k tokens? Chunking strategy?
**Priority**: High
**Blocker**: Yes - need strategy before production

**Recommended Answer**: Implement chunking with overlap:
- Split bills >100k tokens into 80k token chunks with 10k overlap
- Run consensus on each chunk separately
- Merge results, de-duplicate findings by semantic similarity

### 1.7 Caching Strategy
**Gap**: No caching for repeated bill analyses
**Question**: Should we cache consensus results? For how long?
**Priority**: Low
**Blocker**: No - performance optimization for later

### 1.8 Prompt Engineering Validation
**Gap**: Prompts not validated against test set
**Question**: Should we create a validation set to optimize prompts before production?
**Priority**: High
**Blocker**: Yes - need validation before claiming <1% FPR

**Recommended Answer**: Create validation set of 100 manually-labeled bills, iterate prompts until FPR <1% on validation set.

### 1.9 Confidence Calibration
**Gap**: Confidence scores (0.95, 0.70, 0.50) are hardcoded
**Question**: Should these be calibrated using a holdout set? What calibration method (Platt scaling, isotonic)?
**Priority**: Medium
**Blocker**: No - can calibrate post-MVP

### 1.10 Cost Monitoring
**Gap**: No cost tracking per analysis
**Question**: Should we log token usage and cost per bill for budgeting?
**Priority**: Medium
**Blocker**: No - nice to have

---

## Component 2: Evolutionary Analysis

### 2.1 Similarity Threshold for Lineage Matching
**Gap**: 0.75 threshold for matching findings across versions not validated
**Question**: Should we A/B test thresholds? What's optimal for different bill types?
**Priority**: Medium
**Blocker**: No - can validate with test set

### 2.2 Modification Detection Granularity
**Gap**: Currently tracks statement-level modifications
**Question**: Should we track sub-clause modifications (e.g., threshold change 50MW→25MW)?
**Priority**: Low
**Blocker**: No - statement-level sufficient for MVP

### 2.3 Section Numbering Changes
**Gap**: If section numbers change (renumbering), lineage tracking may break
**Question**: How to handle section renumbering between versions?
**Priority**: Medium
**Blocker**: Yes - need strategy

**Recommended Answer**: Use content-based matching instead of section numbers. Match sections by cosine similarity of content, not section number.

### 2.4 Definition Extraction Accuracy
**Gap**: Regex patterns may miss non-standard definition formats
**Question**: Should we use NER or more sophisticated parsing?
**Priority**: Low
**Blocker**: No - regex adequate for common patterns

### 2.5 Contentious Threshold
**Gap**: 3+ modifications marked as contentious, but not validated
**Question**: Is 3 the right threshold? Vary by bill type?
**Priority**: Low
**Blocker**: No - reasonable heuristic for MVP

### 2.6 Vocabulary Growth Interpretation
**Gap**: Vocabulary size increases, but unclear what's meaningful
**Question**: What vocabulary growth rate indicates problematic complexity?
**Priority**: Low
**Blocker**: No - informational metric only

### 2.7 Missing Versions
**Gap**: No handling for gaps in version sequence (v1, v3, v5)
**Question**: How should we handle missing intermediate versions?
**Priority**: Medium
**Blocker**: Yes - need decision

**Recommended Answer**: Process available versions only. Mark lineages as "incomplete version history" if gaps detected.

### 2.8 Performance with Many Versions
**Gap**: Lineage tracking is O(n²) in number of versions
**Question**: Optimize for bills with 20+ versions?
**Priority**: Low
**Blocker**: No - most bills have <10 versions

---

## Component 3: Causal Chain Reasoning

### 3.1 LLM Model Selection for Reasoning
**Gap**: Which model to use for causal reasoning?
**Question**: Use GPT-4o for all steps? Or ensemble across models? Cost vs accuracy tradeoff?
**Priority**: High
**Blocker**: Yes - affects cost and accuracy

**Recommended Answer**: Use GPT-4o for chain-of-thought (best at structured reasoning). Reserve ensemble for final verification only.

### 3.2 Quote Verification Strictness
**Gap**: How strict should quote matching be?
**Question**: Exact substring? Fuzzy match? What edit distance threshold?
**Priority**: Medium
**Blocker**: No - substring matching acceptable for MVP

### 3.3 Alternative Likelihood Calibration
**Gap**: How to calibrate likelihood scores (0.05, 0.65, 0.30)?
**Question**: Based on LLM confidence? Historical accuracy? Expert judgment?
**Priority**: Medium
**Blocker**: No - use LLM-provided likelihoods initially

### 3.4 Counterfactual Parameters
**Gap**: Which parameters to vary for counterfactuals?
**Question**: How to automatically identify key parameters (thresholds, dates, exemptions)?
**Priority**: Low
**Blocker**: No - manual specification acceptable for MVP

### 3.5 Step Failure Handling
**Gap**: What if step 3 fails but others succeed?
**Question**: Return partial chain? Retry failed step? Fail entire analysis?
**Priority**: High
**Blocker**: Yes - need decision

**Recommended Answer**: Return partial chain with failed steps marked. Lower causal strength score proportionally.

### 3.6 Business Impact Quantification
**Gap**: Qualitative impact ("$500K-$2M") not structured
**Question**: Should we extract structured impact (min, max, currency)?
**Priority**: Medium
**Blocker**: No - nice to have for reporting

### 3.7 DAG Construction
**Gap**: DAG structure defined but not implemented
**Question**: When/how to build DAG? What triggers multi-path analysis?
**Priority**: Low
**Blocker**: No - optional enhancement

### 3.8 Chain Caching
**Gap**: Expensive to rebuild chains for same findings
**Question**: Cache chains by finding statement? For how long?
**Priority**: Low
**Blocker**: No - optimization for later

---

## Component 4: Confidence Aggregation

### 4.1 Confidence Interval Calculation
**Gap**: Simple ±15% margin, not calibrated
**Question**: Should we calibrate using historical accuracy data? What method (bootstrap, quantile)?
**Priority**: High
**Blocker**: Yes - confidence intervals critical for decision-making

**Recommended Answer**: Start with ±15% margin for MVP. Post-MVP, calibrate using quantile regression on validation set (map predicted confidence to actual accuracy).

### 4.2 Component Weights
**Gap**: Weights (0.40, 0.30, 0.20, 0.10) not validated
**Question**: A/B test different weight schemes? Optimize using validation set?
**Priority**: High
**Blocker**: Yes - weights directly affect routing decisions

**Recommended Answer**: Validate current weights on test set. If FPR >1%, grid search over weight combinations to minimize FPR while maintaining recall >90%.

### 4.3 Evidence Quality Criteria
**Gap**: Equal weighting of 5 criteria may not be optimal
**Question**: Should some criteria (quote verification) weigh more?
**Priority**: Medium
**Blocker**: No - equal weighting reasonable for MVP

### 4.4 Missing Data Handling
**Gap**: What if chain or lineage is None?
**Question**: Default to 0.50? Use only available signals? Flag as incomplete?
**Priority**: High
**Blocker**: Yes - will occur frequently

**Recommended Answer**:
- If chain is None: Set causal_strength = 0.50, increase evidence_quality weight from 0.40 to 0.50
- If lineage is None: Set evolution_stability = 0.50, redistribute weight proportionally
- Flag finding as "incomplete_analysis"

---

## Component 5: Routing Decision

### 5.1 SLA Enforcement
**Gap**: SLA times specified but no enforcement mechanism
**Question**: How to monitor SLA compliance? Alerts? Automatic escalation?
**Priority**: Medium
**Blocker**: No - monitoring feature for post-MVP

### 5.2 Expertise Availability
**Gap**: Routing assumes experts are available
**Question**: What if energy_law specialist is unavailable? Fallback to general counsel?
**Priority**: High
**Blocker**: Yes - will happen in production

**Recommended Answer**: Implement fallback chain:
1. Try primary expertise (e.g., energy_law)
2. If unavailable, try related expertise (regulatory_compliance)
3. If unavailable, escalate to general_counsel
4. If unavailable, queue for review when expert available

### 5.3 Cost Tracking
**Gap**: No cost consideration in routing
**Question**: Should we factor in expert hourly rates when routing?
**Priority**: Low
**Blocker**: No - accuracy more important than cost for MVP

### 5.4 Dynamic Threshold Adjustment
**Gap**: Thresholds (0.95, 0.85, 0.70) are static
**Question**: Should thresholds adjust based on bill importance or risk?
**Priority**: Medium
**Blocker**: No - static thresholds acceptable for MVP

### 5.5 Topic Detection
**Gap**: `finding_topic` must be provided externally
**Question**: How to automatically detect topic from finding text?
**Priority**: High
**Blocker**: Yes - affects expertise routing

**Recommended Answer**: Use keyword matching initially:
- "tax", "tariff", "levy" → tax
- "energy", "power", "renewable" → energy
- "labor", "employment", "wage" → labor
- etc.

Post-MVP: Train classifier on finding text → topic.

### 5.6 Priority Escalation
**Gap**: Priority is binary (normal/high)
**Question**: Should we have more granular priority levels?
**Priority**: Low
**Blocker**: No - binary sufficient for MVP

### 5.7 Routing Audit Trail
**Gap**: No logging of routing decisions
**Question**: Should we log all routing decisions for review/optimization?
**Priority**: Medium
**Blocker**: No - nice to have for continuous improvement

---

## Cross-Component Questions

### CC.1 End-to-End Integration
**Gap**: Components designed independently
**Question**: How do components pass data? Synchronous or async? Message queue?
**Priority**: High
**Blocker**: Yes - architectural decision needed

**Recommended Answer**:
- Synchronous for MVP (single process, sequential component execution)
- Async with message queue for production scale (Celery + Redis)

### CC.2 Error Propagation
**Gap**: How do errors in one component affect downstream components?
**Question**: Fail fast or best-effort with partial results?
**Priority**: High
**Blocker**: Yes - affects reliability

**Recommended Answer**: Best-effort with partial results:
- If consensus fails: Return empty findings, skip causal/evolution
- If causal fails: Use consensus only, mark as incomplete
- If evolution fails: Use consensus + causal, default stability to 0.50
- Log all failures for review

### CC.3 Testing Strategy
**Gap**: Individual component tests exist, but no integration tests
**Question**: What integration tests are needed? What's the test bill corpus?
**Priority**: High
**Blocker**: Yes - need test set before production

**Recommended Answer**:
1. Create test corpus of 100 bills with manual labels (impact, findings, confidence)
2. Run full pipeline, compare to labels
3. Measure: FPR, FNR, F1, precision, recall
4. Target: FPR <1%, recall >85%

### CC.4 Performance Benchmarking
**Gap**: No performance targets specified
**Question**: What's acceptable latency? Throughput?
**Priority**: Medium
**Blocker**: No - but needed for production SLA

**Recommended Answer**: Based on architecture doc:
- Simple bills (70%): <30 seconds
- Complex bills (30%): <5 minutes
- Throughput: 200 bills/hour peak

### CC.5 Monitoring and Alerting
**Gap**: No monitoring strategy
**Question**: What metrics to track? When to alert?
**Priority**: Medium
**Blocker**: No - operational concern for production

**Recommended Answer**: Track:
- Per-component latency (p50, p95, p99)
- Error rates by component
- FPR estimate (sample manual review)
- Model API failures
- SLA compliance

Alert on:
- FPR >2% (weekly sample)
- Error rate >5%
- Latency p95 >10 minutes

### CC.6 Version Control for Prompts
**Gap**: Prompts are code strings, but may need frequent iteration
**Question**: Store prompts in DB? Version in git? A/B test framework?
**Priority**: Low
**Blocker**: No - git versioning sufficient for MVP

### CC.7 Human Feedback Loop
**Gap**: No mechanism to incorporate expert corrections
**Question**: How do expert reviews improve the system? Retrain? Update prompts?
**Priority**: Medium
**Blocker**: No - continuous improvement feature for post-MVP

### CC.8 Multi-Language Support
**Gap**: Architecture assumes English bills
**Question**: Support for Spanish? French? Other languages?
**Priority**: Low
**Blocker**: No - English only for MVP

---

## Summary of Blocking Questions

These questions must be answered before implementation can proceed:

| ID | Question | Recommended Answer | Component |
|----|----------|-------------------|-----------|
| 1.4 | Handle model failures? | Proceed with 2/3 models, lower confidence | Consensus |
| 1.6 | Handle token limits? | Chunk with overlap, merge results | Consensus |
| 1.8 | Validate prompts? | Create 100-bill validation set | Consensus |
| 2.3 | Handle section renumbering? | Content-based matching | Evolution |
| 2.7 | Handle missing versions? | Process available only, mark incomplete | Evolution |
| 3.1 | Which model for reasoning? | GPT-4o for CoT, ensemble for verification | Causal |
| 3.5 | Handle step failures? | Return partial chain, lower score | Causal |
| 4.1 | Calibrate confidence intervals? | ±15% for MVP, calibrate post-MVP | Confidence |
| 4.2 | Validate component weights? | Test current weights, optimize if FPR >1% | Confidence |
| 4.4 | Handle missing components? | Default to 0.50, redistribute weights | Confidence |
| 5.2 | Handle unavailable experts? | Fallback chain to general counsel | Routing |
| 5.5 | Detect finding topic? | Keyword matching for MVP | Routing |
| CC.1 | Integration architecture? | Synchronous for MVP, async for production | All |
| CC.2 | Error propagation? | Best-effort with partial results | All |
| CC.3 | Testing strategy? | 100-bill corpus, FPR <1% target | All |

---

## Next Steps

1. **Review this document with stakeholders**
2. **Get decisions on all blocking questions**
3. **Create 100-bill validation/test corpus**
4. **Proceed with implementation following approved plans**
5. **Validate each component against test set**
6. **Iterate prompts/thresholds until FPR <1%**
7. **Production hardening (monitoring, alerts, error handling)**
8. **Staged rollout with manual review shadow mode**

---

Last Updated: 2026-01-15
Document Owner: Implementation Team
Review Cadence: Weekly during implementation
