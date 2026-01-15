# SOTA Bill Analysis System - Component Deep Dive with Visual Diagrams

This document provides detailed visual breakdowns of each system component with What, Why, and How sections.

---

## Component 1: Consensus Ensemble - Detailed Breakdown

```
INPUT: Bill Text
│
└─→ Parallel Model Calls (60 second timeout)
    │
    ├─→ Model A: Gemini 1.5 Pro
    │   ├─ Instruction: Analyze bill with structured JSON output
    │   ├─ Response: {"findings": [...], "confidence": [...]}
    │   └─ Processing time: ~35s
    │
    ├─→ Model B: GPT-4o
    │   ├─ Instruction: Same structured prompt as Model A
    │   ├─ Response: {"findings": [...], "confidence": [...]}
    │   └─ Processing time: ~40s
    │
    └─→ Model C: Claude 3.5 Sonnet
        ├─ Instruction: Same structured prompt as Model A
        ├─ Response: {"findings": [...], "confidence": [...]}
        └─ Processing time: ~38s

Wait for all three (max 60 seconds)
│
└─→ Response Parsing
    │
    ├─ Extract finding text from each model
    ├─ Extract supporting quotes
    ├─ Extract confidence scores
    └─ Normalize text (lowercase, remove punctuation)

Semantic Grouping (Cosine Similarity > 0.85)
│
├─ Finding A: "Tax applies to >50MW"
│  ├─ Model A: Similar (0.92)
│  ├─ Model B: Exact match
│  └─ Model C: Similar (0.88)
│
├─ Finding B: "Renewable exempt"
│  ├─ Model A: Not found
│  ├─ Model B: Exact match
│  └─ Model C: Exact match
│
└─ Finding C: "Quarterly reporting"
   ├─ Model A: Not found
   ├─ Model B: Exact match
   └─ Model C: Exact match

Agreement Counting & Classification
│
├─ UNANIMOUS (3/3 models):
│  └─ Finding: "Effective January 1, 2026"
│     ├─ Found by: All 3 models
│     ├─ Confidence: 0.95
│     └─ Action: Direct inclusion
│
├─ MAJORITY (2/3 models):
│  ├─ Finding: "Tax on >50MW"
│  │  ├─ Found by: GPT-4o, Claude
│  │  ├─ Confidence: 0.70
│  │  └─ Action: Include with medium confidence
│  │
│  └─ Finding: "Renewable exempt"
│     ├─ Found by: GPT-4o, Claude
│     ├─ Confidence: 0.68
│     └─ Action: Include with medium confidence
│
└─ DISPUTED (1 or conflicting):
   └─ Finding: Scope of tax
      ├─ Gemini says: "All energy companies"
      ├─ GPT-4o says: ">50MW only"
      ├─ Claude says: ">50MW only"
      ├─ Agreement: 2 vs 1
      └─ Action: Trigger resolution

Evidence Resolution (for Disputed findings)
│
└─ Request to models: "Provide exact bill quote supporting your claim"
   │
   ├─ Gemini response: No quote provided / Generic paraphrase
   │  └─ Verdict: Hallucination detected
   │     ├─ Confidence: Reduced to 0.20
   │     └─ Flag: "Model A hallucinated scope"
   │
   ├─ GPT-4o response: "Section 2.1b: exceeding fifty megawatts"
   │  ├─ Verification: Quote found in bill? YES
   │  └─ Verdict: Accurate, verified
   │
   └─ Claude response: "Section 2.1b: exceeding fifty megawatts"
      ├─ Verification: Quote found in bill? YES
      └─ Verdict: Accurate, verified

Final Output: ConsensusAnalysis
│
└─ Findings Array:
   ├─ {
   │   statement: "Tax applies to energy >50MW",
   │   consensus: "MAJORITY",
   │   confidence: 0.70,
   │   supporting_quotes: ["Section 2.1b: exceeding fifty megawatts"],
   │   found_by: ["GPT-4o", "Claude"],
   │   verification_status: "verified"
   │  }
   │
   ├─ {
   │   statement: "Renewable energy exempt",
   │   consensus: "MAJORITY",
   │   confidence: 0.68,
   │   supporting_quotes: ["Section 3.2: renewable energy exempt"],
   │   found_by: ["GPT-4o", "Claude"],
   │   verification_status: "verified"
   │  }
   │
   └─ {
       statement: "Effective January 1, 2026",
       consensus: "UNANIMOUS",
       confidence: 0.95,
       supporting_quotes: ["Section 5.1: effective Jan 1 2026"],
       found_by: ["Gemini", "GPT-4o", "Claude"],
       verification_status: "verified"
      }
```

What:
Three LLM models analyze the same bill in parallel. Findings are grouped by semantic similarity and agreement level. Disputed findings trigger quote verification to detect hallucinations.

Why This Works:
Model diversity reduces errors. Claude excels at legal reasoning but may miss details. GPT-4o is comprehensive but sometimes over-interprets. Gemini structures output well but sometimes misses nuance. Single model errors (e.g., misread threshold) are caught by other models. Voting with 3 models means one hallucination doesn't corrupt results. Quote verification explicitly catches when models claim something not in bill text.

How It Works:
1. Send identical prompt to all 3 models simultaneously (parallel saves time)
2. Parse JSON responses into Finding objects
3. Embed findings semantically to measure similarity
4. Cluster findings with similarity threshold 0.85
5. Count which models found each clustered finding
6. For disputed findings, request supporting quotes
7. Verify quotes exist in actual bill text
8. Assign confidence based on agreement and verification

Inputs and Outputs:
- Input: Bill text, structured prompt template
- Output: ConsensusAnalysis with findings grouped by agreement level

---

## Component 2: Evolutionary Analysis - Detailed Breakdown

```
INPUT: [Bill Version 1, Bill Version 2, ..., Bill Version N]
        Each with ConsensusAnalysis findings
│
└─→ Version Diff Computation
    │
    ├─ Compare Version 1 → Version 2
    │  │
    │  ├─ Additions:
    │  │  └─ "Tax on energy generation exceeding 50 megawatts"
    │  │
    │  ├─ Deletions:
    │  │  └─ None
    │  │
    │  └─ Modifications:
    │     └─ "Tax on energy companies" → "Tax on energy >50MW"
    │
    ├─ Compare Version 2 → Version 3
    │  │
    │  ├─ Additions:
    │  │  ├─ "Renewable energy exempt"
    │  │  ├─ "Effective January 1, 2026"
    │  │  └─ "Quarterly reporting requirement"
    │  │
    │  ├─ Deletions:
    │  │  └─ None
    │  │
    │  └─ Modifications:
    │     └─ Wording clarifications
    │
    └─ Note which bill sections changed

Finding Lineage Tracking
│
├─ Finding: "Tax applies to energy"
│  ├─ Origin version: v1 (introduced)
│  ├─ v1→v2 change: YES (scope narrowed ">50MW")
│  ├─ v2→v3 change: NO (stable)
│  ├─ Modification count: 1
│  └─ Lineage: [v1, modified-v2, stable-v3]
│
├─ Finding: "Renewable exempt"
│  ├─ Origin version: v3 (introduced late)
│  ├─ v1→v2 change: N/A (didn't exist)
│  ├─ v2→v3 change: N/A (new in v3)
│  ├─ Modification count: 0 (never modified)
│  └─ Lineage: [N/A, N/A, v3]
│
└─ Finding: "Quarterly reporting"
   ├─ Origin version: v3 (introduced late)
   ├─ Modification count: 0
   └─ Lineage: [N/A, N/A, v3]

Complexity Metrics Calculation
│
├─ Section Count Growth
│  ├─ Version 1: 2 sections
│  ├─ Version 2: 4 sections
│  ├─ Version 3: 6 sections
│  ├─ Growth rate: +50% per version
│  └─ Interpretation: Bill becomes more detailed/specific
│
├─ Definition Count
│  ├─ Version 1: 1 definition ("energy companies")
│  ├─ Version 2: 3 definitions (+ ">50MW" threshold)
│  ├─ Version 3: 5 definitions (+ renewable definition)
│  └─ Interpretation: Scope narrowing, precision increasing
│
├─ Modification Frequency Per Section
│  ├─ Section 1 (Core): 1 modification (v1→v2)
│  ├─ Section 2 (Tax calc): 0 modifications
│  ├─ Section 3 (Exemptions): 1 modification (v2→v3)
│  ├─ Section 4 (Reporting): 0 modifications
│  ├─ Section 5 (Effective): 0 modifications
│  └─ Highest contention: Sections 1 & 3
│
└─ Vocabulary Expansion (Heap's Law)
   ├─ Version 1: 500 unique tokens
   ├─ Version 2: 650 unique tokens (+30%)
   ├─ Version 3: 750 unique tokens (+15%)
   └─ Interpretation: Incremental expansion (settling down)

Stability Scoring (per finding)
│
├─ Finding: "Tax applies to >50MW"
│  ├─ v1: Present as "tax on energy"
│  ├─ v2: Modified (threshold added)
│  ├─ v3: Stable
│  ├─ Modification count: 1
│  ├─ Calculation: 0.95 (present v1) - 0.10 (modified once) = 0.85
│  └─ Stability score: 0.85
│
├─ Finding: "Renewable exempt"
│  ├─ v1: Not present
│  ├─ v2: Not present
│  ├─ v3: Present (new)
│  ├─ Modification count: 0
│  ├─ Calculation: 0.20 (recently added, v3 only)
│  └─ Stability score: 0.20
│
└─ Finding: "Quarterly reporting"
   ├─ v1: Not present
   ├─ v2: Not present
   ├─ v3: Present (new)
   ├─ Modification count: 0
   ├─ Calculation: 0.20 (recently added, v3 only)
   └─ Stability score: 0.20

Contentious Area Identification
│
├─ Modification count >= 3: Flag as "highly contentious"
├─ Modification count 1-2: Flag as "moderately contentious"
├─ Modification count 0: Mark as "settled"
│
└─ This bill:
   ├─ Highly contentious areas: None (max 1 modification)
   ├─ Moderately contentious: Sections 1, 3
   └─ Settled: Sections 2, 4, 5

OUTPUT: EvolutionAnalysis
│
└─ Data structure:
   {
     "findings_timeline": [
       {
         "finding": "Tax applies to >50MW",
         "origin_version": 1,
         "modifications": 1,
         "stability_score": 0.85,
         "contentious": false
       },
       {
         "finding": "Renewable exempt",
         "origin_version": 3,
         "modifications": 0,
         "stability_score": 0.20,
         "contentious": false,
         "note": "Recently added, least stable"
       }
     ],
     "complexity_trajectory": {
       "section_count": [2, 4, 6],
       "definition_count": [1, 3, 5],
       "growth_rate": 0.50
     },
     "overall_assessment": "Bill became more specific and detailed.
                          Core concepts settled after v2.
                          Late additions (v3 only) are newest."
   }
```

What:
Track how bill provisions change across versions. Measure stability (unchanged = reliable) and identify contentious areas (modified multiple times = uncertain).

Why This Works:
Legislative process involves negotiation. Early drafts are vague, later versions add specificity. Provisions present from introduction (v1) to final (vN) unchanged are core intent, highly stable. Late additions (final version only) represent compromises and are less stable. Tracking modification count quantifies contention: provision modified 5 times signals major disagreement. Complexity growth shows bill becoming more detailed (safer interpretation).

How It Works:
1. For each consecutive version pair, compute textual diffs
2. Identify additions, deletions, modifications
3. For each final-version finding, trace back through versions
4. Record origin version and modification count
5. Calculate stability score (0-1) based on presence and modifications
6. Compute complexity metrics (section count, definition count)
7. Identify contentious areas (high modification frequency)

Inputs and Outputs:
- Input: Multiple bill versions with consensus analysis for each
- Output: EvolutionAnalysis with stability scores and complexity metrics

---

## Component 3: Causal Chain Reasoning - Detailed Breakdown

```
INPUT: Bill text, Version 3 findings, Consensus results
│
└─→ Root Cause Identification
    │
    └─ Primary bill amendment: "Section 2.1: Tax on energy generation
                                exceeding 50 megawatts"

Chain-of-Thought Reasoning (Structured 5-Step Process)
│
├─ STEP 1: How does this amend existing law?
│  │
│  ├─ Prompt: "What did prior law say about energy taxation?"
│  │
│  ├─ Model reasoning:
│  │  └─ "Prior law had general corporate tax but no per-MW tax.
│  │     This bill adds specific per-MW tax on energy generation."
│  │
│  ├─ Supporting quote from bill:
│  │  └─ Section 1: "Amending Section 3.1 of Energy Code"
│  │
│  ├─ Quote verification: Found? YES
│  │
│  └─ Evidence strength: 0.95 (direct reference)
│
├─ STEP 2: Who is affected by this change?
│  │
│  ├─ Prompt: "What entities qualify as 'energy generation'?"
│  │
│  ├─ Model reasoning:
│  │  └─ "Commercial facilities converting fuel to electricity.
│  │     Includes: coal plants, natural gas, nuclear, hydro.
│  │     Excludes: Residential solar, individual wind turbines."
│  │
│  ├─ Supporting quote from bill:
│  │  └─ Section 2.1a: "Commercial facility converting fuel to
│  │                    electricity for commercial purposes"
│  │
│  ├─ Quote verification: Found? YES
│  │
│  ├─ Alternative interpretation:
│  │  └─ Could "generation" include distribution? (5% likelihood)
│  │
│  └─ Evidence strength: 0.92 (clear definition)
│
├─ STEP 3: What threshold applies?
│  │
│  ├─ Prompt: "Why 50MW and what does it mean?"
│  │
│  ├─ Model reasoning:
│  │  └─ "50MW threshold likely chosen to target commercial-scale
│  │     generators, exempt small operations."
│  │
│  ├─ Supporting quote from bill:
│  │  └─ Section 2.1b: "Exceeding fifty megawatts capacity"
│  │
│  ├─ Quote verification: Found? YES
│  │
│  └─ Evidence strength: 0.98 (explicit and unambiguous)
│
├─ STEP 4: Who is exempt?
│  │
│  ├─ Prompt: "Are there carve-outs or exemptions?"
│  │
│  ├─ Model reasoning:
│  │  └─ "Yes, renewable energy facilities are exempt.
│  │     This incentivizes transition away from fossil fuels."
│  │
│  ├─ Supporting quote from bill:
│  │  └─ Section 3.2: "Renewable energy facilities exempt as
│  │                   defined in Section 5.2"
│  │
│  ├─ Quote verification: Found? YES
│  │
│  ├─ Alternative interpretation:
│  │  ├─ NARROW: "Renewable" = solar/wind only (60% likelihood)
│  │  ├─ BROAD: "Renewable" = solar/wind/biomass/geothermal (40%)
│  │  └─ Impact: If narrow, more facilities taxed
│  │
│  └─ Evidence strength: 0.85 (clear but depends on Sec 5.2)
│
└─ STEP 5: What is the business impact?
   │
   ├─ Prompt: "What must companies do and what costs?"
   │
   ├─ Model reasoning:
   │  └─ "Companies must file quarterly reports, annual tax returns.
   │     Tax rate: [rate] per MW. Expected liability: $500K-$2M/year
   │     depending on whether facilities qualify for renewal exempt."
   │
   ├─ Supporting quotes from bill:
   │  ├─ Section 4.1: "Quarterly generation reports required"
   │  ├─ Section 4.2: "Annual tax return filing required"
   │  └─ Section 4.3: "Tax rate $X per MW annually"
   │
   ├─ Quote verification: All found? YES
   │
   └─ Evidence strength: 0.95 (compliance requirements explicit)

Evidence Coverage Assessment
│
├─ Step 1: 95% confidence (direct reference)
├─ Step 2: 92% confidence (definition provided)
├─ Step 3: 98% confidence (explicit threshold)
├─ Step 4: 85% confidence (definition depends on interpretation)
├─ Step 5: 95% confidence (requirements stated)
│
└─ Overall coverage: (95+92+98+85+95)/5 = 93% (4 of 5 steps directly quoted)

Alternative Interpretations Exploration
│
├─ Alternative A: NARROW INTERPRETATION
│  ├─ Claim: "Tax applies only to fossil fuel generation"
│  ├─ Supporting evidence: Section 2.1 says "energy generation"
│  │                       generic but intent is fossil fuels
│  ├─ Likelihood: 5% (Section 2.1a says "any fuel", contradicts)
│  ├─ Impact if true: Tax applies narrowly, lower liability
│  └─ Probability-weighted impact: Low
│
├─ Alternative B: STANDARD INTERPRETATION
│  ├─ Claim: "Tax on in-state generation >50MW, excludes renewables"
│  ├─ Supporting evidence: Explicit >50MW threshold, renewable exemption
│  ├─ Likelihood: 65% (strong bill language, clear intent)
│  ├─ Impact if true: Tax applies to ~60% of generation, liability ~$1M
│  └─ Probability-weighted impact: PRIMARY SCENARIO
│
└─ Alternative C: BROAD INTERPRETATION
   ├─ Claim: "Tax on all generation including out-of-state imports"
   ├─ Supporting evidence: Section language slightly ambiguous on
   │                       "within jurisdiction"
   ├─ Likelihood: 30% (ambiguity in jurisdictional language)
   ├─ Impact if true: Tax applies broadly, liability ~$2M
   └─ Probability-weighted impact: Moderate

Counterfactual Analysis
│
├─ Parameter: If threshold were 25MW instead of 50MW
│  └─ Impact: More facilities subject to tax, liability increase 40-60%
│
├─ Parameter: If renewable exemption applied narrowly (solar/wind only)
│  └─ Impact: Biomass/geothermal facilities taxed, liability increase 20-30%
│
├─ Parameter: If effective date were 6 months later (July 1)
│  └─ Impact: Company has 6 months to optimize generation mix
│
└─ Parameter robustness: Finding holds under most counterfactuals
   (except exemption definition) → Robust conclusion

DAG Construction (Directed Acyclic Graph)
│
├─ Node: "Tax on >50MW" (Root)
│
├─ Node: "Commercial facility definition" → connects to Root (0.92 strength)
│
├─ Node: "50MW threshold" → connects to Root (0.98 strength)
│
├─ Node: "Renewable exemption" → converges to Root (0.85 strength)
│
├─ Convergence: Multiple paths lead to same impact
│  └─ Path 1: >50MW threshold → scope narrowed → lower FPR
│  └─ Path 2: Renewable exemption → escape route available → lower FPR
│  └─ Both paths confirmed → robust finding
│
└─ Leaf: "Business impact: $500K-$2M liability"

OUTPUT: CausalChain + CausalDAG
│
└─ Data structure:
   {
     "root_cause": "Section 2.1: Tax on >50MW",
     "root_quote": "Exceeding fifty megawatts capacity",
     "chain_of_thought": [
       {
         "step": 1,
         "question": "How does this amend existing law?",
         "answer": "Adds per-MW tax on energy generation",
         "quote": "Amending Section 3.1 of Energy Code",
         "strength": 0.95
       },
       ... (steps 2-5 similar)
     ],
     "alternatives": [
       {
         "interpretation": "Narrow (fossil fuels only)",
         "likelihood": 0.05,
         "impact": "Tax applies narrowly"
       },
       {
         "interpretation": "Standard (in-state, excludes renewables)",
         "likelihood": 0.65,
         "impact": "Tax applies to 60% of generation"
       },
       {
         "interpretation": "Broad (includes imports)",
         "likelihood": 0.30,
         "impact": "Tax applies broadly, 100%+ generation"
       }
     ],
     "coverage": 0.93,
     "confidence": 0.82
   }
```

What:
Build 5-step reasoning chains from bill amendments to business impacts. Each step backed by bill text quotes. Alternative interpretations ranked by likelihood. Counterfactual analysis tests robustness.

Why This Works:
Simple finding "Tax applies" is insufficient for decision-making. Why does it apply? To whom? Under what conditions? What are real consequences? Chain-of-thought reasoning breaks complexity into verifiable steps. Quote verification prevents speculation. Alternative interpretations acknowledge uncertainty honestly. Counterfactual analysis shows whether conclusions are robust or dependent on specific interpretations that could change.

How It Works:
1. Identify root cause (primary bill amendment)
2. Use 5-step CoT prompt (how, who, threshold, exemptions, impact)
3. For each step, extract supporting quote
4. Verify quote exists in bill
5. Assess evidence strength (direct, indirect, speculative)
6. Explore alternative interpretations with likelihood
7. Test counterfactuals (threshold variation, definition variation)
8. Build DAG showing causal connections
9. Output chain with confidence and alternatives

Inputs and Outputs:
- Input: Bill text, findings from consensus, consensus results
- Output: CausalChain with quoted reasoning, alternatives, counterfactuals

---

## Component 4: Confidence Aggregation - Detailed Breakdown

```
INPUT: ConsensusAnalysis + EvolutionAnalysis + CausalDAG
│
└─→ Component 1: Evidence Quality Score
    │
    ├─ Criterion 1: Quote existence
    │  ├─ Does finding have supporting quote? YES → +0.20
    │  ├─ Does finding have supporting quote? NO → +0.00
    │  └─ Applied: YES, quote "Section 2.1b"
    │
    ├─ Criterion 2: Quote clarity
    │  ├─ Clear/unambiguous quote → +0.20
    │  ├─ Paraphrased/inferred → +0.10
    │  ├─ Speculative/implied → +0.00
    │  └─ Applied: Clear (explicit language)
    │
    ├─ Criterion 3: Supporting definitions
    │  ├─ Cross-referenced defined term → +0.20
    │  ├─ No supporting definition → +0.00
    │  └─ Applied: YES, definition in Section 2.1a
    │
    ├─ Criterion 4: Quote count
    │  ├─ Single quote → +0.05
    │  ├─ Multiple quotes (2-3) → +0.10
    │  ├─ Many quotes (4+) → +0.15
    │  └─ Applied: 3 supporting quotes
    │
    ├─ Criterion 5: Hallucination risk
    │  ├─ None detected → +0.15
    │  ├─ Potential issue → +0.07
    │  ├─ High risk → +0.00
    │  └─ Applied: None detected (all quoted)
    │
    └─ Evidence Quality Score: 0.20 + 0.20 + 0.20 + 0.10 + 0.15 = 0.85
       Average: (0.20+0.20+0.20+0.10+0.15) / 5 = 0.90

Component 2: Model Agreement Score
│
├─ All 3 models agree → 0.95
│ │  (Unanimous consensus)
│ │
│ ├─ 2 of 3 models agree → 0.70
│ │  (Majority consensus)
│ │
│ ├─ 1 model only, quote verified → 0.50
│ │  (Verified but not consensus)
│ │
│ ├─ 1 model only, quote not verified → 0.25
│ │  (Unverified dispute)
│ │
│ └─ Models in conflict → 0.10
│    (Complete disagreement)
│
└─ Applied: 2 of 3 models (GPT-4o, Claude) agree
   ├─ Gemini disagreed initially
   ├─ Gemini corrected via quote verification
   ├─ Net result: Majority consensus
   └─ Model Agreement Score: 0.70

Component 3: Causal Strength Score
│
├─ Criterion 1: Root cause clarity
│  ├─ Clear, single root cause → +0.25
│  ├─ Multiple possible causes → +0.12
│  ├─ Unclear cause → +0.00
│  └─ Applied: Clear (Section 2.1)
│
├─ Criterion 2: Step verification
│  ├─ Count verified steps / total steps
│  ├─ Applied: 4 of 5 verified → +0.25 × (4/5)
│  └─ Result: +0.20
│
├─ Criterion 3: Alternative exploration
│  ├─ 3+ alternatives considered → +0.15
│  ├─ 1-2 alternatives considered → +0.07
│  ├─ No alternatives → +0.00
│  └─ Applied: 3 alternatives
│
├─ Criterion 4: Counterfactual robustness
│  ├─ Finding holds under most scenarios → +0.15
│  ├─ Finding fragile to parameter change → +0.05
│  ├─ Finding depends on single interpretation → +0.00
│  └─ Applied: Robust except exemption definition
│
├─ Criterion 5: Evidence coverage
│  ├─ 80%+ steps backed by direct quotes → +0.20
│  ├─ 50-80% coverage → +0.10
│  ├─ <50% coverage → +0.00
│  └─ Applied: 93% coverage
│
└─ Causal Strength Score: 0.25 + 0.20 + 0.15 + 0.15 + 0.20 = 0.95
   (Sum capped at 1.0, so final = 0.80 after calibration)

Component 4: Evolution Stability Score
│
├─ Present in v1 and unchanged through v3 → 0.95
│  └─ Applied: Finding present v1-v3, modified once → Not applicable
│
├─ Present in v1, settled after v2 → 0.85
│  ├─ Applied: Finding core concept from v1
│  ├─ Modified once in v2 (threshold added)
│  ├─ Unchanged in v3 (stable)
│  └─ Result: 0.85
│
├─ Modified 1-2 times across versions → 0.70
│  └─ Not applicable (scoring 0.85 above)
│
├─ Modified 3+ times (contentious) → 0.40
│  └─ Not applicable
│
├─ Recently added (final version only) → 0.20
│  └─ Not applicable
│
└─ Evolution Stability Score: 0.85

Weighted Aggregation Formula
│
├─ Component weights (optimized for bill analysis):
│  ├─ Evidence Quality: 0.40 (hallucination prevention)
│  ├─ Model Agreement: 0.30 (model diversity)
│  ├─ Causal Strength: 0.20 (reasoning quality)
│  └─ Evolution Stability: 0.10 (version history)
│
├─ Calculation:
│  └─ Confidence = (0.90 × 0.40) + (0.70 × 0.30) + (0.80 × 0.20) + (0.85 × 0.10)
│     = 0.36 + 0.21 + 0.16 + 0.085
│     = 0.815
│
└─ Final Confidence (before calibration): 0.815

Confidence Interval Calculation (90% CI)
│
├─ From calibration set (historical accuracy data):
│  │
│  ├─ Find findings with similar component scores
│  │  ├─ Evidence: 0.85-0.95
│  │  ├─ Agreement: 0.65-0.75
│  │  ├─ Causality: 0.75-0.85
│  │  └─ Stability: 0.80-0.90
│  │
│  ├─ Historical findings with similar scores: 47 examples
│  │  ├─ Accuracy distribution: [0.65, 0.70, 0.75, ..., 0.90, 0.92]
│  │  ├─ 5th percentile: 0.70
│  │  ├─ 95th percentile: 0.86
│  │  └─ Median: 0.82
│  │
│  └─ Apply to current finding:
│     ├─ Our confidence: 0.815 (close to median 0.82)
│     ├─ Lower bound (5th percentile): 0.70
│     ├─ Upper bound (95th percentile): 0.86
│     └─ Interpretation: "90% chance true accuracy between 70% and 86%"

OUTPUT: ConfidenceBreakdown
│
└─ Data structure:
   {
     "overall_confidence": 0.815,
     "confidence_interval": {
       "lower_bound": 0.70,
       "upper_bound": 0.86,
       "confidence_level": 0.90
     },
     "components": {
       "evidence_quality": 0.90,
       "model_agreement": 0.70,
       "causal_strength": 0.80,
       "evolution_stability": 0.85
     },
     "weakest_component": "model_agreement",
     "strong_components": ["evidence_quality", "causal_strength"],
     "interpretation": "Good overall confidence (0.815). Main weakness is
                       model agreement (only 2 of 3). Recommend expert
                       review to resolve disagreement."
   }
```

What:
Aggregate four independent confidence signals into single calibrated score with uncertainty bounds. Decomposition reveals which components are weak, guiding escalation.

Why This Works:
Single confidence number is meaningless without context. Does 0.815 mean evidence is strong or models agree? Decomposition shows evidence 0.90 (strong), agreement 0.70 (weak), causality 0.80 (strong), stability 0.85 (strong). Weak agreement immediately suggests: "Expert review needed." Weights reflect priorities: evidence (40%) prevents hallucination, agreement (30%) detects model errors, causality (20%) ensures reasoning quality, stability (10%) ties to version history. Confidence intervals bound uncertainty.

How It Works:
1. Score each component (evidence, agreement, causality, stability) independently
2. Apply component-specific criteria and thresholds
3. Weight each component by importance for bill analysis
4. Aggregate: multiply each component by weight, sum
5. From calibration set, find similar historical findings
6. Compute lower/upper bounds using percentiles
7. Output confidence with breakdown and interval

Inputs and Outputs:
- Input: ConsensusAnalysis, EvolutionAnalysis, CausalDAG
- Output: ConfidenceBreakdown with overall score, components, interval

---

## Component 5: Routing Decision - Detailed Breakdown

```
INPUT: ConfidenceBreakdown + Consensus Level + Dispute Status
│
└─→ Decision Tree Evaluation
    │
    Step 1: Is confidence >= 0.95?
    │
    ├─ YES
    │  └─→ Step 2a: Is consensus == "UNANIMOUS"?
    │      │
    │      ├─ YES → AUTO-PUBLISH
    │      │        ├─ Rationale: Very high confidence + all models agree
    │      │        ├─ Human review: None
    │      │        ├─ SLA: Publish immediately
    │      │        └─ Cost: Minimal ($0 human time)
    │      │
    │      └─ NO → Step 3: Continue to next threshold
    │
    ├─ NO → Step 3
    │
    └─→ Step 3: Is confidence >= 0.85?
        │
        ├─ YES
        │  └─→ Step 4a: Is consensus in ["UNANIMOUS", "STRONG"]?
        │      │
        │      ├─ YES → FLAGGED-PUBLISH
        │      │        ├─ Rationale: High confidence, clear consensus
        │      │        ├─ Human review: Light (compliance check only)
        │      │        ├─ SLA: 24 hours
        │      │        └─ Cost: Minimal ($50 per bill)
        │      │
        │      └─ NO → Step 5: Continue
        │
        ├─ NO → Step 5
        │
        └─→ Step 5: Is confidence >= 0.70?
            │
            ├─ YES
            │  └─→ Step 6: Is there unresolved dispute?
            │      │
            │      ├─ YES → ESCALATION
            │      │        ├─ Reason: Dispute not resolved via verification
            │      │        ├─ SLA: 4 hours
            │      │        └─ Cost: High ($300+ expert time)
            │      │
            │      └─ NO → Step 7: Is causal chain complete?
            │          │
            │          ├─ YES → EXPERT-REVIEW
            │          │        ├─ Rationale: Medium confidence needs specialist
            │          │        ├─ Expertise: Domain-specific (energy law, etc.)
            │          │        ├─ SLA: 48 hours
            │          │        └─ Cost: High ($200 specialist time)
            │          │
            │          └─ NO → ESCALATION
            │                 ├─ Reason: Chain incomplete, gaps in reasoning
            │                 ├─ SLA: 4 hours
            │                 └─ Cost: Very high ($300+)
            │
            └─ NO (confidence < 0.70)
               │
               └─→ ESCALATION
                   ├─ Rationale: Low confidence requires senior review
                   ├─ Expertise: Senior analyst + domain expert + legal
                   ├─ SLA: 4 hours (urgent)
                   └─ Cost: Very high ($400+)

Applied to HB-123:
│
├─ Confidence: 0.815 [0.70, 0.86]
├─ Consensus: MAJORITY (2 of 3)
├─ Dispute: Resolved (Gemini corrected via quote)
├─ Causal chain: Complete (5 steps, 93% coverage)
│
└─ Decision tree traversal:
   ├─ Step 1: Is 0.815 >= 0.95? NO → Continue
   ├─ Step 3: Is 0.815 >= 0.85? NO → Continue
   ├─ Step 5: Is 0.815 >= 0.70? YES
   ├─ Step 6: Any unresolved dispute? NO
   ├─ Step 7: Chain complete? YES
   └─ Result: EXPERT-REVIEW

Determine Required Expertise
│
├─ Analysis topic: Energy taxation
│  ├─ Primary expertise: Energy law
│  ├─ Secondary expertise: Tax compliance
│  └─ Tertiary expertise: Regulatory policy
│
└─ Assigned specialists: "energy_law_specialist, tax_counsel, director"

Set SLA and Escalation Path
│
├─ Action: EXPERT-REVIEW
├─ SLA: 48 hours
├─ Escalation path:
│  ├─ Level 1: Energy law specialist (expert review)
│  ├─ Level 2: Senior tax counsel (if specialist uncertain)
│  └─ Level 3: Director (if counsel uncertain)
│
└─ SLA enforcement:
   ├─ If specialist reviews within 24h: On track
   ├─ If not reviewed by 36h: Escalate to counsel automatically
   ├─ If not resolved by 48h: Mark as "Pending escalation"

Generate Routing Recommendation
│
└─ Output message:
   {
     "action": "EXPERT_REVIEW",
     "confidence": 0.815,
     "confidence_interval": [0.70, 0.86],
     "required_expertise": ["energy_law", "tax_compliance"],
     "escalation_path": ["energy_law_specialist", "senior_counsel", "director"],
     "sla_hours": 48,
     "rationale": "Majority consensus (2 of 3 models) on key provisions.
                  Confidence 0.815 indicates medium-high confidence, but model
                  disagreement suggests human specialist should review. Particular
                  focus: Renewable exemption definition (Section 5.2) - interpretation
                  affects tax liability by 20-30%. Recommend energy law specialist
                  clarify exemption scope before client implementation.",
     "priority": "NORMAL",
     "risk_level": "MEDIUM"
   }
```

What:
Route analysis result to appropriate action (auto-publish, flagged-publish, expert-review, escalation) based on confidence and consensus. Automate high-confidence findings while escalating uncertain ones.

Why This Works:
High-confidence unanimous findings need no review (saves cost). Medium-confidence findings need targeted expertise (specialist vs. generalist). Low-confidence or disputed findings need senior review (catches edge cases). SLA varies inversely with confidence: auto-publish same day, escalation 4 hours urgent. This ensures routine findings move quickly while high-risk findings get immediate attention.

How It Works:
1. Check confidence >= 0.95 and consensus == UNANIMOUS → AUTO-PUBLISH
2. Check confidence >= 0.85 and strong consensus → FLAGGED-PUBLISH
3. Check confidence >= 0.70 and no disputes → EXPERT-REVIEW
4. Otherwise → ESCALATION
5. Determine required expertise from finding topic
6. Set SLA based on action type
7. Output recommendation with rationale

Inputs and Outputs:
- Input: ConfidenceBreakdown, consensus level, dispute status
- Output: ActionRecommendation with action type, SLA, expertise, escalation path

---

## Summary: Component Interactions

```
STAGE 1              STAGE 2               STAGE 3              STAGE 4           STAGE 5
Consensus        Evolutionary         Causal Chains       Confidence        Routing
Ensemble         Analysis             Reasoning            Aggregation       Decision
│                │                    │                   │                 │
├─ Finding A     ├─ Stability: 0.85   ├─ Root: Sec 2.1   ├─ Evidence: 0.90 ├─ Action: 
├─ Finding B     ├─ Stable v3         ├─ 5-step chain    ├─ Agreement: 0.70├─ EXPERT_REVIEW
├─ Finding C     └─ Modified once     ├─ All quoted      ├─ Causality: 0.80│
│                                      ├─ Alternatives    ├─ Stability: 0.85│
└─ Consensus:    Confidence           └─ Confidence      │                 │
   2/3 agree     impact: ±10%            0.82            └─ Final: 0.815   └─ SLA: 48h

Data flow: Input → Component 1 → Component 2 → Component 3 → Component 4 → Component 5 → Output
           Bill      Consensus    Evolution    Causal       Aggregation    Routing      Final
                     findings     stability    strength     confidence     decision    Analysis
```

Each component output feeds into the next. Findings with consensus grouping flow to evolutionary analysis. Evolution stability modifies confidence. Causal chains provide evidence quality signal. Aggregation combines all signals. Routing decision ensures appropriate human involvement.

