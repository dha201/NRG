# Gemini Deep Research Agent - Complete Documentation

> **Source:** Official Google AI Documentation (January 2026)  
> **Last Updated:** January 14, 2026

---

## Overview

**Gemini Deep Research** is an autonomous AI agent designed to perform complex, long-running context gathering and synthesis tasks. It is powered by **Gemini 3 Pro**, Google's most factual model, and is specifically trained to reduce hallucinations and maximize report quality during complex research tasks.

### What is Deep Research?

Deep Research is an **agent, not just a model**. It autonomously:
- Plans multi-step research investigations
- Executes searches and reads web content
- Synthesizes findings into detailed, cited reports
- Identifies knowledge gaps and iteratively refines research

### Key Characteristics

- **Agentic Workflow:** A single request triggers an autonomous loop of planning, searching, reading, and reasoning
- **Long-Running:** Research tasks typically take several minutes to complete (max 60 minutes)
- **Analyst-in-a-Box:** Best suited for workloads requiring comprehensive analysis rather than low-latency chat
- **Citation-Rich:** Provides granular sourcing for all claims

---

## How It Works

### Research Process

```
1. User submits research prompt
   ↓
2. Agent generates research plan
   ↓
3. Autonomous Loop:
   ├── Formulate queries
   ├── Execute web searches
   ├── Read and analyze results
   ├── Identify knowledge gaps
   └── Search again (iterate)
   ↓
4. Synthesize findings
   ↓
5. Generate detailed report with citations
```

### Agent Capabilities

- **Iterative Planning:** Formulates queries, reads results, identifies gaps, and searches again
- **Deep Web Navigation:** Can navigate deep into websites for specific data (improved in latest release)
- **Multi-Source Integration:** Combines web search with user-provided data (PDFs, CSVs, docs)
- **Context Management:** Handles large context gracefully
- **Factuality:** Uses Gemini 3 Pro's superior factuality to minimize hallucinations

---

## Core Capabilities

### 1. Web Search (Default)

- **Tool:** `google_search` and `url_context`
- **Enabled by default** - no configuration needed
- Can navigate deep into websites for specific information
- Subject to [Google Search API restrictions](https://ai.google.dev/gemini-api/terms)

### 2. Research with Your Own Data

- **Tool:** `file_search` (must be explicitly added)
- Supports: PDFs, CSVs, text files, documents
- Combines proprietary data with web search
- Example use cases:
  - Due diligence with internal documents
  - Competitive analysis with market reports
  - Research contextualized by company data

### 3. Multimodal Inputs

**Supported Formats:**
- Images (JPG, PNG, etc.)
- PDFs
- Audio (except audio-only inputs)
- Video
- Text files

**Use Cases:**
- Analyze photograph → identify subjects → research behavior
- Review PDF → conduct related web research
- Process video → investigate topics mentioned

**⚠️ Caution:** Multimodal inputs increase costs and risk context window overflow.

### 4. Report Steerability

Control output format via explicit prompting:

**Formatting Options:**
- Structure reports into specific sections/subsections
- Include data tables
- Adjust tone (technical, executive, casual)
- Define headers and subheaders
- Specify JSON schema outputs

**Example Prompt:**
```
"Research competitor landscape for energy bill tracking software.
Format as: Executive Summary, Market Size, Top 5 Competitors (table), 
Key Differentiators. Use technical tone."
```

### 5. Follow-Up Interactions

- Continue conversation after initial report
- Ask for clarification, summarization, or elaboration
- No need to restart entire research task
- Use `previous_interaction_id` parameter

---

## Technical Specifications

### Model

- **Base Model:** Gemini 3 Pro
- **Agent ID:** `deep-research-pro-preview-12-2025`
- **Training:** Multi-step reinforcement learning for search
- **Optimization:** Factuality, report quality, cost efficiency

### API Access

**Method:** Interactions API (Beta)

**Requirements:**
- Google AI Studio API key OR Google Cloud credentials
- Beta status: Features and schemas subject to change

**Supported SDKs:**
- Python (`google-genai`)
- JavaScript (`@google/genai`)
- REST API

### Agent Name

```
agent='deep-research-pro-preview-12-2025'
```

---

## API Usage

### Basic Usage Pattern

**Python:**
```python
import time
from google import genai

client = genai.Client()

# Start research task (background required)
interaction = client.interactions.create(
    input="Research the history of Google TPUs.",
    agent='deep-research-pro-preview-12-2025',
    background=True  # Required for Deep Research
)

print(f"Research started: {interaction.id}")

# Poll for results
while True:
    interaction = client.interactions.get(interaction.id)
    
    if interaction.status == "completed":
        print(interaction.outputs[-1].text)
        break
    elif interaction.status == "failed":
        print(f"Research failed: {interaction.error}")
        break
    
    time.sleep(10)  # Poll every 10 seconds
```

**JavaScript:**
```javascript
import { GoogleGenAI } from '@google/genai';

const client = new GoogleGenAI({});

const interaction = await client.interactions.create({
    input: 'Research the history of Google TPUs.',
    agent: 'deep-research-pro-preview-12-2025',
    background: true
});

console.log(`Research started: ${interaction.id}`);

while (true) {
    const result = await client.interactions.get(interaction.id);
    
    if (result.status === 'completed') {
        console.log(result.outputs[result.outputs.length - 1].text);
        break;
    } else if (result.status === 'failed') {
        console.log(`Research failed: ${result.error}`);
        break;
    }
    
    await new Promise(resolve => setTimeout(resolve, 10000));
}
```

**REST API:**
```bash
# 1. Start research task
curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -d '{
    "input": "Research the history of Google TPUs.",
    "agent": "deep-research-pro-preview-12-2025",
    "background": true
  }'

# 2. Poll for results (replace INTERACTION_ID)
curl -X GET "https://generativelanguage.googleapis.com/v1beta/interactions/INTERACTION_ID" \
  -H "x-goog-api-key: $GEMINI_API_KEY"
```

### Adding File Search

```python
# Python example with file search
interaction = client.interactions.create(
    input="Analyze our Q4 financial report and research competitor performance.",
    agent='deep-research-pro-preview-12-2025',
    tools=[{'file_search': {'corpus_id': 'your-corpus-id'}}],
    background=True
)
```

### Follow-Up Questions

```python
# Continue conversation from previous research
follow_up = client.interactions.create(
    input="Can you elaborate on the section about market trends?",
    agent='deep-research-pro-preview-12-2025',
    previous_interaction_id=interaction.id,
    background=True
)
```

---

## Pricing

### Model

**Pay-as-you-go based on:**
- Gemini 3 Pro token usage (input/output)
- Tool usage (search queries)
- Context caching (50-70% cache hit typical)

### Cost Estimates

**Standard Research Task:**
- **Usage:** ~80 search queries, ~250k input tokens (50-70% cached), ~60k output tokens
- **Estimated Cost:** $2.00 - $3.00 per task

**Complex Research Task:**
- **Usage:** ~160 search queries, ~900k input tokens (50-70% cached), ~80k output tokens
- **Estimated Cost:** $3.00 - $5.00 per task

**Cost Factors:**
- Depth of research required
- Number of sources analyzed
- Multimodal input complexity
- Follow-up interactions

**Note:** Unlike standard chat (1 request = 1 output), Deep Research is an agentic workflow where a single request triggers multiple autonomous steps.

---

## Performance Benchmarks

### Latest Results (December 2025 Release)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **Humanity's Last Exam (HLE)** | 46.4% | State-of-the-art on full set |
| **DeepSearchQA** | 66.1% | Multi-step web research tasks |
| **BrowseComp** | 59.2% | Best Google performance |

### DeepSearchQA Benchmark

**Open-Source Benchmark for Research Agents**

- **Tasks:** 900 hand-crafted "causal chain" tasks
- **Fields:** 17 different domains
- **Complexity:** Each step depends on prior analysis
- **Metrics:** Comprehensiveness (exhaustive answer sets), precision, recall

**Access:**
- [Dataset](https://www.kaggle.com/datasets/deepmind/deepsearchqa/data)
- [Leaderboard](https://www.kaggle.com/benchmarks/google/dsqa/leaderboard)
- [Technical Report](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)
- [Starter Colab](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code)

---

## Best Practices

### 1. Prompt for Unknowns

Instruct the agent on how to handle missing data:

**Example:**
```
"If specific figures for 2025 are not available, explicitly state they 
are projections or unavailable rather than estimating."
```

### 2. Provide Context

Ground the agent's research by providing background information:

**Example:**
```
"Research renewable energy legislation in Texas. Context: We are a 
solar panel manufacturer targeting commercial buildings with 50+ employees."
```

### 3. Define Output Structure

Be explicit about desired format:

**Example:**
```
"Structure report as: 1) Executive Summary (3 paragraphs), 2) Key Findings 
(bullet points), 3) Competitive Landscape (comparison table), 4) Recommendations 
(numbered list)"
```

### 4. Use Multimodal Inputs Cautiously

- Increases costs significantly
- Risk of context window overflow
- Best for when visual/audio analysis is critical

### 5. Leverage Citations

- Review citations to verify source quality
- Use citations to validate claims
- Check for source diversity

### 6. Background Execution Required

```python
background=True  # Always required for Deep Research
```

---

## Limitations

### Current Restrictions

| Limitation | Details |
|-----------|---------|
| **Beta Status** | API in public beta - schemas may change |
| **Custom Tools** | Cannot provide custom Function Calling tools or remote MCP servers |
| **Structured Output** | No human-approved planning or structured outputs (yet) |
| **Max Research Time** | 60 minutes maximum (most tasks complete in 20 minutes) |
| **Store Requirement** | `background=True` requires `store=True` |
| **Audio Inputs** | Audio-only inputs not supported |
| **Google Search** | Enabled by default, subject to [specific restrictions](https://ai.google.dev/gemini-api/terms) |

### Planned Features (Not Yet Available)

- Human-in-the-loop plan approval
- Custom tool integration
- Structured output support for reports

---

## Safety Considerations

### 1. Prompt Injection via Files

**Risk:** Malicious files could contain hidden text to manipulate agent output

**Mitigation:**
- Only upload documents from trusted sources
- Review file contents before processing
- Validate agent output against known facts

### 2. Web Content Risks

**Risk:** Agent may encounter malicious web pages

**Mitigation:**
- Google implements robust safety filters
- Review citations to verify source quality
- Cross-check critical claims

### 3. Data Exfiltration

**Risk:** Sensitive internal data could be leaked via web searches

**Mitigation:**
- Be cautious when summarizing sensitive data AND allowing web browsing
- Consider separate research tasks (internal-only vs. web-enabled)
- Review prompts for inadvertent data exposure

---

## When to Use Deep Research

### ✅ Best Use Cases

- **Due Diligence:** Financial analysis, competitor research, market sizing
- **Literature Review:** Scientific research, patent analysis, academic surveys
- **Market Research:** Trend analysis, customer insights, industry reports
- **Regulatory Analysis:** Legislative tracking, compliance research, policy impacts
- **Strategic Planning:** SWOT analysis, opportunity assessment, risk evaluation

### ❌ Not Suitable For

- **Low-latency chat:** Use standard Gemini models instead
- **Simple fact lookup:** Overkill for straightforward queries
- **Real-time interactions:** Tasks take minutes, not seconds
- **Streaming conversations:** Agent works asynchronously

---

## Real-World Applications

### Financial Services

- **Use Case:** Automate initial due diligence stages
- **Workflow:** Aggregate market signals, competitor analysis, compliance risks
- **Impact:** Massive force multiplier for investment teams

### Biotech & Drug Discovery

**Example: Axiom Bio**
- **Challenge:** Predict drug toxicity across biomedical literature
- **Solution:** Deep Research provides unprecedented research depth/granularity
- **Impact:** Accelerated drug discovery pipelines

### Legislative Intelligence

- **Use Case:** Bill discovery and impact analysis
- **Workflow:** Weekly scans for new legislation, analyze business impacts
- **Tools:** Web search + File Search (for business context documents)

---

## Integration with Google Products

**Current Availability:**
- ✅ Google AI Studio
- ✅ Gemini API
- ✅ Gemini Enterprise (with allowlist)

**Upcoming:**
- 🔜 Google Search
- 🔜 NotebookLM
- 🔜 Google Finance
- ✅ Gemini App (upgraded)

---

## Interactions API (Beta)

Deep Research uses the **Interactions API**, a next-generation interface for Gemini models and agents.

### Key Concepts

**Interaction Resource:**
- Represents a complete turn in conversation/task
- Contains entire history: inputs, thoughts, tool calls, results, outputs
- Acts as session record

**State Management:**
- Server-side state management (no client-side tracking needed)
- Stateful conversations via `previous_interaction_id`
- Automatic tool orchestration

**Status Lifecycle:**
```
in_progress → completed (or) failed
```

### Supported Models & Agents

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-3-pro-preview`
- `gemini-3-flash-preview`
- `deep-research-pro-preview-12-2025` ⭐

---

## Getting Started

### Prerequisites

1. **API Key:** Get from [Google AI Studio](https://aistudio.google.com/)
2. **SDK:** Install Python or JavaScript SDK
3. **Beta Access:** Interactions API is in public beta

### Quick Start

```python
# Install SDK
pip install google-genai

# Set API key
export GEMINI_API_KEY='your-api-key'

# Run first research task
from google import genai

client = genai.Client()
interaction = client.interactions.create(
    input="Research the latest developments in quantum computing.",
    agent='deep-research-pro-preview-12-2025',
    background=True
)

# Poll and get results
import time
while True:
    status = client.interactions.get(interaction.id)
    if status.status == "completed":
        print(status.outputs[-1].text)
        break
    time.sleep(10)
```

---

## Documentation Links

### Official Resources

- **API Documentation:** https://ai.google.dev/gemini-api/docs/deep-research
- **Interactions API:** https://ai.google.dev/gemini-api/docs/interactions
- **Blog Announcement:** https://blog.google/technology/developers/deep-research-agent-gemini-api/
- **Pricing:** https://ai.google.dev/gemini-api/docs/pricing
- **Terms of Service:** https://ai.google.dev/gemini-api/terms

### Benchmarks

- **DeepSearchQA Dataset:** https://www.kaggle.com/datasets/deepmind/deepsearchqa/data
- **Leaderboard:** https://www.kaggle.com/benchmarks/google/dsqa/leaderboard
- **Technical Report:** https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf

### Related Tools

- **File Search:** https://blog.google/technology/developers/file-search-gemini-api/
- **Google AI Studio:** https://aistudio.google.com/
- **Gemini App:** https://gemini.google/overview/deep-research/

---

## Summary

**Gemini Deep Research** represents a significant advancement in autonomous AI research capabilities:

✅ **Autonomous:** Plans, executes, and synthesizes research without intervention  
✅ **Comprehensive:** Iteratively searches and reads until knowledge gaps are filled  
✅ **Factual:** Powered by Gemini 3 Pro, optimized for accuracy  
✅ **Flexible:** Supports web + proprietary data, multimodal inputs  
✅ **Production-Ready:** API access via Interactions API (Beta)  
✅ **Cost-Effective:** $2-5 per task for analyst-level research  

**Ideal for:** Due diligence, market research, legislative tracking, literature reviews, strategic analysis

**Key Consideration:** Long-running (minutes to hours) - not suitable for low-latency chat use cases
