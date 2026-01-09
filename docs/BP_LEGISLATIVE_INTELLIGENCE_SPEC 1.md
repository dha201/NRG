# BP Legislative Intelligence Agent - Requirements Specification

**Version:** 1.0
**Date:** December 2025
**Prepared for:** BP Energy

---

## Executive Summary

Build a legislative intelligence agent that automates monitoring, analysis, and comparison of federal and state legislation affecting BP Energy's business operations. The solution will leverage AI/LLM capabilities to provide actionable insights grounded in BP-specific business context.

---

## Functional Requirements

### 1. Bill Retrieval and Storage

- Process to periodically scan BP-approved legislative API sources for new bills that match the specified search configuration
- Storage of bill data in a BP-provided data lake

### 2. Bill Analysis and Storage

- AI analysis and summary generation for each tracked bill, including version tracking
- BP-grounded analysis using business context
- Impact scoring and recommended actions based on configuration file
- Related media and social media search, summary, and sentiment analysis based on available APIs
- Storage of bill and media analysis elements stored in a BP-provided data lake

### 3. Report Generation and Notifications

- Reports on bill analysis will be generated in DOCX or PDF format and stored to BP location
- Periodic email that is sent to Legal or their designated group mailbox with links to the bill analysis

---

## Assumptions

- Solution will be deployed on BP cloud infrastructure using BP service providers
- Detailed requirements will be mutually agreed upon to fit available resources
- Services will be delivered between Jan 2026 and June 2026
- Bill information and analysis stored in data lake can be used as RAG corpus by BP for chatbot

---

## API Data Sources

The solution leverages three primary API sources for comprehensive legislative and regulatory coverage.

### API Stack Overview

| Purpose | API | Cost | Authentication |
|---------|-----|------|----------------|
| **Federal Legislation** | Congress.gov | Free | API key (api.data.gov) |
| **Federal Regulations** | Regulations.gov | Free | API key (api.data.gov) |
| **State Legislation** | Open States / Plural Policy | $59+/month | X-API-KEY header |

**Estimated Monthly Cost:** $70-110/month (Open States $59 + LLM analysis $10-50)

---

## 1. Congress.gov API

### Overview
Official U.S. government API providing comprehensive federal legislative data from the Library of Congress.

### Documentation & Links
- **API Documentation:** https://api.congress.gov/
- **API Registration:** https://api.congress.gov/sign-up/
- **Interactive Docs:** https://api.congress.gov/#/bill

### Technical Specifications

| Attribute | Value |
|-----------|-------|
| **Base URL** | `https://api.congress.gov/v3/` |
| **Authentication** | API key via `?api_key=` query parameter |
| **Rate Limit** | 5,000 requests/hour |
| **Cost** | Free (no paid tier) |
| **Data Format** | JSON (default), XML |
| **Update Frequency** | 6x daily for bills, 30 min for votes |

### Key Endpoints

```
GET /bill/{congress}/{type}                    # List bills
GET /bill/{congress}/{type}/{number}           # Bill details
GET /bill/{congress}/{type}/{number}/subjects  # Bill subjects/policy areas
GET /bill/{congress}/{type}/{number}/summaries # CRS summaries
GET /bill/{congress}/{type}/{number}/actions   # Bill actions/history
GET /bill/{congress}/{type}/{number}/text      # Bill text links (to govinfo.gov)
```

### Data Available
- Bills (HR, S, HJRES, SJRES, HCONRES, SCONRES, HRES, SRES)
- Amendments
- CRS Summaries (high-quality professional summaries)
- Bill actions and status timeline
- Sponsors and cosponsors
- Committee assignments
- Policy areas and legislative subjects
- Bill text versions (via govinfo.gov links)

### Oil/Gas Filtering
- **Method:** Policy Area filtering (post-fetch)
- **Relevant Policy Areas:** "Energy"
- **Relevant Subjects:** "Oil and gas", "Petroleum", "Natural gas", "Pipeline safety", "Offshore drilling", "Hydraulic fracturing"

### Limitations
- Federal only (no state coverage)
- No direct keyword search on bill text
- Must fetch then filter by subject (high API call volume initially)

---

## 2. Regulations.gov API

### Overview
Official U.S. government API providing access to federal regulatory and rulemaking processes, maintained by GSA.

### Documentation & Links
- **API Documentation:** https://open.gsa.gov/api/regulationsgov/
- **API Registration:** https://api.data.gov/signup/ (same key as Congress.gov)
- **OpenAPI Spec:** https://api.regulations.gov/v4/openapi.yaml
- **Website:** https://www.regulations.gov/

### Technical Specifications

| Attribute | Value |
|-----------|-------|
| **Base URL** | `https://api.regulations.gov/v4/` |
| **Authentication** | API key via `?api_key=` query parameter |
| **Rate Limit** | 1,000 requests/hour |
| **Cost** | Free (no paid tier) |
| **Data Format** | JSON |
| **Update Frequency** | As agencies post (typically daily) |

### Key Endpoints

```
GET /documents                     # Search documents
GET /documents/{documentId}        # Document details
GET /comments                      # Search public comments
GET /comments/{commentId}          # Comment details
GET /dockets                       # Search dockets (grouped proceedings)
GET /dockets/{docketId}            # Docket details
```

### Data Available
- Proposed Rules (NPRM - Notice of Proposed Rulemaking)
- Final Rules
- Notices
- Supporting & Related Materials
- Public Comments
- Comment periods and effective dates
- Dockets (collections of related documents)

### Filtering Capabilities (Excellent)
```
# Keyword search
filter[searchTerm]=oil and gas

# Agency filtering (energy-relevant agencies)
filter[agencyId]=epa    # Environmental Protection Agency
filter[agencyId]=doe    # Department of Energy
filter[agencyId]=ferc   # Federal Energy Regulatory Commission
filter[agencyId]=phmsa  # Pipeline & Hazardous Materials Safety
filter[agencyId]=doi-blm # Bureau of Land Management

# Document type
filter[documentType]=Proposed Rule
filter[documentType]=Rule

# Date range
filter[postedDate][gte]=2024-01-01

# Combined example
GET /documents?filter[agencyId]=epa&filter[searchTerm]=oil and gas&filter[documentType]=Proposed Rule
```

### Limitations
- Regulations only (not legislation)
- Federal only (no state regulations)
- 2,500 result limit per query (use date chunking for larger datasets)

---

## 3. Open States / Plural Policy API

### Overview
Comprehensive state legislative data platform providing unified access to legislation from all 50 US states, DC, Puerto Rico, and some municipalities through a modern REST API (v3).

### Documentation & Links
- **API Documentation:** https://docs.openstates.org/
- **Interactive API Docs:** https://v3.openstates.org/docs
- **Alternative Docs (ReDoc):** https://v3.openstates.org/redoc
- **API Key Registration:** https://open.pluralpolicy.com/accounts/profile/
- **Plural Policy Platform:** https://pluralpolicy.com/

**Important:** API keys are managed through `open.pluralpolicy.com` (separate account from `pluralpolicy.com`)

### Technical Specifications

| Attribute | Value |
|-----------|-------|
| **Base URL** | `https://v3.openstates.org/` |
| **Authentication** | `X-API-KEY` header or `?apikey=` query param |
| **Rate Limit** | Not publicly disclosed (sufficient for production) |
| **Cost** | $59/month minimum for production |
| **Data Format** | JSON (REST v3) |
| **Update Frequency** | Varies by state |

### Key Endpoints

```
GET /jurisdictions                             # List all states/jurisdictions
GET /jurisdictions/{jurisdiction_id}           # State details
GET /bills?jurisdiction={state}&q={search}     # Search bills (full-text)
GET /bills/{jurisdiction}/{session}/{bill_id}  # Get bill by ID
GET /bills/ocd-bill/{openstates_id}            # Get bill by internal UUID
GET /people?jurisdiction={state}               # List legislators
GET /people.geo?lat={lat}&lng={lng}            # Find reps by location
GET /committees?jurisdiction={state}           # List committees
GET /events?jurisdiction={state}               # List events
```

### Data Available
- Bills (all types across all jurisdictions)
- Bill versions (Introduced, Committee Reports, Engrossed, Enrolled)
- Bill text (PDF links to state legislature sources)
- Bill status and progress
- Sponsors and cosponsors
- Legislative subjects
- Actions and history
- Legislators
- Committees
- Geographic legislator lookup

### Bill Version Tracking (Key Feature)

| Version Type | Description |
|--------------|-------------|
| Introduced | Original filed version |
| House Committee Report | After House committee markup |
| Engrossed | Passed originating chamber |
| Senate Committee Report | After Senate committee markup |
| Enrolled | Passed both chambers (final) |

### Filtering Capabilities (Excellent)
```
# Full-text search
GET /bills?jurisdiction=TX&q=oil and gas
GET /bills?jurisdiction=TX&q=natural gas production

# Subject filtering
GET /bills?jurisdiction=TX&subject=Energy
GET /bills?jurisdiction=TX&subject=Public Utilities

# Date filtering
GET /bills?jurisdiction=TX&updated_since=2025-01-01
GET /bills?jurisdiction=TX&action_since=2025-01-01

# Combined
GET /bills?jurisdiction=TX&session=89&q=energy&updated_since=2025-01-01&sort=updated_desc
```

### Pricing Tiers

| Tier | Monthly Cost | Features | Recommendation |
|------|--------------|----------|----------------|
| Free | $0 | Track 5 bills only | Testing only |
| **Unlimited Tracking** | **$59** | Unlimited bill tracking, 1 seat | **Minimum for production** |
| Premium | $499+ | AI insights, exports, unlimited users | If team features needed |
| API Plans | Contact sales | Higher rate limits, bulk access | If volume issues |

### Limitations
- No federal coverage (use Congress.gov)
- PDF text extraction required (bill text not inline)
- $59/month minimum for production use
- Rate limits not publicly documented

---

## API Comparison Matrix

| Feature | Congress.gov | Regulations.gov | Open States |
|---------|-------------|-----------------|-------------|
| **Geographic Scope** | Federal only | Federal only | 50 states + DC + PR |
| **Data Type** | Legislation | Regulations | State Legislation |
| **Cost** | Free | Free | $59+/month |
| **Rate Limit** | 5,000/hour | 1,000/hour | Undisclosed |
| **Bill Versions** | Full support | N/A | Full support |
| **Full-Text Search** | No | Yes | Yes |
| **Oil/Gas Filtering** | Indirect | Excellent | Excellent |
| **Official Source** | Yes (LoC) | Yes (GSA) | Aggregator |
| **API Style** | REST | REST | REST v3 |

---

## Environment Variables Required

```bash
# Congress.gov & Regulations.gov (same key via api.data.gov)
CONGRESS_API_KEY=your_key

# Open States / Plural Policy (separate account)
OPENSTATES_API_KEY=your_key

# LLM Analysis (choose one or more)
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key  # For Gemini
```

---

## Monthly Cost Estimate

| Component | Cost | Notes |
|-----------|------|-------|
| Congress.gov | $0 | Free tier sufficient |
| Regulations.gov | $0 | Free tier sufficient |
| Open States | $59 | Unlimited Tracking tier |
| LLM Analysis | $10-50 | Depends on volume |
| **Total** | **$70-110** | Production minimum |

---

## Deliverables Timeline

| Phase | Timeframe | Deliverables |
|-------|-----------|--------------|
| Phase 1 | Jan 2026 | API integration, data pipeline |
| Phase 2 | Feb-Mar 2026 | AI analysis engine, business context |
| Phase 3 | Apr 2026 | Report generation, notifications |
| Phase 4 | May-Jun 2026 | Testing, deployment, documentation |

---

## Technical Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────┐
│                     BP Legislative Intelligence                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Congress.gov │  │Regulations   │  │ Open States  │          │
│  │    API       │  │   .gov API   │  │    API       │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         └─────────────────┼──────────────────┘                   │
│                           │                                      │
│                    ┌──────▼───────┐                              │
│                    │  Data Lake   │                              │
│                    │  (BP Cloud)  │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│                    ┌──────▼───────┐                              │
│                    │  AI/LLM      │ ◄─── BP Business Context     │
│                    │  Analysis    │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                    │
│         │                 │                 │                    │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌─────▼────────┐          │
│  │   Reports    │  │    Email     │  │  RAG/Chatbot │          │
│  │ (DOCX/PDF)   │  │   Alerts     │  │   Corpus     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contact & Resources

### API Documentation Links

| API | Documentation | Registration |
|-----|---------------|--------------|
| Congress.gov | https://api.congress.gov/ | https://api.congress.gov/sign-up/ |
| Regulations.gov | https://open.gsa.gov/api/regulationsgov/ | https://api.data.gov/signup/ |
| Open States | https://docs.openstates.org/ | https://open.pluralpolicy.com/accounts/profile/ |

### Interactive API Explorers

| API | Interactive Docs |
|-----|------------------|
| Congress.gov | https://api.congress.gov/#/bill |
| Regulations.gov | https://api.regulations.gov/v4/openapi.yaml |
| Open States | https://v3.openstates.org/docs |

---

*Document prepared for BP Energy legislative intelligence requirements*
