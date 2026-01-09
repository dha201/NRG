# Azure Deployment Options for NRG Legislative Intelligence App

## App Architecture Summary

**Type:** Python CLI application with scheduled execution  
**Key Components:**
- API integrations (Congress.gov, Open States, Regulations.gov)
- LLM processing (Google Gemini / OpenAI GPT)
- **Pluggable storage layer** (Snowflake, BigQuery, Azure Synapse, SQLite)
- Report generation (JSON, Markdown, DOCX via Pandoc)
- Scheduled execution pattern (designed to run periodically)

**⚠️ IMPORTANT: Storage Architecture**  
The app uses **Repository Pattern** for pluggable storage backends. Customer can store data in their own data lake (Snowflake, BigQuery, Azure Synapse, etc.) by changing environment variables. See `@/Users/thamac/Documents/NRG/docs/PLUGGABLE_STORAGE_ARCHITECTURE.md` for detailed implementation guide.

**Dependencies:**
- Python 3.9+
- httpx, openai, google-genai, PyYAML, rich, PyPDF2, pdfplumber
- Pandoc (for DOCX conversion)
- Environment variables for API keys

**Execution Pattern:**
- Not a web server (no HTTP endpoints)
- Runs as a batch job/scheduled task
- ~0.10-0.50 USD per execution (LLM API costs)
- Output: Files (JSON/MD/DOCX) and SQLite database updates

---

## Recommended Azure Services

### 1. **Azure Functions (Timer Trigger)** RECOMMENDED

**What it does:**  
Serverless compute service that runs your code in response to events (timers, HTTP requests, queue messages, etc.). No infrastructure management required.

**Why suitable:**
- Native support for scheduled execution via Timer Triggers (CRON expressions)
- Pay-per-execution model aligns with your periodic usage pattern
- Built-in support for environment variables and secrets
- Python 3.9+ runtime fully supported
- **Connects to customer's data lake** (Snowflake, BigQuery, Azure Synapse) via connection strings
- Stateless execution (no local storage needed with data lake backend)

**Pricing:**
- **Consumption Plan:** 1M free executions + 400,000 GB-s per month
- After free tier: ~$0.20 per million executions + $0.000016/GB-s
- **Estimated monthly cost:** $5-15 for daily runs (mostly LLM API costs, minimal compute)

**Pros:**
- Zero infrastructure management
- Auto-scaling (though not needed for your use case)
- Very cost-effective for scheduled tasks
- 1M free executions monthly
- Easy integration with Azure Storage for output files
- Built-in monitoring via Application Insights

**Cons:**
- Consumption plan has 10-minute execution timeout (Premium plan removes this)
- Requires adapting CLI code to Functions framework (entry point changes)
- Cold start latency (1-2 seconds) - not critical for scheduled jobs
- Pandoc needs custom installation in deployment package

**Source:** https://azure.microsoft.com/en-us/pricing/details/functions/

---

### 2. **Azure App Service WebJobs**

**What it does:**  
Background task feature built into Azure App Service. Runs scripts/programs alongside or independently from a web app.

**Why suitable:**
- Designed specifically for background/scheduled tasks
- Supports Python scripts natively on Linux App Service
- CRON scheduling via NCRONTAB expressions
- Shares App Service plan with potential future web interface

**Pricing:**
- **Basic B1:** ~$13/month (1 vCPU, 1.75 GB RAM, always-on)
- **Standard S1:** ~$70/month (1 vCPU, 1.75 GB RAM, better SLA)
- No per-execution charges - flat monthly rate

**Pros:**
- Minimal code changes (runs your script as-is)
- "Always On" feature prevents cold starts
- Easy to deploy via ZIP or Git
- Direct file system access for SQLite database
- Pandoc can be installed via SSH/Kudu console
- WebJobs supported at no extra cost on App Service plan

**Cons:**
- More expensive than Functions for infrequent runs
- Requires Basic tier or higher (~$13/month minimum)
- Pay for always-running instance even when idle
- Overkill if you only need scheduled tasks (no web app)
- Manual scaling if workload increases

**Sources:**
- https://learn.microsoft.com/en-us/azure/app-service/webjobs-create
- https://learn.microsoft.com/en-us/azure/app-service/webjobs-execution

---

### 3. **Azure Container Apps (Jobs)**

**What it does:**  
Modern container orchestration service built on Kubernetes/KEDA. Supports scheduled and event-driven container jobs without K8s complexity.

**Why suitable:**
- Native support for scheduled jobs (CRON)
- Containers allow packaging Python + Pandoc + all dependencies
- Scale-to-zero pricing model
- No infrastructure management

**Pricing:**
- **Consumption Plan:** $0.000012/vCPU-second + $0.000002/GiB-second (active usage only)
- Free grants: 180,000 vCPU-seconds + 360,000 GiB-seconds per month
- **Estimated:** $2-8/month for daily 5-minute jobs

**Pros:**
- Complete control over runtime environment (Docker container)
- Scale to zero when not running (pay only for execution time)
- Easy to package dependencies (Pandoc, Python libs)
- Modern platform with good future-proofing
- Supports manual triggers + scheduled execution
- Built-in secrets management

**Cons:**
- Requires Docker containerization knowledge
- More complex setup than Functions
- Learning curve for Container Apps concepts
- Persistent storage requires Azure Files mounting for SQLite

**Sources:**
- https://learn.microsoft.com/en-us/azure/container-apps/compare-options
- https://azure.microsoft.com/en-us/pricing/details/container-apps/

---

### 4. **Azure Logic Apps**

**What it does:**  
Low-code/no-code workflow automation platform. Designed for integration scenarios with visual designer.

**Why suitable:**
- Built-in scheduling capabilities
- Can trigger external scripts via HTTP or containers
- Good for orchestrating multiple services

**Pricing:**
- **Consumption:** ~$0.000025 per action execution
- **Standard:** Starts at ~$200/month for dedicated instance

**Pros:**
- Visual workflow designer
- No code deployment needed
- Built-in connectors for 400+ services
- Easy scheduling interface

**Cons:**
- ❌ **Not recommended** - overkill for this use case
- Requires wrapping Python script in HTTP endpoint or container
- More expensive than Functions for simple scheduled tasks
- Best for complex multi-service workflows, not single scripts
- Adds unnecessary complexity layer

**Source:** https://learn.microsoft.com/en-us/azure/azure-functions/functions-compare-logic-apps-ms-flow-webjobs

---

### 5. **Azure Batch**

**What it does:**  
Large-scale parallel computing service for HPC and batch processing workloads.

**Why suitable:**
- Designed for batch jobs
- Can run Python scripts on VM pools

**Pricing:**
- No charge for Batch service itself
- Pay for underlying VMs (similar to App Service)

**Cons:**
- ❌ **Not recommended** - massive overkill
- Designed for compute-intensive parallel workloads (rendering, simulations)
- Your app is I/O bound (API calls), not CPU-bound
- Complex setup with VM pools, task scheduling
- Higher operational overhead

**Source:** https://learn.microsoft.com/en-us/azure/batch/quick-run-python

---

## Comparison Matrix

| Service | Monthly Cost | Setup Complexity | Best For | Cold Start | Data Lake Support |
|---------|-------------|------------------|----------|------------|-------------------|
| **Azure Functions** | $5-15 | Low | Scheduled events | Yes (1-2s) | ✅ All backends |
| **App Service WebJobs** | $13+ | Very Low | Always-on tasks | No | ✅ All backends |
| **Container Apps Jobs** | $2-8 | Medium | Containerized jobs | No | ✅ All backends |
| **Logic Apps** | $20+ | Low | Multi-service workflows | No | ⚠️ Limited |
| **Azure Batch** | $50+ | High | HPC/parallel compute | No | ✅ All backends |

---

## Detailed Recommendations

### 🥇 Best Choice: **Azure Functions (Consumption Plan with Timer Trigger)**

**Why:**
1. **Cost-effective:** Free tier covers most usage, minimal compute cost beyond LLM APIs
2. **Purpose-built:** Timer Triggers designed exactly for scheduled task execution
3. **Low maintenance:** Fully serverless, no infrastructure to manage
4. **Quick setup:** Deploy directly from VS Code or CLI
5. **Scalable:** Can easily add HTTP triggers later for on-demand analysis

**Migration steps:**
1. **Refactor code** to use Repository Pattern (see PLUGGABLE_STORAGE_ARCHITECTURE.md)
2. Create Azure Function App (Python 3.9+ runtime)
3. Convert `main()` execution to Function handler
4. Add Timer Trigger with CRON schedule
5. **Configure storage backend** via environment variables:
   - `STORAGE_BACKEND=snowflake` (or bigquery, synapse, etc.)
   - Add connection strings for chosen data lake
6. Store API keys and DB credentials in Azure Key Vault
7. Deploy via Azure Functions Core Tools or GitHub Actions

**Caveats:**
- 10-minute timeout on Consumption plan (upgrade to Premium if runs exceed this)
- Need to handle Pandoc installation (include in deployment or use custom container)

---

### 🥈 Alternative: **Azure App Service WebJobs (Basic B1 Plan)**

**When to choose this:**
- You plan to add a web dashboard/interface later (can share App Service plan)
- You want minimal code changes (runs script as-is)
- 10-minute timeout is too restrictive (WebJobs allow longer runs)
- You prefer traditional VM-like environment with filesystem access

**Trade-off:**
- Higher fixed cost (~$13/month vs Functions' pay-per-use)
- Better for always-on or long-running processes

---

### 🥉 Future-Proof Option: **Azure Container Apps Jobs**

**When to choose this:**
- You want complete environment control
- Planning to containerize for portability
- May need to scale to multiple parallel jobs later
- Want modern cloud-native architecture

**Trade-off:**
- Steeper learning curve (Docker + Container Apps concepts)
- More setup time initially
- Best ROI if you'll use containers elsewhere

---

## Storage Recommendations

### Data Storage (Customer's Data Lake)

**Primary storage** for bill data, analyses, and change tracking:

**Option 1: Customer's Existing Data Lake** (Recommended)
- **Snowflake:** ~$50-200/month (compute + storage)
- **BigQuery:** ~$20-100/month (queries + storage)
- **Azure Synapse:** ~$50-200/month (compute + storage)
- **Benefit:** Data stays in customer's infrastructure, no data duplication

**Option 2: SQLite (Dev/Test Only)**
- Azure Files mount: ~$0.06/GB/month
- **Not recommended for production** (single-instance limitation)

**See `PLUGGABLE_STORAGE_ARCHITECTURE.md` for implementation details**

### Auxiliary Storage

1. **Azure Storage Account** (for output files)
   - Blob Storage: Store JSON/MD/DOCX reports (~$0.02/GB/month)
   - Cost: ~$1-5/month for typical usage

2. **Azure Key Vault** (for secrets - **required**)
   - Store API keys + data lake credentials
   - Secrets storage: $0.03 per 10,000 operations
   - Cost: <$1/month

---

## Getting Started Guide

### Quick Start with Azure Functions

```bash
# Install Azure Functions Core Tools
brew install azure-functions-core-tools@4  # macOS

# Create new Function App
func init NRGFunctionApp --python
cd NRGFunctionApp

# Create Timer Trigger
func new --name NRGAnalysis --template "Timer trigger"

# Edit function_app.py to:
# 1. Import refactored POC code with Repository Pattern
# 2. Use StorageFactory to get repository instance
# 3. Pass repository to business logic functions

# Configure App Settings (via Azure Portal or CLI)
az functionapp config appsettings set \
  --name <YOUR_FUNCTION_APP_NAME> \
  --resource-group <YOUR_RG> \
  --settings \
    STORAGE_BACKEND=snowflake \
    SNOWFLAKE_ACCOUNT=xy12345.us-east-1 \
    SNOWFLAKE_USER=nrg_etl_user \
    SNOWFLAKE_PASSWORD=@Microsoft.KeyVault(...) \
    SNOWFLAKE_WAREHOUSE=NRG_ETL_WH \
    SNOWFLAKE_DATABASE=NRG_DATA \
    SNOWFLAKE_SCHEMA=LEGISLATIVE

# Deploy
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

### CRON Schedule Examples
- Daily at 8 AM UTC: `0 0 8 * * *`
- Every 6 hours: `0 0 */6 * * *`
- Weekdays at 9 AM: `0 0 9 * * 1-5`

---

## Additional Considerations

### Monitoring
- **Application Insights** (built-in for Functions): Track execution time, failures, LLM API costs
- **Azure Monitor**: Set alerts for failed runs or cost thresholds

### Security
- **Critical:** Store all secrets in **Azure Key Vault**:
  - API keys (Congress.gov, Open States, Gemini, OpenAI)
  - Data lake credentials (Snowflake password, BigQuery service account, etc.)
- Reference Key Vault secrets in App Settings: `@Microsoft.KeyVault(SecretUri=...)`
- Use **Managed Identity** for Azure-to-Azure authentication (Synapse, Blob Storage)
- **Never** hard-code credentials in code or config files
- Enable **HTTPS only** if adding HTTP triggers later
- Rotate data lake credentials regularly per customer security policy

### CI/CD
- **GitHub Actions** recommended for automated deployment
- Functions/WebJobs/Container Apps all support Git-based deployment

---

## Cost Estimates (Monthly)

**Scenario: Daily execution at 8 AM, 5-minute runtime**

| Component | Azure Functions | App Service WebJobs | Container Apps Jobs |
|-----------|----------------|---------------------|---------------------|
| Compute | $0.50 | $13.00 (B1) | $2.00 |
| Blob Storage (reports) | $2.00 | $2.00 | $2.00 |
| **Data Lake** (customer's) | **$50-200** | **$50-200** | **$50-200** |
| LLM APIs (external) | $15.00 | $15.00 | $15.00 |
| **Total** | **~$67-217** | **~$80-230** | **~$69-219** |

**Notes:**
- **Data lake costs** depend on customer's chosen platform (Snowflake/BigQuery/Synapse)
- Compute choice has minimal impact (~$12 difference between cheapest and most expensive)
- **LLM + Data Lake** costs dominate the budget
- If customer already has data lake, marginal cost may be minimal

---

## Sources

- [Azure Functions vs Container Apps vs App Service Comparison](https://learn.microsoft.com/en-us/azure/container-apps/compare-options)
- [Integration Services Comparison (Functions/WebJobs/Logic Apps)](https://learn.microsoft.com/en-us/azure/azure-functions/functions-compare-logic-apps-ms-flow-webjobs)
- [Azure Functions Pricing](https://azure.microsoft.com/en-us/pricing/details/functions/)
- [Azure App Service Pricing](https://azure.microsoft.com/en-us/pricing/details/app-service/windows/)
- [Azure Container Apps Pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/)
- [WebJobs on App Service Linux](https://learn.microsoft.com/en-us/azure/app-service/webjobs-create)
- [Python WebJobs Guide](https://techcommunity.microsoft.com/blog/appsonazureblog/getting-started-with-python-webjobs-on-app-service-linux/4399325)

---

## Final Recommendation

**Start with Azure Functions (Consumption Plan)** for the lowest cost and fastest time-to-value. Migrate to App Service WebJobs or Container Apps Jobs only if you encounter specific limitations (execution timeout, need for web dashboard, containerization requirements).
