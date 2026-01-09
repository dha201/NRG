
## Prerequisites Checklist

### 1. Install Required Tools

**Azure CLI:**
```bash
# macOS (you already have this)
brew update && brew install azure-cli

# Verify installation
az --version
```

**Azure Functions Core Tools v4:**
```bash
# macOS
brew tap azure/functions
brew install azure-functions-core-tools@4

# Verify installation
func --version  # Should show 4.x
```

**Python 3.9+:**
```bash
# Check your version
python3 --version  # You're using Python 3.9+ based on config
```

### 2. Azure Account Setup

**Login to Azure:**
```bash
az login
# Browser will open for authentication
```

**Set subscription (if you have multiple):**
```bash
# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "<SUBSCRIPTION_ID>"
```

---

## Step 1: Create Azure Resources (5 minutes)

### Variables (customize these):
```bash
# Set your variables
RESOURCE_GROUP="nrg-legislative-rg"
LOCATION="eastus"
STORAGE_ACCOUNT="nrglegstorage$(date +%s)"  # Unique name
FUNCTION_APP="nrg-legislative-prod"
```

### Create Resources:

**1. Resource Group:**
```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

**2. Storage Account:**
```bash
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS
```

**3. Function App (Consumption Plan - Free Tier):**
```bash
az functionapp create \
  --resource-group $RESOURCE_GROUP \
  --consumption-plan-location $LOCATION \
  --os-type Linux \
  --runtime python \
  --runtime-version 3.9 \
  --functions-version 4 \
  --name $FUNCTION_APP \
  --storage-account $STORAGE_ACCOUNT
```

**Expected output:** Your Function App URL will be `https://nrg-legislative-prod.azurewebsites.net`

---

## Step 2: Prepare Your Code for Azure Functions (15 minutes)

### Create Function App Structure

**1. Create new directory for Azure Function:**
```bash
cd /Users/thamac/Documents/NRG
mkdir azure-function
cd azure-function
```

**2. Initialize Function App:**
```bash
func init . --python --model V2
```

**3. Create Timer Trigger:**
```bash
func new --template "Timer trigger" --name NRGAnalysis
```

This creates:
```
azure-function/
├── function_app.py        # Main function entry point
├── requirements.txt       # Dependencies
├── host.json             # Function host config
└── local.settings.json   # Local environment variables
```

### 4. Copy Your POC Code

**Create a `nrg_core` package:**
```bash
mkdir nrg_core
touch nrg_core/__init__.py
```

**Copy POC file:**
```bash
cp ../poc\ 2.py nrg_core/legislative_tracker.py
```

### 5. Update `requirements.txt`

**Replace content with:**
```txt
# Core Azure Functions
azure-functions>=1.19.0

# Your existing dependencies
httpx>=0.27.0
openai>=1.14.0
google-genai>=0.3.0
python-dotenv>=1.0.0
PyYAML>=6.0.1
rich>=13.7.0
PyPDF2>=3.0.1
pdfplumber>=0.9.0
```

### 6. Update `function_app.py`

**Replace content with:**
```python
import azure.functions as func
import logging
import os
import sys
from datetime import datetime

# Add nrg_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nrg_core'))

app = func.FunctionApp()

@app.schedule(schedule="0 0 8 * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def NRGAnalysis(myTimer: func.TimerRequest) -> None:
    """
    Runs daily at 8:00 AM UTC
    CRON: 0 0 8 * * * (sec min hour day month dayOfWeek)
    """
    utc_timestamp = datetime.utcnow().isoformat()
    
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function started at %s', utc_timestamp)
    
    try:
        # Import and run your main analysis
        from nrg_core import legislative_tracker
        
        # Run main analysis
        logging.info("Starting legislative analysis...")
        legislative_tracker.main()
        
        logging.info("Legislative analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in legislative analysis: {str(e)}", exc_info=True)
        raise
```

**Why this works:**
- **Timer trigger** runs on schedule (daily at 8 AM UTC)
- **Imports your POC** as a module
- **Logs to App Insights** automatically
- **Error handling** with stack traces

### 7. Copy Configuration Files

**Copy config files:**
```bash
cp ../config.yaml nrg_core/
cp ../nrg_business_context.txt nrg_core/
cp ../.env .env.production  # Don't deploy this - see Step 3
```

### 8. Create `.funcignore` (exclude from deployment)

```bash
cat > .funcignore << 'EOF'
.git
.vscode
__pycache__
*.pyc
.env*
*.db
local.settings.json
test_*
EOF
```

---

## Step 3: Configure Environment Variables (Critical!)

### ⚠️ NEVER commit API keys to git!

**1. Create Key Vault for Secrets:**
```bash
KEY_VAULT_NAME="nrg-legislative-kv"

az keyvault create \
  --name $KEY_VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

**2. Add Secrets to Key Vault:**
```bash
# Read from your .env file and add each secret
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "CONGRESS-API-KEY" --value "<your_key>"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "OPENSTATES-API-KEY" --value "<your_key>"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "REGULATIONS-API-KEY" --value "<your_key>"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "OPENAI-API-KEY" --value "<your_key>"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "GOOGLE-API-KEY" --value "<your_key>"
```

**3. Grant Function App Access to Key Vault:**
```bash
# Enable system-assigned managed identity
az functionapp identity assign \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP

# Get the identity's principal ID
PRINCIPAL_ID=$(az functionapp identity show \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --query principalId -o tsv)

# Grant secret read access
az keyvault set-policy \
  --name $KEY_VAULT_NAME \
  --object-id $PRINCIPAL_ID \
  --secret-permissions get list
```

**4. Configure App Settings (Reference Key Vault):**
```bash
# Get Key Vault URI
VAULT_URI=$(az keyvault show --name $KEY_VAULT_NAME --query properties.vaultUri -o tsv)

# Set app settings to reference Key Vault
az functionapp config appsettings set \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --settings \
    CONGRESS_API_KEY="@Microsoft.KeyVault(SecretUri=${VAULT_URI}secrets/CONGRESS-API-KEY/)" \
    OPENSTATES_API_KEY="@Microsoft.KeyVault(SecretUri=${VAULT_URI}secrets/OPENSTATES-API-KEY/)" \
    REGULATIONS_API_KEY="@Microsoft.KeyVault(SecretUri=${VAULT_URI}secrets/REGULATIONS-API-KEY/)" \
    OPENAI_API_KEY="@Microsoft.KeyVault(SecretUri=${VAULT_URI}secrets/OPENAI-API-KEY/)" \
    GOOGLE_API_KEY="@Microsoft.KeyVault(SecretUri=${VAULT_URI}secrets/GOOGLE-API-KEY/)" \
    PYTHON_ISOLATE_WORKER_DEPENDENCIES=1 \
    GOOGLE_GENAI_DISABLE_NON_TEXT_WARNINGS="true"
```

---

## Step 4: Test Locally Before Deployment (10 minutes)

**1. Create `local.settings.json` for local testing:**
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "CONGRESS_API_KEY": "your_key_here",
    "OPENSTATES_API_KEY": "your_key_here",
    "REGULATIONS_API_KEY": "your_key_here",
    "OPENAI_API_KEY": "your_key_here",
    "GOOGLE_API_KEY": "your_key_here",
    "GOOGLE_GENAI_DISABLE_NON_TEXT_WARNINGS": "true"
  }
}
```

**2. Run locally:**
```bash
func start
```

**Expected output:**
```
Azure Functions Core Tools
Core Tools Version:       4.x
Function Runtime Version: 4.x

Functions:
  NRGAnalysis: timerTrigger

For detailed output, run func with --verbose flag.
[2026-01-09T15:30:00.000] Executing 'NRGAnalysis' (Reason='Timer fired at 2026-01-09T15:30:00...')
```

**3. Test the function manually (optional):**
```bash
# Trigger immediately without waiting for schedule
func start --verbose
# Press Ctrl+C after it runs once
```

---

## Step 5: Deploy to Azure (5 minutes)

### Deploy Your Function

**From the `azure-function` directory:**
```bash
func azure functionapp publish $FUNCTION_APP
```

**Expected output:**
```
Getting site publishing info...
Preparing archive...
Uploading content...
Upload completed successfully.
Syncing triggers...
Functions in nrg-legislative-prod:
    NRGAnalysis - [timerTrigger]
        Schedule: 0 0 8 * * *
```

**Deployment complete!** ✅

---

## Step 6: Verify Deployment (5 minutes)

### 1. Check Function Status

**View function details:**
```bash
az functionapp function show \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP \
  --function-name NRGAnalysis
```

### 2. Monitor Logs

**Stream live logs:**
```bash
func azure functionapp logstream $FUNCTION_APP
```

**Or via Azure CLI:**
```bash
az webapp log tail \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP
```

### 3. Trigger Manual Test

**Run function immediately (don't wait for schedule):**
```bash
az functionapp function keys list \
  --function-name NRGAnalysis \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP

# Get the master key and trigger via HTTP POST
curl -X POST "https://$FUNCTION_APP.azurewebsites.net/admin/functions/NRGAnalysis" \
  -H "x-functions-key: <master-key>"
```

### 4. View Execution History

**Azure Portal:**
1. Go to https://portal.azure.com
2. Navigate to your Function App: `nrg-legislative-prod`
3. Click **Functions** → **NRGAnalysis**
4. Click **Monitor** tab
5. View execution history, logs, and errors

---

## Step 7: Set Up Output Storage (10 minutes)

### Create Blob Container for Reports

**1. Create container:**
```bash
# Get storage account connection string
STORAGE_CONNECTION=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

# Create container
az storage container create \
  --name legislative-reports \
  --connection-string "$STORAGE_CONNECTION" \
  --public-access off
```

**2. Update function to write to Blob Storage:**

**Add to `requirements.txt`:**
```txt
azure-storage-blob>=12.19.0
```

**Add to your POC code (or create wrapper):**
```python
from azure.storage.blob import BlobServiceClient
import os

def upload_report_to_azure(local_file_path, blob_name):
    """Upload report to Azure Blob Storage"""
    connection_string = os.getenv("AzureWebJobsStorage")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client("legislative-reports")
    
    with open(local_file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/legislative-reports/{blob_name}"
```

---

## Step 8: Configure Alerts & Monitoring (5 minutes)

### Set Up Application Insights

**1. Enable Application Insights:**
```bash
# Create App Insights
az monitor app-insights component create \
  --app nrg-legislative-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --kind web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app nrg-legislative-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)

# Link to Function App
az functionapp config appsettings set \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

### 2. Create Alert Rules

**Alert on function failures:**
```bash
az monitor metrics alert create \
  --name "NRG Function Failures" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$FUNCTION_APP" \
  --condition "count FunctionExecutionCount where FunctionName = 'NRGAnalysis' and Succeeded = 'False' >= 2" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action <action-group-id>  # Create action group for email/SMS
```

---

## Troubleshooting

### Issue: Deployment fails with "No module named 'nrg_core'"

**Solution:** Ensure directory structure is correct:
```bash
azure-function/
├── nrg_core/
│   ├── __init__.py
│   ├── legislative_tracker.py
│   └── config.yaml
└── function_app.py
```

### Issue: Function times out after 10 minutes

**Solution:** Upgrade to Premium Plan:
```bash
az functionapp plan create \
  --name nrg-legislative-plan \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku EP1 \
  --is-linux

az functionapp update \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --plan nrg-legislative-plan
```

### Issue: Cannot import `pandoc` for DOCX generation

**Solution:** Pandoc not available in Function App runtime. Options:
1. **Skip DOCX in MVP** - Generate JSON/Markdown only
2. **Use container deployment** with custom Dockerfile
3. **Use Python-docx library** instead of pandoc

**Quick fix - disable DOCX:**
```yaml
# config.yaml
output_formats:
  docx_enabled: false
```

### Issue: SQLite database doesn't persist between runs

**Solution:** SQLite doesn't work with serverless. Options:
1. **Use Azure Files mount** (for MVP):
```bash
az webapp config storage-account add \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP \
  --custom-id bills-db \
  --storage-type AzureFiles \
  --account-name $STORAGE_ACCOUNT \
  --share-name bills-cache \
  --mount-path /mnt/bills
```

2. **Migrate to Azure SQL/PostgreSQL** (production)

---

## Quick Reference Commands

### View Logs
```bash
func azure functionapp logstream $FUNCTION_APP
```

### Redeploy After Code Changes
```bash
cd /Users/thamac/Documents/NRG/azure-function
func azure functionapp publish $FUNCTION_APP
```

### Check Function Status
```bash
az functionapp show \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --query state -o tsv
```

### Update Environment Variable
```bash
az functionapp config appsettings set \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --settings NEW_VAR="value"
```

### Delete Resources (cleanup)
```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

---

## Cost Estimate (MVP)

**Consumption Plan (Pay-per-execution):**
- Function executions: 30/month × $0.000002 = **$0.00**
- Execution time: 30 × 5 min × $0.000016/GB-s = **$0.50**
- Storage: 5GB × $0.06 = **$0.30**
- **Total Azure: ~$1/month** (under free tier limits)

**LLM API costs** (external): $15-50/month

---

## Next Steps After MVP

1. **Monitor for 1 week** - Check logs, execution times, errors
2. **Gather feedback** from evaluation team
3. **Implement Repository Pattern** - Swap SQLite for data lake
4. **Add email notifications** - Use Azure Logic Apps or SendGrid
5. **Set up CI/CD** - GitHub Actions for automated deployment

---

## Support & Resources

**Azure Portal:** https://portal.azure.com  
**Function Logs:** Azure Portal → Function App → Monitor  
**Documentation:** https://learn.microsoft.com/en-us/azure/azure-functions/  
**Pricing Calculator:** https://azure.microsoft.com/en-us/pricing/calculator/

---

## Summary Checklist

- [ ] Install Azure CLI + Functions Core Tools
- [ ] Create Azure resources (RG, Storage, Function App)
- [ ] Prepare code structure (`azure-function` directory)
- [ ] Copy POC code to `nrg_core/legislative_tracker.py`
- [ ] Update `function_app.py` with timer trigger
- [ ] Create Key Vault and add API keys
- [ ] Configure app settings with Key Vault references
- [ ] Test locally with `func start`
- [ ] Deploy with `func azure functionapp publish`
- [ ] Verify in Azure Portal
- [ ] Set up monitoring and alerts
- [ ] Document for evaluation team

**Deployment time: ~1-2 hours**  
**Result: Production-ready MVP running daily at 8 AM UTC**
