# NRG Energy Legislative Intelligence POC
## Automated Legislative Monitoring and Analysis System

### Overview
This system monitors federal and state legislation affecting NRG Energy's oil/gas and power generation operations. It fetches bills from multiple APIs, analyzes them with LLMs (Gemini/GPT-5), tracks changes over time, and generates comprehensive reports.

### Architecture
- **Data Collection**: Congress.gov, Regulations.gov, Open States APIs
- **LLM Analysis**: Google Gemini (default) or OpenAI GPT-5
- **Change Tracking**: SQLite database with version-by-version monitoring
- **Output**: JSON, Markdown, and Word document reports

---

## Setup Instructions

### Step 1: System Requirements
- Python 3.9+
- Homebrew (macOS) or equivalent package manager
- Git

### Step 2: Install System Dependencies
```bash
# Install pandoc for Word document generation
brew install pandoc

# Verify installation
pandoc --version
```

### Step 3: Set Up Python Environment

#### Option A: Modern Setup with uv (Recommended)
```bash
# Install uv (fast Rust-based package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv

# Verify installation
uv --version
```

**Benefits:**
- 10-100x faster than pip
- Automatic environment management
- No manual venv activation needed
- Dependencies auto-installed on first run

#### Option B: Traditional Setup with venv
```bash
# Create virtual environment
python3 -m venv venv

# Activate and upgrade pip
source venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Step 4: Configure API Keys
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- **GOOGLE_API_KEY** (Primary LLM): https://makersuite.google.com/app/apikey
- **OPENAI_API_KEY** (Alternative LLM): https://platform.openai.com/api-keys
- **CONGRESS_API_KEY** (Federal bills): https://api.congress.gov/
- **OPENSTATES_API_KEY** (State bills): https://openstates.org/api
- **REGULATIONS_API_KEY** (Federal regulations): https://api.regulations.gov/
- **LEGISCAN_API_KEY** (Backup): https://legiscan.com/legiscan-api

### Step 5: Verify Configuration
```bash
# Check configuration files
ls -la config.yaml nrg_business_context.txt

# Test dependencies (uv)
uv run python -c "import httpx, openai, google.genai, yaml, sqlite3; print('All dependencies OK')"

# Or with traditional venv
source venv/bin/activate
python -c "import httpx, openai, google.genai, yaml, sqlite3; print('All dependencies OK')"
```

---

## Running the System

### Quick Start

#### With uv (Recommended)
```bash
# Run directly - handles environment automatically
uv run "poc 2.py"
```

#### With traditional venv
```bash
# Activate virtual environment
source venv/bin/activate

# Run legislative analysis
python "poc 2.py"
```

**Note:** Database initialization (`bill_cache.db`) happens automatically on first run.

### Configuration Options
Edit `config.yaml` to customize:
- LLM provider (gemini/openai)
- Data sources to enable/disable
- Number of bills to fetch
- Change tracking settings
- Output formats

### Output Files
Generated reports (timestamped):
- `nrg_analysis_YYYYMMDD_HHMMSS.json` - Structured data
- `nrg_analysis_YYYYMMDD_HHMMSS.md` - Human-readable report
- `nrg_analysis_YYYYMMDD_HHMMSS.docx` - Word document

---

## Database Schema

The SQLite database (`bill_cache.db`) contains:
- **bills** - Core bill metadata
- **bill_versions** - Version-by-version tracking
- **bill_changes** - Change detection logs
- **amendments** - Amendment tracking
- **version_analyses** - LLM analysis results
