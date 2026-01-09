# Pluggable Storage Architecture for NRG App

## Problem Statement

Current implementation uses SQLite for bill caching/change tracking. Customer needs to store data in their own data lake:
- **Snowflake**
- **Google BigQuery** 
- **Azure Synapse Analytics**
- **AWS Redshift**
- Or any future storage system

**Goal:** Design architecture that allows swapping storage backends without changing business logic.

---

## Solution: Repository Pattern

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Azure Function                         │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Business Logic Layer                      │  │
│  │  (analyze_bills, track_changes, generate_reports)│  │
│  └────────────────┬─────────────────────────────────┘  │
│                   │                                      │
│                   │ Uses interface only                 │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐  │
│  │      IBillRepository (Abstract Interface)        │  │
│  │  • save_bill(bill_data)                          │  │
│  │  • get_bill(bill_id)                             │  │
│  │  • save_analysis(bill_id, analysis)              │  │
│  │  • get_previous_version(bill_id)                 │  │
│  │  • check_for_changes(bill_id, current_hash)      │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   │                                      │
│          ┌────────┴────────┬──────────────┬──────────┐ │
│          │                 │              │          │ │
│  ┌───────▼──────┐  ┌──────▼──────┐  ┌───▼────┐  ┌──▼─┐│
│  │SQLiteAdapter │  │ SnowflakeAdp│  │BigQueryA│  │Etc │││
│  └──────────────┘  └─────────────┘  └─────────┘  └────┘│
└─────────────────────────────────────────────────────────┘
                      │          │          │
                      ▼          ▼          ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ SQLite   │ │Snowflake │ │ BigQuery │
              │   File   │ │Data Lake │ │Data Lake │
              └──────────┘ └──────────┘ └──────────┘
```

### Key Principle: Dependency Inversion

Business logic depends on **abstraction** (interface), not concrete implementations. Storage adapters implement the interface.

**Sources:**
- [Repository Pattern - Cosmic Python](https://www.cosmicpython.com/book/chapter_02_repository.html)
- [Repository Pattern in Python - Pybites](https://pybit.es/articles/repository-pattern-in-python/)

---

## Implementation Design

### 1. Abstract Repository Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

class IBillRepository(ABC):
    """Abstract interface for bill storage operations.
    
    Business logic interacts ONLY with this interface.
    Concrete adapters implement backend-specific logic.
    """
    
    @abstractmethod
    def save_bill(self, bill_data: Dict[str, Any]) -> str:
        """Store bill metadata and text.
        
        Args:
            bill_data: Dict with keys:
                - bill_id: str (unique identifier)
                - source: str (Congress.gov, Open States, etc.)
                - bill_number: str
                - title: str
                - full_text: str
                - text_hash: str (for change detection)
                - last_updated: datetime
                - status: str
                
        Returns:
            bill_id: Stored bill identifier
        """
        pass
    
    @abstractmethod
    def get_bill(self, bill_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve bill by ID.
        
        Returns:
            Bill data dict or None if not found
        """
        pass
    
    @abstractmethod
    def save_bill_version(self, bill_id: str, version_data: Dict[str, Any]) -> str:
        """Store bill version (Introduced, Engrossed, Enrolled, etc.).
        
        Args:
            version_data: Dict with keys:
                - version_number: int
                - version_type: str (IH, EH, ENR, etc.)
                - version_date: datetime
                - full_text: str
                - text_hash: str
                - word_count: int
                
        Returns:
            version_id: Stored version identifier
        """
        pass
    
    @abstractmethod
    def get_bill_versions(self, bill_id: str) -> List[Dict[str, Any]]:
        """Retrieve all versions for a bill, ordered by version_number."""
        pass
    
    @abstractmethod
    def save_analysis(self, bill_id: str, version_id: Optional[str], 
                     analysis: Dict[str, Any]) -> str:
        """Store LLM analysis results.
        
        Args:
            bill_id: Bill identifier
            version_id: Specific version (None for overall bill analysis)
            analysis: LLM output (JSON structure)
            
        Returns:
            analysis_id: Stored analysis identifier
        """
        pass
    
    @abstractmethod
    def get_latest_analysis(self, bill_id: str, 
                           version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve most recent analysis for bill/version."""
        pass
    
    @abstractmethod
    def check_for_changes(self, bill_id: str, current_hash: str) -> Dict[str, Any]:
        """Check if bill has changed since last run.
        
        Returns:
            Dict with keys:
                - has_changes: bool
                - is_new: bool (first time seeing this bill)
                - previous_hash: str or None
                - last_checked: datetime or None
        """
        pass
    
    @abstractmethod
    def get_bills_for_monitoring(self, source: Optional[str] = None) -> List[str]:
        """Get list of bill IDs currently being tracked."""
        pass
```

---

### 2. SQLite Adapter (Current Implementation)

```python
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

class SQLiteBillRepository(IBillRepository):
    """SQLite implementation for local/dev environments."""
    
    def __init__(self, db_path: str = "bill_cache.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bills (
                bill_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                bill_number TEXT,
                title TEXT,
                full_text TEXT,
                text_hash TEXT,
                last_updated TIMESTAMP,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bill_versions (
                version_id TEXT PRIMARY KEY,
                bill_id TEXT NOT NULL,
                version_number INTEGER,
                version_type TEXT,
                version_date TIMESTAMP,
                full_text TEXT,
                text_hash TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                bill_id TEXT NOT NULL,
                version_id TEXT,
                analysis_json TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bill_id) REFERENCES bills(bill_id),
                FOREIGN KEY (version_id) REFERENCES bill_versions(version_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_bill(self, bill_data: Dict[str, Any]) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO bills 
            (bill_id, source, bill_number, title, full_text, text_hash, last_updated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bill_data['bill_id'],
            bill_data['source'],
            bill_data.get('bill_number'),
            bill_data.get('title'),
            bill_data.get('full_text'),
            bill_data['text_hash'],
            bill_data.get('last_updated', datetime.now()),
            bill_data.get('status')
        ))
        
        conn.commit()
        conn.close()
        return bill_data['bill_id']
    
    def get_bill(self, bill_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bills WHERE bill_id = ?', (bill_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def check_for_changes(self, bill_id: str, current_hash: str) -> Dict[str, Any]:
        existing = self.get_bill(bill_id)
        
        if not existing:
            return {
                'has_changes': True,
                'is_new': True,
                'previous_hash': None,
                'last_checked': None
            }
        
        return {
            'has_changes': existing['text_hash'] != current_hash,
            'is_new': False,
            'previous_hash': existing['text_hash'],
            'last_checked': existing['last_updated']
        }
    
    # Implement remaining methods...
```

---

### 3. Snowflake Adapter

```python
import snowflake.connector
from typing import Optional, List, Dict, Any
import json
import os

class SnowflakeBillRepository(IBillRepository):
    """Snowflake data lake implementation."""
    
    def __init__(self):
        # Connection details from environment variables
        self.conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'NRG_LEGISLATIVE')
        )
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Create tables in Snowflake if they don't exist."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS BILLS (
                BILL_ID VARCHAR PRIMARY KEY,
                SOURCE VARCHAR NOT NULL,
                BILL_NUMBER VARCHAR,
                TITLE TEXT,
                FULL_TEXT TEXT,
                TEXT_HASH VARCHAR,
                LAST_UPDATED TIMESTAMP_NTZ,
                STATUS VARCHAR,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS BILL_VERSIONS (
                VERSION_ID VARCHAR PRIMARY KEY,
                BILL_ID VARCHAR NOT NULL,
                VERSION_NUMBER INTEGER,
                VERSION_TYPE VARCHAR,
                VERSION_DATE TIMESTAMP_NTZ,
                FULL_TEXT TEXT,
                TEXT_HASH VARCHAR,
                WORD_COUNT INTEGER,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (BILL_ID) REFERENCES BILLS(BILL_ID)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ANALYSES (
                ANALYSIS_ID VARCHAR PRIMARY KEY,
                BILL_ID VARCHAR NOT NULL,
                VERSION_ID VARCHAR,
                ANALYSIS_JSON VARIANT,  -- Snowflake native JSON type
                ANALYZED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (BILL_ID) REFERENCES BILLS(BILL_ID),
                FOREIGN KEY (VERSION_ID) REFERENCES BILL_VERSIONS(VERSION_ID)
            )
        ''')
        
        self.conn.commit()
    
    def save_bill(self, bill_data: Dict[str, Any]) -> str:
        cursor = self.conn.cursor()
        
        cursor.execute('''
            MERGE INTO BILLS AS target
            USING (SELECT 
                %(bill_id)s AS BILL_ID,
                %(source)s AS SOURCE,
                %(bill_number)s AS BILL_NUMBER,
                %(title)s AS TITLE,
                %(full_text)s AS FULL_TEXT,
                %(text_hash)s AS TEXT_HASH,
                %(last_updated)s AS LAST_UPDATED,
                %(status)s AS STATUS
            ) AS source
            ON target.BILL_ID = source.BILL_ID
            WHEN MATCHED THEN UPDATE SET
                FULL_TEXT = source.FULL_TEXT,
                TEXT_HASH = source.TEXT_HASH,
                LAST_UPDATED = source.LAST_UPDATED,
                STATUS = source.STATUS
            WHEN NOT MATCHED THEN INSERT (
                BILL_ID, SOURCE, BILL_NUMBER, TITLE, FULL_TEXT, 
                TEXT_HASH, LAST_UPDATED, STATUS
            ) VALUES (
                source.BILL_ID, source.SOURCE, source.BILL_NUMBER, 
                source.TITLE, source.FULL_TEXT, source.TEXT_HASH, 
                source.LAST_UPDATED, source.STATUS
            )
        ''', bill_data)
        
        self.conn.commit()
        return bill_data['bill_id']
    
    def get_bill(self, bill_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM BILLS WHERE BILL_ID = %s',
            (bill_id,)
        )
        
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return None
    
    # Implement remaining methods...
```

---

### 4. BigQuery Adapter

```python
from google.cloud import bigquery
from typing import Optional, List, Dict, Any
import json
import os

class BigQueryBillRepository(IBillRepository):
    """Google BigQuery data lake implementation."""
    
    def __init__(self):
        # Initialize BigQuery client
        self.client = bigquery.Client(
            project=os.getenv('GCP_PROJECT_ID')
        )
        self.dataset_id = os.getenv('BIGQUERY_DATASET', 'nrg_legislative')
        self._ensure_dataset_and_tables()
    
    def _ensure_dataset_and_tables(self):
        """Create dataset and tables if they don't exist."""
        # Create dataset
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.client.create_dataset(dataset)
        
        # Create bills table
        bills_schema = [
            bigquery.SchemaField("bill_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("bill_number", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("full_text", "STRING"),
            bigquery.SchemaField("text_hash", "STRING"),
            bigquery.SchemaField("last_updated", "TIMESTAMP"),
            bigquery.SchemaField("status", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        table_ref = dataset_ref.table("bills")
        try:
            self.client.get_table(table_ref)
        except:
            table = bigquery.Table(table_ref, schema=bills_schema)
            self.client.create_table(table)
        
        # Create similar tables for versions and analyses...
    
    def save_bill(self, bill_data: Dict[str, Any]) -> str:
        table_id = f"{self.client.project}.{self.dataset_id}.bills"
        
        # BigQuery uses INSERT with deduplication
        query = f"""
            MERGE `{table_id}` T
            USING (SELECT 
                @bill_id AS bill_id,
                @source AS source,
                @bill_number AS bill_number,
                @title AS title,
                @full_text AS full_text,
                @text_hash AS text_hash,
                @last_updated AS last_updated,
                @status AS status
            ) S
            ON T.bill_id = S.bill_id
            WHEN MATCHED THEN UPDATE SET
                full_text = S.full_text,
                text_hash = S.text_hash,
                last_updated = S.last_updated,
                status = S.status
            WHEN NOT MATCHED THEN INSERT (
                bill_id, source, bill_number, title, full_text,
                text_hash, last_updated, status, created_at
            ) VALUES (
                S.bill_id, S.source, S.bill_number, S.title, S.full_text,
                S.text_hash, S.last_updated, S.status, CURRENT_TIMESTAMP()
            )
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("bill_id", "STRING", bill_data['bill_id']),
                bigquery.ScalarQueryParameter("source", "STRING", bill_data['source']),
                # Add remaining parameters...
            ]
        )
        
        query_job = self.client.query(query, job_config=job_config)
        query_job.result()
        return bill_data['bill_id']
    
    # Implement remaining methods...
```

---

## Configuration Management

### Environment Variable Strategy

**Azure Function App Settings** store connection details securely:

```bash
# SQLite (local dev)
STORAGE_BACKEND=sqlite
SQLITE_DB_PATH=/tmp/bill_cache.db

# Snowflake
STORAGE_BACKEND=snowflake
SNOWFLAKE_USER=nrg_etl_user
SNOWFLAKE_PASSWORD=<from_key_vault>
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_WAREHOUSE=NRG_ETL_WH
SNOWFLAKE_DATABASE=NRG_DATA
SNOWFLAKE_SCHEMA=LEGISLATIVE

# BigQuery
STORAGE_BACKEND=bigquery
GCP_PROJECT_ID=nrg-data-platform
BIGQUERY_DATASET=legislative_intelligence
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Azure Synapse
STORAGE_BACKEND=synapse
SYNAPSE_SERVER=nrg-synapse.database.windows.net
SYNAPSE_DATABASE=NRGDataWarehouse
SYNAPSE_USER=<managed_identity>
```

**Source:** [Azure Functions App Settings](https://learn.microsoft.com/en-us/azure/azure-functions/functions-app-settings)

---

## Factory Pattern for Adapter Selection

```python
# storage_factory.py
import os
from typing import Optional
from repositories.base import IBillRepository
from repositories.sqlite_adapter import SQLiteBillRepository
from repositories.snowflake_adapter import SnowflakeBillRepository
from repositories.bigquery_adapter import BigQueryBillRepository

class StorageFactory:
    """Factory for creating storage adapters based on configuration."""
    
    @staticmethod
    def create_repository() -> IBillRepository:
        """Create appropriate repository based on STORAGE_BACKEND env var.
        
        Returns:
            Concrete repository implementation
            
        Raises:
            ValueError: If STORAGE_BACKEND is invalid or required env vars missing
        """
        backend = os.getenv('STORAGE_BACKEND', 'sqlite').lower()
        
        if backend == 'sqlite':
            db_path = os.getenv('SQLITE_DB_PATH', 'bill_cache.db')
            return SQLiteBillRepository(db_path)
        
        elif backend == 'snowflake':
            # Validate required env vars
            required_vars = [
                'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 
                'SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_WAREHOUSE'
            ]
            missing = [v for v in required_vars if not os.getenv(v)]
            if missing:
                raise ValueError(f"Missing Snowflake config: {', '.join(missing)}")
            return SnowflakeBillRepository()
        
        elif backend == 'bigquery':
            if not os.getenv('GCP_PROJECT_ID'):
                raise ValueError("Missing GCP_PROJECT_ID for BigQuery")
            return BigQueryBillRepository()
        
        elif backend == 'synapse':
            # Azure Synapse adapter
            from repositories.synapse_adapter import SynapseRepository
            return SynapseRepository()
        
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
```

---

## Updated Azure Function Code

### Before (Tight Coupling to SQLite)

```python
import sqlite3

def main(timer: func.TimerRequest):
    # Direct SQLite usage throughout
    conn = sqlite3.connect('bill_cache.db')
    cursor = conn.cursor()
    
    # Fetch bills...
    bills = fetch_congress_bills()
    
    for bill in bills:
        # Check for changes
        cursor.execute('SELECT text_hash FROM bills WHERE bill_id = ?', (bill['id'],))
        row = cursor.fetchone()
        
        if not row or row[0] != bill['hash']:
            # Analyze with LLM
            analysis = analyze_with_gemini(bill)
            
            # Save to DB
            cursor.execute('INSERT INTO bills VALUES (?, ?, ?)', (...))
            cursor.execute('INSERT INTO analyses VALUES (?, ?)', (...))
    
    conn.commit()
    conn.close()
```

### After (Loose Coupling via Repository)

```python
from storage_factory import StorageFactory

def main(timer: func.TimerRequest):
    # Get repository (backend determined by env var)
    repo = StorageFactory.create_repository()
    
    # Fetch bills...
    bills = fetch_congress_bills()
    
    for bill in bills:
        # Check for changes (works with ANY backend)
        change_check = repo.check_for_changes(bill['id'], bill['hash'])
        
        if change_check['has_changes']:
            # Save bill
            repo.save_bill(bill)
            
            # Analyze with LLM
            analysis = analyze_with_gemini(bill)
            
            # Save analysis (works with ANY backend)
            repo.save_analysis(bill['id'], None, analysis)
```

**No business logic changes needed when switching storage!**

---

## Migration Path

### Phase 1: Refactor Current Code (1-2 days)
1. Extract SQLite logic into `SQLiteBillRepository` class
2. Define `IBillRepository` abstract interface
3. Create `StorageFactory`
4. Update `poc 2.py` to use repository pattern
5. Test locally with SQLite (should behave identically)

### Phase 2: Implement Target Adapter (1-2 days)
1. Identify customer's data lake (Snowflake, BigQuery, Synapse, etc.)
2. Implement concrete adapter (e.g., `SnowflakeBillRepository`)
3. Add integration tests against customer's data lake
4. Update Azure Function app settings with connection strings

### Phase 3: Deploy (1 day)
1. Deploy updated Function code to Azure
2. Set `STORAGE_BACKEND=snowflake` in App Settings
3. Verify data flows to customer's data lake
4. Monitor first few executions

### Phase 4: Add More Adapters (ongoing)
- Easy to add new backends as requirements evolve
- Each adapter is independent (no impact on existing code)

---

## Testing Strategy

### Unit Tests (Fast)
```python
# Use fake in-memory repository for testing business logic
class FakeRepository(IBillRepository):
    def __init__(self):
        self.bills = {}
        self.versions = {}
        self.analyses = {}
    
    def save_bill(self, bill_data):
        self.bills[bill_data['bill_id']] = bill_data
        return bill_data['bill_id']
    
    def get_bill(self, bill_id):
        return self.bills.get(bill_id)
    
    # Implement remaining methods...

# Test business logic without real database
def test_change_detection():
    repo = FakeRepository()
    
    # First run - new bill
    change_check = repo.check_for_changes('HR123', 'hash1')
    assert change_check['is_new'] == True
    
    # Save bill
    repo.save_bill({'bill_id': 'HR123', 'text_hash': 'hash1', ...})
    
    # Second run - no changes
    change_check = repo.check_for_changes('HR123', 'hash1')
    assert change_check['has_changes'] == False
    
    # Third run - bill changed
    change_check = repo.check_for_changes('HR123', 'hash2')
    assert change_check['has_changes'] == True
```

### Integration Tests (Real backends)
```python
import pytest

@pytest.mark.integration
def test_snowflake_adapter():
    repo = SnowflakeBillRepository()
    
    bill_data = {
        'bill_id': 'TEST_HR999',
        'source': 'Congress.gov',
        'text_hash': 'test_hash_123',
        # ...
    }
    
    # Test save
    bill_id = repo.save_bill(bill_data)
    assert bill_id == 'TEST_HR999'
    
    # Test retrieve
    retrieved = repo.get_bill('TEST_HR999')
    assert retrieved['text_hash'] == 'test_hash_123'
    
    # Cleanup
    # ...
```

---

## Security Best Practices

### 1. Never Hard-Code Credentials
```python
# ❌ BAD
snowflake_conn = snowflake.connector.connect(
    user='admin',
    password='P@ssw0rd123',  # NEVER DO THIS
    account='xy12345'
)

# ✅ GOOD
snowflake_conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT')
)
```

### 2. Use Azure Key Vault for Secrets
```bash
# Reference Key Vault secrets in Function App Settings
SNOWFLAKE_PASSWORD=@Microsoft.KeyVault(SecretUri=https://nrg-vault.vault.azure.net/secrets/snowflake-pwd/)
```

### 3. Managed Identity (Azure Synapse)
```python
# No credentials needed - uses Function's Managed Identity
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
# Synapse connection uses credential automatically
```

### 4. Least Privilege Access
- Grant Function only necessary permissions (INSERT, SELECT on specific tables)
- Create dedicated service account for each environment (dev, staging, prod)
- Rotate credentials regularly

---

## Cost Considerations

### SQLite
- **Cost:** $0 (local file)
- **Pros:** Free, simple, fast for dev/test
- **Cons:** Not suitable for production multi-instance scenarios

### Snowflake
- **Cost:** ~$2/hour compute + $23/TB/month storage
- **Pros:** Best for analytics, handles large volumes, separation of compute/storage
- **Estimate:** $50-200/month depending on usage

### BigQuery
- **Cost:** $5/TB queried + $0.02/GB storage
- **Pros:** Serverless, pay-per-query, integrates with Google ecosystem
- **Estimate:** $20-100/month for typical usage

### Azure Synapse
- **Cost:** Similar to Snowflake pricing model
- **Pros:** Native Azure integration, unified analytics
- **Estimate:** $50-200/month

**Recommendation:** For NRG's use case (small data volumes, infrequent queries), any option is cost-effective. Choose based on existing infrastructure.

---

## Dependencies

### New Python Packages Required

```txt
# Add to requirements.txt based on chosen backend

# Snowflake
snowflake-connector-python>=3.0.0

# BigQuery
google-cloud-bigquery>=3.10.0
google-auth>=2.17.0

# Azure Synapse
pyodbc>=4.0.39
azure-identity>=1.12.0

# AWS Redshift (if needed)
psycopg2-binary>=2.9.5
```

---

## Rollback Strategy

If issues arise with new backend:

1. **Immediate:** Change `STORAGE_BACKEND=sqlite` in App Settings (reverts to local storage)
2. **Data migration:** Copy data from data lake back to SQLite if needed
3. **Code rollback:** Deploy previous version via Azure Functions deployment slots

---

## Summary

### What We Get
✅ **Flexibility:** Swap storage backends by changing 1 env var  
✅ **Testability:** Mock repository for unit tests  
✅ **Maintainability:** Storage logic isolated from business logic  
✅ **Scalability:** Easy to add new adapters (Redshift, Databricks, etc.)  
✅ **Customer control:** Data stays in their infrastructure  

### What It Costs
⚠️ **Initial effort:** 3-5 days to refactor and implement first adapter  
⚠️ **Complexity:** Additional abstraction layer  
⚠️ **Learning curve:** Team needs to understand pattern  

### Decision
For multi-customer SaaS or enterprise deployments where data sovereignty matters, **Repository pattern is essential**. One-time investment pays off with each new customer/backend.

---

## Next Steps

1. **Review with stakeholders:** Confirm target data lake (Snowflake? BigQuery?)
2. **Spike:** 1-day proof-of-concept implementing chosen adapter
3. **Refactor:** Extract current SQLite code into repository pattern
4. **Test:** Validate with fake repository and integration tests
5. **Deploy:** Update Azure Function with new architecture
6. **Monitor:** Track first week of production usage

---

## References

- **Repository Pattern:** https://www.cosmicpython.com/book/chapter_02_repository.html
- **Python Repository Example:** https://pybit.es/articles/repository-pattern-in-python/
- **Azure Functions Config:** https://learn.microsoft.com/en-us/azure/azure-functions/functions-app-settings
- **Snowflake Python Connector:** https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example
- **BigQuery Python Client:** https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries
