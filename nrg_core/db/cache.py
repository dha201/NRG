import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Any, Optional


def compute_bill_hash(bill_text: str) -> str:
    if not bill_text:
        return ""
    return hashlib.sha256(bill_text.encode('utf-8')).hexdigest()


def init_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bills (
            bill_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            bill_number TEXT,
            title TEXT,
            text_hash TEXT,
            status TEXT,
            full_data_json TEXT,
            last_checked TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bill_changes (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id TEXT,
            change_type TEXT,
            change_detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            old_value TEXT,
            new_value TEXT,
            diff_summary TEXT,
            impact_analysis TEXT,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS amendments (
            amendment_id TEXT PRIMARY KEY,
            bill_id TEXT,
            amendment_number TEXT,
            amendment_text TEXT,
            introduced_date TEXT,
            status TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bill_versions (
            version_id TEXT PRIMARY KEY,
            bill_id TEXT NOT NULL,
            version_type TEXT,
            version_date TEXT,
            version_number INTEGER,
            pdf_url TEXT,
            full_text TEXT,
            text_hash TEXT,
            word_count INTEGER,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS version_analyses (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL,
            impact_score INTEGER,
            impact_type TEXT,
            impact_summary TEXT,
            affected_assets TEXT,
            recommended_action TEXT,
            stakeholders TEXT,
            full_analysis_json TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (version_id) REFERENCES bill_versions(version_id)
        )
    ''')

    # Indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_bill ON bill_versions(bill_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_type ON bill_versions(version_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_hash ON bill_versions(text_hash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_version ON version_analyses(version_id)')

    conn.commit()
    return conn


def get_cached_bill(bill_id: str, conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM bills WHERE bill_id = ?', (bill_id,))
    row = cursor.fetchone()

    if row:
        return {
            "bill_id": row[0],
            "source": row[1],
            "bill_number": row[2],
            "title": row[3],
            "text_hash": row[4],
            "status": row[5],
            "full_data_json": row[6],
            "last_checked": row[7],
            "created_at": row[8]
        }
    return None


def save_bill_to_cache(bill: dict[str, Any], conn: sqlite3.Connection) -> None:
    bill_id = f"{bill['source']}:{bill['number']}"
    bill_text = bill.get('summary', '')
    text_hash = compute_bill_hash(bill_text)

    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO bills
        (bill_id, source, bill_number, title, text_hash, status, full_data_json, last_checked)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        bill_id,
        bill['source'],
        bill['number'],
        bill['title'],
        text_hash,
        bill.get('status', 'Unknown'),
        json.dumps(bill),
        datetime.now().isoformat()
    ))

    # Save amendments if present
    if 'amendments' in bill and bill['amendments']:
        for amendment in bill['amendments']:
            amendment_id = f"{bill_id}:amendment:{amendment.get('amendment_id', len(bill['amendments']))}"
            cursor.execute('''
                INSERT OR IGNORE INTO amendments
                (amendment_id, bill_id, amendment_number, amendment_text, introduced_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                amendment_id,
                bill_id,
                amendment.get('amendment_number', 'Unknown'),
                json.dumps(amendment),
                amendment.get('date', 'Unknown'),
                amendment.get('status', 'Unknown')
            ))

    conn.commit()


def save_bill_version(bill_id: str, version: dict[str, Any], conn: sqlite3.Connection) -> str:
    version_id = f"{bill_id}:v{version['version_number']}:{version['version_type'].replace(' ', '_')}"

    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO bill_versions
        (version_id, bill_id, version_type, version_date, version_number, pdf_url, full_text, text_hash, word_count, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        version_id,
        bill_id,
        version['version_type'],
        version.get('version_date', ''),
        version['version_number'],
        version.get('pdf_url', ''),
        version.get('full_text', ''),
        version['text_hash'],
        version.get('word_count', 0),
        datetime.now().isoformat()
    ))

    conn.commit()
    return version_id


def save_version_analysis(version_id: str, analysis: dict[str, Any], conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO version_analyses
        (version_id, impact_score, impact_type, impact_summary, affected_assets, recommended_action, stakeholders, full_analysis_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        version_id,
        analysis.get('business_impact_score', 0),
        analysis.get('impact_type', 'unknown'),
        analysis.get('impact_summary', ''),
        json.dumps(analysis.get('affected_nrg_assets', {})),
        analysis.get('recommended_action', 'monitor'),
        json.dumps(analysis.get('internal_stakeholders', [])),
        json.dumps(analysis),
        datetime.now().isoformat()
    ))

    conn.commit()


def get_bill_versions(bill_id: str, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT version_id, version_type, version_date, version_number, pdf_url, full_text, text_hash, word_count
        FROM bill_versions
        WHERE bill_id = ?
        ORDER BY version_number
    ''', (bill_id,))

    rows = cursor.fetchall()
    versions: list[dict[str, Any]] = []
    for row in rows:
        versions.append({
            "version_id": row[0],
            "version_type": row[1],
            "version_date": row[2],
            "version_number": row[3],
            "pdf_url": row[4],
            "full_text": row[5],
            "text_hash": row[6],
            "word_count": row[7]
        })

    return versions


def get_version_analysis(version_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT full_analysis_json
        FROM version_analyses
        WHERE version_id = ?
    ''', (version_id,))

    row = cursor.fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return None


def get_cached_analysis_by_hash(text_hash: str, conn: sqlite3.Connection) -> Optional[dict]:
    """
    Look up cached LLM analysis by text hash.
    If the same text was analyzed before (regardless of bill/version_id), return that analysis.
    
    Args:
        text_hash: SHA-256 hash of version text
        conn: Database connection
        
    Returns:
        Analysis dict if found, None otherwise
    """
    if not text_hash or not conn:
        return None
    
    cursor = conn.cursor()
    cursor.execute('''
        SELECT va.full_analysis_json
        FROM bill_versions bv
        JOIN version_analyses va ON bv.version_id = va.version_id
        WHERE bv.text_hash = ?
        ORDER BY va.analyzed_at DESC
        LIMIT 1
    ''', (text_hash,))
    
    row = cursor.fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return None
