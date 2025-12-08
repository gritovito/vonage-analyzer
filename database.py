"""
Database operations for Call Analyzer
SQLite database with tables: documents, facts, faq, summary
"""
import sqlite3
import json
from datetime import datetime, date
from contextlib import contextmanager
from config import DATABASE_PATH


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize database with all required tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Documents table - stores all source documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                processed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Facts table - extracted facts from documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                key TEXT,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        # FAQ table - auto-generated FAQ entries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_documents TEXT,
                times_asked INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Summary table - daily statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                total_calls INTEGER DEFAULT 0,
                new_facts INTEGER DEFAULT 0,
                new_faq INTEGER DEFAULT 0,
                top_questions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faq_times_asked ON faq(times_asked DESC)")


# Document operations
def add_document(filename, doc_type, content=None, status="pending"):
    """Add a new document to the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO documents (filename, type, content, status)
                VALUES (?, ?, ?, ?)
            """, (filename, doc_type, content, status))
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Document already exists
            return None


def get_document(doc_id):
    """Get document by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return cursor.fetchone()


def get_document_by_filename(filename):
    """Get document by filename"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
        return cursor.fetchone()


def get_documents(doc_type=None, status=None, limit=100, offset=0):
    """Get list of documents with optional filtering"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if doc_type:
            query += " AND type = ?"
            params.append(doc_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def update_document_status(doc_id, status, content=None):
    """Update document status and optionally content"""
    with get_db() as conn:
        cursor = conn.cursor()
        if content:
            cursor.execute("""
                UPDATE documents
                SET status = ?, content = ?, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, content, doc_id))
        else:
            cursor.execute("""
                UPDATE documents
                SET status = ?, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, doc_id))


def get_documents_count(doc_type=None, status=None):
    """Get count of documents"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM documents WHERE 1=1"
        params = []

        if doc_type:
            query += " AND type = ?"
            params.append(doc_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        return cursor.fetchone()[0]


# Fact operations
def add_fact(document_id, category, key, value, confidence=1.0):
    """Add a new fact"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO facts (document_id, category, key, value, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (document_id, category, key, value, confidence))
        return cursor.lastrowid


def add_facts_bulk(document_id, facts_list):
    """Add multiple facts at once"""
    with get_db() as conn:
        cursor = conn.cursor()
        for fact in facts_list:
            cursor.execute("""
                INSERT INTO facts (document_id, category, key, value, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (document_id, fact['category'], fact.get('key', ''),
                  fact['value'], fact.get('confidence', 1.0)))


def get_facts(document_id=None, category=None, limit=100, offset=0):
    """Get facts with optional filtering"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT f.*, d.filename
            FROM facts f
            JOIN documents d ON f.document_id = d.id
            WHERE 1=1
        """
        params = []

        if document_id:
            query += " AND f.document_id = ?"
            params.append(document_id)
        if category:
            query += " AND f.category = ?"
            params.append(category)

        query += " ORDER BY f.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def get_facts_count(category=None):
    """Get count of facts"""
    with get_db() as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT COUNT(*) FROM facts WHERE category = ?", (category,))
        else:
            cursor.execute("SELECT COUNT(*) FROM facts")
        return cursor.fetchone()[0]


def get_facts_by_category():
    """Get fact counts grouped by category"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM facts
            GROUP BY category
            ORDER BY count DESC
        """)
        return cursor.fetchall()


# FAQ operations
def add_or_update_faq(question, answer, source_doc_id):
    """Add new FAQ or update existing one"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if similar question exists (simple matching)
        cursor.execute("""
            SELECT id, source_documents, times_asked
            FROM faq
            WHERE question = ?
        """, (question,))
        existing = cursor.fetchone()

        if existing:
            # Update existing FAQ
            sources = json.loads(existing['source_documents'] or '[]')
            if source_doc_id not in sources:
                sources.append(source_doc_id)
            cursor.execute("""
                UPDATE faq
                SET times_asked = times_asked + 1,
                    source_documents = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(sources), existing['id']))
            return existing['id']
        else:
            # Add new FAQ
            cursor.execute("""
                INSERT INTO faq (question, answer, source_documents)
                VALUES (?, ?, ?)
            """, (question, answer, json.dumps([source_doc_id])))
            return cursor.lastrowid


def get_faq(limit=50, offset=0, sort_by='times_asked'):
    """Get FAQ entries"""
    with get_db() as conn:
        cursor = conn.cursor()
        order = "times_asked DESC" if sort_by == 'times_asked' else "last_updated DESC"
        cursor.execute(f"""
            SELECT * FROM faq
            ORDER BY {order}
            LIMIT ? OFFSET ?
        """, (limit, offset))
        return cursor.fetchall()


def get_faq_count():
    """Get total FAQ count"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faq")
        return cursor.fetchone()[0]


def search_faq(query, limit=20):
    """Search FAQ by question text"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM faq
            WHERE question LIKE ? OR answer LIKE ?
            ORDER BY times_asked DESC
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        return cursor.fetchall()


# Summary operations
def update_daily_summary(calls=0, facts=0, faq_count=0):
    """Update or create daily summary"""
    today = date.today().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO summary (date, total_calls, new_facts, new_faq)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_calls = total_calls + ?,
                new_facts = new_facts + ?,
                new_faq = new_faq + ?
        """, (today, calls, facts, faq_count, calls, facts, faq_count))


def get_summary(days=30):
    """Get summary for last N days"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM summary
            ORDER BY date DESC
            LIMIT ?
        """, (days,))
        return cursor.fetchall()


def get_total_stats():
    """Get overall statistics"""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['total_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE type = 'transcription'")
        stats['total_transcriptions'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'processed'")
        stats['processed_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM facts")
        stats['total_facts'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM faq")
        stats['total_faq'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents
            WHERE date(created_at) = date('now')
        """)
        stats['today_documents'] = cursor.fetchone()[0]

        return stats


# Search operations
def search_all(query, limit=50):
    """Search across all tables"""
    results = {
        'documents': [],
        'facts': [],
        'faq': []
    }

    with get_db() as conn:
        cursor = conn.cursor()

        # Search documents
        cursor.execute("""
            SELECT id, filename, type, status, created_at
            FROM documents
            WHERE filename LIKE ? OR content LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        results['documents'] = cursor.fetchall()

        # Search facts
        cursor.execute("""
            SELECT f.*, d.filename
            FROM facts f
            JOIN documents d ON f.document_id = d.id
            WHERE f.key LIKE ? OR f.value LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        results['facts'] = cursor.fetchall()

        # Search FAQ
        cursor.execute("""
            SELECT * FROM faq
            WHERE question LIKE ? OR answer LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        results['faq'] = cursor.fetchall()

    return results


# Initialize database on import
init_db()
