"""
Knowledge Hub Database - Clustered Q&A System
SQLite database with semantic clustering and answer effectiveness tracking
"""
import sqlite3
import os
import logging
import struct
from datetime import datetime, date
from contextlib import contextmanager
from config import DATABASE_PATH, DATA_DIR

logger = logging.getLogger(__name__)


def ensure_data_dir():
    """Ensure data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, mode=0o755, exist_ok=True)


@contextmanager
def get_db():
    """Context manager for database connections"""
    ensure_data_dir()
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def serialize_embedding(embedding):
    """Serialize embedding list to bytes"""
    if embedding is None:
        return None
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob):
    """Deserialize bytes to embedding list"""
    if blob is None:
        return None
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


def init_db():
    """Initialize database with Knowledge Hub schema"""
    ensure_data_dir()

    with get_db() as conn:
        cursor = conn.cursor()

        # Clusters table - categories for questions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                icon TEXT,
                color TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Questions table - canonical questions with embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER,
                canonical_text TEXT NOT NULL,
                embedding BLOB,
                status TEXT DEFAULT 'no_answer',
                best_answer_id INTEGER,
                times_asked INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES clusters(id)
            )
        """)

        # Question variants - different phrasings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                variant_text TEXT NOT NULL,
                source_document_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (source_document_id) REFERENCES documents(id)
            )
        """)

        # Answers table - with effectiveness tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                answer_text TEXT NOT NULL,
                source_document_id INTEGER,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                effectiveness_percent REAL DEFAULT 0.0,
                is_best BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (source_document_id) REFERENCES documents(id)
            )
        """)

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                content TEXT,
                processed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_calls INTEGER DEFAULT 0,
                new_questions INTEGER DEFAULT 0,
                new_answers INTEGER DEFAULT 0,
                resolved_count INTEGER DEFAULT 0,
                unresolved_count INTEGER DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_cluster ON questions(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_times ON questions(times_asked DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_effectiveness ON answers(effectiveness_percent DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_question ON question_variants(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

        # Insert default clusters
        default_clusters = [
            ("Messaging", "SMS, MMS, messages delayed, not sending", "💬", "#9b59b6"),
            ("Calls & Voice", "Call quality, can't make calls, voicemail", "📞", "#3498db"),
            ("Data & Internet", "Slow data, no internet, WiFi issues", "🌐", "#2ecc71"),
            ("Device Issues", "Screen, battery, charging, buttons", "📱", "#e74c3c"),
            ("Apps & Software", "App crashes, updates, settings", "💻", "#f39c12"),
            ("Account & Billing", "Payments, plans, account issues", "💳", "#1abc9c"),
            ("Store & Service", "Pickup, repair, warranty", "🏪", "#e67e22"),
            ("General Inquiry", "Other questions", "❓", "#95a5a6"),
        ]

        for name, desc, icon, color in default_clusters:
            cursor.execute("""
                INSERT OR IGNORE INTO clusters (name, description, icon, color)
                VALUES (?, ?, ?, ?)
            """, (name, desc, icon, color))

        logger.info("Knowledge Hub database initialized")


# ==================== CLUSTER OPERATIONS ====================

def get_clusters():
    """Get all clusters with stats"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*,
                   COUNT(q.id) as question_count,
                   SUM(CASE WHEN q.status = 'resolved' THEN 1 ELSE 0 END) as resolved_count
            FROM clusters c
            LEFT JOIN questions q ON c.id = q.cluster_id
            GROUP BY c.id
            ORDER BY question_count DESC
        """)
        return cursor.fetchall()


def get_cluster(cluster_id):
    """Get cluster by ID with stats"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*,
                   COUNT(q.id) as question_count,
                   SUM(CASE WHEN q.status = 'resolved' THEN 1 ELSE 0 END) as resolved_count
            FROM clusters c
            LEFT JOIN questions q ON c.id = q.cluster_id
            WHERE c.id = ?
            GROUP BY c.id
        """, (cluster_id,))
        return cursor.fetchone()


def get_cluster_by_name(name):
    """Get cluster by name"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clusters WHERE name = ?", (name,))
        return cursor.fetchone()


# ==================== QUESTION OPERATIONS ====================

def add_question(cluster_id, canonical_text, embedding=None):
    """Add a new question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO questions (cluster_id, canonical_text, embedding, status)
            VALUES (?, ?, ?, 'no_answer')
        """, (cluster_id, canonical_text, serialize_embedding(embedding)))
        return cursor.lastrowid


def get_question(question_id):
    """Get question with full details"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   (SELECT COUNT(*) FROM question_variants WHERE question_id = q.id) as variant_count,
                   (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            WHERE q.id = ?
        """, (question_id,))
        return cursor.fetchone()


def get_questions(cluster_id=None, status=None, limit=100, offset=0, sort_by='times_asked'):
    """Get questions with optional filters"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   (SELECT COUNT(*) FROM question_variants WHERE question_id = q.id) as variant_count,
                   (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            WHERE 1=1
        """
        params = []

        if cluster_id:
            query += " AND q.cluster_id = ?"
            params.append(cluster_id)

        if status:
            query += " AND q.status = ?"
            params.append(status)

        if sort_by == 'times_asked':
            query += " ORDER BY q.times_asked DESC"
        elif sort_by == 'status':
            query += " ORDER BY CASE q.status WHEN 'needs_work' THEN 1 WHEN 'no_answer' THEN 2 ELSE 3 END, q.times_asked DESC"
        else:
            query += " ORDER BY q.updated_at DESC"

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def get_questions_count(cluster_id=None, status=None):
    """Get count of questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM questions WHERE 1=1"
        params = []

        if cluster_id:
            query += " AND cluster_id = ?"
            params.append(cluster_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        return cursor.fetchone()[0]


def get_all_questions_with_embeddings():
    """Get all questions with embeddings for similarity search"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, canonical_text, embedding, cluster_id FROM questions WHERE embedding IS NOT NULL")
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'canonical_text': row['canonical_text'],
                'embedding': deserialize_embedding(row['embedding']),
                'cluster_id': row['cluster_id']
            })
        return results


def update_question_embedding(question_id, embedding):
    """Update question's embedding"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE questions SET embedding = ? WHERE id = ?",
                      (serialize_embedding(embedding), question_id))


def increment_question_asked(question_id):
    """Increment times_asked counter"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE questions
            SET times_asked = times_asked + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (question_id,))


def update_question_status(question_id, status):
    """Update question status (resolved/needs_work/no_answer)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE questions SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (status, question_id))


def update_question_best_answer(question_id, answer_id):
    """Set the best answer for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE questions SET best_answer_id = ? WHERE id = ?",
                      (answer_id, question_id))


def get_needs_work_questions(limit=100):
    """Get questions that need work"""
    return get_questions(status='needs_work', limit=limit, sort_by='times_asked')


def get_top_questions(limit=10):
    """Get most frequently asked questions"""
    return get_questions(limit=limit, sort_by='times_asked')


# ==================== QUESTION VARIANTS ====================

def add_question_variant(question_id, variant_text, source_document_id=None):
    """Add a variant phrasing"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO question_variants (question_id, variant_text, source_document_id)
            VALUES (?, ?, ?)
        """, (question_id, variant_text, source_document_id))
        return cursor.lastrowid


def get_question_variants(question_id):
    """Get all variants for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT qv.*, d.filename as source_filename
            FROM question_variants qv
            LEFT JOIN documents d ON qv.source_document_id = d.id
            WHERE qv.question_id = ?
            ORDER BY qv.created_at DESC
        """, (question_id,))
        return cursor.fetchall()


# ==================== ANSWER OPERATIONS ====================

def add_answer(question_id, answer_text, source_document_id=None, resolution_status='unknown', customer_satisfaction='neutral'):
    """Add answer and update effectiveness"""
    # Determine success/fail based on resolution and satisfaction
    resolved = resolution_status == 'resolved'
    satisfied = customer_satisfaction in ('positive', 'neutral')
    success = 1 if resolved else 0
    fail = 1 if not resolved else 0

    with get_db() as conn:
        cursor = conn.cursor()

        # Check if similar answer exists
        cursor.execute("""
            SELECT id, success_count, fail_count FROM answers
            WHERE question_id = ? AND answer_text = ?
        """, (question_id, answer_text))
        existing = cursor.fetchone()

        if existing:
            # Update existing answer
            cursor.execute("""
                UPDATE answers
                SET success_count = success_count + ?,
                    fail_count = fail_count + ?
                WHERE id = ?
            """, (success, fail, existing['id']))
            answer_id = existing['id']
        else:
            # Create new answer
            cursor.execute("""
                INSERT INTO answers (question_id, answer_text, source_document_id, success_count, fail_count)
                VALUES (?, ?, ?, ?, ?)
            """, (question_id, answer_text, source_document_id, success, fail))
            answer_id = cursor.lastrowid

        # Update effectiveness
        _update_answer_effectiveness(cursor, answer_id)

        # Update best answer for question
        _update_best_answer(cursor, question_id)

        return answer_id


def _update_answer_effectiveness(cursor, answer_id):
    """Calculate effectiveness percentage"""
    cursor.execute("SELECT success_count, fail_count FROM answers WHERE id = ?", (answer_id,))
    row = cursor.fetchone()
    if row:
        total = row['success_count'] + row['fail_count']
        if total > 0:
            effectiveness = (row['success_count'] / total) * 100
        else:
            effectiveness = 50.0
        cursor.execute("UPDATE answers SET effectiveness_percent = ? WHERE id = ?",
                      (effectiveness, answer_id))


def _update_best_answer(cursor, question_id):
    """Update best answer and question status"""
    # Reset all is_best
    cursor.execute("UPDATE answers SET is_best = 0 WHERE question_id = ?", (question_id,))

    # Find best answer
    cursor.execute("""
        SELECT id, effectiveness_percent FROM answers
        WHERE question_id = ?
        ORDER BY effectiveness_percent DESC, success_count DESC
        LIMIT 1
    """, (question_id,))
    best = cursor.fetchone()

    if best:
        cursor.execute("UPDATE answers SET is_best = 1 WHERE id = ?", (best['id'],))
        cursor.execute("UPDATE questions SET best_answer_id = ? WHERE id = ?",
                      (best['id'], question_id))

        # Update question status based on effectiveness
        if best['effectiveness_percent'] >= 70:
            status = 'resolved'
        elif best['effectiveness_percent'] > 0:
            status = 'needs_work'
        else:
            status = 'no_answer'

        cursor.execute("UPDATE questions SET status = ? WHERE id = ?", (status, question_id))
    else:
        cursor.execute("UPDATE questions SET status = 'no_answer', best_answer_id = NULL WHERE id = ?",
                      (question_id,))


def get_answers(question_id):
    """Get all answers for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, d.filename as source_filename
            FROM answers a
            LEFT JOIN documents d ON a.source_document_id = d.id
            WHERE a.question_id = ?
            ORDER BY a.is_best DESC, a.effectiveness_percent DESC
        """, (question_id,))
        return cursor.fetchall()


def get_best_answer(question_id):
    """Get the best answer for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, d.filename as source_filename
            FROM answers a
            LEFT JOIN documents d ON a.source_document_id = d.id
            WHERE a.question_id = ? AND a.is_best = 1
        """, (question_id,))
        return cursor.fetchone()


def update_answer_feedback(answer_id, helpful):
    """Update answer based on user feedback"""
    with get_db() as conn:
        cursor = conn.cursor()

        if helpful:
            cursor.execute("UPDATE answers SET success_count = success_count + 1 WHERE id = ?", (answer_id,))
        else:
            cursor.execute("UPDATE answers SET fail_count = fail_count + 1 WHERE id = ?", (answer_id,))

        _update_answer_effectiveness(cursor, answer_id)

        # Get question_id and update best answer
        cursor.execute("SELECT question_id FROM answers WHERE id = ?", (answer_id,))
        row = cursor.fetchone()
        if row:
            _update_best_answer(cursor, row['question_id'])


# ==================== DOCUMENT OPERATIONS ====================

def add_document(filename, content=None, status="pending"):
    """Add a new document"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO documents (filename, content, status)
                VALUES (?, ?, ?)
            """, (filename, content, status))
            return cursor.lastrowid
        except sqlite3.IntegrityError:
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


def get_documents(status=None, limit=100, offset=0):
    """Get documents with optional status filter"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT id, filename, status, processed_at, created_at FROM documents WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def get_pending_documents(limit=50):
    """Get pending documents"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM documents WHERE status = 'pending'
            ORDER BY created_at ASC LIMIT ?
        """, (limit,))
        return cursor.fetchall()


def update_document_status(doc_id, status, error_message=None, analysis_result=None):
    """Update document status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET status = ?, processed_at = CURRENT_TIMESTAMP, error_message = ?, analysis_result = ?
            WHERE id = ?
        """, (status, error_message, analysis_result, doc_id))


def get_documents_count(status=None):
    """Get document count"""
    with get_db() as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]


# ==================== STATISTICS ====================

def get_stats():
    """Get overall statistics"""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM clusters")
        stats['total_clusters'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions")
        stats['total_questions'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM answers")
        stats['total_answers'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions WHERE status = 'resolved'")
        stats['resolved_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions WHERE status = 'needs_work'")
        stats['needs_work_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions WHERE status = 'no_answer'")
        stats['no_answer_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['total_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'processed'")
        stats['processed_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'")
        stats['pending_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM question_variants")
        stats['total_variants'] = cursor.fetchone()[0]

        return stats


# ==================== SEARCH ====================

def search_questions_text(query, limit=20):
    """Text search in questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*, c.name as cluster_name, c.icon as cluster_icon
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            WHERE q.canonical_text LIKE ?
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (f'%{query}%', limit))
        return cursor.fetchall()


# ==================== DAILY SUMMARY ====================

def update_daily_summary(calls=0, questions=0, answers=0, resolved=0, unresolved=0):
    """Update daily summary counters"""
    today = date.today().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO daily_summary (date, total_calls, new_questions, new_answers, resolved_count, unresolved_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_calls = total_calls + excluded.total_calls,
                new_questions = new_questions + excluded.new_questions,
                new_answers = new_answers + excluded.new_answers,
                resolved_count = resolved_count + excluded.resolved_count,
                unresolved_count = unresolved_count + excluded.unresolved_count
        """, (today, calls, questions, answers, resolved, unresolved))


def get_summary(days=30):
    """Get daily summary for last N days"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM daily_summary
            ORDER BY date DESC
            LIMIT ?
        """, (days,))
        return cursor.fetchall()


def get_questions_by_cluster():
    """Get question count by cluster"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.name, c.icon, c.color, COUNT(q.id) as count
            FROM clusters c
            LEFT JOIN questions q ON c.id = q.cluster_id
            GROUP BY c.id
            ORDER BY count DESC
        """)
        return cursor.fetchall()


# Initialize on import
init_db()
