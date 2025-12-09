"""
Database operations for Call Analyzer v2.0
SQLite database with semantic matching support using embeddings
"""
import sqlite3
import json
import os
import logging
import struct
from datetime import datetime, date
from contextlib import contextmanager
from config import DATABASE_PATH, DATA_DIR

# Setup logging
logger = logging.getLogger(__name__)


def ensure_data_dir():
    """Ensure data directory exists with proper permissions"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, mode=0o755, exist_ok=True)
        logger.info(f"Created data directory: {DATA_DIR}")


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
    """Serialize embedding list to bytes for SQLite BLOB storage"""
    if embedding is None:
        return None
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob):
    """Deserialize bytes back to embedding list"""
    if blob is None:
        return None
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


def init_db():
    """Initialize database with all required tables for v2.0"""
    ensure_data_dir()

    with get_db() as conn:
        cursor = conn.cursor()

        # Topics table - categories for organizing questions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                parent_id INTEGER,
                icon TEXT,
                color TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES topics(id)
            )
        """)

        # Questions table - canonical questions with embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                canonical_text TEXT NOT NULL,
                embedding BLOB,
                times_asked INTEGER DEFAULT 1,
                last_asked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )
        """)

        # Question variants - different phrasings of the same question
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

        # Answers table - solutions with effectiveness ratings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                answer_text TEXT NOT NULL,
                source_document_id INTEGER,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                is_recommended BOOLEAN DEFAULT 0,
                resolution_status TEXT,
                customer_satisfaction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (source_document_id) REFERENCES documents(id)
            )
        """)

        # Documents table - stores all source documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                processed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Knowledge base table - for onboarding materials
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                source_document_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_document_id) REFERENCES documents(id)
            )
        """)

        # Summary table - daily statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                total_calls INTEGER DEFAULT 0,
                new_questions INTEGER DEFAULT 0,
                new_answers INTEGER DEFAULT 0,
                resolved_count INTEGER DEFAULT 0,
                unresolved_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_times_asked ON questions(times_asked DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_effectiveness ON answers(effectiveness_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_question ON question_variants(question_id)")

        # Insert default topics if not exist
        default_topics = [
            ("Device Issues", "Hardware problems: screen, battery, buttons, physical damage", None, "📱", "#e74c3c"),
            ("Software & Apps", "Software updates, settings, applications, OS issues", None, "💻", "#3498db"),
            ("Account & Billing", "Payments, subscriptions, account management", None, "💳", "#2ecc71"),
            ("Connectivity", "Network, WiFi, Bluetooth, calls, data issues", None, "📡", "#9b59b6"),
            ("Store & Pickup", "Orders, delivery, store pickup, returns", None, "🏪", "#f39c12"),
            ("General Inquiry", "Other questions and general information", None, "❓", "#95a5a6"),
        ]

        for name, desc, parent, icon, color in default_topics:
            cursor.execute("""
                INSERT OR IGNORE INTO topics (name, description, parent_id, icon, color)
                VALUES (?, ?, ?, ?, ?)
            """, (name, desc, parent, icon, color))

        logger.info("Database v2.0 initialized successfully")


# ==================== TOPIC OPERATIONS ====================

def get_topics():
    """Get all topics"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topics ORDER BY name")
        return cursor.fetchall()


def get_topic(topic_id):
    """Get topic by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
        return cursor.fetchone()


def get_topic_by_name(name):
    """Get topic by name"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topics WHERE name = ?", (name,))
        return cursor.fetchone()


# ==================== QUESTION OPERATIONS ====================

def add_question(topic_id, canonical_text, embedding=None):
    """Add a new canonical question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO questions (topic_id, canonical_text, embedding)
            VALUES (?, ?, ?)
        """, (topic_id, canonical_text, serialize_embedding(embedding)))
        return cursor.lastrowid


def get_question(question_id):
    """Get question by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*, t.name as topic_name, t.icon as topic_icon, t.color as topic_color
            FROM questions q
            LEFT JOIN topics t ON q.topic_id = t.id
            WHERE q.id = ?
        """, (question_id,))
        return cursor.fetchone()


def get_questions(topic_id=None, limit=100, offset=0, sort_by='times_asked'):
    """Get questions with optional topic filter"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT q.*, t.name as topic_name, t.icon as topic_icon, t.color as topic_color,
                   (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
            FROM questions q
            LEFT JOIN topics t ON q.topic_id = t.id
            WHERE 1=1
        """
        params = []

        if topic_id:
            query += " AND q.topic_id = ?"
            params.append(topic_id)

        if sort_by == 'times_asked':
            query += " ORDER BY q.times_asked DESC"
        elif sort_by == 'recent':
            query += " ORDER BY q.last_asked DESC"
        else:
            query += " ORDER BY q.created_at DESC"

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def get_questions_count(topic_id=None):
    """Get count of questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        if topic_id:
            cursor.execute("SELECT COUNT(*) FROM questions WHERE topic_id = ?", (topic_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM questions")
        return cursor.fetchone()[0]


def get_all_questions_with_embeddings():
    """Get all questions with their embeddings for similarity search"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, canonical_text, embedding FROM questions WHERE embedding IS NOT NULL")
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'canonical_text': row['canonical_text'],
                'embedding': deserialize_embedding(row['embedding'])
            })
        return results


def update_question_embedding(question_id, embedding):
    """Update question's embedding"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE questions SET embedding = ? WHERE id = ?
        """, (serialize_embedding(embedding), question_id))


def increment_question_asked(question_id):
    """Increment times_asked counter for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE questions
            SET times_asked = times_asked + 1, last_asked = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (question_id,))


def get_questions_by_topic():
    """Get question counts grouped by topic"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.id, t.name, t.icon, t.color, COUNT(q.id) as count
            FROM topics t
            LEFT JOIN questions q ON t.id = q.topic_id
            GROUP BY t.id
            ORDER BY count DESC
        """)
        return cursor.fetchall()


# ==================== QUESTION VARIANT OPERATIONS ====================

def add_question_variant(question_id, variant_text, source_document_id=None):
    """Add a variant phrasing of a question"""
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

def add_answer(question_id, answer_text, source_document_id=None, resolution_status=None, customer_satisfaction=None):
    """Add a new answer to a question"""
    # Calculate initial effectiveness
    success = 0
    fail = 0

    if resolution_status == 'resolved':
        if customer_satisfaction == 'positive':
            success = 2
        elif customer_satisfaction == 'neutral':
            success = 1
    elif resolution_status == 'partial':
        pass  # 0
    elif resolution_status == 'unresolved':
        fail = 1

    if customer_satisfaction == 'negative':
        fail += 1

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO answers (question_id, answer_text, source_document_id,
                                success_count, fail_count, resolution_status, customer_satisfaction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (question_id, answer_text, source_document_id, success, fail, resolution_status, customer_satisfaction))
        answer_id = cursor.lastrowid

        # Update effectiveness score
        update_answer_effectiveness(answer_id, conn)

        # Update recommended answer for this question
        update_recommended_answer(question_id, conn)

        return answer_id


def get_answers(question_id, limit=20):
    """Get answers for a question, ordered by effectiveness"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, d.filename as source_filename
            FROM answers a
            LEFT JOIN documents d ON a.source_document_id = d.id
            WHERE a.question_id = ?
            ORDER BY a.is_recommended DESC, a.effectiveness_score DESC
            LIMIT ?
        """, (question_id, limit))
        return cursor.fetchall()


def get_recommended_answer(question_id):
    """Get the recommended (best) answer for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, d.filename as source_filename
            FROM answers a
            LEFT JOIN documents d ON a.source_document_id = d.id
            WHERE a.question_id = ? AND a.is_recommended = 1
        """, (question_id,))
        return cursor.fetchone()


def update_answer_effectiveness(answer_id, conn=None):
    """Recalculate effectiveness score for an answer"""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT success_count, fail_count FROM answers WHERE id = ?", (answer_id,))
        row = cursor.fetchone()

        if row:
            total = row['success_count'] + row['fail_count']
            if total > 0:
                score = (row['success_count'] / total) * 100
            else:
                score = 50.0  # Neutral score for new answers

            cursor.execute("UPDATE answers SET effectiveness_score = ? WHERE id = ?", (score, answer_id))

        if should_close:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def update_recommended_answer(question_id, conn=None):
    """Update the recommended answer for a question based on effectiveness"""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        should_close = True

    try:
        cursor = conn.cursor()

        # Reset all recommendations for this question
        cursor.execute("UPDATE answers SET is_recommended = 0 WHERE question_id = ?", (question_id,))

        # Find and set the best answer
        cursor.execute("""
            UPDATE answers SET is_recommended = 1
            WHERE id = (
                SELECT id FROM answers
                WHERE question_id = ?
                ORDER BY effectiveness_score DESC, success_count DESC
                LIMIT 1
            )
        """, (question_id,))

        if should_close:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def update_answer_outcome(answer_id, resolved, satisfied):
    """Update answer statistics based on outcome"""
    with get_db() as conn:
        cursor = conn.cursor()

        success_delta = 0
        fail_delta = 0

        if resolved:
            if satisfied:
                success_delta = 2
            else:
                success_delta = 1
        else:
            fail_delta = 1

        if not satisfied:
            fail_delta += 1

        cursor.execute("""
            UPDATE answers
            SET success_count = success_count + ?, fail_count = fail_count + ?
            WHERE id = ?
        """, (success_delta, fail_delta, answer_id))

        # Recalculate effectiveness
        update_answer_effectiveness(answer_id, conn)

        # Get question_id and update recommended
        cursor.execute("SELECT question_id FROM answers WHERE id = ?", (answer_id,))
        row = cursor.fetchone()
        if row:
            update_recommended_answer(row['question_id'], conn)


def get_answers_count():
    """Get total answer count"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM answers")
        return cursor.fetchone()[0]


# ==================== DOCUMENT OPERATIONS ====================

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


def get_pending_documents(limit=50):
    """Get all pending documents for processing"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM documents
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
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


# ==================== KNOWLEDGE BASE OPERATIONS ====================

def add_knowledge(title, content, category=None, source_document_id=None):
    """Add knowledge base entry"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge_base (title, content, category, source_document_id)
            VALUES (?, ?, ?, ?)
        """, (title, content, category, source_document_id))
        return cursor.lastrowid


def get_knowledge(category=None, limit=100, offset=0):
    """Get knowledge base entries"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT kb.*, d.filename as source_filename
            FROM knowledge_base kb
            LEFT JOIN documents d ON kb.source_document_id = d.id
            WHERE 1=1
        """
        params = []

        if category:
            query += " AND kb.category = ?"
            params.append(category)

        query += " ORDER BY kb.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return cursor.fetchall()


def get_knowledge_count(category=None):
    """Get count of knowledge entries"""
    with get_db() as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE category = ?", (category,))
        else:
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        return cursor.fetchone()[0]


def search_knowledge(query, limit=20):
    """Search knowledge base"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM knowledge_base
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        return cursor.fetchall()


# ==================== SUMMARY OPERATIONS ====================

def update_daily_summary(calls=0, questions=0, answers=0, resolved=0, unresolved=0):
    """Update or create daily summary"""
    today = date.today().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO summary (date, total_calls, new_questions, new_answers, resolved_count, unresolved_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_calls = total_calls + ?,
                new_questions = new_questions + ?,
                new_answers = new_answers + ?,
                resolved_count = resolved_count + ?,
                unresolved_count = unresolved_count + ?
        """, (today, calls, questions, answers, resolved, unresolved,
              calls, questions, answers, resolved, unresolved))


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

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'")
        stats['pending_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'error'")
        stats['error_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions")
        stats['total_questions'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM answers")
        stats['total_answers'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        stats['total_knowledge'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM answers WHERE resolution_status = 'resolved'")
        stats['resolved_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM answers WHERE resolution_status = 'unresolved'")
        stats['unresolved_count'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents
            WHERE date(created_at) = date('now')
        """)
        stats['today_documents'] = cursor.fetchone()[0]

        return stats


# ==================== SEARCH OPERATIONS ====================

def search_questions(query, limit=20):
    """Search questions by text"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*, t.name as topic_name, t.icon as topic_icon
            FROM questions q
            LEFT JOIN topics t ON q.topic_id = t.id
            WHERE q.canonical_text LIKE ?
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (f'%{query}%', limit))
        return cursor.fetchall()


def search_all(query, limit=50):
    """Search across all tables"""
    results = {
        'questions': [],
        'answers': [],
        'knowledge': [],
        'documents': []
    }

    with get_db() as conn:
        cursor = conn.cursor()

        # Search questions
        cursor.execute("""
            SELECT q.*, t.name as topic_name, t.icon as topic_icon
            FROM questions q
            LEFT JOIN topics t ON q.topic_id = t.id
            WHERE q.canonical_text LIKE ?
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (f'%{query}%', limit))
        results['questions'] = cursor.fetchall()

        # Search answers
        cursor.execute("""
            SELECT a.*, q.canonical_text as question_text
            FROM answers a
            JOIN questions q ON a.question_id = q.id
            WHERE a.answer_text LIKE ?
            ORDER BY a.effectiveness_score DESC
            LIMIT ?
        """, (f'%{query}%', limit))
        results['answers'] = cursor.fetchall()

        # Search knowledge
        cursor.execute("""
            SELECT * FROM knowledge_base
            WHERE title LIKE ? OR content LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        results['knowledge'] = cursor.fetchall()

        # Search documents
        cursor.execute("""
            SELECT id, filename, type, status, created_at
            FROM documents
            WHERE filename LIKE ? OR content LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))
        results['documents'] = cursor.fetchall()

    return results


# ==================== MIGRATION ====================

def migrate_from_v1():
    """Migrate data from v1 schema to v2"""
    logger.info("Checking for v1 data to migrate...")

    with get_db() as conn:
        cursor = conn.cursor()

        # Check if old tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faq'")
        if not cursor.fetchone():
            logger.info("No v1 data found, skipping migration")
            return

        # Get default topic for migration
        cursor.execute("SELECT id FROM topics WHERE name = 'General Inquiry'")
        default_topic = cursor.fetchone()
        default_topic_id = default_topic['id'] if default_topic else 1

        # Migrate FAQ to questions/answers
        cursor.execute("SELECT * FROM faq")
        faq_items = cursor.fetchall()

        migrated = 0
        for faq in faq_items:
            # Add question
            cursor.execute("""
                INSERT INTO questions (topic_id, canonical_text, times_asked)
                VALUES (?, ?, ?)
            """, (default_topic_id, faq['question'], faq['times_asked']))
            question_id = cursor.lastrowid

            # Add answer
            cursor.execute("""
                INSERT INTO answers (question_id, answer_text, is_recommended, effectiveness_score)
                VALUES (?, ?, 1, 50.0)
            """, (question_id, faq['answer']))

            migrated += 1

        logger.info(f"Migrated {migrated} FAQ items to v2 schema")

        # Optionally rename old table
        try:
            cursor.execute("ALTER TABLE faq RENAME TO faq_v1_backup")
            cursor.execute("ALTER TABLE facts RENAME TO facts_v1_backup")
            logger.info("Old tables renamed to *_v1_backup")
        except:
            pass


# Initialize database on import
init_db()
