"""
Knowledge Hub Database v3.1 - Response Scripts Library with Subcategories
SQLite database with hierarchical categorization and operator script extraction
"""
import sqlite3
import os
import logging
import struct
from datetime import datetime, date
from contextlib import contextmanager
from config import DATABASE_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# Predefined subcategories for each cluster
SUBCATEGORIES = {
    "Device Issues": [
        "Battery & Charging",
        "Screen & Display",
        "Audio & Sound",
        "Power & Boot",
        "Buttons & Controls",
        "Camera",
        "Physical Damage"
    ],
    "Messaging": [
        "SMS Text Messages",
        "MMS & Media Messages",
        "Group Messages",
        "Messaging Apps"
    ],
    "Calls & Voice": [
        "Can't Make Calls",
        "Can't Receive Calls",
        "Call Quality",
        "Voicemail"
    ],
    "Data & Internet": [
        "No Data Connection",
        "Slow Data",
        "WiFi Issues",
        "Hotspot"
    ],
    "Apps & Software": [
        "App Crashes",
        "Software Updates",
        "Settings Issues",
        "Storage"
    ],
    "Account & Billing": [
        "Bill Questions",
        "Payment Issues",
        "Plan Changes",
        "Account Access"
    ],
    "Store & Service": [
        "Store Visit",
        "Device Repair",
        "Order Pickup",
        "Warranty"
    ],
    "General Inquiry": [
        "Product Information",
        "Service Questions",
        "Other"
    ]
}


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

        # Clusters table - main categories
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                icon TEXT,
                color TEXT,
                question_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Subcategories table - nested under clusters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subcategories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                question_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES clusters(id),
                UNIQUE(cluster_id, name)
            )
        """)

        # Questions table - with subcategory reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER,
                subcategory_id INTEGER,
                canonical_text TEXT NOT NULL,
                embedding BLOB,
                status TEXT DEFAULT 'no_answer',
                best_script_id INTEGER,
                times_asked INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES clusters(id),
                FOREIGN KEY (subcategory_id) REFERENCES subcategories(id)
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

        # Scripts table - ready-to-use operator response scripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                script_text TEXT NOT NULL,
                script_type TEXT DEFAULT 'instruction',
                has_steps BOOLEAN DEFAULT 0,
                success_count INTEGER DEFAULT 1,
                fail_count INTEGER DEFAULT 0,
                effectiveness REAL DEFAULT 50.0,
                is_best BOOLEAN DEFAULT 0,
                source_document_id INTEGER,
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
                new_scripts INTEGER DEFAULT 0,
                resolved_count INTEGER DEFAULT 0,
                unresolved_count INTEGER DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_cluster ON questions(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_subcategory ON questions(subcategory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_times ON questions(times_asked DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subcategories_cluster ON subcategories(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_question ON question_variants(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_question ON scripts(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_effectiveness ON scripts(effectiveness DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_is_best ON scripts(is_best)")

        # Insert default clusters
        default_clusters = [
            ("Device Issues", "Hardware problems with phones and devices", "📱", "#e74c3c"),
            ("Messaging", "SMS, MMS, and messaging app issues", "💬", "#9b59b6"),
            ("Calls & Voice", "Call quality, voicemail, phone calls", "📞", "#3498db"),
            ("Data & Internet", "Mobile data, WiFi, connectivity", "🌐", "#2ecc71"),
            ("Apps & Software", "Applications, updates, settings", "💻", "#f39c12"),
            ("Account & Billing", "Payments, plans, account issues", "💳", "#1abc9c"),
            ("Store & Service", "Store visits, repairs, pickup", "🏪", "#e67e22"),
            ("General Inquiry", "Other questions", "❓", "#95a5a6"),
        ]

        for name, desc, icon, color in default_clusters:
            cursor.execute("""
                INSERT OR IGNORE INTO clusters (name, description, icon, color)
                VALUES (?, ?, ?, ?)
            """, (name, desc, icon, color))

        # Insert subcategories
        for cluster_name, subcats in SUBCATEGORIES.items():
            cursor.execute("SELECT id FROM clusters WHERE name = ?", (cluster_name,))
            row = cursor.fetchone()
            if row:
                cluster_id = row['id']
                for subcat_name in subcats:
                    cursor.execute("""
                        INSERT OR IGNORE INTO subcategories (cluster_id, name)
                        VALUES (?, ?)
                    """, (cluster_id, subcat_name))

        logger.info("Knowledge Hub database initialized with subcategories")


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


def get_cluster_with_subcategories(cluster_id):
    """Get cluster with all its subcategories and top questions"""
    cluster = get_cluster(cluster_id)
    if not cluster:
        return None

    with get_db() as conn:
        cursor = conn.cursor()

        # Get subcategories with question counts
        cursor.execute("""
            SELECT s.*,
                   COUNT(q.id) as question_count
            FROM subcategories s
            LEFT JOIN questions q ON s.id = q.subcategory_id
            WHERE s.cluster_id = ?
            GROUP BY s.id
            ORDER BY question_count DESC
        """, (cluster_id,))
        subcategories = cursor.fetchall()

        # For each subcategory, get top 3 questions
        result = dict(cluster)
        result['subcategories'] = []

        for sub in subcategories:
            sub_dict = dict(sub)
            cursor.execute("""
                SELECT q.id, q.canonical_text, q.times_asked, q.status
                FROM questions q
                WHERE q.subcategory_id = ?
                ORDER BY q.times_asked DESC
                LIMIT 3
            """, (sub['id'],))
            sub_dict['top_questions'] = [dict(q) for q in cursor.fetchall()]
            result['subcategories'].append(sub_dict)

        return result


# ==================== SUBCATEGORY OPERATIONS ====================

def get_subcategory(subcategory_id):
    """Get subcategory by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color
            FROM subcategories s
            JOIN clusters c ON s.cluster_id = c.id
            WHERE s.id = ?
        """, (subcategory_id,))
        return cursor.fetchone()


def get_subcategory_by_name(cluster_id, name):
    """Get subcategory by cluster_id and name"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM subcategories
            WHERE cluster_id = ? AND name = ?
        """, (cluster_id, name))
        return cursor.fetchone()


def get_or_create_subcategory(cluster_id, name):
    """Get existing subcategory or create new one"""
    sub = get_subcategory_by_name(cluster_id, name)
    if sub:
        return sub['id']

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO subcategories (cluster_id, name)
            VALUES (?, ?)
        """, (cluster_id, name))
        return cursor.lastrowid


def get_subcategory_with_questions(subcategory_id, limit=50):
    """Get subcategory with all its questions"""
    sub = get_subcategory(subcategory_id)
    if not sub:
        return None

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*,
                   (SELECT COUNT(*) FROM scripts WHERE question_id = q.id) as script_count
            FROM questions q
            WHERE q.subcategory_id = ?
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (subcategory_id, limit))
        questions = cursor.fetchall()

        result = dict(sub)
        result['questions'] = [dict(q) for q in questions]
        return result


def get_subcategories_for_cluster(cluster_id):
    """Get all subcategories for a cluster"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*,
                   COUNT(q.id) as question_count
            FROM subcategories s
            LEFT JOIN questions q ON s.id = q.subcategory_id
            WHERE s.cluster_id = ?
            GROUP BY s.id
            ORDER BY question_count DESC
        """, (cluster_id,))
        return cursor.fetchall()


# ==================== QUESTION OPERATIONS ====================

def add_question(cluster_id, canonical_text, embedding=None, subcategory_id=None):
    """Add a new question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO questions (cluster_id, subcategory_id, canonical_text, embedding, status)
            VALUES (?, ?, ?, ?, 'no_answer')
        """, (cluster_id, subcategory_id, canonical_text, serialize_embedding(embedding)))
        return cursor.lastrowid


def get_question(question_id):
    """Get question with full details"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   s.name as subcategory_name,
                   (SELECT COUNT(*) FROM question_variants WHERE question_id = q.id) as variant_count,
                   (SELECT COUNT(*) FROM scripts WHERE question_id = q.id) as script_count
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            LEFT JOIN subcategories s ON q.subcategory_id = s.id
            WHERE q.id = ?
        """, (question_id,))
        return cursor.fetchone()


def get_question_detail(question_id):
    """Get question with variants and all scripts"""
    question = get_question(question_id)
    if not question:
        return None

    result = dict(question)
    result['variants'] = [dict(v) for v in get_question_variants(question_id)]

    scripts = get_scripts(question_id)
    result['best_script'] = None
    result['other_scripts'] = []

    for s in scripts:
        if s['is_best']:
            result['best_script'] = dict(s)
        else:
            result['other_scripts'].append(dict(s))

    return result


def get_questions(cluster_id=None, subcategory_id=None, status=None, limit=100, offset=0, sort_by='times_asked'):
    """Get questions with optional filters"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   s.name as subcategory_name,
                   (SELECT COUNT(*) FROM scripts WHERE question_id = q.id) as script_count
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            LEFT JOIN subcategories s ON q.subcategory_id = s.id
            WHERE 1=1
        """
        params = []

        if cluster_id:
            query += " AND q.cluster_id = ?"
            params.append(cluster_id)

        if subcategory_id:
            query += " AND q.subcategory_id = ?"
            params.append(subcategory_id)

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


def get_questions_count(cluster_id=None, subcategory_id=None, status=None):
    """Get count of questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM questions WHERE 1=1"
        params = []

        if cluster_id:
            query += " AND cluster_id = ?"
            params.append(cluster_id)
        if subcategory_id:
            query += " AND subcategory_id = ?"
            params.append(subcategory_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        return cursor.fetchone()[0]


def get_all_questions_with_embeddings():
    """Get all questions with embeddings for similarity search"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, canonical_text, embedding, cluster_id, subcategory_id
            FROM questions WHERE embedding IS NOT NULL
        """)
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'canonical_text': row['canonical_text'],
                'embedding': deserialize_embedding(row['embedding']),
                'cluster_id': row['cluster_id'],
                'subcategory_id': row['subcategory_id']
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


def get_needs_work_questions(limit=100):
    """Get questions that need work"""
    return get_questions(status='needs_work', limit=limit, sort_by='times_asked')


def get_top_questions(limit=10):
    """Get most frequently asked questions with best scripts"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   s.name as subcategory_name
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            LEFT JOIN subcategories s ON q.subcategory_id = s.id
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (limit,))
        questions = cursor.fetchall()

        results = []
        for q in questions:
            q_dict = dict(q)
            best = get_best_script(q['id'])
            q_dict['best_script'] = dict(best) if best else None
            results.append(q_dict)

        return results


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


# ==================== SCRIPT OPERATIONS ====================

def add_script(question_id, script_text, script_type='instruction', has_steps=False, resolved=True, source_doc_id=None):
    """Add a new script for a question"""
    # Calculate initial effectiveness based on resolution
    effectiveness = 70.0 if resolved else 30.0
    if has_steps:
        effectiveness += 10
    if script_type == 'instruction':
        effectiveness += 5
    effectiveness = min(effectiveness, 100.0)

    success = 1 if resolved else 0
    fail = 0 if resolved else 1

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scripts (question_id, script_text, script_type, has_steps,
                                success_count, fail_count, effectiveness, source_document_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (question_id, script_text, script_type, has_steps, success, fail, effectiveness, source_doc_id))
        script_id = cursor.lastrowid

        # Update best script
        _update_best_script(cursor, question_id)

        return script_id


def get_scripts(question_id):
    """Get all scripts for a question, best first"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, d.filename as source_filename
            FROM scripts s
            LEFT JOIN documents d ON s.source_document_id = d.id
            WHERE s.question_id = ?
            ORDER BY s.is_best DESC, s.effectiveness DESC
        """, (question_id,))
        return cursor.fetchall()


def get_best_script(question_id):
    """Get the best script for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, d.filename as source_filename
            FROM scripts s
            LEFT JOIN documents d ON s.source_document_id = d.id
            WHERE s.question_id = ? AND s.is_best = 1
        """, (question_id,))
        return cursor.fetchone()


def get_script(script_id):
    """Get a single script by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, d.filename as source_filename
            FROM scripts s
            LEFT JOIN documents d ON s.source_document_id = d.id
            WHERE s.id = ?
        """, (script_id,))
        return cursor.fetchone()


def find_similar_script(question_id, script_text, similarity_threshold=0.9):
    """Find if a similar script already exists for this question"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, script_text, success_count, fail_count FROM scripts
            WHERE question_id = ?
        """, (question_id,))

        normalized_new = script_text.lower().strip()
        for row in cursor.fetchall():
            normalized_existing = row['script_text'].lower().strip()
            if normalized_new == normalized_existing:
                return row['id']
            if len(normalized_new) > 50 and len(normalized_existing) > 50:
                shorter = min(normalized_new, normalized_existing, key=len)
                longer = max(normalized_new, normalized_existing, key=len)
                if shorter in longer:
                    return row['id']
                words_new = set(normalized_new.split())
                words_existing = set(normalized_existing.split())
                if len(words_new) > 0:
                    overlap = len(words_new & words_existing) / len(words_new)
                    if overlap > similarity_threshold:
                        return row['id']
        return None


def update_script_count(script_id, success=True):
    """Update script success/fail count"""
    with get_db() as conn:
        cursor = conn.cursor()
        if success:
            cursor.execute("UPDATE scripts SET success_count = success_count + 1 WHERE id = ?", (script_id,))
        else:
            cursor.execute("UPDATE scripts SET fail_count = fail_count + 1 WHERE id = ?", (script_id,))

        _update_script_effectiveness(cursor, script_id)

        cursor.execute("SELECT question_id FROM scripts WHERE id = ?", (script_id,))
        row = cursor.fetchone()
        if row:
            _update_best_script(cursor, row['question_id'])


def update_script_feedback(script_id, helpful):
    """Update script based on user feedback"""
    with get_db() as conn:
        cursor = conn.cursor()

        if helpful:
            cursor.execute("UPDATE scripts SET success_count = success_count + 1 WHERE id = ?", (script_id,))
        else:
            cursor.execute("UPDATE scripts SET fail_count = fail_count + 1 WHERE id = ?", (script_id,))

        _update_script_effectiveness(cursor, script_id)

        cursor.execute("SELECT question_id FROM scripts WHERE id = ?", (script_id,))
        row = cursor.fetchone()
        if row:
            _update_best_script(cursor, row['question_id'])


def _update_script_effectiveness(cursor, script_id):
    """Calculate effectiveness for a script"""
    cursor.execute("SELECT success_count, fail_count, has_steps, script_type FROM scripts WHERE id = ?", (script_id,))
    row = cursor.fetchone()
    if row:
        total = row['success_count'] + row['fail_count']
        if total > 0:
            base = (row['success_count'] / total) * 100
        else:
            base = 50.0

        if row['has_steps']:
            base += 10
        if row['script_type'] == 'instruction':
            base += 5

        effectiveness = min(base, 100.0)
        cursor.execute("UPDATE scripts SET effectiveness = ? WHERE id = ?", (effectiveness, script_id))


def _update_best_script(cursor, question_id):
    """Update best script and question status"""
    cursor.execute("UPDATE scripts SET is_best = 0 WHERE question_id = ?", (question_id,))

    cursor.execute("""
        SELECT id, effectiveness FROM scripts
        WHERE question_id = ?
        ORDER BY effectiveness DESC, success_count DESC
        LIMIT 1
    """, (question_id,))
    best = cursor.fetchone()

    if best:
        cursor.execute("UPDATE scripts SET is_best = 1 WHERE id = ?", (best['id'],))

        if best['effectiveness'] >= 70:
            status = 'resolved'
        elif best['effectiveness'] > 0:
            status = 'needs_work'
        else:
            status = 'no_answer'

        cursor.execute("UPDATE questions SET status = ?, best_script_id = ? WHERE id = ?",
                      (status, best['id'], question_id))
    else:
        cursor.execute("UPDATE questions SET status = 'no_answer', best_script_id = NULL WHERE id = ?",
                      (question_id,))


def recalculate_best_script(question_id):
    """Recalculate best script for a question"""
    with get_db() as conn:
        cursor = conn.cursor()
        _update_best_script(cursor, question_id)


def get_scripts_count(question_id=None):
    """Get count of scripts"""
    with get_db() as conn:
        cursor = conn.cursor()
        if question_id:
            cursor.execute("SELECT COUNT(*) FROM scripts WHERE question_id = ?", (question_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM scripts")
        return cursor.fetchone()[0]


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


# ==================== SEARCH & AUTOCOMPLETE ====================

def search_questions_text(query, limit=20):
    """Text search in questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*, c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   s.name as subcategory_name
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            LEFT JOIN subcategories s ON q.subcategory_id = s.id
            WHERE q.canonical_text LIKE ?
            ORDER BY q.times_asked DESC
            LIMIT ?
        """, (f'%{query}%', limit))
        return cursor.fetchall()


def get_autocomplete_suggestions(text, limit=5):
    """Get autocomplete suggestions for search"""
    if not text or len(text) < 2:
        return []

    with get_db() as conn:
        cursor = conn.cursor()
        # Search in canonical text and variants
        cursor.execute("""
            SELECT DISTINCT canonical_text as suggestion
            FROM questions
            WHERE canonical_text LIKE ?
            ORDER BY times_asked DESC
            LIMIT ?
        """, (f'%{text}%', limit))
        results = [row['suggestion'] for row in cursor.fetchall()]
        return results


# ==================== STATISTICS ====================

def get_stats():
    """Get overall statistics"""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM clusters")
        stats['total_clusters'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM subcategories")
        stats['total_subcategories'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions")
        stats['total_questions'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scripts")
        stats['total_scripts'] = cursor.fetchone()[0]

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

        cursor.execute("SELECT AVG(effectiveness) FROM scripts WHERE is_best = 1")
        avg_eff = cursor.fetchone()[0]
        stats['avg_script_effectiveness'] = round(avg_eff, 1) if avg_eff else 0

        return stats


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


# ==================== DAILY SUMMARY ====================

def update_daily_summary(calls=0, questions=0, scripts=0, resolved=0, unresolved=0):
    """Update daily summary counters"""
    today = date.today().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO daily_summary (date, total_calls, new_questions, new_scripts, resolved_count, unresolved_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_calls = total_calls + excluded.total_calls,
                new_questions = new_questions + excluded.new_questions,
                new_scripts = new_scripts + excluded.new_scripts,
                resolved_count = resolved_count + excluded.resolved_count,
                unresolved_count = unresolved_count + excluded.unresolved_count
        """, (today, calls, questions, scripts, resolved, unresolved))


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


# Initialize on import
init_db()
