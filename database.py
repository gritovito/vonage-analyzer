"""
Knowledge Hub Database v3.1 - Response Scripts Library with Subcategories
SQLite database with hierarchical categorization and operator script extraction
"""
import sqlite3
import os
import logging
import struct
import re
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

        # Filter rules table - for auto-moderation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filter_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_type TEXT NOT NULL,
                condition_value TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'auto_reject',
                description TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Moderation log table - history of all moderation actions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moderation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER,
                script_id INTEGER,
                action TEXT NOT NULL,
                reason TEXT,
                old_value TEXT,
                new_value TEXT,
                admin_user TEXT DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (script_id) REFERENCES scripts(id)
            )
        """)

        # Merged questions table - history of question merges
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS merged_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_question_id INTEGER NOT NULL,
                target_question_id INTEGER NOT NULL,
                source_canonical_text TEXT,
                merged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        # Questions table - with subcategory reference and moderation fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER,
                subcategory_id INTEGER,
                canonical_text TEXT NOT NULL,
                embedding BLOB,
                status TEXT DEFAULT 'no_answer',
                moderation_status TEXT DEFAULT 'pending',
                best_script_id INTEGER,
                times_asked INTEGER DEFAULT 1,
                reviewed_at TIMESTAMP,
                reviewed_by TEXT,
                source_filename TEXT,
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_moderation ON questions(moderation_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_times ON questions(times_asked DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subcategories_cluster ON subcategories(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_question ON question_variants(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_question ON scripts(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_effectiveness ON scripts(effectiveness DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_is_best ON scripts(is_best)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_moderation_log_question ON moderation_log(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_filter_rules_active ON filter_rules(is_active)")

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

        # Insert default filter rules
        default_filter_rules = [
            ('contains', 'Thank you for calling', 'auto_reject', 'Auto-answer message'),
            ('contains', 'Your call is important', 'auto_reject', 'Auto-answer message'),
            ('contains', 'Please hold', 'auto_reject', 'Hold message'),
            ('contains', 'Leave a message', 'auto_reject', 'Voicemail prompt'),
            ('contains', 'Press 1', 'auto_reject', 'IVR menu'),
            ('contains', 'Press 2', 'auto_reject', 'IVR menu'),
            ('contains', 'office hours', 'auto_reject', 'Office hours message'),
            ('contains', 'currently closed', 'auto_reject', 'Office closed message'),
            ('contains', 'mailbox is full', 'auto_reject', 'Voicemail full message'),
            ('contains', 'beep', 'auto_reject', 'Voicemail beep'),
            ('word_count_lt', '10', 'auto_reject', 'Too short (less than 10 words)'),
        ]

        for rule_type, condition_value, action, description in default_filter_rules:
            cursor.execute("""
                INSERT OR IGNORE INTO filter_rules (rule_type, condition_value, action, description)
                SELECT ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM filter_rules
                    WHERE rule_type = ? AND condition_value = ?
                )
            """, (rule_type, condition_value, action, description, rule_type, condition_value))

        logger.info("Knowledge Hub database initialized with subcategories and filter rules")


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


def get_cluster_with_subcategories(cluster_id, approved_only=True):
    """Get cluster with all its subcategories and top questions"""
    cluster = get_cluster(cluster_id)
    if not cluster:
        return None

    with get_db() as conn:
        cursor = conn.cursor()

        # Get subcategories with question counts (only approved if specified)
        if approved_only:
            cursor.execute("""
                SELECT s.*,
                       COUNT(q.id) as question_count
                FROM subcategories s
                LEFT JOIN questions q ON s.id = q.subcategory_id AND q.moderation_status = 'approved'
                WHERE s.cluster_id = ?
                GROUP BY s.id
                ORDER BY question_count DESC
            """, (cluster_id,))
        else:
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
            if approved_only:
                cursor.execute("""
                    SELECT q.id, q.canonical_text, q.times_asked, q.status
                    FROM questions q
                    WHERE q.subcategory_id = ? AND q.moderation_status = 'approved'
                    ORDER BY q.times_asked DESC
                    LIMIT 3
                """, (sub['id'],))
            else:
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


def get_subcategory_with_questions(subcategory_id, limit=50, approved_only=True):
    """Get subcategory with all its questions"""
    sub = get_subcategory(subcategory_id)
    if not sub:
        return None

    with get_db() as conn:
        cursor = conn.cursor()
        if approved_only:
            cursor.execute("""
                SELECT q.*,
                       (SELECT COUNT(*) FROM scripts WHERE question_id = q.id) as script_count
                FROM questions q
                WHERE q.subcategory_id = ? AND q.moderation_status = 'approved'
                ORDER BY q.times_asked DESC
                LIMIT ?
            """, (subcategory_id, limit))
        else:
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
    result['scripts'] = [dict(s) for s in scripts]
    result['best_script'] = None
    result['other_scripts'] = []

    for s in scripts:
        if s['is_best']:
            result['best_script'] = dict(s)
        else:
            result['other_scripts'].append(dict(s))

    return result


def get_questions(cluster_id=None, subcategory_id=None, status=None, moderation_status=None, approved_only=True, limit=100, offset=0, sort_by='times_asked'):
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

        if moderation_status:
            query += " AND q.moderation_status = ?"
            params.append(moderation_status)
        elif approved_only:
            query += " AND q.moderation_status = 'approved'"

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


def get_questions_count(cluster_id=None, subcategory_id=None, status=None, moderation_status=None, approved_only=True):
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
        if moderation_status:
            query += " AND moderation_status = ?"
            params.append(moderation_status)
        elif approved_only:
            query += " AND moderation_status = 'approved'"

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

def search_questions_text(query, limit=20, approved_only=True):
    """Text search in questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        if approved_only:
            cursor.execute("""
                SELECT q.*, c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                       s.name as subcategory_name
                FROM questions q
                LEFT JOIN clusters c ON q.cluster_id = c.id
                LEFT JOIN subcategories s ON q.subcategory_id = s.id
                WHERE q.canonical_text LIKE ? AND q.moderation_status = 'approved'
                ORDER BY q.times_asked DESC
                LIMIT ?
            """, (f'%{query}%', limit))
        else:
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


def get_autocomplete_suggestions(text, limit=5, approved_only=True):
    """Get autocomplete suggestions for search"""
    if not text or len(text) < 2:
        return []

    with get_db() as conn:
        cursor = conn.cursor()
        # Search in canonical text and variants
        if approved_only:
            cursor.execute("""
                SELECT DISTINCT canonical_text as suggestion
                FROM questions
                WHERE canonical_text LIKE ? AND moderation_status = 'approved'
                ORDER BY times_asked DESC
                LIMIT ?
            """, (f'%{text}%', limit))
        else:
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


# ==================== FILTER RULES OPERATIONS ====================

def get_filter_rules(active_only=False):
    """Get all filter rules"""
    with get_db() as conn:
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT * FROM filter_rules WHERE is_active = 1 ORDER BY created_at DESC")
        else:
            cursor.execute("SELECT * FROM filter_rules ORDER BY created_at DESC")
        return cursor.fetchall()


def get_filter_rule(rule_id):
    """Get a single filter rule"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM filter_rules WHERE id = ?", (rule_id,))
        return cursor.fetchone()


def add_filter_rule(rule_type, condition_value, action, description=None):
    """Add a new filter rule"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO filter_rules (rule_type, condition_value, action, description)
            VALUES (?, ?, ?, ?)
        """, (rule_type, condition_value, action, description))
        return cursor.lastrowid


def update_filter_rule(rule_id, rule_type=None, condition_value=None, action=None, description=None):
    """Update a filter rule"""
    with get_db() as conn:
        cursor = conn.cursor()
        rule = get_filter_rule(rule_id)
        if not rule:
            return False
        cursor.execute("""
            UPDATE filter_rules
            SET rule_type = ?, condition_value = ?, action = ?, description = ?
            WHERE id = ?
        """, (
            rule_type or rule['rule_type'],
            condition_value or rule['condition_value'],
            action or rule['action'],
            description if description is not None else rule['description'],
            rule_id
        ))
        return True


def toggle_filter_rule(rule_id):
    """Toggle filter rule active status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE filter_rules SET is_active = NOT is_active WHERE id = ?", (rule_id,))
        return cursor.rowcount > 0


def delete_filter_rule(rule_id):
    """Delete a filter rule"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM filter_rules WHERE id = ?", (rule_id,))
        return cursor.rowcount > 0


def get_filter_rules_count():
    """Get count of active filter rules"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM filter_rules WHERE is_active = 1")
        return cursor.fetchone()[0]


def apply_filter_rules(text):
    """
    Apply filter rules to text and return the moderation action.
    Returns: 'pending', 'approved', or 'rejected'
    """
    if not text:
        return 'rejected'

    rules = get_filter_rules(active_only=True)
    text_lower = text.lower().strip()
    word_count = len(text.split())

    for rule in rules:
        rule_type = rule['rule_type']
        condition = rule['condition_value']
        action = rule['action']

        matched = False

        if rule_type == 'contains':
            matched = condition.lower() in text_lower
        elif rule_type == 'not_contains':
            matched = condition.lower() not in text_lower
        elif rule_type == 'word_count_lt':
            try:
                matched = word_count < int(condition)
            except ValueError:
                pass
        elif rule_type == 'word_count_gt':
            try:
                matched = word_count > int(condition)
            except ValueError:
                pass
        elif rule_type == 'starts_with':
            matched = text_lower.startswith(condition.lower())
        elif rule_type == 'regex':
            try:
                matched = bool(re.search(condition, text, re.IGNORECASE))
            except re.error:
                pass

        if matched:
            if action == 'auto_reject':
                return 'rejected'
            elif action == 'auto_approve':
                return 'approved'
            # mark_pending just continues to check other rules

    return 'pending'


# ==================== MODERATION OPERATIONS ====================

def get_questions_for_moderation(moderation_status='pending', limit=50, offset=0):
    """Get questions for moderation with full details"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.*,
                   c.name as cluster_name, c.icon as cluster_icon, c.color as cluster_color,
                   s.name as subcategory_name,
                   (SELECT COUNT(*) FROM question_variants WHERE question_id = q.id) as variant_count,
                   (SELECT COUNT(*) FROM scripts WHERE question_id = q.id) as script_count,
                   (SELECT script_text FROM scripts WHERE question_id = q.id AND is_best = 1 LIMIT 1) as best_script_preview
            FROM questions q
            LEFT JOIN clusters c ON q.cluster_id = c.id
            LEFT JOIN subcategories s ON q.subcategory_id = s.id
            WHERE q.moderation_status = ?
            ORDER BY q.created_at DESC
            LIMIT ? OFFSET ?
        """, (moderation_status, limit, offset))
        return cursor.fetchall()


def get_moderation_counts():
    """Get counts for each moderation status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT moderation_status, COUNT(*) as count
            FROM questions
            GROUP BY moderation_status
        """)
        counts = {'pending': 0, 'approved': 0, 'rejected': 0}
        for row in cursor.fetchall():
            status = row['moderation_status'] or 'pending'
            counts[status] = row['count']
        return counts


def set_question_moderation_status(question_id, moderation_status, reason=None, admin_user='admin'):
    """Set moderation status for a question and log the action"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get old status
        cursor.execute("SELECT moderation_status FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        old_status = row['moderation_status'] if row else 'pending'

        # Update status
        cursor.execute("""
            UPDATE questions
            SET moderation_status = ?, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
            WHERE id = ?
        """, (moderation_status, admin_user, question_id))

        # Log the action
        action = f"status_changed_to_{moderation_status}"
        add_moderation_log(question_id, None, action, reason, old_status, moderation_status, admin_user, conn)

        return cursor.rowcount > 0


def bulk_set_moderation_status(question_ids, moderation_status, reason=None, admin_user='admin'):
    """Set moderation status for multiple questions"""
    with get_db() as conn:
        cursor = conn.cursor()
        for qid in question_ids:
            cursor.execute("SELECT moderation_status FROM questions WHERE id = ?", (qid,))
            row = cursor.fetchone()
            old_status = row['moderation_status'] if row else 'pending'

            cursor.execute("""
                UPDATE questions
                SET moderation_status = ?, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
                WHERE id = ?
            """, (moderation_status, admin_user, qid))

            action = f"bulk_status_changed_to_{moderation_status}"
            add_moderation_log(qid, None, action, reason, old_status, moderation_status, admin_user, conn)

        return len(question_ids)


def delete_question(question_id, reason=None, admin_user='admin'):
    """Delete a question and all related data"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get question text for logging
        cursor.execute("SELECT canonical_text FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        question_text = row['canonical_text'] if row else ''

        # Delete related data
        cursor.execute("DELETE FROM question_variants WHERE question_id = ?", (question_id,))
        cursor.execute("DELETE FROM scripts WHERE question_id = ?", (question_id,))
        cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))

        # Log deletion
        add_moderation_log(question_id, None, 'deleted', reason, question_text, None, admin_user, conn)

        return cursor.rowcount > 0


def bulk_delete_questions(question_ids, reason=None, admin_user='admin'):
    """Delete multiple questions"""
    deleted = 0
    with get_db() as conn:
        cursor = conn.cursor()
        for qid in question_ids:
            cursor.execute("SELECT canonical_text FROM questions WHERE id = ?", (qid,))
            row = cursor.fetchone()
            question_text = row['canonical_text'] if row else ''

            cursor.execute("DELETE FROM question_variants WHERE question_id = ?", (qid,))
            cursor.execute("DELETE FROM scripts WHERE question_id = ?", (qid,))
            cursor.execute("DELETE FROM questions WHERE id = ?", (qid,))

            if cursor.rowcount > 0:
                deleted += 1
                add_moderation_log(qid, None, 'bulk_deleted', reason, question_text, None, admin_user, conn)

    return deleted


def update_question_text(question_id, new_text, admin_user='admin'):
    """Update canonical text of a question"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get old text
        cursor.execute("SELECT canonical_text FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        old_text = row['canonical_text'] if row else ''

        # Update text
        cursor.execute("""
            UPDATE questions SET canonical_text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (new_text, question_id))

        # Log the change
        add_moderation_log(question_id, None, 'text_edited', None, old_text, new_text, admin_user, conn)

        return cursor.rowcount > 0


def update_question_cluster(question_id, cluster_id, subcategory_id=None, admin_user='admin'):
    """Update cluster and subcategory of a question"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get old values
        cursor.execute("SELECT cluster_id, subcategory_id FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        old_cluster = row['cluster_id'] if row else None

        # Update
        cursor.execute("""
            UPDATE questions SET cluster_id = ?, subcategory_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (cluster_id, subcategory_id, question_id))

        # Log
        add_moderation_log(question_id, None, 'cluster_changed', None, str(old_cluster), str(cluster_id), admin_user, conn)

        return cursor.rowcount > 0


# ==================== MODERATION LOG ====================

def add_moderation_log(question_id, script_id, action, reason, old_value, new_value, admin_user='admin', conn=None):
    """Add entry to moderation log"""
    def _insert(cursor):
        cursor.execute("""
            INSERT INTO moderation_log (question_id, script_id, action, reason, old_value, new_value, admin_user)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (question_id, script_id, action, reason, old_value, new_value, admin_user))

    if conn:
        cursor = conn.cursor()
        _insert(cursor)
    else:
        with get_db() as conn:
            cursor = conn.cursor()
            _insert(cursor)


def get_moderation_log(limit=100, offset=0, question_id=None):
    """Get moderation log entries"""
    with get_db() as conn:
        cursor = conn.cursor()
        if question_id:
            cursor.execute("""
                SELECT ml.*, q.canonical_text as question_text
                FROM moderation_log ml
                LEFT JOIN questions q ON ml.question_id = q.id
                WHERE ml.question_id = ?
                ORDER BY ml.created_at DESC
                LIMIT ? OFFSET ?
            """, (question_id, limit, offset))
        else:
            cursor.execute("""
                SELECT ml.*, q.canonical_text as question_text
                FROM moderation_log ml
                LEFT JOIN questions q ON ml.question_id = q.id
                ORDER BY ml.created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
        return cursor.fetchall()


def get_moderation_log_count(question_id=None):
    """Get count of moderation log entries"""
    with get_db() as conn:
        cursor = conn.cursor()
        if question_id:
            cursor.execute("SELECT COUNT(*) FROM moderation_log WHERE question_id = ?", (question_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM moderation_log")
        return cursor.fetchone()[0]


# ==================== MERGE OPERATIONS ====================

def merge_questions(source_id, target_id, admin_user='admin'):
    """
    Merge source question into target question.
    - Move all variants from source to target
    - Move all scripts from source to target
    - Add times_asked from source to target
    - Delete source question
    - Log the merge
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Get source question info
        cursor.execute("SELECT canonical_text, times_asked FROM questions WHERE id = ?", (source_id,))
        source = cursor.fetchone()
        if not source:
            return False, "Source question not found"

        # Check target exists
        cursor.execute("SELECT id FROM questions WHERE id = ?", (target_id,))
        if not cursor.fetchone():
            return False, "Target question not found"

        source_text = source['canonical_text']
        source_times = source['times_asked'] or 1

        # Move variants
        cursor.execute("""
            UPDATE question_variants SET question_id = ? WHERE question_id = ?
        """, (target_id, source_id))

        # Add source canonical text as variant of target
        cursor.execute("""
            INSERT INTO question_variants (question_id, variant_text)
            VALUES (?, ?)
        """, (target_id, source_text))

        # Move scripts
        cursor.execute("""
            UPDATE scripts SET question_id = ? WHERE question_id = ?
        """, (target_id, source_id))

        # Update times_asked
        cursor.execute("""
            UPDATE questions SET times_asked = times_asked + ? WHERE id = ?
        """, (source_times, target_id))

        # Record merge in merged_questions table
        cursor.execute("""
            INSERT INTO merged_questions (source_question_id, target_question_id, source_canonical_text)
            VALUES (?, ?, ?)
        """, (source_id, target_id, source_text))

        # Delete source question
        cursor.execute("DELETE FROM questions WHERE id = ?", (source_id,))

        # Recalculate best script for target
        _update_best_script(cursor, target_id)

        # Log the merge
        add_moderation_log(
            target_id, None, 'merged',
            f"Merged question #{source_id} into this question",
            source_text, None, admin_user, conn
        )

        return True, "Questions merged successfully"


def get_similar_questions_for_merge(question_id, limit=10):
    """Get similar questions for merge suggestion"""
    question = get_question(question_id)
    if not question:
        return []

    # Import embeddings here to avoid circular import
    from database import deserialize_embedding

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        if not row or not row['embedding']:
            return []

        source_embedding = deserialize_embedding(row['embedding'])
        if not source_embedding:
            return []

    # Get all questions with embeddings
    all_questions = get_all_questions_with_embeddings()

    import math
    def cosine_sim(vec1, vec2):
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    results = []
    for q in all_questions:
        if q['id'] == question_id:
            continue
        if q.get('embedding'):
            similarity = cosine_sim(source_embedding, q['embedding'])
            if similarity > 0.5:  # Only show reasonably similar questions
                full_q = get_question(q['id'])
                if full_q:
                    results.append({
                        'id': q['id'],
                        'canonical_text': q['canonical_text'],
                        'similarity': round(similarity * 100, 1),
                        'cluster_name': full_q['cluster_name'] if full_q else None,
                        'times_asked': full_q['times_asked'] if full_q else 0
                    })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:limit]


# ==================== SCRIPT OPERATIONS FOR ADMIN ====================

def update_script_text(script_id, new_text, admin_user='admin'):
    """Update script text"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get old text
        cursor.execute("SELECT script_text, question_id FROM scripts WHERE id = ?", (script_id,))
        row = cursor.fetchone()
        if not row:
            return False
        old_text = row['script_text']
        question_id = row['question_id']

        # Update
        cursor.execute("UPDATE scripts SET script_text = ? WHERE id = ?", (new_text, script_id))

        # Log
        add_moderation_log(question_id, script_id, 'script_edited', None, old_text, new_text, admin_user, conn)

        return True


def delete_script(script_id, admin_user='admin'):
    """Delete a script"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get script info
        cursor.execute("SELECT script_text, question_id FROM scripts WHERE id = ?", (script_id,))
        row = cursor.fetchone()
        if not row:
            return False
        script_text = row['script_text']
        question_id = row['question_id']

        # Delete
        cursor.execute("DELETE FROM scripts WHERE id = ?", (script_id,))

        # Recalculate best script
        _update_best_script(cursor, question_id)

        # Log
        add_moderation_log(question_id, script_id, 'script_deleted', None, script_text, None, admin_user, conn)

        return True


def set_best_script(question_id, script_id, admin_user='admin'):
    """Manually set a script as best"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Reset all scripts for this question
        cursor.execute("UPDATE scripts SET is_best = 0 WHERE question_id = ?", (question_id,))

        # Set new best
        cursor.execute("UPDATE scripts SET is_best = 1 WHERE id = ? AND question_id = ?", (script_id, question_id))

        # Update question
        cursor.execute("UPDATE questions SET best_script_id = ? WHERE id = ?", (script_id, question_id))

        # Log
        add_moderation_log(question_id, script_id, 'best_script_set', None, None, str(script_id), admin_user, conn)

        return cursor.rowcount > 0


def delete_variant(variant_id, admin_user='admin'):
    """Delete a question variant"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get variant info
        cursor.execute("SELECT variant_text, question_id FROM question_variants WHERE id = ?", (variant_id,))
        row = cursor.fetchone()
        if not row:
            return False
        variant_text = row['variant_text']
        question_id = row['question_id']

        # Delete
        cursor.execute("DELETE FROM question_variants WHERE id = ?", (variant_id,))

        # Log
        add_moderation_log(question_id, None, 'variant_deleted', None, variant_text, None, admin_user, conn)

        return True


# ==================== ADMIN STATS ====================

def get_admin_stats():
    """Get statistics for admin dashboard"""
    with get_db() as conn:
        cursor = conn.cursor()
        stats = {}

        # Total questions
        cursor.execute("SELECT COUNT(*) FROM questions")
        stats['total_questions'] = cursor.fetchone()[0]

        # Moderation counts
        cursor.execute("SELECT COUNT(*) FROM questions WHERE moderation_status = 'pending'")
        stats['pending_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions WHERE moderation_status = 'approved'")
        stats['approved_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions WHERE moderation_status = 'rejected'")
        stats['rejected_count'] = cursor.fetchone()[0]

        # Scripts
        cursor.execute("SELECT COUNT(*) FROM scripts")
        stats['total_scripts'] = cursor.fetchone()[0]

        # Filter rules
        cursor.execute("SELECT COUNT(*) FROM filter_rules WHERE is_active = 1")
        stats['active_rules'] = cursor.fetchone()[0]

        # Recent moderation actions
        cursor.execute("SELECT COUNT(*) FROM moderation_log")
        stats['total_log_entries'] = cursor.fetchone()[0]

        return stats


def apply_rules_to_existing_questions():
    """Apply filter rules to all existing questions and update their moderation status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, canonical_text FROM questions")
        questions = cursor.fetchall()

        updated = {'pending': 0, 'approved': 0, 'rejected': 0}

        for q in questions:
            new_status = apply_filter_rules(q['canonical_text'])
            cursor.execute("""
                UPDATE questions SET moderation_status = ? WHERE id = ?
            """, (new_status, q['id']))
            updated[new_status] += 1

        return updated


# Initialize on import
init_db()
