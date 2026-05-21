import sqlite3
import bcrypt
import os
import json

# ABSOLUTE PATH FIX
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "database", "ai_portal.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def initialize_extended_schema():
    """SaaS Upgrade: Initializes tables and safely migrates older schemas."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. User Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # --- SQLITE FIX: Safe Auto-Migration ---
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'created_at' not in columns:
        # SQLite workaround: Add as NULL (constant), then backfill with current timestamp
        cursor.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NULL")
        cursor.execute("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    
    # 2. Chat History Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            app_type TEXT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. UNIVERSAL DOCUMENT VAULT
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS universal_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            doc_category TEXT,
            raw_markdown TEXT,
            json_metadata TEXT,
            confidence_score FLOAT,
            extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def register_user(username, plain_password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    try:
        # Explicitly passing CURRENT_TIMESTAMP to bypass any schema default issues
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)", 
            (username, hashed.decode('utf-8'))
        )
        conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_login(username, plain_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        user_id, hashed_password = result
        if bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8')):
            return True, user_id
    return False, None

def archive_document(user_id, filename, category, markdown, confidence, metadata={}):
    """SaaS Feature: Saves OCR/RAG results permanently to the Vault."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO universal_docs (user_id, filename, doc_category, raw_markdown, confidence_score, json_metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, filename, category, markdown, confidence, json.dumps(metadata)))
    conn.commit()
    conn.close()

def get_user_vault(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, doc_category, confidence_score, extraction_date 
        FROM universal_docs WHERE user_id = ? ORDER BY extraction_date DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_document_markdown(doc_id, user_id):
    """Retrieves raw text for the Vault viewer."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_markdown FROM universal_docs WHERE id = ? AND user_id = ?", (doc_id, user_id))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_chat_message(user_id, app_type, session_id, role, content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (user_id, app_type, session_id, role, content)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, app_type, session_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(user_id, app_type, session_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT role, content FROM chat_history WHERE user_id = ? AND app_type = ?"
    params = [user_id, app_type]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]