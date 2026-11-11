"""业务数据库模块：MySQL 连接、建表、documents / qa_history 的增删查。

分工：MySQL 存"业务数据"（谁传了什么文档、谁问了什么问题），
Chroma 向量库存"文档切片"（负责语义检索）——两个库各管一摊。
"""
import json

import pymysql
from pymysql.cursors import DictCursor

from app.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB,
)


def get_conn():
    """拿一条新连接。autocommit=True：每条 SQL 立即生效，不用手动 commit。"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=DictCursor,
    )


def init_db():
    """建三张表（已存在就跳过）。应用启动时调用一次。"""
    sqls = [
        # 用户表（第 2 周注册/登录用）
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            username      VARCHAR(50)  NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        # 文档表：每上传一个文件记一行
        # chroma_ids 存该文档在 Chroma 里的切片 id 列表（JSON），删除文档时靠它删向量
        """
        CREATE TABLE IF NOT EXISTS documents (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            filename   VARCHAR(255) NOT NULL,
            uploader   VARCHAR(50)  NOT NULL DEFAULT 'anonymous',
            chunks     INT          NOT NULL DEFAULT 0,
            chroma_ids TEXT,
            created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        # 问答历史表：每次提问记一行（多端互动的"联动"数据源）
        """
        CREATE TABLE IF NOT EXISTS qa_history (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(64)  NOT NULL,
            question   TEXT         NOT NULL,
            answer     TEXT         NOT NULL,
            created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]
    conn = get_conn()
    with conn.cursor() as cur:
        for sql in sqls:
            cur.execute(sql)
    conn.close()

    # 旧表补列（列已存在会报 1060，忽略）
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE qa_history ADD COLUMN source_doc TEXT")
        conn.close()
    except pymysql.err.OperationalError:
        pass
    # 文件哈希列：重复上传检测用（优化5）
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64)")
        conn.close()
    except pymysql.err.OperationalError:
        pass


# ===== users 表操作 =====

def create_user(username, password_hash):
    """插入一行用户，返回自增 id；用户名重复返回 None（UNIQUE 约束兜底）。"""
    conn = get_conn()
    with conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
            uid = cur.lastrowid
        except pymysql.err.IntegrityError:
            uid = None
    conn.close()
    return uid


def get_user_by_username(username):
    """按用户名查用户完整记录，查不到返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    conn.close()
    return row


# ===== documents 表操作 =====

def add_document(filename, uploader, chunks, chroma_ids, file_hash=""):
    """插入一行文档记录，返回自增 id。chroma_ids 是列表，存成 JSON 字符串。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (filename, uploader, chunks, chroma_ids, file_hash) VALUES (%s, %s, %s, %s, %s)",
            (filename, uploader, chunks, json.dumps(list(chroma_ids)), file_hash),
        )
        doc_id = cur.lastrowid
    conn.close()
    return doc_id


def get_document_by_hash(file_hash):
    """按文件哈希查文档，命中返回该行（重复上传检测用），没有返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename FROM documents WHERE file_hash = %s", (file_hash,))
        row = cur.fetchone()
    conn.close()
    return row


def list_documents():
    """返回所有文档记录（不含 chroma_ids，那是内部数据）。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, uploader, chunks, created_at FROM documents ORDER BY id DESC"
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def get_document(doc_id):
    """按 id 查单条完整记录，查不到返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
    conn.close()
    return row


def delete_document(doc_id):
    """删一行，返回是否真的删了。"""
    conn = get_conn()
    with conn.cursor() as cur:
        n = cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.close()
    return n > 0


# ===== qa_history 表操作 =====

def add_qa_record(session_id, question, answer, source_doc="[]"):
    """记一次问答。source_doc 是命中来源文件名的 JSON 列表。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO qa_history (session_id, question, answer, source_doc) VALUES (%s, %s, %s, %s)",
            (session_id, question, answer, source_doc),
        )
    conn.close()


def doc_question_stats():
    """互动看板数据：每个文档被提问几次 + 最近一次问题。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename, uploader FROM documents ORDER BY id")
        docs = cur.fetchall()
        cur.execute("SELECT question, source_doc, created_at FROM qa_history ORDER BY id DESC")
        qas = cur.fetchall()
    conn.close()

    stats = []
    for d in docs:
        # source_doc 里存的是来源文件名 JSON，和 documents.filename 匹配
        hits = [q for q in qas if d["filename"] in _parse_sources(q["source_doc"])]
        stats.append({
            "filename": d["filename"],
            "uploader": d["uploader"],
            "asks": len(hits),
            "last_question": hits[0]["question"] if hits else None,
            "last_at": str(hits[0]["created_at"]) if hits else None,
        })
    return stats


def _parse_sources(raw):
    """把 qa_history.source_doc 的 JSON 文本解析成列表；空/坏数据返回空列表。"""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except ValueError:
        return []
