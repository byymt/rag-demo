"""业务数据库模块：MySQL 连接池、建表、documents / qa_history 的增删查。

分工：MySQL 存"业务数据"（谁传了什么文档、谁问了什么问题），
Chroma 向量库存"文档切片"（负责语义检索）——两个库各管一摊。
"""
import json
from collections import defaultdict
from queue import Queue, Empty

import pymysql
from pymysql.cursors import DictCursor

from app.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB,
)

# ===== 连接池：避免每次请求新建连接的开销 =====
# 课程项目并发不高，池大小设 5 够用；连接取走后用完归还，断连自动重建。
_POOL_SIZE = 5
_pool: Queue = Queue(maxsize=_POOL_SIZE)


def _new_conn():
    """新建一条连接。autocommit=True：每条 SQL 立即生效，不用手动 commit。"""
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


def get_conn():
    """从池里拿一条连接；池空就新建。拿到后先 ping 一下，断连就重建。"""
    try:
        conn = _pool.get_nowait()
        conn.ping(reconnect=True)
        return conn
    except Empty:
        return _new_conn()
    except Exception:
        # 池里的连接坏了，丢弃再新建
        try:
            conn.close()
        except Exception:
            pass
        return _new_conn()


def release_conn(conn):
    """用完归还连接池；池满就直接关闭。"""
    try:
        _pool.put_nowait(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


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
    release_conn(conn)

    # 旧表补列（列已存在报 1060，只忽略这个错误；其他错误照常抛出）
    for col_sql in [
        "ALTER TABLE qa_history ADD COLUMN source_doc TEXT",
        "ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64)",
    ]:
        try:
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute(col_sql)
            release_conn(conn)
        except pymysql.err.OperationalError as e:
            if e.args[0] != 1060:  # 1060 = Duplicate column name
                release_conn(conn)
                raise
            release_conn(conn)


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
    release_conn(conn)
    return uid


def get_user_by_username(username):
    """按用户名查用户完整记录，查不到返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    release_conn(conn)
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
    release_conn(conn)
    return doc_id


def get_document_by_hash(file_hash):
    """按文件哈希查文档，命中返回该行（重复上传检测用），没有返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename FROM documents WHERE file_hash = %s", (file_hash,))
        row = cur.fetchone()
    release_conn(conn)
    return row


def list_documents():
    """返回所有文档记录（不含 chroma_ids，那是内部数据）。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, uploader, chunks, created_at FROM documents ORDER BY id DESC"
        )
        rows = cur.fetchall()
    release_conn(conn)
    return rows


def get_document(doc_id):
    """按 id 查单条完整记录，查不到返回 None。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
    release_conn(conn)
    return row


def delete_document(doc_id):
    """删一行，返回是否真的删了。"""
    conn = get_conn()
    with conn.cursor() as cur:
        n = cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    release_conn(conn)
    return n > 0


def update_document_name(doc_id, new_filename):
    """修改文档文件名（用户上传时可能手滑写错，允许改名）。返回是否真的改了。"""
    conn = get_conn()
    with conn.cursor() as cur:
        n = cur.execute(
            "UPDATE documents SET filename = %s WHERE id = %s",
            (new_filename[:200], doc_id),
        )
    release_conn(conn)
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
    release_conn(conn)


def doc_question_stats():
    """互动看板数据：每个文档被提问几次 + 最近一次问题。
    优化：先一次性把 qa_history 按来源文件名建索引（O(qas)），
    再遍历 documents 查表（O(docs)），整体 O(n) 而非原来的 O(docs*qas)。"""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename, uploader FROM documents ORDER BY id")
        docs = cur.fetchall()
        cur.execute("SELECT question, source_doc, created_at FROM qa_history ORDER BY id DESC")
        qas = cur.fetchall()
    release_conn(conn)

    # 来源文件名 -> 命中的 qa 列表（qa_history 已按 id 倒序，所以列表也是时间倒序）
    file_hits = defaultdict(list)
    for q in qas:
        for fname in _parse_sources(q["source_doc"]):
            file_hits[fname].append(q)

    stats = []
    for d in docs:
        hits = file_hits.get(d["filename"], [])
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


def get_recent_activities(limit=10):
    """取最近 limit 条"活动时间线"——上传文档 + 提问 + 删除。
    多端互动的核心数据源：A 上传/提问，B 的前端轮询这个接口就能感知到。"""
    conn = get_conn()
    with conn.cursor() as cur:
        # 最近上传
        cur.execute(
            "SELECT 'upload' AS type, filename AS detail, uploader AS actor, created_at "
            "FROM documents ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        uploads = cur.fetchall()
        # 最近提问（只取前 50 条再内存排序，避免大表 SQL UNION 复杂度）
        cur.execute(
            "SELECT 'question' AS type, question AS detail, session_id AS actor, created_at "
            "FROM qa_history ORDER BY id DESC LIMIT %s",
            (limit * 5,),
        )
        questions = cur.fetchall()
    release_conn(conn)

    # 合并 + 按时间倒序
    all_rows = uploads + questions
    all_rows.sort(key=lambda r: r["created_at"], reverse=True)
    # 同一秒内多条合并？课程项目不用，直接去重（session_id 是 u:user:... 格式，提取用户名）
    seen_keys = set()
    activities = []
    for r in all_rows:
        # 去重：同一人同秒内的操作只保留一条
        key = (str(r["created_at"]), r["actor"], r["type"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        actor = r["actor"]
        if actor.startswith("u:"):   # session_id 格式 u:username:...
            actor = actor.split(":")[1]
        elif actor.startswith("ip:"):
            actor = actor.split(":")[1]
        activities.append({
            "type": r["type"],
            "actor": actor,
            "detail": r["detail"][:60],    # 截断长问题/文件名
            "time": str(r["created_at"]),
        })
        if len(activities) >= limit:
            break
    return activities
