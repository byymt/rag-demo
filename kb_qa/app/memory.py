"""长期记忆模块：把用户偏好/学习进度持久化到 JSON，重启后仍可用。

大作业必做第 5 项要求"两种记忆"：
- 短期记忆：当前对话历史（agent.py 的 save_node + LangGraph checkpointer）
- 长期记忆：本模块——存 JSON 文件，程序重启后读取并注入提示词，
  提供"更新"（Agent 调 save_memory 工具 / main.py 的 clear 接口）与"清除"方法。

存储结构（data/memory.json）：
{
  "ymt": {"facts": {"进度": "已完成第3章"}, "updated_at": "2026-11-19 12:00:00"}
}
课程项目用单文件 + threading.Lock（并发量低够用）；生产应换数据库或 Redis。
"""
import json
import os
import threading
from datetime import datetime

from app.config import DATA_DIR

_MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
_lock = threading.Lock()


def _read_all() -> dict:
    """读全量记忆文件；文件不存在或损坏返回空表（损坏不抛异常，避免整个应用挂掉）。"""
    if not os.path.exists(_MEMORY_FILE):
        return {}
    try:
        with open(_MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def _write_all(data: dict) -> None:
    """全量写回。先写临时文件再改名，避免写一半崩溃留下坏文件。"""
    os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
    tmp = _MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _MEMORY_FILE)


def update_user_memory(username: str, key: str, value: str) -> bool:
    """更新一条长期记忆（存在则覆盖，不存在则新增）。"""
    if not username or not key:
        return False
    with _lock:
        data = _read_all()
        user = data.setdefault(username, {"facts": {}, "updated_at": ""})
        user["facts"][str(key)[:50]] = str(value)[:500]
        user["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_all(data)
    return True


def clear_user_memory(username: str) -> bool:
    """清除某用户的全部长期记忆。返回是否确实删了东西。"""
    with _lock:
        data = _read_all()
        if username in data:
            del data[username]
            _write_all(data)
            return True
    return False


def load_user_memory(username: str) -> str:
    """把用户的长期记忆读成一段文本，注入系统提示词；没有任何记忆返回空串。"""
    data = _read_all().get(username)
    if not data or not data.get("facts"):
        return ""
    lines = [f"- {k}: {v}" for k, v in data["facts"].items()]
    return "\n".join(lines)
