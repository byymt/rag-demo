"""认证模块：注册 / 登录 / token 校验。

两个核心设计（面试可讲）：
1. 密码绝不存明文：存 "盐$哈希"，哈希 = sha256(盐 + 密码)。
   盐是每个用户随机的字符串——即使两个用户密码相同，库里存的也不同。
2. token 是登录后发的随机编号：谁持 token 谁已登录。
   存在内存 dict 里，重启后需重新登录（课程项目够用；
   生产环境用 Redis 存会话或签 JWT，面试可展开）。
"""
import hashlib
import uuid

from app import db

# token -> username 的映射（进程内存）
_tokens = {}


def _hash_password(password: str, salt: str) -> str:
    """sha256(盐 + 密码)，返回 64 位十六进制。"""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register(username: str, password: str):
    """注册。返回 (是否成功, 提示信息)。"""
    if not username or not password:
        return False, "用户名和密码不能为空"
    if len(password) < 6:
        return False, "密码至少 6 位"
    if len(username) > 50:
        return False, "用户名过长"
    if db.get_user_by_username(username):
        return False, "用户名已被占用"
    salt = uuid.uuid4().hex[:16]
    stored = f"{salt}${_hash_password(password, salt)}"
    db.create_user(username, stored)
    return True, "注册成功"


def login(username: str, password: str):
    """登录。返回 (是否成功, token 或错误提示)。"""
    user = db.get_user_by_username(username)
    if user is None:
        return False, "用户不存在"
    salt, expected = user["password_hash"].split("$", 1)
    if _hash_password(password, salt) != expected:
        return False, "密码错误"
    token = uuid.uuid4().hex
    _tokens[token] = username
    return True, token


def current_user(token: str):
    """校验 token：有效返回用户名，无效返回 None。"""
    return _tokens.get(token or "")
