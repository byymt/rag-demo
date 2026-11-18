"""认证模块：注册 / 登录 / token 校验 / 登出。

两个核心设计（面试可讲）：
1. 密码绝不存明文：存 "盐$哈希"，哈希 = pbkdf2_hmac(sha256, 盐, 密码, 10万次)。
   盐是每个用户随机的字符串——即使两个用户密码相同，库里存的也不同。
   用 pbkdf2 而非单次 sha256：单次 sha256 太快，GPU 每秒数十亿次；
   pbkdf2 故意慢 10 万次，暴力破解成本指数级上升。
2. token 是登录后发的随机编号：谁持 token 谁已登录。
   存在内存 dict 里，带 7 天过期时间；可登出主动失效。
   （课程项目够用；生产环境用 Redis 存会话或签 JWT，面试可展开）
"""
import hashlib
import time
import uuid

from app import db

# token -> {"user": 用户名, "exp": 过期时间戳}
_tokens = {}
TOKEN_TTL = 7 * 24 * 3600  # 7 天
PBKDF2_ITERATIONS = 100_000


def _hash_password(password: str, salt: str) -> str:
    """pbkdf2_hmac(sha256, 盐, 密码, 10万次)，返回 64 位十六进制。"""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()


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
    # 防御：库里哈希格式被破坏时不抛异常，而是友好提示
    try:
        salt, expected = user["password_hash"].split("$", 1)
    except (ValueError, AttributeError):
        return False, "账户数据异常，请联系管理员"
    if _hash_password(password, salt) != expected:
        return False, "密码错误"
    token = uuid.uuid4().hex
    _tokens[token] = {"user": username, "exp": time.time() + TOKEN_TTL}
    return True, token


def current_user(token: str):
    """校验 token：有效返回用户名，无效/过期返回 None。"""
    info = _tokens.get(token or "")
    if not info:
        return None
    if time.time() > info["exp"]:
        _tokens.pop(token, None)  # 过期清理
        return None
    return info["user"]


def logout(token: str):
    """登出：从内存里删掉 token，使其立即失效。"""
    _tokens.pop(token or "", None)


def is_admin(username: str) -> bool:
    """查库判断是否管理员（审核接口用）。"""
    row = db.get_user_by_username(username)
    return bool(row and row.get("is_admin"))
