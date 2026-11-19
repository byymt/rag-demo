"""结构化日志：记录智能体执行的关键信息（大作业必做第 6 项）。

记录四类内容，输出到 logs/agent.log + 控制台：
1. 模型调用：请求了哪个模型、耗时、Token 用量（输入/输出/总计）
2. 工具调用：工具名称、主要参数、结果摘要
3. 任务耗时：每轮 / 每次完整问答的总耗时
4. 异常：检索失败、工具报错等

用 logging 而不是 print：print 进不了文件、没法按级别过滤、
多线程并发时会互相截断——生产代码统一用 logging。
"""
import logging
import os
import re

from app.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("kbqa.agent")
logger.setLevel(logging.INFO)
if not logger.handlers:  # 防止重复加载（uvicorn reload）时 handler 翻倍
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    _fh = logging.FileHandler(
        os.path.join(LOG_DIR, "agent.log"), encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    # 控制台只放 WARNING 以上，避免刷屏（INFO 全量进文件）
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


def _clean(text: str, limit: int = 120) -> str:
    """压缩成单行并截断——日志一行一条，方便 grep。"""
    one_line = re.sub(r"\s+", " ", str(text)).strip()
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def log_model_call(step: int, model: str, seconds: float, usage: dict) -> None:
    """模型调用日志：模型名、第几轮、耗时、Token 用量。"""
    u = usage or {}
    logger.info(
        "模型调用 round=%d model=%s 耗时=%.2fs tokens(输入=%s 输出=%s 总计=%s)",
        step, model, seconds,
        u.get("prompt_tokens", "?"), u.get("completion_tokens", "?"),
        u.get("total_tokens", "?"),
    )


def log_tool_call(tool: str, action_input: str, result: str, seconds: float) -> None:
    """工具调用日志：名称、参数、结果摘要、耗时。"""
    logger.info(
        "工具调用 tool=%s 参数=%s 耗时=%.2fs 结果=%s",
        tool, _clean(action_input, 80), seconds, _clean(result),
    )


def log_task(question: str, seconds: float, rounds: int, note: str = "") -> None:
    """完整问答任务日志：问题、总耗时、走了几轮。"""
    logger.info("任务完成 问题=%s 总耗时=%.2fs 推理轮数=%d %s", _clean(question, 60), seconds, rounds, note)


def log_error(where: str, err: Exception) -> None:
    logger.error("异常 position=%s err=%s", where, err)
