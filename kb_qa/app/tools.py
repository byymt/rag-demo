"""工具模块：智能体可调用的 4 个 Python 工具（大作业必做第 2、7 项）。

每个工具都是"名称明确、用途独立、参数清晰"的真工具，不是拆动作凑数：
1. search_knowledge  资料检索  —— 知识库混合检索，返回带来源的片段
2. calculator        结果计算  —— ast 白名单四则运算（防 eval 注入）
3. save_note         成果输出  —— 把回答整理成 Markdown 笔记文件（outputs/）
4. save_memory       状态记忆  —— 把用户偏好/学习进度写入 JSON 长期记忆

用户上下文：uvicorn 多线程并发，用 contextvars 保存"当前提问的用户名"，
save_note/save_memory 由此知道给谁存。ContextVar 在线程池里随请求上下文拷贝，天然线程安全。
"""
import ast
import contextvars
import operator
import os
import re
from datetime import datetime

from app import memory
from app.config import OUTPUTS_DIR
from app.kb import search as kb_search

# 当前请求的用户名（agent.chat 入口设置）
current_user: contextvars.ContextVar = contextvars.ContextVar("current_user", default="anonymous")

# 每个工具的元信息：作业要求"为工具写好名称、用途和参数说明"
TOOL_SPECS = {
    "search_knowledge": "在知识库中检索课程资料片段。参数：检索关键词或问句。",
    "calculator": "计算数学表达式（+-*/%**）。参数：算式字符串，如 2*(3+4)/7。",
    "save_note": "把回答保存为 Markdown 学习笔记（成果文件）。参数：标题|内容。",
    "save_memory": "记住用户长期偏好/进度，存入 JSON，重启后仍生效。参数：键|值。",
}


def search_knowledge(query: str) -> str:
    """在知识库中检索资料，返回带来源文件名的片段。"""
    return kb_search(query)


def calculator(expression: str) -> str:
    """计算数学表达式的结果，输入是算式，例如 2*(3+4)/7。
    用 ast 解析只允许数字和 +-*/%** 运算，绝不调用 eval——
    eval 会执行任意代码，若课件藏 prompt injection 诱导调用恶意表达式就完了。"""
    _allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _allowed_ops:
            return _allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _allowed_ops:
            return _allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("只支持数字和 +-*/%** 运算")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree))
    except ZeroDivisionError:
        return "计算失败: 除数不能为 0"
    except Exception as e:
        return f"计算失败: {e}"


def save_note(title_content: str) -> str:
    """把整理好的回答保存为 Markdown 学习笔记（可直接打开查看的成果文件）。
    参数格式"标题|内容"，竖线分隔；文件名带时间戳防覆盖，存到 outputs/。"""
    parts = title_content.split("|", 1)
    title = (parts[0] if len(parts) > 0 else "").strip() or "学习笔记"
    content = (parts[1] if len(parts) > 1 else title_content).strip()
    if not content:
        return "保存失败: 笔记内容为空"
    # 模型在单行"行动输入"里只能写字面量 \n，落盘前还原成真换行
    content = content.replace("\\n", "\n")

    # 文件名消毒：只留安全字符，防止特殊字符在文件系统上报错
    safe_title = re.sub(r'[\\/:*?"<>|\s]+', "_", title)[:40]
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    username = current_user.get() or "anonymous"
    filename = f"{ts}_{safe_title}_{username}.md"
    path = os.path.join(OUTPUTS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"> 由 kb_qa 智能体整理 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 用户 {username}\n\n")
        f.write(content + "\n")
    return f"已保存笔记: outputs/{filename}"


def save_memory(key_value: str) -> str:
    """把用户的长期偏好/学习进度写入 JSON 记忆文件。
    参数格式"键|值"；重启后 main.py 加载并注入系统提示词，跨会话生效。"""
    parts = key_value.split("|", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return "保存失败: 参数格式应为 键|值，例如 进度|已完成第3章复习"
    username = current_user.get() or "anonymous"
    ok = memory.update_user_memory(username, parts[0].strip(), parts[1].strip())
    return f"已记住 ({username}): {parts[0].strip()} = {parts[1].strip()}" if ok else "保存失败"


# Agent 运行时的工具表：名字 -> 函数
TOOLS = {
    "search_knowledge": search_knowledge,
    "calculator": calculator,
    "save_note": save_note,
    "save_memory": save_memory,
}
