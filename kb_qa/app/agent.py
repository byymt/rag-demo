"""Agent 模块：LangGraph 图 + 工具表，对外只暴露 chat()。

大作业结构对照：
- 工具函数在 app/tools.py（名称/用途/参数说明见 TOOL_SPECS）
- 系统提示词在项目根 system_prompt.md（职责、工具规则、输出要求）
- 本模块负责：模型调用、ReAct 双层循环编排、日志埋点

双层循环：
- 外层 = Web 服务连续对话（main.py 的 /chat 接口，thread_id 区分会话）
- 内层 = LangGraph 的 agent→tool→agent… 迭代，直到给出最终答案或到达最大轮数
"""
import os
import re
import time
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from app.config import (
    BASE_DIR, CHAT_MODEL, TEMPERATURE,
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
)
from app.logger import log_model_call, log_tool_call, log_task, log_error
from app.tools import TOOLS, TOOL_SPECS, current_user

llm = ChatOpenAI(
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
)

# ===== 系统提示词外置（system_prompt.md）：职责/工具规则/输出要求都在里面 =====
# .format() 填五个变量：memory（长期记忆）/ search_query（改写后问题）/ chat_history / question / scratchpad
with open(os.path.join(BASE_DIR, "system_prompt.md"), encoding="utf-8") as _f:
    REACT_PROMPT = _f.read()

MAX_STEPS = 5  # 内层循环最大轮数：防模型陷入"检索→不满意→再检索"死循环


class AgentState(TypedDict):
    question:str
    search_query:str
    chat_history:str
    memory:str          # 长期记忆文本（重启后仍生效的那部分）
    scratchpad:str
    action:str
    action_input:str
    steps:int
    answer:str


# ===== 优化1：查询改写 =====
# 多轮对话的痛点："那要带什么材料"直接拿去检索效果差。
# 先让 LLM 结合历史把追问改写成独立问题，再让 Agent 拿它去检索。
REWRITE_PROMPT = """根据历史对话，把用户的最新问题改写成一个不依赖上下文、可以直接拿去知识库检索的独立问题。只输出改写后的问题，不要解释。

历史对话:
{chat_history}

最新问题: {question}

独立问题:"""


def rewrite_question(question, chat_history):
    """首轮不改写；有历史时改写，失败则退回原问题。"""
    if not chat_history:
        return question
    try:
        out = llm.invoke(REWRITE_PROMPT.format(chat_history=chat_history, question=question))
        return out.content.strip() or question
    except Exception as e:
        log_error("查询改写", e)
        return question


def extract(text,pattern):
    m = re.search(pattern,text)
    return m.group(1).strip() if m else ""


def _token_usage(resp) -> dict:
    """从 LangChain 的 AIMessage 里取 Token 用量（openai 兼容字段，取不到就空表）。"""
    meta = getattr(resp, "response_metadata", None) or {}
    return meta.get("token_usage") or {}


def agent_node(state:AgentState):
    """大脑：组装提示词 → 调模型 → 解析输出 → 决定下一步。"""
    if state["steps"] >= MAX_STEPS:
        return {"answer":"已到达最大轮数，仍无法给出答案。"}

    prompt = REACT_PROMPT.format(
        question=state["question"],
        search_query=state.get("search_query", "") or state["question"],
        chat_history = state.get("chat_history",""),
        memory = state.get("memory","") or "（暂无）",
        scratchpad = state["scratchpad"]
    )
    t0 = time.time()
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        log_model_call(state["steps"] + 1, CHAT_MODEL, time.time() - t0, _token_usage(resp))
    except Exception as e:
        log_error("模型调用", e)
        return {"answer":"模型调用失败，请稍后重试。", "steps": state["steps"] + 1}

    text = re.split(r"观察[:：]",text)[0].strip()
    print(f"-----第{state['steps']+1}轮------")
    print(text)

    if "最终答案" in text:
        answer = re.split(r"最终答案[:：]",text)[-1].strip()
        return {"answer":answer,"steps":state["steps"]+1}

    action =extract(text,r"行动[:：]\s*(.+)")
    action_input = extract(text,r"行动输入[:：]\s*(.+)")
    if action:
        return {
            "action":action,
            "action_input":action_input,
            "scratchpad":state["scratchpad"]+text+"\n",
            "steps":state["steps"]+1
        }
    return {
        "answer":text,
        "steps":state["steps"]+1
    }


def tool_node(state:AgentState):
    """手：查 TOOLS 表执行工具，把观察写回 scratchpad，并记录工具日志。"""
    tool = TOOLS.get(state["action"])
    t0 = time.time()
    if tool is None:
        observation = f"没有名为{state['action']}的工具，可用工具:{list(TOOLS)}"
    else:
        try:
            observation = tool(state["action_input"])
        except Exception as e:
            log_error(f"工具{state['action']}执行", e)
            observation = f"工具执行出错: {e}"
    log_tool_call(state["action"], state["action_input"], observation, time.time() - t0)
    return {"scratchpad":state["scratchpad"]+f"观察:{observation}\n"}


def save_node(state:AgentState):
    """档案员：把本轮问答写进 chat_history，实现跨轮记忆。
    只保留最近 10 轮（20 行），防止多轮对话把 prompt 撑爆 token 上限。"""
    history = state.get("chat_history","") + f"用户:{state['question']}\n助手:{state['answer']}\n"
    lines = [l for l in history.split("\n") if l]
    if len(lines) > 20:
        history = "\n".join(lines[-20:]) + "\n"
    return {"chat_history": history}


def route(state:AgentState):
    """交通指挥：有答案去 save，没答案去 tool。"""
    if state["answer"]:
        return "save"
    return "tool"


builder = StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tool",tool_node)
builder.add_node("save",save_node)

builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",route,{"tool":"tool","save":"save"})
builder.add_edge("tool","agent")
builder.add_edge("save",END)

memory = MemorySaver()
graph =builder.compile(checkpointer=memory)


def _prepare(question, thread_id):
    """chat / chat_stream 共用的前置：查历史、改写问题、设用户上下文。"""
    config = {"configurable":{"thread_id":thread_id}}
    prev = graph.get_state(config).values or {}
    history = prev.get("chat_history","") or ""
    rewritten = rewrite_question(question, history)
    return config, history, rewritten


def chat(question, thread_id, username="anonymous", memory_text=""):
    """每轮重置本轮字段，保留 chat_history，实现多轮记忆。
    追问先改写再入库检索；长期记忆注入提示词；工具经 current_user 知道给谁存。"""
    token = current_user.set(username)
    t0 = time.time()
    try:
        config, history, rewritten = _prepare(question, thread_id)
        result = graph.invoke(
            {
                "answer":"",
                "question":question,
                "search_query":rewritten,
                "memory":memory_text,
                "scratchpad":"",
                "action":"",
                "action_input":"",
                "steps":0
            },
            config = config
        )
        rounds = result["steps"]
        note = f"memory={'有' if memory_text else '无'}"
        log_task(question, time.time() - t0, rounds, note)
        return result["answer"]
    finally:
        current_user.reset(token)


def extract_final(raw):
    """从流式拼出的全文里截取"最终答案:"之后的部分（给 qa_history 存干净答案用）。"""
    if "最终答案" in raw:
        return re.split(r"最终答案[:：]", raw)[-1].strip()
    return raw.strip()


def chat_stream(question, thread_id, username="anonymous", memory_text=""):
    """流式版 chat：逐段产出模型的原始 ReAct 输出，前端按"最终答案:"切分显示。
    思考/行动也实时流出——演示时能看到 Agent 的推理过程。"""
    token = current_user.set(username)
    t0 = time.time()
    try:
        config, history, rewritten = _prepare(question, thread_id)
        for msg, _meta in graph.stream(
            {
                "answer":"",
                "question":question,
                "search_query":rewritten,
                "memory":memory_text,
                "scratchpad":"",
                "action":"",
                "action_input":"",
                "steps":0
            },
            config = config,
            stream_mode = "messages",
        ):
            # 只放行模型的文字 token（跳过工具消息和空块）
            if type(msg).__name__ in ("AIMessageChunk", "AIMessage"):
                content = msg.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    text = "".join(part.get("text","") for part in content if isinstance(part, dict))
                    if text:
                        yield text
        log_task(question, time.time() - t0, -1, "流式")
    finally:
        current_user.reset(token)


if __name__ == "__main__":
    print(chat("校园卡丢了怎么补办？", "t1"))
    print(chat("那要带什么材料？", "t1"))          # 多轮记忆
    print(chat("帮我算一下 128 * 365 等于多少", "t2"))  # calculator 工具
    print(chat("记住我在准备软件工程期末考试", "t1"))    # save_memory 工具
    print(chat("把刚才的回答整理成笔记保存", "t1"))      # save_note 工具
