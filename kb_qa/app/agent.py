"""Agent 模块：LangGraph 图 + 工具表，对外只暴露 chat()。"""
from typing import TypedDict
import re

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from app.config import (
    CHAT_MODEL, TEMPERATURE,
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
)
from app.kb import search as kb_search

llm = ChatOpenAI(
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
)
def search_knowledge(query: str) -> str:
    """在知识库中检索资料，返回带来源文件名的片段。"""
    return kb_search(query)

def calculator(expression: str) -> str:
    """计算数学表达式的结果，输入是算式，例如 2*(3+4)/7。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算失败: {e}"

TOOLS = {"search_knowledge": search_knowledge, "calculator": calculator}

REACT_PROMPT = """你是一个知识库问答助手，你有两个工具：

工具1: search_knowledge
作用: 在知识库中检索资料，参数是检索用的关键词或问句。

工具2: calculator
作用: 计算数学表达式，参数是一个算式，例如 2*(3+4)/7。

请严格按格式输出，每轮只输出一种内容：

思考: 你当前的推理（是否需要检索、检索什么）
行动: search_knowledge
行动输入: 检索的关键词或问句

当你认为资料已足够回答问题时，停止检索并输出：
思考: 我已经掌握足够的资料
最终答案: 优先依据检索到的资料作答。若资料未覆盖问题，先说明"知识库资料未覆盖该问题"，再基于通用知识给出参考回答，并在回答末尾标注"（注：以上补充内容来自模型通用知识，非知识库课件，请以教材为准）"。禁止把通用知识伪装成课件内容。

重要: "观察"内容由系统填写，你不要自己编造"观察:"。

检索参考（已结合上下文改写，行动输入应与其保持一致）: {search_query}

历史对话:
{chat_history}

问题: {question}

{scratchpad}"""

class AgentState(TypedDict):
    question:str
    search_query:str
    chat_history:str
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
    except Exception:
        return question

def extract(text,pattern):
    m = re.search(pattern,text)
    return m.group(1).strip() if m else ""

def agent_node(state:AgentState):
    """大脑：组装提示词 → 调模型 → 解析输出 → 决定下一步。"""
    Max_steps=5
    if state["steps"] >= Max_steps:
        return {"answer":"已到达最大轮数，仍无法给出答案。"}

    prompt = REACT_PROMPT.format(
        question=state["question"],
        search_query=state.get("search_query", "") or state["question"],
        chat_history = state.get("chat_history",""),
        scratchpad = state["scratchpad"]
    )
    text = llm.invoke(prompt).content.strip()
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
    """手：查 TOOLS 表执行工具，把观察写回 scratchpad。"""
    tool = TOOLS.get(state["action"])
    if tool is None:
        observation = f"没有名为{state['action']}的工具，可用工具:{list(TOOLS)}"
    else:
        observation = tool(state["action_input"])
    return {"scratchpad":state["scratchpad"]+f"观察:{observation}\n"}

def save_node(state:AgentState):
    """档案员：把本轮问答写进 chat_history，实现跨轮记忆。"""
    return {"chat_history":state.get("chat_history","")+f"用户:{state['question']}\n助手:{state['answer']}\n"}

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

def chat(question,thread_id):
    """每轮重置本轮字段，保留 chat_history，实现多轮记忆。追问先改写再入库检索。"""
    config = {"configurable":{"thread_id":thread_id}}
    prev = graph.get_state(config).values or {}
    history = prev.get("chat_history","") or ""
    rewritten = rewrite_question(question, history)
    result = graph.invoke(
        {
            "answer":"",
            "question":question,
            "search_query":rewritten,
            "scratchpad":"",
            "action":"",
            "action_input":"",
            "steps":0
        },
        config = config
    )
    return result["answer"]


def extract_final(raw):
    """从流式拼出的全文里截取"最终答案:"之后的部分（给 qa_history 存干净答案用）。"""
    if "最终答案" in raw:
        return re.split(r"最终答案[:：]", raw)[-1].strip()
    return raw.strip()


def chat_stream(question, thread_id):
    """流式版 chat：逐段产出模型的原始 ReAct 输出，前端按"最终答案:"切分显示。
    思考/行动也实时流出——演示时能看到 Agent 的推理过程。"""
    config = {"configurable":{"thread_id":thread_id}}
    prev = graph.get_state(config).values or {}
    history = prev.get("chat_history","") or ""
    rewritten = rewrite_question(question, history)
    for msg, _meta in graph.stream(
        {
            "answer":"",
            "question":question,
            "search_query":rewritten,
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

if __name__ == "__main__":
    print(chat("校园卡丢了怎么补办？", "t1"))
    print(chat("那要带什么材料？", "t1"))          # 多轮记忆
    print(chat("帮我算一下 128 * 365 等于多少", "t2"))  # calculator 工具