import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,END,START
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.7-plus",
    temperature= 0.0
)

def load_document(folder='./docs'):
    docs = []
    for name in os.listdir(folder):
        path = os.path.join(folder,name)
        ext = os.path.splitext(name)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(path)
        elif ext in ('.md',".txt"):
            loader = TextLoader(path,encoding ="utf-8")
        else:
            continue
        loaded = loader.load()
        docs.extend(loaded)
    return docs

embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = Chroma(
    collection_name="day17_langgraph",
    embedding_function=embeddings,
    persist_directory="./chroma"
)
if len(vectorstore.get()["ids"])==0:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(load_document())
    vectorstore.add_documents(chunks)
    print(f"【知识库】首次入库{len(chunks)}个切片")
retriever = vectorstore.as_retriever(search_kwargs = {"k":3})

def search_knowledge(query:str)->str:
    """在知识库中检索资料，返回带来源文件名的片段。"""
    parts = []
    hits = retriever.invoke(query)
    if not hits:
        return "没有检索到相关资料"
    for i ,doc in enumerate(hits,1):
        source =os.path.basename(doc.metadata.get("source","未知来源"))
        parts.append(f"[{i}] 来源:{source}\n{doc.page_content}")
    return "\n\n".join(parts)

TOOLS = {"search_knowledge":search_knowledge}

REACT_PROMPT = """你是一个可以自主检索知识库的助手，你有且只有一个工具：

工具名: search_knowledge
作用: 在知识库中检索资料，参数是检索用的关键词或问句。

请严格按格式输出，每轮只输出一种内容：

思考: 你当前的推理（是否需要检索、检索什么）
行动: search_knowledge
行动输入: 检索的关键词或问句

当你认为资料已足够回答问题时，停止检索并输出：
思考: 我已经掌握足够的资料
最终答案: 只能依据检索到的资料作答；资料里没有的就回答"根据现有资料无法回答"，禁止编造。

重要: "观察"内容由系统填写，你不要自己编造"观察:"。

历史对话:
{chat_history}

问题: {question}

{scratchpad}"""

class AgentState(TypedDict):
    question:str
    chat_history:str
    scratchpad:str
    action:str
    action_input:str
    steps:int
    answer:str

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
    """每轮重置本轮字段，保留 chat_history，实现多轮记忆。"""
    result = graph.invoke(
        {
            "answer":"",
            "question":question,
            "scratchpad":"",
            "action":"",
            "action_input":"",
            "steps":0
        },
        config = {"configurable":{"thread_id":thread_id}}
    )
    return result["answer"]

if __name__ =="__main__":

    print(chat("校园卡丢了怎么补办，要多少钱？","multi"))
    print(chat("那要带什么材料","multi"))

    print(chat("那要带什么材料？", "other"))
