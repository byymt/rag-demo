import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.7-plus",
    temperature=0.0
)

def load_document(folder = "./docs"):
    docs=[]
    for name in os.listdir(folder):
        path = os.path.join(folder,name)
        ext = os.path.splitext(name)[1].lower()
        if ext ==".pdf":
            loaded = PyPDFLoader(path)
        elif ext in (".md",".txt"):
            loaded = TextLoader(path,encoding="utf-8")
        else:
            continue
        loader = loaded.load()
        docs.extend(loader)
    return docs

docs = load_document()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3"
)
vectorstore  = Chroma(
    embedding_function=embeddings,
    collection_name="Day16_react_agent",
    persist_directory="./chroma"
)
if len(vectorstore.get()["ids"])==0:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    print(f"【知识库】首次入库 {len(chunks)}个切片")

retriever = vectorstore.as_retriever(search_kwargs ={"k":3})

def search_knowledge(query:str)->str:
    hits = retriever.invoke(query)
    if not hits:
        return "没有检索到相关资料"
    parts=[]
    for i,doc in enumerate(hits,1):
        source = os.path.basename(doc.metadata.get("source","未知来源"))
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

问题: {question}

{history}"""

def extract(text,pattern):
    """用正则从模型输出中取出某个字段的值，取不到返回空字符串"""
    m=re.search(pattern,text)
    return m.group(1).strip() if m else ""

def run_agent(question,max_step=5):
    history=""
    for step in range(1,max_step+1):
        prompt = REACT_PROMPT.format(question=question,history=history)
        text = llm.invoke(prompt).content.strip()
        text = re.split(r"观察[:：]",text)[0].strip()
        print(f"=====第{step}轮=====")
        print(text)
        if "最终答案" in text:
            return re.split(r"最终答案[:：]",text)[-1].strip()
        action = extract(text,r"行动[:：]\s*(.+)")
        action_input = extract(text,r"行动输入[:：]\s*(.+)")

        if not action:
            return text
        tool =TOOLS.get(action)
        if tool is None:
            observation = f"没有名为{action}的工具，可用工具:{list(TOOLS)}"
        else:
            observation = tool(action_input)
        history += f"{text}\n观察：{observation}\n"

    return "已达到最大轮数，仍无法给出最终答案。"


if __name__ == "__main__":
    questions = [
        "校园卡丢了怎么补办，要多少钱？",
        "图书馆普通图书能借多久？逾期一天罚多少钱？",
        "怎么炒番茄炒蛋？",
    ]
    for q in questions:
        print("\n" + "=" * 50)
        print("问:", q)
        print("答:", run_agent(q))


