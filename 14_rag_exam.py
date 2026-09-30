# 14_rag_exam.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# TODO: 补 import(loader/splitter/embedding/chroma/llm)
load_dotenv()

# 1. 加载文档
def load_documents(folder="./docs"):
    docs = []
    for name in os.listdir(folder):
        path = os.path.join(folder,name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt",".md"):
            loader = TextLoader(path,encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        docs.extend(loaded)
    return docs

# 2. 切片
docs = load_documents()
spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
spliter_chunks = spliter.split_documents(docs)
# 3. 向量化入库 + 检索器
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = Chroma.from_documents(
    spliter_chunks,
    embedding=embeddings,
    collection_name="day_14_rag_exam"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. LLM
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.7-plus",
    temperature=0.0
)

# 5. 测试问题 → 检索 → 拼prompt(带编号)
question = "没按时还书会怎么样"
print("====回答问题====")
context_parts = []
sources = []
raw = retriever.invoke(question)
for i,doc in enumerate(raw,1):
    source = doc.metadata.get("source","未知来源")
    source_name = os.path.basename(source)
    context_parts.append(f"[{i}] {doc.page_content}")
    sources.append(source_name)
context= "\n\n".join(context_parts)

prompt = f"""请你根据参考文档回答问题，要在关键信息后面标注引用编号[1]
参考文档
{context}
问题
{question}
回答："""
# 6. 生成回答 + 打印引用来源
answer = llm.invoke(prompt)
print("回答")
print(answer.content)
print("===引用来源===")
for i,name in enumerate(sources,1):
    print(f"[{i}] {name}")