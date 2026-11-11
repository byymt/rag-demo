"""知识库模块：读文件、切片、入库、检索——知识库的一切操作都从这里走。"""
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, RETRIEVER_K

# ===== 模块级：创建一次，全模块共用 =====
embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name="kb_qa",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


# ===== 函数1：读单个文件（上传接口以后会传一个路径进来）=====
def load_any_file(path):
    """按扩展名选 loader，返回文档列表。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    return loader.load()


# ===== 函数2：入库（上传接口调用）=====
def add_file_to_kb(path):
    """读文件 → 切片 → 入库 → 返回 Chroma 切片 id 列表（删除文档时要用）。"""
    docs = load_any_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    ids = vectorstore.add_documents(chunks)
    return ids


# ===== 函数2b：按 id 删切片（删除文档接口调用）=====
def delete_by_ids(ids):
    """按 Chroma id 删除切片；空列表直接跳过。"""
    if ids:
        vectorstore.delete(ids=ids)


# ===== 函数3：检索（Agent 的工具以后调它）=====
# 最近一次检索命中的来源文件名列表（chat 接口记录"问答命中哪个文档"用）
last_sources = []


def search(query):
    """检索 RETRIEVER_K 条，返回带来源的拼接文本；检索不到返回提示语。"""
    global last_sources
    hits = vectorstore.similarity_search(query, k=RETRIEVER_K)
    last_sources = []  # 每次检索先清空，避免残留上一次的结果
    if not hits:
        return "没有检索到相关资料"
    parts = []
    for i, doc in enumerate(hits, 1):
        source = os.path.basename(doc.metadata.get("source", "未知来源"))
        last_sources.append(source)
        parts.append(f"[{i}] 来源:{source}\n{doc.page_content}")
    return "\n\n".join(parts)

