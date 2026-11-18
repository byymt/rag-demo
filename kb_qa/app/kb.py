"""知识库模块：读文件、切片、入库、检索——知识库的一切操作都从这里走。"""
import os
import re
import threading

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

from app.config import (
    CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, RETRIEVER_K,
    SIM_THRESHOLD, VECTOR_K, RRF_K, BM25_STRONG_SCORE,
)

# ===== 模块级：创建一次，全模块共用 =====
# hnsw:space=cosine：用余弦距离而不是默认 L2。
# L2 的问题：相似度换算（1-距离）失真，相关片段才 0.3 左右，和无关片段拉不开差距；
# cosine 下相关片段 0.7+、无关 0.4 以下，阈值才有清晰的分界线
embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name="kb_qa",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)


# ===== 函数1：读单个文件（上传接口以后会传一个路径进来）=====
def _load_pptx(path):
    """提取 PPTX 每页文字。选 python-pptx 而不是 langchain 的
    UnstructuredPowerPointLoader：后者依赖重量级 unstructured 库，安装慢且
    Windows 上易出问题；提取文字这一件事用 python-pptx 十几行就够了。"""
    from pptx import Presentation
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides, 1):
        texts = [shape.text for shape in slide.shapes if shape.has_text_frame and shape.text.strip()]
        if texts:
            docs.append(Document(page_content="\n".join(texts), metadata={"source": path, "page": i}))
    return docs


def load_any_file(path):
    """按扩展名选 loader，返回文档列表。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext == ".pptx":
        return _load_pptx(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {ext}（支持 pdf/docx/pptx/txt/md）")
    return loader.load()


# ===== 函数2：入库（上传接口调用）=====
def add_file_to_kb(path):
    """读文件 → 切片 → 入库 → 返回 Chroma 切片 id 列表（删除文档时要用）。"""
    global _bm25
    docs = load_any_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    ids = vectorstore.add_documents(chunks)
    _bm25 = None  # 新切片入库，强制下次重建 BM25 索引
    return ids


# ===== 函数2b：按 id 删切片（删除文档接口调用）=====
def delete_by_ids(ids):
    """按 Chroma id 删除切片；空列表直接跳过。"""
    global _bm25
    if ids:
        vectorstore.delete(ids=ids)
        _bm25 = None  # 切片变了，强制下次重建 BM25 索引


# ===== 函数3：检索（Agent 的工具以后调它）=====
# 最近一次检索命中的来源文件名列表（chat 接口记录"问答命中哪个文档"用）。
# 用 threading.local 而非全局变量：FastAPI 的 def 接口跑在线程池里，
# 多个请求并发时全局变量会被互相覆盖（A 的回答挂 B 的来源）。
# 线程本地存储保证每个请求线程各有一份，互不干扰。
_thread_local = threading.local()


def _get_sources():
    if not hasattr(_thread_local, "last_sources"):
        _thread_local.last_sources = []
    return _thread_local.last_sources


def _clear_sources():
    _thread_local.last_sources = []


def __getattr__(name):
    """模块级属性钩子：kb.last_sources 返回当前线程的来源列表（真实 list，可直接 json 序列化）。
    这样 main.py 不用改读取逻辑，且每个线程拿到的都是自己那份。"""
    if name == "last_sources":
        return _get_sources()
    raise AttributeError(f"module 'app.kb' has no attribute {name!r}")


def search(query):
    """混合检索 + 阈值过滤，返回带来源的拼接文本；检索不到返回提示语。"""
    _clear_sources()
    docs = hybrid_search(query, k=RETRIEVER_K)
    if not docs:
        return "没有检索到相关资料"
    parts = []
    sources = _get_sources()
    for i, doc in enumerate(docs, 1):
        name = os.path.basename(doc.metadata.get("source", "未知来源"))
        page = doc.metadata.get("page")  # pptx/pdf 切片带页码，md/txt 没有
        loc = f"{name}(第{page}页)" if page else name
        sources.append({"name": name, "page": page})
        parts.append(f"[{i}] 来源:{loc}\n{doc.page_content}")
    return "\n\n".join(parts)


# ===== 优化2：混合检索（BM25 关键词 + 向量语义，RRF 融合）=====
from types import SimpleNamespace

from rank_bm25 import BM25Okapi
import jieba

# BM25 索引缓存：切片数量变了才重建（课程项目数据量小，全量重建足够快）
_bm25 = None
_bm25_ids = []    # 与 BM25 语料按位置对应的 Chroma 切片 id
_bm25_docs = []   # 与语料对应的文档对象（只含 page_content 和 metadata）


def _get_bm25():
    """从 Chroma 全量切片构建（或取缓存）BM25 索引。空库返回 (None, [], [])。"""
    global _bm25, _bm25_ids, _bm25_docs
    data = vectorstore.get(include=["documents", "metadatas"])
    ids = data["ids"]
    if not ids:
        return None, [], []
    if _bm25 is not None and set(_bm25_ids) == set(ids):
        return _bm25, _bm25_ids, _bm25_docs
    docs = [
        SimpleNamespace(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    _bm25 = BM25Okapi([jieba.lcut(d.page_content) for d in docs])
    _bm25_ids = ids
    _bm25_docs = docs
    return _bm25, _bm25_ids, _bm25_docs


def _rewrite_query(query: str) -> str:
    """短问法规则改写：课件写的是"类图的作用是…"，用户问的是"什么是类图"，
    两种措辞咬合差（本地实测 BM25 0.95 vs 1.58，向量同理），导致定义型问题被
    防幻觉阈值误杀。把问法改写成贴近课件原文的措辞，比调低阈值更安全——
    阈值放松会伤库外拒答，改写只是让真问题更容易过门。"""
    q = query.strip().rstrip("？?！!。. ")
    m = re.fullmatch(r"什么是(.+)", q)
    if m:
        return f"{m.group(1)}的定义和作用"
    m = re.fullmatch(r"(.+?)和(.+?)有什么区别", q)
    if m:
        return f"{m.group(1)}和{m.group(2)}的区别 比较"
    return query


def hybrid_search(query, k=RETRIEVER_K):
    """向量检索管"语义相近"，BM25 管"关键词精确命中"，两路结果用 RRF 融合排序取 top-k。

    放行规则（防幻觉的关键）：向量路有片段过阈值 → 正常返回；
    向量路全灭时，只有 BM25 出现强关键词信号（≥BM25_STRONG_SCORE）才放行——
    避免无关问题被 BM25 的弱匹配（单字碰上语料）带出不相干的片段。
    """
    candidates = {}  # chroma_id -> {"doc":..., "rrf":融合分}
    has_vec_hit = False
    query = _rewrite_query(query)  # 定义型/对比型短问法先改写，再进两路检索

    # 第 1 路：向量检索（带分数）。Chroma 默认 L2 距离，越小越相似 → 相似度 = 1 - 距离
    vec_hits = vectorstore.similarity_search_with_score(query, k=VECTOR_K)
    for rank, (doc, dist) in enumerate(vec_hits):
        sim = max(0.0, 1.0 - dist)
        if sim < SIM_THRESHOLD:  # 优化3：低于阈值的片段直接丢弃，宁可不答也不硬答
            continue
        has_vec_hit = True
        entry = candidates.setdefault(doc.id, {"doc": doc, "rrf": 0.0})
        entry["rrf"] += 1.0 / (RRF_K + rank + 1)

    # 第 2 路：BM25 关键词检索（精确命中编号、专有名词这类向量不敏感的词）
    bm25, ids, docs = _get_bm25()
    top_bm25 = 0.0
    if bm25 is not None:
        scores = bm25.get_scores(jieba.lcut(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:VECTOR_K]
        if ranked:
            top_bm25 = scores[ranked[0]]
        for rank, i in enumerate(ranked):
            if scores[i] <= 0:
                break
            cid = ids[i]
            if cid in candidates:
                candidates[cid]["rrf"] += 1.0 / (RRF_K + rank + 1)
            else:
                candidates[cid] = {"doc": docs[i], "rrf": 1.0 / (RRF_K + rank + 1)}

    # 放行判定：向量路全灭 且 BM25 不是强命中 → 判定"资料里没有"，返回空（触发拒答）
    if not has_vec_hit and top_bm25 < BM25_STRONG_SCORE:
        return []

    # 融合：按 RRF 分排序，取 top-k
    winners = sorted(candidates.values(), key=lambda e: e["rrf"], reverse=True)
    return [e["doc"] for e in winners[:k]]

