"""知识库模块：读文件、切片、入库、检索——知识库的一切操作都从这里走。"""
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
    """混合检索 + 阈值过滤，返回带来源的拼接文本；检索不到返回提示语。"""
    global last_sources
    last_sources = []
    docs = hybrid_search(query, k=RETRIEVER_K)
    if not docs:
        return "没有检索到相关资料"
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "未知来源"))
        last_sources.append(source)
        parts.append(f"[{i}] 来源:{source}\n{doc.page_content}")
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
    if _bm25 is not None and len(_bm25_ids) == len(ids):
        return _bm25, _bm25_ids, _bm25_docs
    docs = [
        SimpleNamespace(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    _bm25 = BM25Okapi([jieba.lcut(d.page_content) for d in docs])
    _bm25_ids = ids
    _bm25_docs = docs
    return _bm25, _bm25_ids, _bm25_docs


def hybrid_search(query, k=RETRIEVER_K):
    """向量检索管"语义相近"，BM25 管"关键词精确命中"，两路结果用 RRF 融合排序取 top-k。

    放行规则（防幻觉的关键）：向量路有片段过阈值 → 正常返回；
    向量路全灭时，只有 BM25 出现强关键词信号（≥BM25_STRONG_SCORE）才放行——
    避免无关问题被 BM25 的弱匹配（单字碰上语料）带出不相干的片段。
    """
    candidates = {}  # chroma_id -> {"doc":..., "rrf":融合分}
    has_vec_hit = False

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

