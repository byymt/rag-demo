import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DashScopeEmbeddings
import jieba
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
load_dotenv()
def load_documents(folder="./docs"):
    docs=[]
    for name in os.listdir(folder):
        path = os.path.join(folder,name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt",".md"):
            loader = TextLoader(path,encoding="utf-8")
        elif ext ==".docx":
            loader = Docx2txtLoader(path)
        else:
            continue
        loaded = loader.load()
        print(f"加载 {name}:{len(loaded)} 个Document")
        docs.extend(loaded)
    return docs
docs = load_documents()
# print("总 Document 数:", len(docs))

embeddings = DashScopeEmbeddings(model="text-embedding-v3")
fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
fixed_chunks = fixed_splitter.split_documents(docs)

full_text = "\n".join(d.page_content for d in docs)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,          # 新增:百分位线降到70
    sentence_split_regex=r"(?<=[。!?])"      # 新增:认识中文标点
)
semantic_chunks = semantic_splitter.create_documents([full_text])
# 过滤掉过短的碎片(末尾可能切出单独一个标点)
semantic_chunks = [c for c in semantic_chunks if len(c.page_content.strip()) > 20]
print("固定切片数:", len(fixed_chunks), " 语义切片数:", len(semantic_chunks))
for c in semantic_chunks:
    print("【一片】", c.page_content[:60].replace("\n",""), "...")
    print("---")

chunks = semantic_chunks

bm25_retriever = BM25Retriever.from_documents(chunks,preprocess_func=jieba.lcut)
bm25_retriever.k = 3
vectorStore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="day12_hybrid"
)
vector_retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
# for q, keyword in [("逾期费一天多少钱", "0.2"), ("没按时还书会怎么样", "逾期")]:
#     hits = ensemble_retriever.invoke(q)
#     print(f"问题: {q}")
#     for rank, d in enumerate(hits, 1):
#         hit_mark = "✅" if keyword in d.page_content else "  "
#         print(f"  {hit_mark} 第{rank}片 是否含答案关键词'{keyword}'")

test_cases = [
    {"q": "云实验室每天几点关门", "keyword": "11 点"},
    {"q": "图书逾期一天罚多少钱", "keyword": "0.2"},
    {"q": "补办校园卡要带什么证件", "keyword": "身份证"},
]
# ========== 阶段 D:Embedding 选型实验 ==========
# 测试集:每个问题的答案藏在不同文件里,keyword 是人工标注的"答案依据词"
test_cases = [
    {"q": "云实验室每天几点关门", "keyword": "11 点"},
    {"q": "图书逾期一天罚多少钱", "keyword": "0.2"},
    {"q": "补办校园卡要带什么证件", "keyword": "身份证"},
]

for model_name in ["text-embedding-v2", "text-embedding-v3", "text-embedding-v4"]:
    emb = DashScopeEmbeddings(model=model_name)
    store = Chroma.from_documents(
        fixed_chunks,
        emb,
        collection_name=f"eval_{model_name.replace('-', '_')}"
    )
    ret = store.as_retriever(search_kwargs={"k": 1})

    hit_count = 0
    print(f"\n===== {model_name} =====")
    for case in test_cases:
        top1 = ret.invoke(case["q"])[0]
        ok = case["keyword"] in top1.page_content
        hit_count += int(ok)
        mark = "✅命中" if ok else "❌失手"
        preview = top1.page_content[:30].replace("\n", "")
        print(f"  {mark} {case['q']} => {preview}")
    print(f"  Top1 命中率: {hit_count}/3")