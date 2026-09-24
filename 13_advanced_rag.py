import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

def load_document(folder="./docs"):
    docs=[]
    for name in os.listdir(folder):
        path = os.path.join(folder,name)
        ext = os.path.splitext(name)[1].lower()
        if ext ==".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt",".md"):
            loader = TextLoader(path,encoding="utf-8")
        else :
            continue
        loaded = loader.load()
        docs.extend(loaded)
    return docs

docs = load_document()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,chunk_overlap=30
)
chunks = splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = Chroma.from_documents(chunks,embeddings,collection_name="day13_rerank")
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.7-plus",
    temperature=0.0
)
def rerank(question,docs,llm):
    scored=[]
    for doc in docs :
        prompt = f"""请给以下文档片段对回答问题的相关性打分（0-10分），
只回答数字，不要解释。
问题:{question}.
文档片段：{doc.page_content[:200]}
分数："""
        result = llm.invoke(prompt)
        score = float(result.content.strip())
        scored.append([doc,score])
        print ( f"  片段预览: {doc.page_content[: 30 ].replace( chr ( 10 ), '' )} → 分数: {score} " )
    scored.sort(key=lambda x:x[1],reverse=True)
    return scored
# ↓↓↓ 从这里开始全部顶格,不属于 rerank 函数 ↓↓↓

# 测试
question = ("怎么炒番茄炒蛋")
print(f"问题: {question}")
print("\n--- 粗检索 5 片 ---")
raw_results = retriever.invoke(question)
for d in raw_results:
    print(f"  {d.page_content[:30].replace(chr(10), '')}")
print("\n--- 重排序后 ---")
reranked = rerank(question, raw_results, llm)
for i, (doc,score) in enumerate(reranked, 1):
    print(f"  第{i}名(分数{score}): {doc.page_content[:30].replace(chr(10), '')}")
# 重排序后只取前3名
top3 = reranked[:3]

# ============ 阶段C:拒答去幻觉 ============
SCORE_THRESHOLD = 5.0          # 相关性阈值(0-10),最高分低于它就拒答
top_score = top3[0][1]         # 第1名的分数

if top_score < SCORE_THRESHOLD:
    # 资料全不相关 → 不调用LLM,直接拒答,从源头防止幻觉
    print("\n===回答===")
    print("根据现有资料无法回答该问题。")
else:
    # ---------- 阶段B:引用溯源(只有资料相关时才执行)----------
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(top3, 1):
        source = doc.metadata.get("source", "未知来源")
        source_name = os.path.basename(source)
        context_parts.append(f"[{i}] {doc.page_content}")
        sources.append(source_name)

    context = "\n\n".join(context_parts)

    prompt = f"""请根据以下资料回答问题。
要求:回答时在关键信息后标注引用编号,引用编号只能是数字1、2、3,用方括号包裹,例如[1]。
如果资料中没有答案,请直接说根据现有资料无法回答。

资料:
{context}

问题:
{question}
回答:
"""
    answer = llm.invoke(prompt)
    print("\n===回答===")
    print(answer.content)

    print("\n===引用来源===")
    for i, name in enumerate(sources, 1):
        print(f"[{i}] {name}")