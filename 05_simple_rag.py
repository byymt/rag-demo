import os
import requests
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥

# ========== 配置区 改这里 ==========
API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBED_URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"
MODEL_NAME = "qwen3.7-plus"
EMBED_MODEL = "text-embedding-v3"
PDF_PATH = "rag_test_doc.pdf"  # 把你的pdf放到同目录，写文件名

# 1.读取PDF文本
def load_pdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text
    return full_text

# 2.文本简单切片（固定字符长度切块，简易版本）
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap重叠，防止一句话被切断丢失上下文
    return chunks

# 3.调用Embedding接口，文本转向量
def get_embedding(text):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": EMBED_MODEL,
        "input": text
    }
    resp = requests.post(EMBED_URL, json=payload, headers=headers, timeout=60)
    data = resp.json()
    vector = data["data"][0]["embedding"]
    return vector

# 4.初始化向量库，存入切块文档向量
def init_vector_db(chunks):
    # 本地持久化chroma，会生成文件夹保存向量
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="pdf_knowledge")

    # 循环每一块文本，生成向量存入（upsert：已存在则更新，避免重复运行报ID冲突）
    for idx, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        collection.upsert(
            embeddings=[vec],
            documents=[chunk],
            ids=[f"id_{idx}"]
        )
    return collection

# 5.检索：用户问题转向量，查询最相似片段
def search_knowledge(collection, question, top_k=2):
    q_vec = get_embedding(question)
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k
    )
    # 拿到检索出来的文档片段
    docs = res["documents"][0]
    return "\n".join(docs)

# 6.调用大模型，把检索片段+问题一起发给AI
def llm_answer(question, context):
    prompt = f"""基于下面文档内容回答用户问题。如果文档没有相关信息，直接说文档找不到答案，不要编造。
【参考文档】
{context}
用户问题：{question}
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role":"system", "content":"你是文档问答助手，只能使用提供的文档内容回答，禁止编造信息。"},
            {"role":"user", "content": prompt}
        ],
        "temperature":0.3
    }
    resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=60)
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ==========主程序入口==========
if __name__ == "__main__":
    print("正在读取PDF...")
    pdf_text = load_pdf(PDF_PATH)
    print("文本切片...")
    text_chunks = split_text(pdf_text)
    print(f"一共切成 {len(text_chunks)} 个文本块")
    print("向量化存入向量库...")
    coll = init_vector_db(text_chunks)
    print("✅知识库准备完成，可以提问！输入quit退出")

    while True:
        user_q = input("\n你的问题：")
        if user_q.strip().lower() == "quit":
            print("程序结束")
            break
        # 检索相关片段
        related_context = search_knowledge(coll, user_q)
        # 传给大模型生成答案
        ans = llm_answer(user_q, related_context)
        print(f"\nAI回答：{ans}")

#
# # 1。读取pdf文档
# def read_pdf(pdf_file_path):
#     read = PdfReader(pdf_file_path)
#     full_text=""
#     for page in read.pages:
#         page_content = page.extract_text()
#         if page_content:
#             full_text+=page_content
#     return full_text
#
# # 2.文本切片读取存入向量库
# def split_text(text,chunk_size=500,overlap=100):
#     chunks=[]
#     start = 0
#     while start <len(text):
#         end = start+chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start = end-overlap
#     return chunks
#
# # 3.调用Embedding接口，文本转向量
# def get_embedding(text):
#     payload={
#         "model":EMBED_MODEL,
#         "input":text
#     }
#     header = {
#         "Authorization":f"Bearer {API_KEY}",
#         "Content-Type":"applicaiton/json"
#     }
#     resp = requests.post(EMBED_URL,json = payload, headers=header,timeout=60)
#     resp.raise_for_status()
#     data = resp.json()
#     vec = data["data"][0]["embedding"]
#     return vec
#
# # 4.初始化向量库，存入切块文档向量
# def init_vector_chromadb(chunks):
#     client = chromadb.PersistentClient(path="./chroma_db")
#     collection = client.get_or_create_collection(name="pdf_knowledge")
#
# # 循环每一块文本，生成向量存入（upsert：已存在则更新，避免重复运行报ID冲突）
#     for idx,chunk in enumerate(chunks):
#         vector = get_embedding(chunk)
#         collection.upsert(
#         embeddings=[vector],
#         documents=[chunk],
#         ids=[f"id_{idx}"]
#    )
#     return collection
#
#
# # # 5.检索：用户问题转向量，查询最相似片段
# def search_knowledge(collection,question,top_k=2):
#     q_vec = get_embedding(question)
#     res=collection.query(
#         query_embeddings=[q_vec],
#         n_results=top_k
#     )
#     docs = res["documents"][0]
#     return "/n".join(docs)
#
# # # 6.调用大模型，把检索片段+问题一起发给AI
#
#
