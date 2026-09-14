import requests
import chromadb
from chromadb.utils import embedding_functions

# API_KEY = "你的密钥已移到 .env 文件"
# EMBED_URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"
#
# # ---------------------1.封装获取embedding向量函数----------------------
# def get_embedding(text_list: list[str]):
#     """输入文本列表，返回向量列表"""
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type":"application/json"
#     }
#     payload = {
#         "model":"text-embedding-v3",
#         "input": text_list
#     }
#     resp = requests.post(EMBED_URL, json=payload, headers=headers, timeout=60)
#     resp.raise_for_status()
#     res_data = resp.json()
#     embeddings = [item["embedding"] for item in res_data["data"]]
#     return embeddings
#
# # ---------------------2.初始化chroma本地向量库----------------------
# # 持久化存储，数据保存在本地文件夹chroma_db，重启程序数据不会丢
# client = chromadb.PersistentClient(path="./chroma_db")
#
# # 创建集合collection，相当于数据库的一张表
# collection = client.get_or_create_collection(name="knowledge_demo")
#
# # ---------------------3.准备测试文档数据----------------------
# docs = [
#     "福州大学至诚学院是福州的独立院校，开设软件工程专业",
#     "软件工程专业可以学习大模型应用开发，做RAG项目",
#     "Embedding可以将文本转为向量，用于知识库检索",
#     "向量数据库用来存放文本向量，实现语义相似度查询"
# ]
# # 给每一篇文档分配唯一id
# doc_ids = ["doc0","doc1","doc2","doc3"]
# # 获取全部文档的向量
# doc_embeddings = get_embedding(docs)
#
# # ---------------------4.把文档、向量存入chroma（存在则更新，不存在则新增）----------------------
# collection.upsert(
#     documents=docs,
#     embeddings=doc_embeddings,
#     ids=doc_ids
# )
# print("✅文档向量入库完成")
#
# # ---------------------5.根据用户问题做相似度检索（核心）----------------------
# user_query = "至诚学院有什么专业可以做AI开发"
# # 问题也要先转向量
# query_emb = get_embedding([user_query])
#
# # n_results=2 返回相似度最高2条文档
# search_result = collection.query(
#     query_embeddings=query_emb,
#     n_results=2
# )
#
# print("\n====检索结果====")
# # search_result["documents"][0] 拿到匹配到的文本列表
# for idx, doc_text in enumerate(search_result["documents"][0]):
#     distance = search_result["distances"][0][idx]
#     print(f"相似度距离:{distance:.4f}  文档内容：{doc_text}")

# ---------------------额外：基础增删改查示例----------------------
# 根据id删除文档
# collection.delete(ids=["doc3"])

# 获取集合全部数据
# all_data = collection.get()
# print(all_data["documents"])




# import requests
# import chromadb


import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥

API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBED_URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"

def get_embeddings(text_list:list[str]):
    payload = {
        "model":"text-embedding-v3",
        "input":text_list
    }
    headers = {
        "Authorization":f"Bearer {API_KEY}",
        "Content-Type":"application/json"
    }
    resp = requests.post(EMBED_URL,json=payload,headers=headers,timeout=60)
    resp.raise_for_status()
    resp_data = resp.json()
    embeddings = []
    for item in resp_data["data"]:
        vec  = item["embedding"]
        embeddings.append(vec)
    return embeddings


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_demo")

# ---------------------3.准备测试文档数据----------------------
docs = [
    "福州大学至诚学院是福州的独立院校，开设软件工程专业",
    "软件工程专业可以学习大模型应用开发，做RAG项目",
    "Embedding可以将文本转为向量，用于知识库检索",
    "向量数据库用来存放文本向量，实现语义相似度查询"
 ]
doc_ids = ["doc0","doc1","doc2","doc3"]

docs_embeddings = get_embeddings(docs)

# # ---------------------4.把文档、向量存入chroma（新增）----------------------
collection.upsert(
    documents=docs,
    embeddings=docs_embeddings,
    ids=doc_ids
)
print("成功存入")
# # ---------------------5.根据用户问题做相似度检索（核心）----------------------
user_query = "至诚学院有什么专业可以做AI开发"
query_emb = get_embeddings([user_query])
search_result = collection.query(
    query_embeddings = query_emb,
    n_results=3
)

for idx, doc_text in enumerate(search_result["documents"][0]):
    distance = search_result["distances"][0][idx]
    print(f"相似度距离:{distance:.4f}  文档内容：{doc_text}")


