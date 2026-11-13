# 1. 导入 os 和 load_dotenv，并调用 load_dotenv()（第16天栽过的坑）
import os
from dotenv import load_dotenv
# 2. 路径配置：三个字符串常量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR,".env"))
#    Chroma 持久化目录、上传目录、提示——用 os.path 拼相对路径
CHROMA_DIR = os.path.join(BASE_DIR,"chroma_db")
UPLOADS_DIR = os.path.join(BASE_DIR,"uploads")

# 3. 模型配置：四个常量
#    CHAT_MODEL / EMBEDDING_MODEL / TEMPERATURE / 检索的 K 值
CHAT_MODEL = "qwen3.7-plus"
EMBEDDING_MODEL = "text-embedding-v3"
TEMPERATURE = 0.0
RETRIEVER_K = 3
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
# 4. 切片配置：两个常量
#    CHUNK_SIZE / CHUNK_OVERLAP
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 5. 检索优化常量
#    SIM_THRESHOLD:    向量路放行线。Chroma 已改用 cosine 余弦距离（sim = 1 - dist），
#                      实测相关片段 sim 0.7+、无关片段 0.4 边缘；0.40 配合提示词"通用知识
#                      标注"兜底——宁可多放行让 LLM 标注来源，也不频繁误杀真问题
#    VECTOR_K:         向量/BM25 各自先取多少个候选，再融合排序
#    RRF_K:            RRF 融合公式的平滑常数（业界惯用 60）
#    BM25_STRONG_SCORE: BM25 强关键词信号线。用途：向量路全军覆没时，只有 BM25 达线才放行。
#                       试点实测（软件工程课件，900+切片）："什么是X"类短问法得分普遍 1.0~1.5，
#                       1.5 会误杀真问题 → 降到 0.9；误放行的兜底由提示词"通用知识标注"接住
SIM_THRESHOLD = 0.40
VECTOR_K = 10
RRF_K = 60
BM25_STRONG_SCORE = 0.9

# 5. MySQL 配置（业务库：用户 / 文档 / 问答记录）
#    本地开发连 127.0.0.1；部署时 docker-compose 会用服务名 mysql 覆盖它
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "kbqa")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB", "kb_qa")