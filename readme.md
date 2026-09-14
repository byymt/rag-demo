# rag-demo — 大模型应用开发第一阶段学习代码

从零开始学习大模型应用开发（LLM API → FastAPI 接口 → Embedding 向量检索 → 完整 PDF 问答 RAG）的练习代码仓库。

## 项目结构

```
rag-demo/
├─ .gitignore               # Git 忽略规则（本地向量库、缓存等不入库）
├─ requirements.txt         # 依赖清单，pip install -r requirements.txt 安装
├─ readme.md                # 本文件
├─ 01_llm_single.py         # 第1天：单轮对话 API 调用（POST /chat/completions）
├─ 02_llm_multi_stream.py   # 第2天：多轮对话 messages 累积 + 流式输出（iter_lines / [DONE]）
├─ 03_fastapi_chat.py       # 第3天：FastAPI 接口 + SSE 流式（StreamingResponse, text/event-stream）
├─ 04_embedding_chroma.py   # 第4天：Embedding 向量化 + Chroma 向量库（持久化 + 相似度检索）
├─ 05_simple_rag.py         # 第5天：完整 PDF 问答 RAG（解析→切块→向量化→检索→拼 prompt→回答）
├─ rag_test_doc.pdf         # 05 的测试文档《校园云实验室使用指南》
└─ notes/
   └─ 第一阶段笔记.md        # 学习笔记（知识点、踩坑记录）
```

## 各脚本内容速览

| 脚本 | 核心知识点 | 运行方式 |
|------|-----------|---------|
| 01 | OpenAI 兼容接口三要素：URL、API_KEY、model；payload 结构（model/messages/temperature） | `python 01_llm_single.py` |
| 02 | 多轮 = 把历史对话全部放进 `messages`；流式 = `stream=True` + 逐行解析 `data: {...}`，遇到 `[DONE]` 结束 | `python 02_llm_multi_stream.py` |
| 03 | FastAPI 定义 POST 接口、Pydantic 校验请求体、SSE 流式返回（`media_type="text/event-stream"`） | `python 03_fastapi_chat.py` 后用 Apifox/浏览器测 `POST /chat/stream` |
| 04 | Embedding 把文本变成向量；Chroma `PersistentClient` 本地持久化；`upsert` 防重复入库；`query` 相似度检索（距离越小越相似） | `python 04_embedding_chroma.py` |
| 05 | RAG 完整链路：PDF 提取文本 → 按长度切块 → 向量化入库 → 问题检索 top2 → 塞进 prompt 让大模型"根据文档回答，禁止编造" | `python 05_simple_rag.py`，输入问题，`quit` 退出 |

## 环境

- Python 3.10+，依赖见 `requirements.txt`
- 大模型：阿里云百炼兼容模式（`qwen3.7-plus` 聊天模型 + `text-embedding-v3` 向量模型）

## 注意事项

1. **API Key 安全**：脚本中 Key 仅为学习用途硬编码，若要上传公开仓库（如 GitHub），务必先改成环境变量或 `.env` 并**作废该 Key**。
2. **05 的 PDF**：`PDF_PATH = "rag_test_doc.pdf"` 是相对路径，需在 `rag-demo` 目录下运行，或把变量改成 PDF 的绝对路径。
3. **04 / 05 的向量库**：首次运行会在当前目录生成 `chroma_db/` 文件夹（已加入 .gitignore），删掉它即可清空知识库重来。
