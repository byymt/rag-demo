"""入口模块：FastAPI 接口 + Gradio 聊天页面。

启动方式（必须在 kb_qa 目录下）：
    d:\\rag-demo\\.venv\\Scripts\\python.exe main.py

启动后：
    - 聊天页面:  http://127.0.0.1:8000/ui
    - 接口文档:  http://127.0.0.1:8000/docs （FastAPI 自带，可以直接点按钮测接口）
"""
import json
import os
import shutil
import uuid

import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import agent, auth, db, kb
from app.config import BASE_DIR, UPLOADS_DIR

# 网页前端目录（index.html 登录页 / docs.html 文档管理 / chat.html 提问）
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 启动时建表。数据库连不上会直接启动失败——这是故意的：没有业务库服务就没意义
db.init_db()

# ==================== 第一部分：API 服务器 ====================

api = FastAPI(title="kb_qa 知识库问答系统")


def require_user(request: Request) -> str:
    """从请求头 X-Token 里校验登录状态；没登录直接抛 401，接口代码不用重复判断。"""
    username = auth.current_user(request.headers.get("X-Token", ""))
    if username is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期，请先登录")
    return username


# ---- 接口0：注册 / 登录 ----

class AuthRequest(BaseModel):
    """注册和登录共用的数据合同。"""
    username: str
    password: str


@api.post("/register")
def register_api(req: AuthRequest) -> dict:
    ok, msg = auth.register(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}


@api.post("/login")
def login_api(req: AuthRequest) -> dict:
    ok, result = auth.login(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    # token 交给前端保存，之后每个请求带上 X-Token 头
    return {"token": result, "username": req.username}


# ---- 接口1：POST /chat 问答 ----

class ChatRequest(BaseModel):
    """请求的数据合同：前端必须按这个格式发 JSON。"""
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """响应的数据合同：后端保证按这个格式回 JSON。"""
    answer: str


@api.post("/chat", response_model=ChatResponse)
def chat_api(req: ChatRequest) -> ChatResponse:
    """收问题 → 调 Agent → 回答案。session_id 区分不同用户的会话。"""
    kb.last_sources = []  # 清空上一问的来源，防止把旧命中记到这一问头上
    answer = agent.chat(req.question, req.session_id)
    # 写入 MySQL：问题、答案、命中了哪些文档（互动看板靠它统计）
    db.add_qa_record(req.session_id, req.question, answer, json.dumps(kb.last_sources))
    return ChatResponse(answer=answer)


# ---- 接口5：GET /kb/stats 互动看板 ----

@api.get("/kb/stats")
def stats_api() -> dict:
    """每个文档被提问几次 + 最近一次问题——多端互动的数据源。"""
    return {"stats": db.doc_question_stats()}


# ---- 接口2：POST /kb/upload 上传入库 ----

ALLOWED_EXTS = (".pdf", ".txt", ".md")


@api.post("/kb/upload")
def upload_api(request: Request, file: UploadFile = File(...)) -> dict:
    """登录后才能调：收文件 → 存到 uploads/ → 切片入库 → 记 MySQL → 返回切片数。"""
    username = require_user(request)  # 未登录会在这里被拦下（401）

    # 1. 先检查扩展名，不对直接拒绝（400 = 客户端参数错误）
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，只支持 pdf/txt/md")

    # 2. 把上传的文件保存到 uploads/ 目录（uploads 目录不存在就创建）
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3. 入向量库；失败返回 500（服务器内部错误）
    try:
        ids = kb.add_file_to_kb(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")

    # 4. 记入 MySQL（谁传的、切了几片、Chroma 切片 id 是什么）
    doc_id = db.add_document(file.filename, username, len(ids), ids)

    return {"filename": file.filename, "chunks": len(ids), "doc_id": doc_id}


# ---- 接口3：GET /kb/docs 文档列表 ----

@api.get("/kb/docs")
def list_docs_api() -> dict:
    """查 MySQL documents 表，返回所有已入库文档。"""
    return {"documents": db.list_documents()}


# ---- 接口4：DELETE /kb/docs/{doc_id} 删除文档 ----

@api.delete("/kb/docs/{doc_id}")
def delete_doc_api(doc_id: int, request: Request) -> dict:
    """删文档 = Chroma 删切片 + 删物理文件 + MySQL 删记录，三处一起删。只能删自己上传的。"""
    username = require_user(request)

    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: id={doc_id}")
    if row["uploader"] != username:
        raise HTTPException(status_code=403, detail="只能删除自己上传的文档")

    # 1. 删 Chroma 向量切片（靠上传时存下的 chroma_ids）
    kb.delete_by_ids(json.loads(row["chroma_ids"] or "[]"))

    # 2. 删上传的物理文件
    file_path = os.path.join(UPLOADS_DIR, row["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    # 3. 删 MySQL 记录
    db.delete_document(doc_id)
    return {"deleted": doc_id, "filename": row["filename"]}


# ==================== 第二部分：Gradio 聊天页面 ====================


def gradio_upload(file) -> str:
    """网页上传回调：拿到文件路径，调入库，返回状态文字。"""
    if file is None:
        return "请先选择文件"
    if isinstance(file, str):
        path = file
    elif isinstance(file, dict):
        path = file.get("path") or file.get("name")
    else:
        path = file.name  # 兼容不同 gradio 版本的返回格式
    try:
        ids = kb.add_file_to_kb(path)
        db.add_document(os.path.basename(path), "gradio_demo", len(ids), ids)
        return f"《{os.path.basename(path)}》已入库，切了 {len(ids)} 片"
    except Exception as e:
        return f"入库失败: {e}"


def gradio_chat(message, history, session_id):
    """网页聊天回调：维护每人一份的 session_id，调 Agent 更新聊天记录。"""
    if session_id is None:  # 每个浏览器标签页第一次聊天时领一个编号
        session_id = uuid.uuid4().hex
    answer = agent.chat(message, session_id)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history, session_id


with gr.Blocks(title="kb_qa 知识库问答") as demo:
    gr.Markdown("# 知识库问答系统\n先在右侧上传文档入库，再提问。答案只基于已上传的资料。")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=420)
            msg = gr.Textbox(label="你的问题", placeholder="输入问题，回车发送...")
        with gr.Column(scale=1):
            session_id = gr.State(None)  # 会话编号，存档用，网页上看不见
            file = gr.File(label="上传文档（pdf / txt / md）")
            upload_btn = gr.Button("入库")
            upload_status = gr.Textbox(label="入库状态", interactive=False)

    msg.submit(gradio_chat, inputs=[msg, chatbot, session_id], outputs=[msg, chatbot, session_id])
    upload_btn.click(gradio_upload, inputs=[file], outputs=[upload_status])

# 把 Gradio 页面挂到 FastAPI 服务器上：一个进程同时提供接口和网页
app = gr.mount_gradio_app(api, demo, path="/ui")

# 静态网页挂载在最后：没被接口、/docs、/ui 占用的路径都交给 static/ 目录
# html=True 表示访问 http://127.0.0.1:8000/ 自动返回 index.html
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
