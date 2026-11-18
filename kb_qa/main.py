"""入口模块：FastAPI 接口 + Gradio 聊天页面。

启动方式（必须在 kb_qa 目录下）：
    d:\\rag-demo\\.venv\\Scripts\\python.exe main.py

启动后：
    - 聊天页面:  http://127.0.0.1:8000/ui
    - 接口文档:  http://127.0.0.1:8000/docs （FastAPI 自带，可以直接点按钮测接口）
"""
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections import defaultdict, deque

import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
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


def require_admin(request: Request) -> str:
    """在 require_user 基础上再查管理员角色；普通用户调审核接口直接 403。"""
    username = require_user(request)
    if not auth.is_admin(username):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return username


# ===== 简易限流：防止 /chat 被刷爆 API 配额（课程项目用内存滑窗，生产换 Redis）=====
RATE_LIMIT_PER_MIN = 30  # 每个 IP 每分钟最多 30 次问答请求
_rate_window: dict = defaultdict(deque)  # ip -> 时间戳队列


def check_rate_limit(request: Request) -> None:
    """按客户端 IP 限流，超限抛 429。"""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    dq = _rate_window[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    dq.append(now)


def resolve_session(request: Request, session_id: str) -> str:
    """把 session_id 和登录用户/客户端 IP 绑定，防止 A 冒充 B 的会话读到别人对话。
    已登录：u:{用户名}:{session_id}；未登录：ip:{客户端IP}:{session_id}。"""
    user = auth.current_user(request.headers.get("X-Token", ""))
    if user:
        return f"u:{user}:{session_id}"
    ip = request.client.host if request.client else "anon"
    return f"ip:{ip}:{session_id}"


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
    # is_admin 决定前端要不要显示"管理端"入口（后端接口仍会独立校验权限）
    return {"token": result, "username": req.username, "is_admin": auth.is_admin(req.username)}


@api.post("/logout")
def logout_api(request: Request) -> dict:
    """登出：让后端的 token 立即失效（前端只需清 localStorage）。"""
    auth.logout(request.headers.get("X-Token", ""))
    return {"message": "已退出登录"}


# ---- 接口1：POST /chat 问答 ----

class ChatRequest(BaseModel):
    """请求的数据合同：前端必须按这个格式发 JSON。"""
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """响应的数据合同：后端保证按这个格式回 JSON。"""
    answer: str
    sources: list = []  # 引用来源：[{"name": 文件名, "page": 页码或None}]


def _cited_sources(answer: str) -> list:
    """只有答案里真的出现了 [n] 角标才返回来源列表——
    降级回答（通用知识兜底）时模型没引用资料，弱命中片段不该展示，
    否则'红烧肉'下面挂'编写代码.pptx'就是误导。"""
    return kb.last_sources if re.search(r"\[\d+\]", answer) else []


@api.post("/chat", response_model=ChatResponse)
def chat_api(request: Request, req: ChatRequest) -> ChatResponse:
    """收问题 → 调 Agent → 回答案。session_id 区分不同用户的会话。"""
    check_rate_limit(request)
    kb._clear_sources()  # 清空当前线程的上一问来源，防止串到这一问
    session_id = resolve_session(request, req.session_id)
    answer = agent.chat(req.question, session_id)
    # 写入 MySQL：问题、答案、命中了哪些文档（互动看板靠它统计）
    db.add_qa_record(session_id, req.question, answer, json.dumps(kb.last_sources))
    return ChatResponse(answer=answer, sources=_cited_sources(answer))


# ---- 接口1b：POST /chat/stream 流式问答（SSE，前端打字机效果） ----

@api.post("/chat/stream")
def chat_stream_api(request: Request, req: ChatRequest) -> StreamingResponse:
    """逐段推送 Agent 的原始 ReAct 输出（思考/行动/最终答案）。
    前端按"最终答案:"切分：之前的是思考过程，之后的进答案气泡。"""
    check_rate_limit(request)
    session_id = resolve_session(request, req.session_id)

    def gen():
        # 注意：生成器跑在事件循环线程里，所以清空来源要放在这里（和 search 同线程），
        # 不能放在外面——外面是线程池线程，线程本地存储互不相通。
        kb._clear_sources()
        raw = ""
        for chunk in agent.chat_stream(req.question, session_id):
            raw += chunk
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        # 流结束：把干净答案 + 命中来源写进 MySQL（互动看板用）；
        # 来源列表单独推一个事件给前端渲染"引用来源"区块（仅当答案真有角标）
        final_answer = agent.extract_final(raw)
        db.add_qa_record(session_id, req.question, final_answer, json.dumps(kb.last_sources))
        yield f"data: {json.dumps({'sources': _cited_sources(final_answer)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---- 接口5：GET /kb/stats 互动看板 ----

@api.get("/kb/stats")
def stats_api(request: Request) -> dict:
    """每个文档被提问几次 + 最近一次问题——多端互动的数据源。需登录。"""
    require_user(request)
    return {"stats": db.doc_question_stats()}


# ---- 接口6：GET /kb/activity 实时活动流（多端互动的"消息"数据源） ----

@api.get("/kb/activity")
def activity_api(request: Request) -> dict:
    """最近 10 条活动：谁上传了什么文档、谁问了什么问题。
    前端每 3 秒轮询一次，发现比上次更新的事件就弹通知——
    这就是验收标准说的"A 下单 B 收到消息"式多端互动。需登录。"""
    require_user(request)
    return {"activities": db.get_recent_activities(10)}


# ---- 接口2：POST /kb/upload 上传入库 ----

ALLOWED_EXTS = (".pdf", ".txt", ".md", ".docx", ".pptx")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 单文件最大 50MB，防止磁盘被打满


@api.post("/kb/upload")
def upload_api(request: Request, file: UploadFile = File(...)) -> dict:
    """登录后才能调：收文件 → 存到 uploads/ → 切片入库 → 记 MySQL → 返回切片数。"""
    username = require_user(request)  # 未登录会在这里被拦下（401）

    # 0. 文件大小限制（content-length 是客户端声明的，先挡一道）
    if file.content_length is not None and file.content_length > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件过大，最大 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

    # 1. 先检查扩展名，不对直接拒绝（400 = 客户端参数错误）
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，只支持 pdf/docx/pptx/txt/md")

    # 1b. 文件名消毒：只留纯文件名（防路径穿越），过长的截断（防文件系统报错）
    safe_name = file.filename.replace("\\", "/").split("/")[-1].strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名不合法")
    safe_name = safe_name[:200]  # 文件名最多 200 字符

    # 2. 把上传的文件保存到 uploads/ 目录（uploads 目录不存在就创建）
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2b. 优化5：按文件内容哈希去重——同一个文件传两遍会产生双倍切片、检索结果重复
    with open(save_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    dup = db.get_document_by_hash(file_hash)
    if dup is not None:
        os.remove(save_path)  # 拒收，刚才落地的临时文件也删掉
        raise HTTPException(status_code=400, detail=f"该文件已上传过（《{dup['filename']}》，编号 {dup['id']}），请勿重复上传")

    # 3. 登记为"待审核"：审核通过前不切片入库——知识库是所有人共享的，
    #    任何人都能传资料会污染全库检索结果，所以必须经过管理员审核（状态机：
    #    pending → approved / rejected）
    try:
        doc_id = db.add_document(safe_name, username, 0, [], file_hash, status="pending")
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"登记失败: {e}")

    return {"filename": safe_name, "doc_id": doc_id, "status": "pending",
            "message": "已提交，等待管理员审核通过后入库"}


# ---- 接口7~9：审核流（管理端专用） ----

@api.get("/kb/pending")
def pending_docs_api(request: Request) -> dict:
    """待审核文档列表。管理端每 3 秒轮询这个接口——用户提交新文档立刻出现在这里。"""
    require_admin(request)
    return {"documents": db.get_pending_documents()}


@api.post("/kb/docs/{doc_id}/approve")
def approve_doc_api(doc_id: int, request: Request) -> dict:
    """审核通过：这时才真正切片入 Chroma，全库可检索。"""
    require_admin(request)
    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: id={doc_id}")
    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"文档状态是 {row['status']}，无法审核")
    path = os.path.join(UPLOADS_DIR, row["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="原始文件已丢失，无法入库")
    try:
        ids = kb.add_file_to_kb(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    db.set_document_ingested(doc_id, ids, len(ids))
    return {"approved": doc_id, "filename": row["filename"], "chunks": len(ids)}


@api.post("/kb/docs/{doc_id}/reject")
def reject_doc_api(doc_id: int, request: Request) -> dict:
    """驳回：不入库、删掉落地文件，记录保留（用户在文档列表能看到"已驳回"，修正后可重新提交）。"""
    require_admin(request)
    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: id={doc_id}")
    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"文档状态是 {row['status']}，无法审核")
    file_path = os.path.join(UPLOADS_DIR, row["filename"])
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"[驳回] 文件删除失败 {file_path}: {e}")
    db.set_document_status(doc_id, "rejected")
    return {"rejected": doc_id, "filename": row["filename"]}


# ---- 接口3：GET /kb/docs 文档列表 ----

@api.get("/kb/docs")
def list_docs_api(request: Request) -> dict:
    """查 MySQL documents 表，返回所有已入库文档。需登录。"""
    require_user(request)
    return {"documents": db.list_documents()}


# ---- 接口3b：PUT /kb/docs/{doc_id} 修改文档名 ----

@api.put("/kb/docs/{doc_id}")
def update_doc_name_api(doc_id: int, req: dict, request: Request) -> dict:
    """修改文档文件名。只能改自己上传的。"""
    username = require_user(request)
    new_name = req.get("filename", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="新文件名不能为空")

    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: id={doc_id}")
    if row["uploader"] != username:
        raise HTTPException(status_code=403, detail="只能修改自己上传的文档")

    ok = db.update_document_name(doc_id, new_name)
    return {"updated": doc_id, "old_name": row["filename"], "new_name": new_name}


# ---- 接口4：DELETE /kb/docs/{doc_id} 删除文档 ----

@api.delete("/kb/docs/{doc_id}")
def delete_doc_api(doc_id: int, request: Request) -> dict:
    """删文档 = Chroma 删切片 + 删物理文件 + MySQL 删记录，三处一起删。只能删自己上传的。
    每步用 try-except 包住，某步失败不影响其他步，尽量保证一致性。"""
    username = require_user(request)

    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: id={doc_id}")
    # 普通用户只能删自己的；管理员可以删任何人的（治理污染文档的权限）
    if row["uploader"] != username and not auth.is_admin(username):
        raise HTTPException(status_code=403, detail="只能删除自己上传的文档")

    # 1. 删 Chroma 向量切片（靠上传时存下的 chroma_ids）
    try:
        kb.delete_by_ids(json.loads(row["chroma_ids"] or "[]"))
    except Exception as e:
        print(f"[删除] Chroma 切片删除失败 doc_id={doc_id}: {e}")

    # 2. 删上传的物理文件
    file_path = os.path.join(UPLOADS_DIR, row["filename"])
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"[删除] 物理文件删除失败 {file_path}: {e}")

    # 3. 删 MySQL 记录
    db.delete_document(doc_id)
    return {"deleted": doc_id, "filename": row["filename"]}


# ==================== 第二部分：Gradio 聊天页面 ====================


def gradio_upload(file) -> str:
    """网页上传回调：拿到文件路径，去重检查 + 调入库，返回状态文字。"""
    if file is None:
        return "请先选择文件"
    if isinstance(file, str):
        path = file
    elif isinstance(file, dict):
        path = file.get("path") or file.get("name")
    else:
        path = file.name  # 兼容不同 gradio 版本的返回格式
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    dup = db.get_document_by_hash(file_hash)
    if dup is not None:
        return f"该文件已上传过（《{dup['filename']}》），请勿重复上传"
    try:
        ids = kb.add_file_to_kb(path)
        db.add_document(os.path.basename(path), "gradio_demo", len(ids), ids, file_hash)
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
