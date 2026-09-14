from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥

app = FastAPI(title="SSE流式大模型接口")

# 跨域配置，前端浏览器访问必须加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class ChatReq(BaseModel):
    question: str

API_KEY = os.getenv("DASHSCOPE_API_KEY")
URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

# ---------------------- 生成器函数：SSE数据流生成器（重点）----------------------
def stream_llm_generator(user_question: str):
    """yield 不断产出SSE格式的数据块"""
    messages = [
        {"role":"system","content":"你是简洁的助手，简短回答问题"}
    ]
    messages.append({"role":"user","content":user_question})

    payload = {
        "model":"qwen3.7-plus",
        "messages": messages,
        "temperature":0.6,
        "stream": True
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(URL, json=payload, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").removeprefix("data: ")
            if line_str.strip() == "[DONE]":
                # SSE结束标记
                yield "data: [DONE]\n\n"
                break
            try:
                chunk = json.loads(line_str)
                # 千问流式最后一个块只含用量统计，choices为空数组，直接跳过
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                # 流式取delta.content！！不是message.content
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    # SSE协议固定格式 data:内容\n\n
                    sse_data = f"data: {delta}\n\n"
                    yield sse_data
            except json.JSONDecodeError:
                continue
    except Exception as e:
        yield f"data: 异常:{str(e)}\n\n"

# ---------------------- 异步SSE流式接口 ----------------------
@app.post("/chat/stream")
async def chat_stream_api(req: ChatReq):
    generator = stream_llm_generator(req.question)
    # media_type必须是text/event-stream，这是SSE协议MIME类型
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)








