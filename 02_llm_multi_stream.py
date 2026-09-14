# import requests
# import json
#
# API_KEY = "你的密钥已移到 .env 文件"
# URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

# # system角色 + 全局对话历史
# messages = [
#     {
#         "role":"system",
#         "content":"你是信息提取助手。用户提问，你严格只返回JSON，字段：answer(回答文本), confidence(0~1置信度)。不要任何额外文字、解释，只输出纯JSON。"
#     }
# ]
#
# def chat_stream(user_text):
#     messages.append({"role":"user", "content":user_text})
#     payload = {
#         "model":"qwen3.7-plus",
#         "messages": messages,
#         "temperature":0.4,
#         "stream": True
#     }
#     headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
#     full_reply = ""
#     try:
#         with requests.post(URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
#             resp.raise_for_status()
#             print("AI：", end="", flush=True)
#             for line in resp.iter_lines():
#                 if not line:
#                     continue
#                 line_str = line.decode("utf-8").removeprefix("data: ")
#                 if line_str == "[DONE]":
#                     break
#                 try:
#                     chunk = json.loads(line_str)
#                     delta_text = chunk["choices"][0]["delta"].get("content", "")
#                     if delta_text:
#                         full_reply += delta_text
#                         print(delta_text, end="", flush=True)
#                 except json.JSONDecodeError:
#                     continue
#         print()
#         messages.append({"role":"assistant", "content": full_reply})
#         # 解析结构化JSON
#         try:
#             parse_data = json.loads(full_reply)
#             print("结构化解析：", parse_data)
#         except json.JSONDecodeError:
#             print("⚠ JSON解析失败")
#         return full_reply
#     except Exception as e:
#         return f"\n请求异常：{str(e)}"
#
# if __name__ == "__main__":
#     print("====多轮流式聊天机器人，输入quit退出====")
#     while True:
#         user_input = input("你：")
#         if user_input.strip().lower() == "quit":
#             print("结束对话")
#             break
#         chat_stream(user_input)







import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥

API_KEY = os.getenv("DASHSCOPE_API_KEY")
URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"


messages = [
    {"role": "system",
     "content": "你是信息提取助手。严格只返回JSON，字段：answer, confidence。不要任何额外文字。"}
]

def chat_stream(user_next):
    messages.append({"role":"user","content":user_next})
    payload={
        "model":"qwen3.7-plus",
        "messages":messages,
        "temperature":0.6,
        "stream":True,
    }
    headers={
        "Authorization":f"Bearer {API_KEY}",
        "Content-Type":"application/json"
    }
    full_answer=""
    try:
        with requests.post(URL,json=payload,headers=headers,timeout=60,stream=True) as resp:
            resp.raise_for_status()
            print("AI:",end="",flush=True)
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8").removeprefix("data: ")
                if line_str =="[DONE]":
                    break
                try:
                    chunk = json.loads(line_str)
                    delta_next =chunk["choices"][0]["delta"].get("content","")
                    if delta_next:
                        full_answer+=delta_next
                        print(delta_next,end="",flush=True)
                except json.JSONDecodeError:
                    continue
        print()
        messages.append({"role":"assistant","content":full_answer})
        try:
            parse_str = json.loads(full_answer)
            print("jiegouhuajiexi:",parse_str)
        except json.JSONDecodeError:
            print("jiexishibai")
        return full_answer
    except Exception as e:
        return f"\nqingqiuyichang:{str(e)}"

if __name__ == "__main__":
    print("qingshuru")
    while True:
        user_input = input("ni:")
        if user_input.strip().lower() =="quit":
            print("jieshu")
            break
        chat_stream(user_input)

