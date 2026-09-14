import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥

# ----------------------配置区----------------------
API_KEY = os.getenv("DASHSCOPE_API_KEY")
URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen3.7-plus"
class ChatRequest(BaseModel):
    question: str

def call_llm(question: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "stream": False
    }
    try:
        resp = requests.post(URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()  # http状态码非200直接抛异常
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return True, answer
    except requests.exceptions.Timeout:
        return False, "请求超时，请检查网络"
    except requests.exceptions.HTTPError as e:
        return False, f"接口返回错误：{e}, 返回内容:{resp.text if 'resp' in locals() else ''}"
    except Exception as e:
        return False, f"未知异常：{str(e)}"


if __name__ == "__main__":
    user_input = input("请输入你的问题：")
    req = ChatRequest(question=user_input)
    ok, result = call_llm(req.question)
    if ok:
        print("大模型回答：\n", result)
    else:
        print("调用失败：", result)
# # 1. 导入请求库
# import requests
# from pydantic import BaseModel
# # 2. 配置密钥和接口地址
# API_KEY =
# URL =
# # 3. 定义调用大模型的函数
# def call_llm(question):
#     question:str
# #    构造请求头
# headers = {"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"}
# #    构造请求参数：模型、用户问题、温度
# def call_llm(question):
#     payload = {
#         "model":"",
#         "message":"",
#         "temperatrue":""
#     }
# #    发送post请求
#     result = requests.post(URL,json=payload,headers=headers,timeout=60)
# #    检查请求是否报错
#     result.raise_for_status()
# #    提取回答返回
#     data = result.json()
#     answer = data["choice"][0]["message"]["content"]
# #    捕获异常返回错误信息
# # 4. 主程序：输入问题，调用函数，打印结果
