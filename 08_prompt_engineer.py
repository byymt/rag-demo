import os

import requests
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥


API_KEY = os.getenv("DASHSCOPE_API_KEY")
URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

def llm_prompt_run(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type":"application/json"}
    payload = {
        "model":"qwen3.7-plus",
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.2
    }
    res = requests.post(URL,json=payload,headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# 1.信息提取模板
def info_extract(text):
    prompt = f"""
你是信息提取专家，只返回纯JSON，不解释。
字段：name,age,school,major，缺失填空。
规则：major 必须使用完整官方专业名(如"软件工程"而非"软件")，原文口语化表达时按官方全称还原，无法确定时填空。
文本：{text}
"""
    return llm_prompt_run(prompt)

# 2.文本总结模板
def summary_text(text):
    prompt = f"""
你是文本总结助手，80字左右，保留核心信息。
文本：{text}
"""
    return llm_prompt_run(prompt)

# 3.文本分类模板
def classify_text(text):
    prompt = f"""
只能输出：正面、负面、中性、咨询、投诉、建议 其中一个
文本：{text}
"""
    return llm_prompt_run(prompt)

if __name__ == "__main__":
    # 回归测试1:完美case(确认没被新约束破坏)
    print("=" * 50)
    print("回归测试1 完美case:")
    print(info_extract("小明，20岁，福州大学至诚学院，软件工程专业"))

    # 回归测试4:口语化专业(看"软件"是否还原为"软件工程")
    print("=" * 50)
    print("回归测试4 口语化专业(新约束要解决的case):")
    print(info_extract("小明和小红是福州大学至诚学院同学，小明22岁学软件，小红学法学"))

    # 回归测试6:模糊学校(确认没被新约束破坏)
    print("=" * 50)
    print("回归测试6 模糊学校:")
    print(info_extract("小王同学是某985高校计算机专业大三在读学生"))
