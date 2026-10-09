# 15_function_calling_raw.py —— 裸写版(理解原理用)
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ============ 1. 真实工具函数(就是普通 Python 函数)============
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
    data = resp.json()
    current = data["current_condition"][0]
    desc = current["weatherDesc"][0]["value"]
    temp = current["temp_C"]
    return f"{city}当前天气:{desc},温度{temp}℃"

# 工具名 → 函数 的映射,后面根据 LLM 返回的名字找函数
TOOL_MAP = {"get_weather": get_weather}

# ============ 2. 工具的 JSON Schema(手动写,等价于 @tool 自动生成的)============
# LLM 看到这个才知道"有个工具叫 get_weather,参数 city 是字符串"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气情况,返回天气描述和温度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市名称,例如:北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ============ 3. 第一轮:发问题 + 工具列表 ============
question = "北京今天天气怎么样"
messages = [{"role": "user", "content": question}]

payload1 = {
    "model": "qwen3.7-plus",
    "messages": messages,
    "tools": TOOLS              # ← 关键:把工具列表塞进去
}

print("=" * 50)
print("【第1轮请求】发出去的 messages + tools:")
print(json.dumps(payload1, ensure_ascii=False, indent=2)[:500], "...")

resp1 = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=HEADERS,
    json=payload1,
    timeout=60
)
resp1.raise_for_status()
data1 = resp1.json()

print("\n【第1轮响应】LLM 的回复(注意 tool_calls 字段):")
print(json.dumps(data1["choices"][0]["message"], ensure_ascii=False, indent=2))

# 取出 LLM 的回复消息
assistant_msg = data1["choices"][0]["message"]
messages.append(assistant_msg)   # ← 必须把 LLM 回复存进 messages

# ============ 4. 判断 LLM 要不要调工具 ============
tool_calls = assistant_msg.get("tool_calls")

if tool_calls:
    print("\n" + "=" * 50)
    print(f"LLM 要调用 {len(tool_calls)} 个工具:")
    for tc in tool_calls:
        print(f"  工具名: {tc['function']['name']}")
        print(f"  参数: {tc['function']['arguments']}")
        print(f"  调用ID: {tc['id']}")

    # ============ 5. 逐个执行工具,把结果塞回 messages ============
    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        tool_args = json.loads(tc["function"]["arguments"])  # arguments 是 JSON 字符串
        tool_call_id = tc["id"]

        # 执行真实函数
        result = TOOL_MAP[tool_name](**tool_args)
        print(f"\n  工具执行结果: {result}")

        # 把结果包成 role=tool 的消息,tool_call_id 必须对应
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)
        })

    # ============ 6. 第二轮:把工具结果发回 LLM,拿最终回答 ============
    payload2 = {
        "model": "qwen3.7-plus",
        "messages": messages       # ← 现在包含:用户问题 + LLM的tool_calls + 工具结果
    }

    print("\n" + "=" * 50)
    print("【第2轮请求】发回去的 messages(含工具结果):")
    print(json.dumps(messages, ensure_ascii=False, indent=2)[:800], "...")

    resp2 = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=HEADERS,
        json=payload2,
        timeout=60
    )
    resp2.raise_for_status()
    data2 = resp2.json()

    final_answer = data2["choices"][0]["message"]["content"]
    print("\n" + "=" * 50)
    print(f"【最终回答】\n{final_answer}")
else:
    # LLM 没调工具,直接回答
    print("\n" + "=" * 50)
    print(f"【直接回答】\n{assistant_msg.get('content', '')}")