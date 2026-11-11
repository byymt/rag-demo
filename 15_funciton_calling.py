# 15_function_calling_langchain.py —— LangChain 框架版
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# ============ 1. 定义工具(@tool 装饰器)============
# @tool
# def get_weather(city: str) -> str:
#     """查询指定城市的实时天气情况,返回天气描述和温度。"""
#     resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
#     data = resp.json()
#     current = data["current_condition"][0]
#     desc = current["weatherDesc"][0]["value"]
#     temp = current["temp_C"]
#     return f"{city}当前天气:{desc},温度{temp}℃"
#
# @tool
# def get_time(city: str) -> str:
#     """查询指定城市的当前时间。"""
#     from datetime import datetime
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     return f"{city}当前时间:{now}"

@tool
def get_weather(city:str)->str:
    """描述指定城市的实时天气情况，返回天气描述和温度"""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=j1",timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"天气服务暂时不可用（{type(e).__name__}），无法查询{city}的天气"
    current = data["current_condition"][0]
    desc = current["weatherDesc"][0]["value"]
    temp = current["temp_C"]
    return f"{city}当前天气{desc},温度{temp}"

@tool
def get_time(city:str)->str:
    """查询指定城市的当前时间"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city}当前时间{now}"

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.7-plus",
    temperature=0.0
)

tools=[get_weather,get_time]
llm_with_tools=llm.bind_tools(tools)

tool_map = {t.name:t for t in tools}

def chat(question:str)->str:
    messages = [HumanMessage(content=question)]
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            print(f"[llm要调用]{tc['name']},参数：{tc['args']}")
            result = tool_map[tc['name']].invoke(tc['args'])
            print(f"工具调用{result}")

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))
    return response.content
#
# # ============ 2. 创建 LLM 并绑定工具 ============
# llm = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=os.getenv("DASHSCOPE_BASE_URL"),
#     model="qwen3.7-plus",
#     temperature=0.0
# )
#
# tools = [get_weather, get_time]
# llm_with_tools = llm.bind_tools(tools)
#
# # 工具名 → 工具对象 的映射(等价于裸写版的 TOOL_MAP)
# tool_map = {t.name: t for t in tools}
#
# # ============ 3. 工具调用主循环 ============
# def chat(question: str) -> str:
#     messages = [HumanMessage(content=question)]
#
#     # 最多循环3轮,防止 LLM 无限调用工具
#     for _ in range(3):
#         # 第一轮/后续轮:把消息发给 LLM
#         response = llm_with_tools.invoke(messages)
#         messages.append(response)          # 必须把 LLM 回复存进消息列表
#
#         # 情况A:LLM 不需要工具,直接返回回答
#         if not response.tool_calls:
#             return response.content
#
#         # 情况B:LLM 要调工具,逐个执行并把结果发回
#         for tc in response.tool_calls:
#             print(f"  [LLM要调用] {tc['name']}, 参数: {tc['args']}")
#
#             # 用 .invoke() 执行工具(不能直接调用被 @tool 装饰的函数)
#             result = tool_map[tc["name"]].invoke(tc["args"])
#             print(f"  [工具返回] {result}")
#
#             # 把结果包成 ToolMessage 发回,id 必须对应
#             messages.append(ToolMessage(
#                 content=str(result),
#                 tool_call_id=tc["id"]
#             ))
#
#     # 循环超过3轮还没结束,返回最后一条回复
#     return response.content
#
# ============ 4. 测试 ============
if __name__ == "__main__":
    print("问: 北京今天天气怎么样")
    print("答:", chat("北京今天天气怎么样"))

    print("\n问: 上海现在几点了")
    print("答:", chat("上海现在几点了"))

    print("\n问: 你好")
    print("答:", chat("你好"))