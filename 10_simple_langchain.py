# ============================================================
# LangChain 第1课：多轮对话机器人
# 四大核心组件：Model(模型) / Prompt(模板) / Memory(记忆) / Chain(链)
# 比喻：llm=厨师，prompt=菜单模板，memory=服务员的小本子，chain=整条流水线
# ============================================================

from langchain_openai import ChatOpenAI                              # 模型对象（阿里百炼兼容OpenAI格式）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示词模板 + 历史占位符
from langchain.chains import LLMChain                               # 链：把模型/模板/记忆串起来
from langchain.memory import ConversationBufferMemory               # 对话记忆：自动存取聊天历史
from dotenv import load_dotenv
import os

# 把 .env 文件里的密钥读进环境变量（之后才能用 os.getenv 取到）
load_dotenv()

# ------------------------------------------------------------
# 组件1：Model —— 初始化大模型（等价于以前手写 headers + URL + model）
# ------------------------------------------------------------
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),    # 密钥，从环境变量取，代码里不写明文
    base_url=os.getenv("DASHSCOPE_BASE_URL"),  # 阿里云接口地址（结尾不带 /chat/completions）
    temperature=0.6,                            # 随机性：0最稳定死板，1最发散；聊天0.6，提取信息用0
    model="qwen3.7-plus"                        # 使用的模型名
)

# ------------------------------------------------------------
# 组件2：Prompt —— 定义“每一轮发给模型的消息结构”，共三层
#   system  : AI的人设/规矩（每轮固定不变）
#   历史坑  : MessagesPlaceholder，每轮由 Memory 自动填入历史对话
#   human   : 用户本轮输入，{input} 是占位符，invoke 时手动传入
# ------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是严谨的对话问答助手，严格按照用户要求回答"),
        MessagesPlaceholder(variable_name="chat_history"),  # 历史对话插入位置；名字必须和memory_key一致
        ("human", "{input}")                                # {input} 必须和 invoke 字典的键一致
    ]
)

# ------------------------------------------------------------
# 组件3：Memory —— 记忆本，每轮结束自动存“问+答”，下轮自动取出
#   memory_key="chat_history" 必须和上面占位符的 variable_name 完全相同，
#   否则历史填不进模板，AI 就会“失忆”。
# ------------------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True   # 以消息对象形式返回历史（而不是纯字符串），配合占位符使用
)

# ------------------------------------------------------------
# 组件4：Chain —— 流水线调度员，自己不产生智能，只按顺序调用前三样
#   invoke 一次内部执行：
#     ① 从 memory 取历史 → ② 历史+输入填进模板 → ③ 发给 llm
#     ④ 拿到回答 → ⑤ 把本轮问答存回 memory → ⑥ 返回 {"text": 回答}
# ------------------------------------------------------------
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True   # 调试用：打印每轮真正发给模型的完整 prompt，能看到历史怎么被塞进去
)

# ------------------------------------------------------------
# 主循环：命令行持续对话，输入 quit 退出
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=====多轮对话机器人=======")
    while True:
        text = input("你：")                          # input 必须在循环内，每轮重新等待输入
        if text.strip().lower() == "quit":            # strip去首尾空格 + lower转小写，方便退出
            print("结束")
            break
        result = chain.invoke({"input": text})        # 键名 "input" 必须和模板里的 {input} 一模一样
        print(f"ai:{result['text']}")                 # 返回是字典，回答文本在 result["text"] 里



# llm = ChatOpenAI(
#     base_url=os.getenv("DASHSCOPE_BASE_URL"),
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     model="qwen3.7-plus",
#     temperature=0.6
# )
# prompt = ChatPromptTemplate.from_messages([
#     ("system","你是严谨的ai对话助手"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human","{input}")
# ])
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
# chain = LLMChain(
#     llm=llm, 
#     prompt=prompt,
#     memory=memory,
# )
# if __name__ =="__main__":
#     print("==========多轮对话助手=========")
#     while True:
#         text = input("你:")
#         if text.strip().lower()=="quit":
#             print("结束")
#             break
#         result = chain.invoke({"input":text})
#         print(f"ai:{result['text']}")