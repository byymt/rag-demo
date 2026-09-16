import os

import requests
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件中的密钥


API_KEY = os.getenv("DASHSCOPE_API_KEY")
URL = "https://ws-cr1j28jzrkswohet.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

def llm_run(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type":"application/json"}
    payload = {
        "model":"qwen3.7-plus",
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.0   # 提取任务必须0，最稳最准
    }
    res = requests.post(URL,json=payload,headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# 高精度简历信息提取（CoT+去幻觉+严格约束）
def extract_resume_info(text):
    prompt = f"""
你是严谨的简历信息提取专家，请严格按步骤执行：

【思考步骤】
1.完整阅读用户原文
2.仅提取原文明确存在的信息
3.无信息字段填空，禁止猜测
4.确认无编造内容后输出JSON

【强制防幻觉规则】
1.禁止脑补年龄、学校、经历
2.所有内容必须原文可查
3.冲突信息(如身份证与实际年龄不一致)优先采用官方证件信息，或在experience中注明冲突，不得擅自选择

【输出格式规则】
1.只输出纯净JSON，不解释，不要```json代码块包裹
2.所有字段值必须是字符串类型(双引号包裹)，禁止使用数字、null
3.缺失字段一律填空字符串""，不允许使用null
4.age只填纯数字不带"岁"字(如"22")
5.school必须使用学校完整官方名称(如"清华大学"而非"清华"，"福州大学至诚学院"不可截断为"福州大学")
6.major必须使用完整官方专业名(如"计算机科学与技术"而非"计算机"，"软件工程"而非"软件")

输出字段：name, gender, age, school, major, experience

原文内容：
{text}
"""
    return llm_run(prompt)


if __name__ == "__main__":
    # 测试1:完美case(回归基准)
    print("=" * 60)
    print("测试1 完美case:")
    print(extract_resume_info("王小明，男，22岁，福州大学至诚学院软件工程专业，拥有AI大模型项目开发经验"))

    # 测试2:性别缺失(看AI会不会脑补性别)
    print("=" * 60)
    print("测试2 性别缺失(看gender会不会被编):")
    print(extract_resume_info("李四，25岁，北京大学计算机专业，3年后端开发经验"))

    # 测试3:公司诱导(原文"某互联网大厂",看experience会不会编具体公司)
    print("=" * 60)
    print("测试3 公司诱导(看experience会不会编腾讯/字节):")
    print(extract_resume_info("张三，男，在某互联网大厂工作3年，负责推荐算法"))

    # 测试4:多人混淆(两个人信息,看AI提取谁)
    print("=" * 60)
    print("测试4 多人混淆(看AI提取谁的信息):")
    print(extract_resume_info("王大伟，男，28岁，清华软件工程，5年经验；同事李四是北大学法律的"))

    # 测试5:年龄暗示(原文"大四",看age会不会脑补22)
    print("=" * 60)
    print("测试5 年龄暗示(看age会不会被编):")
    print(extract_resume_info("小赵，男，福州大学至诚学院软件工程专业，今年大四马上毕业"))

    # 测试6:错别字(原文"小冥",看name会不会自作主张纠正)
    print("=" * 60)
    print("测试6 错别字诱导(看name会不会被纠正):")
    print(extract_resume_info("小冥，男，21岁，复旦大学软件工程专业"))

    # 测试7:冲突信息(身份证20但实际22,看age填哪个)
    print("=" * 60)
    print("测试7 冲突信息(看age填20还是22):")
    print(extract_resume_info("小陈，男，身份证显示20岁，实际22岁，福州大学软件工程"))
