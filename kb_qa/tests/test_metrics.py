"""kb_qa 指标跑分脚本：对运行中的服务测三件事——
1. 库内命中率：10 个库内问题的答案是否包含预期关键词
2. 拒答准确率：10 个库外问题是否被拒绝
3. 响应耗时：首 token 时间（流式）+ 每问完整耗时

运行前提：本地服务已启动（kb_qa 目录下 python main.py），且知识库里已上传
《AI大模型应用开发基础知识.md》。本脚本不需要登录（/chat 是开放接口）。

运行方式：
    cd d:\\rag-demo\\kb_qa
    d:\\rag-demo\\.venv\\Scripts\\python.exe tests\\test_metrics.py
"""
import json
import os
import re
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"
TESTSET = os.path.join(os.path.dirname(__file__), "testset.json")

# 拒答特征词：答案里出现任意一个即判定为"拒绝回答"
REFUSAL_MARKERS = [
    "没有检索到", "没有找到", "未检索到", "无法回答", "无法从",
    "知识库中没有", "知识库中未", "没有相关资料", "没有相关内容",
    "不在知识库", "超出了", "超出我的知识库", "仅能根据", "只能根据",
]


def is_refusal(text: str) -> bool:
    return any(m in text for m in REFUSAL_MARKERS)


def measure_first_token(question: str) -> float:
    """用流式接口测首 token 时间（秒）。失败返回 -1。"""
    try:
        t0 = time.perf_counter()
        first = None
        with requests.post(
            f"{BASE}/chat/stream",
            json={"question": question, "session_id": "metrics-probe"},
            stream=True,
            timeout=120,
        ) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:") and line[5:].strip():
                    first = time.perf_counter() - t0
                    break
        return round(first, 2) if first is not None else -1
    except Exception:
        return -1


def main():
    with open(TESTSET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    print(f"共 {len(questions)} 题，开始跑分（每题独立 session，避免多轮记忆干扰）...\n")

    in_hit = in_total = 0
    out_ok = out_total = 0
    latencies = []
    rows = []

    for i, item in enumerate(questions, 1):
        q, typ = item["q"], item["type"]
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{BASE}/chat",
                json={"question": q, "session_id": f"metrics-{i}"},
                timeout=180,
            )
            answer = r.json().get("answer", "") if r.status_code == 200 else f"[HTTP {r.status_code}]"
        except Exception as e:
            answer = f"[请求失败 {type(e).__name__}]"
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        if typ == "in":
            in_total += 1
            kws = item["keywords"]
            missing = [k for k in kws if k not in answer]
            hit = len(missing) < len(kws)  # 命中任一关键词即算通过
            in_hit += hit
            verdict = "✅" if hit else f"❌ 缺关键词 {missing}"
            rows.append((i, "库内", q, f"{elapsed:.1f}s", verdict))
        else:
            out_total += 1
            refused = is_refusal(answer)
            out_ok += refused
            verdict = "✅ 已拒答" if refused else "❌ 误答"
            rows.append((i, "库外", q, f"{elapsed:.1f}s", verdict))

        print(f"[{i:>2}/{len(questions)}] {typ} {q} -> {verdict}")

    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(f"库内命中率   : {in_hit}/{in_total} = {in_hit / in_total:.0%}")
    print(f"库外拒答率   : {out_ok}/{out_total} = {out_ok / out_total:.0%}")
    print(f"综合正确率   : {(in_hit + out_ok) / (in_total + out_total):.0%}")
    lat_sorted = sorted(latencies)
    print(f"响应耗时     : 平均 {sum(latencies)/len(latencies):.1f}s | "
          f"中位 {lat_sorted[len(lat_sorted)//2]:.1f}s | 最慢 {lat_sorted[-1]:.1f}s")
    ft = measure_first_token("什么是RAG？")
    print(f"首 token 延迟: {ft}s（流式接口）" if ft > 0 else "首 token 延迟: 流式探测失败（服务未开流式接口？）")
    print("=" * 70)

    # 失败明细（可贴给我分析）
    fails = [r for r in rows if r[4].startswith("❌")]
    if fails:
        print("\n失败明细：")
        for i, typ, q, cost, v in fails:
            print(f"  [{i}] {typ} | {q} | {cost} | {v}")


if __name__ == "__main__":
    main()
