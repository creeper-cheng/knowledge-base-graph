# -*- coding: utf-8 -*-
"""验收闸门：把用户问过的两个问题跑完整链路（检索 + DeepSeek），看能不能答对。"""
import json
import re
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = "http://127.0.0.1:8765"

CASES = [
    {
        "q": "泰和新材2025年的营业收入和归属于母公司股东的净利润分别是多少？",
        "want": ["3,595", "41,464,894.92"],   # 营收 35.95亿 / 归母净利 4,146万
        "label": "泰和新材 2025 营收与归母净利润",
    },
    {
        "q": "新乡化纤2025年经营活动产生的现金流量净额与净利润相差多少？主要差异来自哪些项目？",
        "want": ["252,382,527.02", "197,496,121.75"],
        "label": "新乡化纤 2025 经营现金流与净利润差异",
    },
]


def post(path, obj, timeout=300):
    req = urllib.request.Request(
        B + path, data=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def get(path):
    with urllib.request.urlopen(B + path, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


h = get("/api/health")
print(f"[health] n={h['n']} dim={h['dim']} model_ready={h['model_ready']} llm_ready={h['llm_ready']}")
if not h["llm_ready"]:
    print("!! DeepSeek 未配置，无法验收")
    sys.exit(2)

passed = 0
for i, case in enumerate(CASES, 1):
    print("\n" + "=" * 82)
    print(f"【用例 {i}】{case['label']}")
    print(f"问题：{case['q']}")
    print("=" * 82)

    r = json.loads(post("/api/retrieve", {"query": case["q"], "k": 20}).read().decode("utf-8"))
    print(f"检索 {r['neighbors']}  " if False else
          f"检索：{len(r['neighbors'])} 块  编码 {r['embed_ms']}ms  检索 {r['search_ms']}ms")
    for n in r["neighbors"]:
        print(f"   [{n['rank']}] RRF={n['rrf']:.4f} 余弦={n['sim']:.4f}(#{n['vrank']}) "
              f"BM25={n['bm25']:.1f}(#{n['brank']})  {n['company']} {n['chapter'][:14]} P{n['page']}")

    resp = post("/api/answer", {"qid": r["qid"], "k": len(r["neighbors"])})
    buf, answer, cites = "", "", []
    for raw in resp:
        buf += raw.decode("utf-8")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if not line.strip():
                continue
            o = json.loads(line)
            if o["type"] == "token":
                answer += o["t"]
            elif o["type"] == "citations":
                cites = o["items"]
            elif o["type"] == "error":
                print("   [流错误]", o["msg"])

    print("\n--- DeepSeek 回答 ---")
    print(answer)
    print("--- 引用 ---")
    for c in cites:
        print(f"   [{c['no']}] {c['company']} {c['year']} {c['chapter']} P{c['page']}")

    # 判定：回答里必须出现关键数字（允许千分位/空格差异）
    norm = re.sub(r"[\s,，]", "", answer)
    hits = [w for w in case["want"] if re.sub(r"[\s,，]", "", w) in norm]
    ok = len(hits) == len(case["want"])
    print(f"\n关键数字命中：{hits} / {case['want']}  →  {'✅ 通过' if ok else '❌ 未通过'}")
    if ok:
        passed += 1

print("\n" + "=" * 82)
print(f"闸门结果：{passed} / {len(CASES)} 通过")
print("=" * 82)
sys.exit(0 if passed == len(CASES) else 1)
