# -*- coding: utf-8 -*-
"""端到端冒烟测试：直接打本地服务，避免中文经 shell 传参被 GBK 弄坏。"""
import json
import sys
import urllib.parse
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = "http://127.0.0.1:8765"


def get(path, raw=False):
    with urllib.request.urlopen(B + path, timeout=120) as r:
        b = r.read()
    return b if raw else json.loads(b.decode("utf-8"))


def post(path, obj):
    req = urllib.request.Request(
        B + path, data=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode("utf-8")


ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}  {detail}")
    else:
        fail += 1
        print(f"  ✗ {name}  {detail}")


print("=" * 72)
h = get("/api/health")
print(f"0. health  n={h['n']}  dim={h['dim']}  model_ready={h['model_ready']}  "
      f"llm_ready={h['llm_ready']}  bm25={'—'}")

print("A. 首页")
html = get("/", raw=True).decode("utf-8")
check("GET / 返回 HTML", "<canvas id=\"base\">" in html and "__stageLog" in html,
      f"{len(html)} 字节")

print("B. 检索")
d = json.loads(post("/api/retrieve", {"query": "吉林化纤2023年主营业务及产能情况", "k": 5}))
check("dim=1024", d["dim"] == 1024)
check("norm≈1", abs(d["norm"] - 1) < 1e-3, f"{d['norm']}")
check("vec_preview 长度 8", len(d["vec_preview"]) == 8, str(d["vec_preview"][:3]))
check("RRF 融合分严格降序",
      all(d["neighbors"][i]["rrf"] >= d["neighbors"][i+1]["rrf"]
          for i in range(len(d["neighbors"]) - 1)),
      f"{d['neighbors'][0]['rrf']:.4f} → {d['neighbors'][-1]['rrf']:.4f}")
_sims = [n["sim"] for n in d["neighbors"]]
check("余弦不再单调（混合召回的正常特征）", _sims != sorted(_sims, reverse=True),
      f"余弦范围 {min(_sims):.4f} ~ {max(_sims):.4f}")
check("每块都带两路得分", all("bm25" in n and "vrank" in n and "brank" in n
                          for n in d["neighbors"]))
check("查询落点在 [0,1]", all(0 <= v <= 1 for v in d["query_xy"]),
      f"({d['query_xy'][0]:.3f},{d['query_xy'][1]:.3f})")
check("邻居 xy 都在 [0,1]", all(all(0 <= v <= 1 for v in n["xy"]) for n in d["neighbors"]))
qid = d["qid"]

print("C. 取原文（O(1) seek）")
c = get(f"/api/chunk?row={d['neighbors'][0]['row']}")
check("chunk_id 一致", c["chunk_id"] == d["neighbors"][0]["chunk_id"],
      f"{c['chunk_id']}")
check("正文长度 == char_len", len(c["text"]) == c["char_len"],
      f"{c['char_len']} 字 / {c['company']} {c['report_year']} {c['chapter']} P{c['page']}")

print("D. BM25 关键词检索（只给搜索框用，不进 RAG）")
kw = get("/api/kw?q=" + urllib.parse.quote("吉林化纤"))
check("返回结果", len(kw["results"]) > 0,
      f"{len(kw['results'])} 条，top score={kw['results'][0]['score']:.2f}" if kw["results"] else "")

print("E. 完整向量")
v = get(f"/api/vec?qid={qid}")
check("返回 1024 维", v["dim"] == 1024 and len(v["vec"]) == 1024)
check("与 preview 一致", [round(x, 6) for x in v["vec"][:8]] == d["vec_preview"])

print("F. 回答流")
try:
    body = post("/api/answer", {"qid": qid, "k": 5})
    lines = [json.loads(x) for x in body.splitlines() if x.strip()]
    types = [x["type"] for x in lines]
    n_tok = types.count("token")
    check("首个事件是 start", types[0] == "start", f"共 {len(types)} 个事件")
    if h.get("llm_ready"):
        # 配了 key：应该真的流出 token，并以 citations + done 收尾
        check("流式输出 token", n_tok > 0, f"{n_tok} 个 token")
        check("以 citations+done 收尾", types[-1] == "done" and "citations" in types,
              f"尾部 {types[-2:]}")
        ci = next(x for x in lines if x["type"] == "citations")
        check("引用项字段完整",
              all({"company", "year", "chapter", "page", "sim"} <= set(c) for c in ci["items"]),
              f"{len(ci['items'])} 项")
    else:
        # 没配 key：必须给出可操作的提示而不是崩
        check("缺 key 时给出明确提示",
              any(x["type"] == "error" and "LLM_API_KEY" in x.get("msg", "") for x in lines),
              next((x["msg"][:60] for x in lines if x["type"] == "error"), ""))
        check("仍以 done 收尾", types[-1] == "done")
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", "replace")[:400]
    check("answer 接口", False, f"HTTP {e.code} body={detail}")
except Exception as e:
    check("answer 接口", False, f"{type(e).__name__}: {e}")

print("=" * 72)
print(f"通过 {ok} / 失败 {fail}")
sys.exit(1 if fail else 0)
