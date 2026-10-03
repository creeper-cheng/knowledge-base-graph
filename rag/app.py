# -*- coding: utf-8 -*-
r"""RAG 过程可视化 —— 本地 Web 服务。

跑在**主环境**（jieba/numpy/requests；不 import torch）。查询编码交给
qwen3-embedding\.venv 里的常驻 worker 子进程（见 embed_worker.py）。

启动: run_rag.bat   或   python rag\app.py [--port 8765]

⚠ 监听 0.0.0.0 且无鉴权，仅限内网演示，不要暴露到公网。
"""
import argparse
import json
import os
import pickle
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DERIVED = REPO / "年报_合成纤维制造" / "_derived"
CHUNK_DIR = DERIVED / "chunks"
VECTOR_DIR = DERIVED / "vectors"
VIZ_DIR = DERIVED / "viz"
WEB_DIR = HERE / "web"
VENV_PY = REPO / "qwen3-embedding" / ".venv" / "Scripts" / "python.exe"

HOST = "0.0.0.0"
PORT_CANDIDATES = [8765, 8766, 8767, 5173, 8000]

TASK = ("Given a Chinese A-share annual report query, "
        "retrieve the passage that answers it")
SNIPPET = 160
RRF_K = 60            # RRF 融合常数；越大越平滑。实测 10/30/60 结果都稳，用通用的 60

# 默认召回条数。原来用 8，但实测「新乡化纤经营现金流与净利润差异」这种需要
# 「现金流量表补充资料」的问题，那张表融合后排在 15–24 名，k=8 直接漏掉，
# 导致模型只能从别处抓一个口径不对的净利润（105,569,879.54 而非 197,496,121.75）。
# 提到 20，配合下面的上下文预算，能覆盖到。
DEFAULT_K = 20
CHUNK_CHAR_BUDGET = 900       # 每块送给 LLM 的最大字数
TOTAL_CHAR_BUDGET = 18000     # 全部上下文的最大字数（DeepSeek 64k 上下文，用得起）
LLM_TIMEOUT = 60
LLM_RETRY = 2

PALETTE = ["#667eea", "#f0883e", "#2ea88f", "#d9455f", "#3d9bd8", "#d99a2b",
           "#8e5bd0", "#5aa469", "#c96fa8", "#7a8b99", "#b0724a"]


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ==================================================================
#  全局状态
# ==================================================================
class State:
    ready = False
    err = None
    idx = None          # index.json
    meta = None         # build_meta.json
    mat = None          # vectors.npy (mmap)
    coords = None
    groups = None
    comps = None
    flags = None
    offsets = None
    rows = None
    row_of_chunk = {}
    comp_rows = {}      # company -> 行号数组（跨文档聚合用）
    bm25 = None
    vocab = None
    qcache = {}         # qid -> {"q", "vec", "xy", "neighbors"}
    qid = 0


S = State()
EMB_LOCK = threading.Lock()
EMB_PROC = None
EMB_INFO = {"ready": False, "error": None, "load_s": None}
ANSWER_SEM = threading.Semaphore(2)


def load_artifacts():
    need = [VECTOR_DIR / "index.json", VECTOR_DIR / "vectors.npy",
            VIZ_DIR / "build_meta.json", VIZ_DIR / "coords.npy"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        S.err = "缺少索引产物：" + "; ".join(missing)
        return False
    S.idx = json.load(open(VECTOR_DIR / "index.json", encoding="utf-8"))
    S.meta = json.load(open(VIZ_DIR / "build_meta.json", encoding="utf-8"))
    S.rows = S.idx["rows"]
    if S.meta["n"] != len(S.rows):
        S.err = (f"索引不一致：build_meta.n={S.meta['n']} 但 index.json 有 {len(S.rows)} 行。"
                 f"请重跑 build_index.bat")
        return False
    S.mat = np.load(VECTOR_DIR / "vectors.npy", mmap_mode="r")
    S.coords = np.load(VIZ_DIR / "coords.npy")
    S.groups = np.load(VIZ_DIR / "groups.npy")
    S.comps = np.load(VIZ_DIR / "comps.npy")
    S.flags = np.load(VIZ_DIR / "flags.npy")
    S.offsets = np.load(VIZ_DIR / "offsets.npy")
    S.row_of_chunk = {r["chunk_id"]: r["row"] for r in S.rows}
    build_company_index()
    S.ready = True
    return True


def load_bm25():
    if S.bm25 is not None:
        return
    d = np.load(VIZ_DIR / "bm25.npz")
    vocab = pickle.load(open(VIZ_DIR / "bm25_vocab.pkl", "rb"))
    idf = np.zeros(len(vocab), dtype=np.float32)
    doc_len = d["doc_len"].astype(np.float32)
    n_doc = len(doc_len)
    avg = doc_len.mean() or 1.0
    df = np.diff(d["offsets"])
    idf = np.log(1 + (n_doc - df + 0.5) / (df + 0.5)).astype(np.float32)
    S.bm25 = {"postings": d["postings"], "tf": d["tf"], "offsets": d["offsets"],
              "doc_len": doc_len, "avg": avg, "idf": idf}
    S.vocab = vocab


# ==================================================================
#  常驻编码 worker
# ==================================================================
def start_worker():
    global EMB_PROC
    if not VENV_PY.exists():
        EMB_INFO["error"] = f"找不到 venv python: {VENV_PY}"
        return
    try:
        EMB_PROC = subprocess.Popen(
            [str(VENV_PY), str(HERE / "embed_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        line = EMB_PROC.stdout.readline()          # 第一行是 ready
        info = json.loads(line) if line.strip() else {"error": "worker 无输出"}
        if info.get("ready"):
            EMB_INFO.update(ready=True, error=None, load_s=info.get("load_s"))
        else:
            EMB_INFO["error"] = info.get("error", "未知错误")
    except Exception as e:
        EMB_INFO["error"] = f"{type(e).__name__}: {e}"


def embed_query(q):
    """串行调用 worker。返回 (vec, ms) 或抛异常。"""
    if EMB_PROC is None or EMB_PROC.poll() is not None:
        with EMB_LOCK:
            start_worker()
    if EMB_PROC is None:
        raise RuntimeError(EMB_INFO.get("error") or "编码 worker 未启动")
    with EMB_LOCK:
        try:
            EMB_PROC.stdin.write(json.dumps({"q": q}, ensure_ascii=False) + "\n")
            EMB_PROC.stdin.flush()
            line = EMB_PROC.stdout.readline()
        except Exception as e:
            raise RuntimeError(f"编码 worker 通信失败: {e}")
    if not line.strip():
        raise RuntimeError("编码 worker 无响应")
    r = json.loads(line)
    if "error" in r:
        raise RuntimeError(r["error"])
    return r["vec"], r.get("ms", 0)


# ==================================================================
#  检索
# ==================================================================
def _rank_of(scores):
    """把打分数组转成名次数组（1 起，分数越高名次越小）。RRF 只看名次不看分值，
    这样余弦(0~1) 和 BM25(0~30) 量纲不同也不会互相压制。"""
    order = np.argsort(-scores)
    rk = np.empty(len(scores), dtype=np.float32)
    rk[order] = np.arange(1, len(scores) + 1)
    return rk


def build_company_index():
    """company -> 该公司的行号数组。跨文档聚合时按公司切子集用。"""
    d = {}
    for i, r in enumerate(S.rows):
        d.setdefault(r["company"], []).append(i)
    S.comp_rows = {k: np.array(v, dtype=np.int64) for k, v in d.items()}


def _pick(sims, bs, k, idx=None):
    """在（可选的）行号子集上做 RRF 融合并取 top-k。

    名次在子集内重新计算——跨公司聚合时每家公司的块数差很多，
    用全局名次会让小公司的块永远排不进来。
    """
    if idx is not None:
        idx = np.asarray(idx)
        if idx.size == 0:
            return [], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), None
        s, b = sims[idx], bs[idx]
    else:
        idx = np.arange(len(sims))
        s, b = sims, bs
    vr, br = _rank_of(s), _rank_of(b)
    fused = 1.0 / (RRF_K + vr) + 1.0 / (RRF_K + br)
    take = np.argsort(-fused)[:k]
    return idx[take], s[take], b[take], fused[take]


def _neighbor_list(rows_idx, s, b, fused, vrank=None):
    nbs = []
    for rank, (r, si, bi, fu) in enumerate(zip(rows_idx, s, b, fused), 1):
        r = int(r)
        row = S.rows[r]
        nbs.append({
            "rank": rank, "row": r, "chunk_id": row["chunk_id"],
            "company": row["company"], "report_year": row["report_year"],
            "chapter": row["chapter"], "page": row["page"],
            "sim": float(si), "bm25": float(bi), "rrf": float(fu),
            "xy": [float(S.coords[r][0]), float(S.coords[r][1])],
            "snippet": chunk_text(r, limit=SNIPPET),
        })
    return nbs


def retrieve(query, k=DEFAULT_K):
    vec, embed_ms = embed_query(query)
    qv = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(qv))
    if norm > 0:
        qv = qv / norm

    t0 = time.time()
    sims = np.asarray(S.mat) @ qv
    bs, toks = bm25_scores(query)
    # RRF 融合：混合召回。纯向量对"精确数字类"查询很弱——实测
    # 「恒逸石化2025年归属于上市公司股东的净利润」的答案块向量排 404 名，
    # BM25 排第 2 名；融合后稳定进 top-3。
    idx, s, b, fused = _pick(sims, bs, k)
    search_ms = (time.time() - t0) * 1000
    vrank_all, brank_all = _rank_of(sims), _rank_of(bs)
    nbs = _neighbor_list(idx, s, b, fused)
    for n in nbs:
        n["vrank"] = int(vrank_all[n["row"]])
        n["brank"] = int(brank_all[n["row"]])

    # 查询向量在散点图上的落点：top-k 邻居二维坐标的相似度加权重心。
    # UMAP 是流形映射，外推新点需要保留整个 reducer；加权重心既廉价又与
    # 后面"以它为中心召回周围这几个点"的叙事天然一致。UI 上如实标注。
    if nbs:
        w = np.array([max(0.0, n["sim"]) for n in nbs], dtype=np.float32)
        w = w / (w.sum() or 1.0)
        p = np.array([n["xy"] for n in nbs], dtype=np.float32)
        xy = (p * w[:, None]).sum(axis=0)
    else:
        xy = [0.5, 0.5]

    S.qid += 1
    qid = S.qid
    S.qcache[qid] = {"q": query, "vec": vec, "xy": [float(xy[0]), float(xy[1])],
                     "neighbors": nbs}
    if len(S.qcache) > 24:                     # 只留最近若干次
        for old in sorted(S.qcache)[:-24]:
            S.qcache.pop(old, None)

    return {
        "qid": qid, "query": query, "dim": int(S.idx["dim"]),
        "vec_preview": [round(float(v), 6) for v in vec[:8]],
        "vec_curve": [round(float(v), 5) for v in vec[:48]],
        "norm": round(float(np.linalg.norm(np.asarray(vec))), 6),
        "embed_ms": round(embed_ms, 1), "search_ms": round(search_ms, 1),
        "query_xy": [float(xy[0]), float(xy[1])],
        "bm25_terms": toks,          # 命中的查询词，给界面显示"关键词这条走了什么"
        "rrf_k": RRF_K,
        "neighbors": nbs,
    }


def chunk_text(row_idx, limit=None):
    """row_idx 是 vectors.npy 的行号。用 offsets 做 O(1) seek，绝不扫描全量。"""
    row = S.rows[row_idx]
    pid = row["pdf_id"]
    off = int(S.offsets[row_idx])
    with open(CHUNK_DIR / f"{pid}.jsonl", "rb") as f:
        f.seek(off)
        line = f.readline()
    c = json.loads(line.decode("utf-8"))
    if c.get("chunk_id") != row["chunk_id"]:
        raise RuntimeError("索引已过期：chunk_id 与 offsets 不匹配，请重跑 build_index.bat")
    t = c.get("text", "")
    return t[:limit] if limit else t


def bm25_scores(q):
    """返回 (全库 BM25 打分数组, 命中的查询词)。打分逻辑与 kw_search 共用。"""
    load_bm25()
    import jieba
    toks = [t.strip().lower() for t in jieba.lcut(q)]
    toks = [t for t in toks if len(t) >= 2 and t in S.vocab]
    scores = np.zeros(len(S.rows), dtype=np.float32)
    if not toks:
        return scores, []
    b = S.bm25
    for t in set(toks):
        tid = S.vocab[t]
        a, e = int(b["offsets"][tid]), int(b["offsets"][tid + 1])
        rows = b["postings"][a:e]
        tf = b["tf"][a:e].astype(np.float32)
        dl = b["doc_len"][rows]
        denom = tf + K1_ * (1 - B_ + B_ * dl / b["avg"])
        contrib = b["idf"][tid] * tf * (K1_ + 1) / denom
        # 倒排里同一 term 的 postings 行号互不重复（建索引时已按文档聚合词频），
        # 所以可以直接花式索引累加。np.add.at 语义上更保险但慢两个数量级
        # （实测 2.6 秒 → 几十毫秒）。
        scores[rows] += contrib
    return scores, toks


def kw_search(q, k=10):
    scores, toks = bm25_scores(q)
    if not toks:
        return []
    top = np.argsort(-scores)[:k]
    out = []
    for r in top:
        r = int(r)
        if scores[r] <= 0:
            continue
        row = S.rows[r]
        out.append({"row": r, "score": float(scores[r]), "chunk_id": row["chunk_id"],
                    "company": row["company"], "report_year": row["report_year"],
                    "chapter": row["chapter"], "page": row["page"],
                    "xy": [float(S.coords[r][0]), float(S.coords[r][1])],
                    "snippet": chunk_text(r, limit=SNIPPET)})
    return out


K1_, B_ = 1.5, 0.75


# ==================================================================
#  DeepSeek
# ==================================================================
SYSTEM_PROMPT = (
    "你是一个严谨的财报分析助手。只能依据用户提供的片段回答问题，"
    "不得引入片段之外的信息。每个结论句末用 [n] 标注来源片段编号（可多个）。"
    "如果片段不足以回答，直接说明「提供的片段不足以回答该问题」，不要猜测。"
    "用简体中文回答，简洁、分点。"
)


def llm_config():
    base = os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("LLM_MODEL") or ("deepseek-chat" if base else None)
    return base, key, model


def build_context(neighbors):
    """返回 (prompt_text, used_neighbors)，受字数预算约束。"""
    used, budget, parts = [], TOTAL_CHAR_BUDGET, []
    for n in neighbors:
        if budget <= 0:
            break
        txt = chunk_text(n["row"])
        txt = txt[:min(CHUNK_CHAR_BUDGET, budget)]
        budget -= len(txt)
        used.append(n)
        parts.append(f"[{n['rank']}] {n['company']} {n['report_year']}年年度报告 "
                     f"{n['chapter']} P{n['page']}\n{txt}")
    return "\n\n".join(parts), used


def stream_llm(query, context, on_event, system=None, user=None):
    """流式调 DeepSeek，逐 token 回调。on_event({"type":...})

    system/user 都是可选覆盖：跨文档聚合的汇总步要用不同的提示词。
    """
    import requests
    base, key, model = llm_config()
    if not key or not base:
        on_event({"type": "error", "msg": "未配置 DeepSeek。请设置环境变量 "
                                          "LLM_API_KEY（以及 LLM_BASE_URL=https://api.deepseek.com、"
                                          "LLM_MODEL=deepseek-chat）后重启服务。"})
        return
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    payload = {
        "model": model, "stream": True, "temperature": 0.2,
        "messages": [{"role": "system", "content": system or SYSTEM_PROMPT},
                     {"role": "user", "content": user or f"片段：\n{context}\n\n问题：{query}"}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(LLM_RETRY + 1):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            r = requests.post(f"{base}/chat/completions", headers=headers,
                              json=payload, timeout=LLM_TIMEOUT, stream=True)
            if r.status_code in (429,) or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                r.close()
                continue
            if r.status_code != 200:
                on_event({"type": "error", "msg": f"DeepSeek HTTP {r.status_code}: {r.text[:200]}"})
                return
            for raw in r.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                s = raw.decode("utf-8", "ignore")
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except Exception:
                    continue
                t = delta.get("content")
                if t:
                    on_event({"type": "token", "t": t})
            r.close()
            return
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    on_event({"type": "error", "msg": f"DeepSeek 调用失败（已重试 {LLM_RETRY} 次）：{last}"})


# ------------------------------------------------------------------
#  跨文档聚合（Map-Reduce）
# ------------------------------------------------------------------
# 为什么要单独一条路径：像「哪些公司计提了存货跌价准备，它们之后的产能计划是什么」
# 这类问题要求**遍历全部年报**，而 top-k 检索最多覆盖三五家——把 k 调大也没用，
# 因为"哪些"本质上是穷举而非相似度排序。所以拆成 map（逐家抽取结构化结论）
# + reduce（汇总 32 份结论）。
AGG_MAP_SYSTEM = (
    "你是财报信息抽取助手。只依据用户给出的片段作答，不得引入片段之外的信息。"
    "严格输出 JSON，不要任何解释性文字。"
)
AGG_MAP_TMPL = """问题：{q}

以下是该公司年报中与问题最相关的片段：
{ctx}

请针对**这一家公司**回答问题，输出 JSON：
{{
  "relevant": true/false,     // 片段里有没有能回答问题的信息（无关就是 false）
  "answer": "针对这家公司的结论，不超过120字；无信息则留空",
  "evidence": "支撑结论的最短原文片段，不超过100字",
  "citations": [片段编号]
}}
注意：区分「会计政策里提到」和「本期实际发生」。比如存货跌价准备，
会计政策章节一定会写"公司计提存货跌价准备"，那不算本期真的计提；
要看资产减值损失或存货附注里的实际金额。"""

AGG_REDUCE_SYSTEM = (
    "你在汇总多家上市公司年报的逐家结论，回答一个需要跨公司比较的问题。"
    "只用给出的结论作答，不要引入外部知识，也不要臆测未出现在结论里的公司。"
    "用简体中文，分点作答，明确指出公司名称。"
)
AGG_REDUCE_TMPL = """问题：{q}

以下是对 {n} 家公司逐家分析得到的结论（只列出有相关信息的）：
{rows}

请据此回答问题。要求：
1. 先直接回答「哪些公司」，用公司简称列出；
2. 再逐家说明关键数字与产能计划；
3. 如果某些公司信息不足，明确说"未披露/片段不足"，不要编造。"""


def llm_json(system, user, timeout=LLM_TIMEOUT):
    """非流式调用，要求返回 JSON 对象。失败抛异常。"""
    import requests
    base, key, model = llm_config()
    if not key or not base:
        raise RuntimeError("未配置 DeepSeek（需要 LLM_API_KEY / LLM_BASE_URL）")
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    payload = {
        "model": model, "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(LLM_RETRY + 1):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            r = requests.post(f"{base}/chat/completions", headers=headers,
                              json=payload, timeout=timeout)
            if r.status_code in (429,) or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:200]}")
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except json.JSONDecodeError as e:
            last = f"返回不是合法 JSON: {e}"
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"DeepSeek 调用失败（重试 {LLM_RETRY} 次）：{last}")


# ==================================================================
#  HTTP
# ==================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "ragviz/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass    # 关掉访问日志

    def log_error(self, fmt, *args):
        # BaseHTTPRequestHandler 默认把未捕获异常经 log_error 打印，而它走的正是
        # log_message —— 上面静音之后就再也看不到 traceback 了（这个坑已经踩过一次）。
        print("[HTTP] " + (fmt % args), file=sys.stderr, flush=True)

    # ---------- helpers ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bin(self, blob, ctype="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _guard(self):
        if not S.ready:
            self._json({"error": S.err or "索引未就绪"}, 503)
            return False
        return True

    # ---------- GET ----------
    def do_GET(self):
        path, _, qs = self.path.partition("?")
        q = {}
        for kv in qs.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                q[k] = requests_unquote(v)

        if path in ("/", "/index.html"):
            f = WEB_DIR / "index.html"
            if not f.exists():
                return self._json({"error": "缺少 web/index.html"}, 500)
            return self._bin(f.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/health":
            base, key, _ = llm_config()
            return self._json({
                "status": "ok", "index_ready": S.ready, "error": S.err,
                "model_ready": EMB_INFO["ready"], "model_error": EMB_INFO["error"],
                "model_load_s": EMB_INFO["load_s"], "worker_alive": _worker_alive(),
                "llm_ready": bool(base and key),
                "n": (S.meta or {}).get("n"), "dim": (S.meta or {}).get("dim"),
                "is_sample": (S.meta or {}).get("is_sample"),
                "port": self.server.server_port,
            })

        if not self._guard():
            return

        if path == "/api/meta":
            m = dict(S.meta)
            m["palette"] = PALETTE
            m["total_chunks_all"] = 34258          # 全量语料块数（样本模式下做对比）
            return self._json(m)

        if path == "/api/coords":
            n = S.meta["n"]
            # 头必须是 32 字节：前端按 new Float32Array(buf, 32, ...) 取的
            head = struct.pack("<IIIIIIII", 0x52414756, n, 2, 0, 0, 0, 0, 0)
            xy = S.coords.astype(np.float32).tobytes()
            g = S.groups.astype(np.uint8).tobytes()
            c = S.comps.astype(np.uint8).tobytes()
            fl = S.flags.astype(np.uint8).tobytes()
            return self._bin(head + xy + g + c + fl)

        if path == "/api/chunk":
            try:
                row = int(q.get("row", -1))
                if not (0 <= row < len(S.rows)):
                    return self._json({"error": "row 越界"}, 400)
                r = S.rows[row]
                text = chunk_text(row)
                return self._json({
                    "row": row, "chunk_id": r["chunk_id"], "company": r["company"],
                    "stock_code": r.get("stock_code"), "report_year": r["report_year"],
                    "chapter": r["chapter"], "page": r["page"],
                    "char_len": len(text), "text": text,
                })
            except RuntimeError as e:
                return self._json({"error": str(e)}, 409)
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        if path == "/api/vec":
            ent = S.qcache.get(int(q.get("qid", -1)))
            if not ent:
                return self._json({"error": "qid 不存在"}, 404)
            return self._json({"dim": len(ent["vec"]), "vec": ent["vec"]})

        if path == "/api/kw":
            try:
                return self._json({"results": kw_search(q.get("q", ""),
                                                        int(q.get("k", 10)))})
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        return self._json({"error": "not found"}, 404)

    # ---------- POST ----------
    def do_POST(self):
        path = self.path.split("?")[0]
        if not self._guard():
            return

        if path == "/api/retrieve":
            body = self._body()
            query = (body.get("query") or "").strip()
            if not query:
                return self._json({"error": "query 为空"}, 400)
            try:
                return self._json(retrieve(query, int(body.get("k", 8))))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        if path == "/api/answer":
            body = self._body()
            ent = S.qcache.get(int(body.get("qid", -1)))
            if not ent:
                return self._json({"error": "qid 不存在，请重新提问"}, 404)
            if not ANSWER_SEM.acquire(blocking=False):
                return self._json({"error": "已有回答正在生成，请稍候"}, 429)
            try:
                context, used = build_context(ent["neighbors"])
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                    self.wfile.write(b"%X\r\n" % len(line) + line + b"\r\n")
                    self.wfile.flush()

                send({"type": "start", "n_context": len(used),
                      "chars": sum(len(chunk_text(n["row"])[:CHUNK_CHAR_BUDGET]) for n in used)})
                stream_llm(ent["q"], context, send)
                send({"type": "citations", "items": [
                    {"no": n["rank"], "row": n["row"], "chunk_id": n["chunk_id"],
                     "company": n["company"], "year": n["report_year"],
                     "chapter": n["chapter"], "page": n["page"], "sim": n["sim"]}
                    for n in used]})
                send({"type": "done"})
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:
                    pass
            finally:
                ANSWER_SEM.release()
            return

        if path == "/api/aggregate":
            body = self._body()
            query = (body.get("query") or "").strip()
            if not query:
                return self._json({"error": "query 为空"}, 400)
            k = int(body.get("k", 6))
            only = body.get("companies")
            if not ANSWER_SEM.acquire(blocking=False):
                return self._json({"error": "已有回答正在生成，请稍候"}, 429)
            try:
                companies = sorted(S.comp_rows)
                if only:
                    want = set(only)
                    companies = [c for c in companies if c in want]
                if not companies:
                    return self._json({"error": "没有可聚合的公司，请先建索引"}, 400)

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                    self.wfile.write(b"%X\r\n" % len(line) + line + b"\r\n")
                    self.wfile.flush()

                send({"type": "start", "companies": len(companies), "k": k})
                # 查询只编码一次，32 家公司共用；每条链路的名次在自己的子集内重算
                vec, embed_ms = embed_query(query)
                qv = np.asarray(vec, dtype=np.float32)
                nrm = float(np.linalg.norm(qv))
                if nrm > 0:
                    qv = qv / nrm
                sims = np.asarray(S.mat) @ qv
                bs, toks = bm25_scores(query)
                send({"type": "query", "dim": len(vec), "embed_ms": round(embed_ms, 1),
                      "vec_preview": [round(float(v), 6) for v in vec[:8]],
                      "bm25_terms": toks})

                results, failed = [], 0
                for i, comp in enumerate(companies, 1):
                    idx, s, b, fu = _pick(sims, bs, k, S.comp_rows[comp])
                    nbs = _neighbor_list(idx, s, b, fu)
                    if not nbs:
                        send({"type": "company", "i": i, "n": len(companies),
                              "company": comp, "state": "empty"})
                        continue
                    ctx = "\n\n".join(
                        f"[{x['rank']}] {x['company']} {x['report_year']}年年度报告 "
                        f"{x['chapter']} P{x['page']}\n{chunk_text(x['row'])[:1000]}"
                        for x in nbs)
                    try:
                        r = llm_json(AGG_MAP_SYSTEM, AGG_MAP_TMPL.format(q=query, ctx=ctx))
                    except Exception as e:
                        failed += 1
                        send({"type": "company", "i": i, "n": len(companies),
                              "company": comp, "state": "error", "reason": str(e)[:120]})
                        continue
                    ans = (r.get("answer") or "").strip()
                    rel = bool(r.get("relevant")) and bool(ans)
                    rec = {"company": comp, "relevant": rel, "answer": ans,
                           "evidence": (r.get("evidence") or "").strip()[:200],
                           "rows": [x["row"] for x in nbs],
                           "first": nbs[0]}
                    if rel:
                        results.append(rec)
                    send({"type": "company", "i": i, "n": len(companies),
                          "company": comp, "state": "relevant" if rel else "irrelevant",
                          "answer": ans, "evidence": rec["evidence"]})

                send({"type": "map_done", "relevant": len(results),
                      "total": len(companies), "failed": failed})

                if not results:
                    send({"type": "token", "t": "逐家扫描了 %d 份年报，没有任何一份的片段"
                                                "足以回答该问题。" % len(companies)})
                    send({"type": "done"})
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return

                rows_txt = "\n".join(
                    f"- {r['company']}：{r['answer']}"
                    + (f"（证据：{r['evidence']}）" if r["evidence"] else "")
                    for r in results)
                send({"type": "reduce_start", "n": len(results)})
                stream_llm(
                    query, "", send,
                    system=AGG_REDUCE_SYSTEM,
                    user=AGG_REDUCE_TMPL.format(q=query, n=len(companies), rows=rows_txt))
                send({"type": "citations", "items": [
                    {"no": i, "company": r["company"],
                     "year": r["first"]["report_year"], "chapter": r["first"]["chapter"],
                     "page": r["first"]["page"], "row": r["first"]["row"],
                     "sim": r["first"]["sim"], "answer": r["answer"]}
                    for i, r in enumerate(results, 1)]})
                send({"type": "done"})
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:
                    pass
            finally:
                ANSWER_SEM.release()
            return

        return self._json({"error": "not found"}, 404)


def requests_unquote(s):
    from urllib.parse import unquote_plus
    return unquote_plus(s)


def _worker_alive():
    return EMB_PROC is not None and EMB_PROC.poll() is None


def pick_port():
    for p in PORT_CANDIDATES:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, p))
                return p
            except OSError:
                continue
    return 0


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ok = load_artifacts()
    if not ok:
        print("=" * 70)
        print(f"[错误] {S.err}")
        print("       建索引步骤：")
        print(r"         1) qwen3-embedding\.venv\Scripts\python.exe rag\embed.py --sample")
        print(r"         2) python rag\merge_vectors.py")
        print(r"         3) python rag\build_viz_index.py")
        print("=" * 70)

    if S.ready:
        try:
            load_bm25()                     # 预热倒排索引，免得首次查询卡一下
            bm25_mb = (VIZ_DIR / "bm25.npz").stat().st_size / 1e6
            print(f"  BM25     : 已载入 {len(S.vocab):,} 词（倒排 {bm25_mb:.1f}MB）")
        except Exception as e:
            print(f"  BM25     : ✗ 载入失败，混合召回将退化为纯向量（{e}）")

    port = args.port or pick_port()
    if not port:
        print("[错误] 候选端口全被占用，请用 --port 指定")
        return 1

    threading.Thread(target=start_worker, daemon=True).start()

    srv = ThreadingHTTPServer((HOST, port), Handler)
    srv.daemon_threads = True
    url = f"http://localhost:{port}/"
    print("=" * 70)
    print(f"  RAG 可视化已启动")
    print(f"  本机访问 : {url}")
    print(f"  局域网   : http://<本机IP>:{port}/")
    if S.ready:
        m = S.meta
        tag = "【样本模式】" if m.get("is_sample") else "【全量】"
        print(f"  语料     : {tag} {m['n']} 块 / {m['n_pdf']} 份 / dim={m['dim']}")
    else:
        print(f"  语料     : ✗ 索引未就绪")
    base, key, model = llm_config()
    print(f"  DeepSeek : {'已配置 ' + str(model) if (base and key) else '未配置（第④步会提示设置方法）'}")
    print(f"  模型加载中（后台，约 5-8 秒）…")
    print("  Ctrl+C 停止")
    print("=" * 70)

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止]")
    finally:
        srv.server_close()
        if EMB_PROC and EMB_PROC.poll() is None:
            try:
                EMB_PROC.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                EMB_PROC.stdin.flush()
                EMB_PROC.wait(timeout=5)
            except Exception:
                EMB_PROC.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
