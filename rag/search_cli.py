# -*- coding: utf-8 -*-
r"""阶段6：检索验收 CLI。

⚠ 用 venv 的 python 跑（要 torch 编码 query）：
    qwen3-embedding\.venv\Scripts\python.exe _work\search_cli.py "恒逸石化2024年净利润"

查询侧加 instruct 前缀、文档侧不加（Qwen3-Embedding 的非对称检索约定）。
向量已 L2 归一化，点积即余弦相似度。

用法:
    python search_cli.py "研发投入占营业收入比例" --k 5
    python search_cli.py --batch           # 跑预置验收查询集
    python search_cli.py "净利润" --company 恒逸石化 --year 2024
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "qwen3-embedding"))

TASK = "Given a Chinese A-share annual report query, retrieve the passage that answers it"

BATCH_QUERIES = [
    ("恒逸石化2024年归属于上市公司股东的净利润是多少", None),
    ("吉林化纤2023年主营业务及产能情况", None),
    ("公司研发投入金额及占营业收入比例", None),
    ("前五名客户销售额占年度销售总额比例", None),
    ("董事会秘书姓名及联系方式", None),
    ("公司涤纶长丝产品的产能和产量", None),
    ("贵州茅台2024年营业收入", "反例-应低分"),
]


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_all():
    # 不 import ann_common —— 它依赖 pymupdf，而本脚本跑在只有 torch 的 venv 里
    import numpy as np

    CHUNK_DIR = REPO / "年报_合成纤维制造" / "_derived" / "chunks"
    VECTOR_DIR = REPO / "年报_合成纤维制造" / "_derived" / "vectors"
    idx_path = VECTOR_DIR / "index.json"
    if not idx_path.exists():
        print("找不到 index.json，先跑 merge_vectors.py")
        return None
    idx = json.load(open(idx_path, encoding="utf-8"))
    mat = np.load(VECTOR_DIR / "vectors.npy", mmap_mode="r")

    # 把块的正文读进内存，按 row 索引
    texts = {}
    for pid in {r["pdf_id"] for r in idx["rows"]}:
        with open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8") as f:
            for i, ln in enumerate(f):
                c = json.loads(ln)
                texts[c["chunk_id"]] = c
    return idx, mat, texts


def main(argv=None):
    setup_console()
    ap = argparse.ArgumentParser(description="年报向量检索")
    ap.add_argument("query", nargs="?", help="查询语句")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--company")
    ap.add_argument("--year", type=int)
    ap.add_argument("--chapter")
    ap.add_argument("--batch", action="store_true", help="跑预置验收查询集")
    ap.add_argument("--show", type=int, default=180, help="每条的正文预览字数")
    args = ap.parse_args(argv)

    loaded = load_all()
    if loaded is None:
        return 1
    idx, mat, texts = loaded

    import numpy as np
    from qwen3_embed import QwenEmbedder

    # 过滤候选的行号
    allow = None
    if args.company or args.year or args.chapter:
        allow = set()
        for r in idx["rows"]:
            if args.company and args.company not in r["company"]:
                continue
            if args.year and r["report_year"] != args.year:
                continue
            if args.chapter and args.chapter not in r["chapter"]:
                continue
            allow.add(r["row"])

    print(f"[库] {mat.shape[0]} 块 × {mat.shape[1]} 维  模型={idx['model_id']} 量化={idx['quant']}")

    emb = QwenEmbedder(dtype=__import__("torch").float32, verbose=False)
    if idx["quant"] == "int8":
        from torch.ao.quantization import quantize_dynamic
        import torch
        emb.model = quantize_dynamic(emb.model, {torch.nn.Linear}, dtype=torch.qint8)
        print("[库] 已按 int8 对齐查询侧编码")

    queries = BATCH_QUERIES if args.batch else [(args.query, None)]
    if not queries or queries[0][0] is None:
        print("给个查询，或用 --batch")
        return 1

    for q, tag in queries:
        qv = emb.encode([q], is_query=True, task=TASK)[0].numpy()
        sims = np.asarray(mat) @ qv
        if allow is not None:
            mask = np.full(sims.shape, -1e9, dtype=np.float32)
            for r in allow:
                mask[r] = 0.0
            sims = sims + mask
        order = np.argsort(-sims)[:args.k]

        print()
        print("=" * 78)
        print(f"查询: {q}" + (f"   [{tag}]" if tag else ""))
        print("-" * 78)
        for rank, r in enumerate(order, 1):
            row = idx["rows"][r]
            c = texts.get(row["chunk_id"], {})
            body = " ".join(c.get("text", "").split())[:args.show]
            print(f"{rank}. [{sims[r]:.4f}] {row['company']} {row['report_year']} | "
                  f"{row['chapter']} | P{row['page']} | #{c.get('chunk_index')} "
                  f"| {c.get('char_len')}字{'  [表]' if c.get('has_table') else ''}")
            print(f"     {body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
