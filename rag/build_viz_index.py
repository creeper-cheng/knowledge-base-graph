# -*- coding: utf-8 -*-
r"""阶段7（可视化索引）：UMAP 降维 + BM25 倒排 + chunk 字节偏移表。

跑在**主环境**（需要 jieba 与 umap-learn；不需要 torch）。
输入是 merge_vectors.py 的产出 vectors.npy + index.json，输出到 _derived/viz/：

    coords.npy     (N,2) float32  归一化到 [0,1]，给散点图用
    groups.npy     (N,)  uint8    chapter_no（默认着色维度）
    comps.npy      (N,)  uint8    公司序号（company 表在 build_meta.json）
    flags.npy      (N,)  uint8    bit0 = has_table
    offsets.npy    (N,)  int64    每行在它所属 chunks/{pdf_id}.jsonl 里的字节偏移
    bm25.npz       倒排 CSR（postings/tf/offsets/doc_len）
    bm25_vocab.pkl term -> tid
    build_meta.json

内存要点：BM25 用 array('i')/array('h') 累加，而不是 Python list of tuple
——后者 1500 万条 postings 要吃掉 1GB+ 的对象开销。

用法: python _work/build_viz_index.py
"""
import array
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DERIVED = REPO / "年报_合成纤维制造" / "_derived"
CHUNK_DIR = DERIVED / "chunks"
VECTOR_DIR = DERIVED / "vectors"
VIZ_DIR = DERIVED / "viz"

K1, B = 1.5, 0.75
MIN_TOKEN_LEN = 2
STOPWORDS = set("的 了 和 与 及 或 为 在 是 等 有 中 上 下 之 其 该 本 我们 公司 以及 其中".split())


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ------------------------------------------------------------------
#  UMAP
# ------------------------------------------------------------------
def normalize_xy(xy):
    """统一半径 + 99 分位裁剪。不要 per-axis min-max——那会破坏簇形状与距离比例。"""
    med = np.median(xy, axis=0)
    r = float(np.percentile(np.abs(xy - med), 99))
    if not np.isfinite(r) or r <= 0:
        r = float(np.abs(xy - med).max()) or 1.0
    xy = np.clip((xy - med) / (r * 1.05), -1.2, 1.2)
    return ((xy + 1.2) / 2.4).astype(np.float32)


def reduce_2d(X):
    n = X.shape[0]
    if n <= 2:
        # 数据太少，UMAP 无意义，铺成一个圆
        ang = np.linspace(0, 2 * np.pi, max(n, 1), endpoint=False)
        return np.stack([np.cos(ang) * 0.35 + 0.5, np.sin(ang) * 0.35 + 0.5], axis=1)[:n].astype(np.float32), "circle"
    import umap
    nn = max(2, min(int(round(n ** 0.5 / 2)), 30, n - 2))
    print(f"[umap] n={n} n_neighbors={nn} ...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=nn, min_dist=0.08, n_components=2, metric="cosine",
        random_state=42, init=("spectral" if n <= 20000 else "random"),
        low_memory=(n > 20000), n_jobs=8, verbose=True,
    )
    xy = reducer.fit_transform(X)
    print(f"[umap] 完成 {time.time() - t0:.1f}s")
    return normalize_xy(xy), f"umap(n_neighbors={nn}, min_dist=0.08, cosine, seed=42)"


# ------------------------------------------------------------------
#  BM25
# ------------------------------------------------------------------
def tokenize(text):
    import jieba
    out = []
    for t in jieba.lcut(text):
        t = t.strip().lower()
        if len(t) < MIN_TOKEN_LEN or t in STOPWORDS:
            continue
        if not any(c.isalnum() or "一" <= c <= "鿿" for c in t):
            continue
        out.append(t)
    return out


def main():
    setup_console()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-coords", action="store_true",
                    help="复用已有 coords.npy（只重建 BM25/元数据时用，省去 UMAP 时间）")
    args = ap.parse_args()

    idx_path = VECTOR_DIR / "index.json"
    npy_path = VECTOR_DIR / "vectors.npy"
    if not idx_path.exists() or not npy_path.exists():
        print(f"[错误] 找不到向量库产物：{idx_path} / {npy_path}")
        print("       请先跑：qwen3-embedding\\.venv\\Scripts\\python.exe _work\\embed.py --sample")
        print("       再跑：python _work\\merge_vectors.py")
        return 1

    idx = json.load(open(idx_path, encoding="utf-8"))
    rows = idx["rows"]
    N = len(rows)
    dim = idx["dim"]
    print(f"[载入] {N} 块 × {dim} 维  模型={idx['model_id']} 量化={idx['quant']}")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. offsets + 公司/章节/标记（一次遍历 chunks）----------
    by_pdf = OrderedDict()
    for i, r in enumerate(rows):
        by_pdf.setdefault(r["pdf_id"], []).append(i)
    print(f"[扫描] {len(by_pdf)} 个 chunk 文件 ...")

    offsets = np.zeros(N, dtype=np.int64)
    chapter_no = np.zeros(N, dtype=np.uint8)
    flags = np.zeros(N, dtype=np.uint8)
    companies = sorted({r["company"] for r in rows})
    comp_of = {c: i for i, c in enumerate(companies)}
    comps = np.zeros(N, dtype=np.uint8)

    t0 = time.time()
    chapter_label = {}
    for pid, row_ids in by_pdf.items():
        p = CHUNK_DIR / f"{pid}.jsonl"
        with open(p, "rb") as f:
            for k, line in enumerate(f):
                if k >= len(row_ids):
                    break
                rid = row_ids[k]
                offsets[rid] = f.tell() - len(line)
                c = json.loads(line.decode("utf-8"))
                cn = min(255, max(0, int(c.get("chapter_no") or 0)))
                chapter_no[rid] = cn
                # 注意：index.json.rows 里没有 chapter_no，标签只能从 chunk 原文取
                chapter_label[cn] = c.get("chapter") or "前置"
                comps[rid] = comp_of.get(c.get("company", ""), 0)
                if c.get("has_table"):
                    flags[rid] |= 1
    print(f"[扫描] 完成 {time.time() - t0:.1f}s  章节 {len(chapter_label)} 类")

    # ---------- 2. BM25 倒排（用 array 累加，避免 Python 对象开销）----------
    print("[bm25] 分词并建倒排 ...")
    t0 = time.time()
    vocab = {}
    post_rows = {}    # tid -> array('i')
    post_tfs = {}     # tid -> array('h')
    doc_len = np.zeros(N, dtype=np.int32)
    n_tok = 0

    for pid, row_ids in by_pdf.items():
        with open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8") as f:
            for k, line in enumerate(f):
                if k >= len(row_ids):
                    break
                rid = row_ids[k]
                text = json.loads(line).get("text", "")
                toks = tokenize(text)
                doc_len[rid] = len(toks)
                n_tok += len(toks)
                if not toks:
                    continue
                tf = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                for t, n in tf.items():
                    tid = vocab.get(t)
                    if tid is None:
                        tid = len(vocab)
                        vocab[t] = tid
                        post_rows[tid] = array.array("i")
                        post_tfs[tid] = array.array("h")
                    post_rows[tid].append(rid)
                    post_tfs[tid].append(min(n, 32767))
    print(f"[bm25] 词表 {len(vocab)} 词 / {n_tok} token / {time.time() - t0:.1f}s")

    V = len(vocab)
    offs = np.zeros(V + 1, dtype=np.int64)
    for tid in range(V):
        offs[tid + 1] = offs[tid] + len(post_rows[tid])
    total = int(offs[-1])
    postings = np.empty(total, dtype=np.int32)
    tfs = np.empty(total, dtype=np.int16)
    for tid in range(V):
        a, b = offs[tid], offs[tid + 1]
        postings[a:b] = np.frombuffer(post_rows[tid], dtype=np.int32)
        tfs[a:b] = np.frombuffer(post_tfs[tid], dtype=np.int16)
        post_rows[tid] = post_tfs[tid] = None   # 及时释放
    del post_rows, post_tfs

    np.savez_compressed(VIZ_DIR / "bm25.npz", postings=postings, tf=tfs,
                        offsets=offs, doc_len=doc_len)
    with open(VIZ_DIR / "bm25_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=4)
    sz = (VIZ_DIR / "bm25.npz").stat().st_size / 1e6
    print(f"[bm25] postings={total} 落盘 {sz:.1f}MB")

    # ---------- 3. UMAP ----------
    cached = VIZ_DIR / "coords.npy"
    if args.keep_coords and cached.exists() and np.load(cached).shape[0] == N:
        coords = np.load(cached)
        how = "reused"
        print("[umap] --keep-coords：复用已有 coords.npy")
    else:
        X = np.asarray(np.load(npy_path, mmap_mode="r"), dtype=np.float32)
        coords, how = reduce_2d(X)
        del X

    # ---------- 4. 落盘 ----------
    np.save(VIZ_DIR / "coords.npy", coords)
    np.save(VIZ_DIR / "groups.npy", chapter_no)
    np.save(VIZ_DIR / "comps.npy", comps)
    np.save(VIZ_DIR / "flags.npy", flags)
    np.save(VIZ_DIR / "offsets.npy", offsets)

    chapters = chapter_label
    meta = {
        "n": N, "dim": dim, "model_id": idx["model_id"], "quant": idx["quant"],
        "n_pdf": len(by_pdf),
        "is_sample": N < 30000,
        "coords": how,
        "build_ms": None,
        "companies": companies,
        "years": sorted({r["report_year"] for r in rows}),
        "chapters": [{"no": k, "label": chapters[k]} for k in sorted(chapters)],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(VIZ_DIR / "build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("-" * 60)
    print(f"coords {coords.shape} 范围 [{coords.min():.3f},{coords.max():.3f}] "
          f"finite={np.isfinite(coords).all()}")
    print(f"章节类数 {len(chapters)}  公司数 {len(companies)}  年份 {meta['years']}")
    print(f"[完成] → {VIZ_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
