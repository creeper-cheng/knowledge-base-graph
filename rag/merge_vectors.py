# -*- coding: utf-8 -*-
"""阶段5：把所有 vectors/{pid}.npy 合并成一个大矩阵 + 行号对齐表。

产出 _derived/vectors/vectors.npy (memmap) 与 index.json：
  第 i 行 ↔ index.json['rows'][i] 的 chunk_id
检索时先算 query·vectors.T，取 top-k 行号，再回 index.json 查 chunk。

会校验所有 done.json 的 model/quant/dim 一致——不一致拒绝建索引（否则相似度分布被污染）。

用法: python merge_vectors.py
"""
import json
import os
import sys

import numpy as np

from ann_common import CHUNK_DIR, DERIVED, VECTOR_DIR


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _chunk_digest(pid):
    """与 embed.py 的 chunk_digest 保持一致：对各块的 chunk_sha1 再取 sha256。"""
    import hashlib
    h = hashlib.sha256()
    with open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                h.update(json.loads(ln)["chunk_sha1"].encode("ascii"))
    return h.hexdigest()


def main():
    setup_console()
    dones = sorted(VECTOR_DIR.glob("*.done.json"))
    if not dones:
        print("没有向量产物，先跑 embed.py")
        return 1

    metas, bad = [], []
    for d in dones:
        pid = d.name[:-len(".done.json")]
        if not (VECTOR_DIR / f"{pid}.npy").exists():
            bad.append((pid, "缺 .npy"))
            continue
        rec = json.load(open(d, encoding="utf-8"))
        n_chunks = sum(1 for _ in open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8"))
        if rec["count"] != n_chunks:
            bad.append((pid, f"行数不符 {rec['count']} != {n_chunks}"))
            continue
        # ⚠ 只比行数不够：切块内容变了但行数恰好相同时，旧向量会被静默错位使用。
        # 必须比对 embed.py 写下的 chunks_sha256（它是对各块 chunk_sha1 的 sha256）。
        want = _chunk_digest(pid)
        if rec.get("chunks_sha256") != want:
            bad.append((pid, "切块已变，向量过期"))
            continue
        metas.append((pid, rec))

    if not metas:
        print("没有可用的向量产物")
        return 1

    keys = {(m["model_id"], m["quant"], m["dim"], m["dtype"]) for _, m in metas}
    if len(keys) > 1:
        print(f"[错误] 向量参数不一致，拒绝建索引：{keys}")
        print("       删掉不一致的 vectors/*.npy 与 *.done.json 后重跑 embed.py")
        return 1
    model_id, quant, dim, dtype = keys.pop()

    total = sum(m["count"] for _, m in metas)
    print(f"合并 {len(metas)} 份 / {total} 块  模型={model_id} 量化={quant} dim={dim}")

    out = np.lib.format.open_memmap(
        VECTOR_DIR / "vectors.npy", mode="w+", dtype=np.float32, shape=(total, dim)
    )
    rows, pos = [], 0
    for pid, m in metas:
        v = np.load(VECTOR_DIR / f"{pid}.npy")
        if v.shape != (m["count"], dim):
            bad.append((pid, f"形状异常 {v.shape}"))
            continue
        out[pos:pos + m["count"]] = v
        # 与 jsonl 行序严格对齐
        with open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8") as f:
            for i, ln in enumerate(f):
                rec = json.loads(ln)
                rows.append({
                    "chunk_id": rec["chunk_id"],
                    "pdf_id": pid,
                    "company": rec["company"],
                    "report_year": rec["report_year"],
                    "chapter": rec["chapter"],
                    "page": rec["page"],
                    "row": pos + i,
                })
        pos += m["count"]

    out.flush()
    del out

    idx = {
        "model_id": model_id, "quant": quant, "dim": dim, "dtype": dtype,
        "total": pos, "rows": rows,
    }
    with open(VECTOR_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"[索引] → {VECTOR_DIR / 'index.json'}  行数 {pos}")
    if bad:
        print(f"[警告] {len(bad)} 份被跳过：{bad[:5]}")
    # 自检：矩阵行数 == 对齐表行数
    mm = np.load(VECTOR_DIR / "vectors.npy", mmap_mode="r")
    print(f"[自检] vectors.npy shape={mm.shape}  index.rows={len(rows)}  "
          f"{'✓' if mm.shape[0] == len(rows) else '✗ 不一致'}")
    norms = np.linalg.norm(mm[:200], axis=1)
    print(f"[自检] 前200行 L2 范数 mean={norms.mean():.4f} min={norms.min():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
