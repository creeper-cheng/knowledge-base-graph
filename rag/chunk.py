# -*- coding: utf-8 -*-
"""阶段2：章节识别 + 800 字切块（重合 120），产出 _derived/chunks/{pdf_id}.jsonl。

章节识别四道闸（详见 CLAUDE 计划，实测得出）：
  1. 行首正则 ^第X节，长度上限 40 字，排除目录引导点行
  2. 尾部特征排除交叉引用（含引号/之/中文数字顿号序号；含"章"或以"规定"开头）
  3. 形态③：编号与标题名分处两行/两 cell（`第一节 | 释义`）→ 回看下一段
  4. 文档级：单页 >=3 个候选判为目录页整页丢弃；再按编号单调递增接受

切块：800 软上限 / 120 重合 / 句末回溯 200 / 硬上限 1100 / 尾块 <60 并入前块。
不跨章节；允许跨页，页码标签取起始页。

用法:
    python chunk.py                # 全量
    python chunk.py --only xxx     # 指定 pdf_id
    python chunk.py --stats        # 只打印统计，不写文件
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter

from ann_common import CHUNK_DIR, DERIVED, TEXT_DIR, is_dotleader

# ---- 切块参数（改这里会让已算的向量全部作废，done.json 的 sha256 会自动检测） ----
TARGET = 800
OVERLAP = 120
SENT_WIN = 200
HARD_MAX = 1100
MIN_TAIL = 60
MIN_KEEP = 10             # 低于这个字数的块视为碎片直接丢弃
SENT_END = tuple("。！？；!?;…")
INCLUDE_FRONT = True      # 第一节之前的内容（封面/重要提示/释义）是否入库
CHUNK_PARAMS = {"target": TARGET, "overlap": OVERLAP, "sent_win": SENT_WIN,
                "hard_max": HARD_MAX, "min_tail": MIN_TAIL}

CN = "一二三四五六七八九十"
CN_NUM = {c: i for i, c in enumerate("一二三四五六七八九", 1)}
HEAD_RE = re.compile(rf"^第([{CN}]+)节(.*)$")
CROSSREF_BAD = '"“”之'
CANON_NAME = {
    "重要提示、目录和释义", "公司简介和主要财务指标", "管理层讨论与分析",
    "公司治理", "环境和社会责任", "环境与社会责任", "公司治理、环境和社会",
    "重要事项", "股份变动及股东情况", "优先股相关情况", "债券相关情况",
    "财务报告", "备查文件目录",
}


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def norm(s):
    """去空白和单元格分隔符——`第一节 | 释义` 与 `第一节释义` 归一成同一种。"""
    return re.sub(r"[\s　|]+", "", s or "")


def cn2int(s):
    if s == "十":
        return 10
    if "十" in s:
        a, b = s.split("十", 1)
        return (CN_NUM.get(a, 1) if a else 1) * 10 + (CN_NUM.get(b, 0) if b else 0)
    return CN_NUM.get(s, 0)


def heading_cand(text, next_text):
    """判断一段文本是否是节标题，是则返回 (节号, 节名)。"""
    if is_dotleader(text):
        return None
    s = norm(text)
    if not s:
        return None

    m = HEAD_RE.match(s)
    if m and len(s) <= 70:
        pass                                    # 形态①②：正常标题行，长度上限挡掉目录行
    else:
        # 兜底：OCR 页没有 bbox、页眉剔不掉，会把页眉和正文粘成一行，例如
        #   2023 年年度报告第一节重要提示、目录和释义公司董事会、监事会及董事…
        #   XX股份有限公司 2024 年年度报告全文第八节优先股相关情况…
        # 这种行很长，不能用总长度挡；改为只校验"第X节"之前的前缀像不像页眉，
        # 避免把交叉引用（详见第三节…）误判成标题。
        m2 = re.search(rf"第([{CN}]+)节", s)
        if not m2:
            return None
        prefix = s[:m2.start()]
        if not (len(prefix) <= 40 and re.search(r"年度报告|股份|有限公司", prefix)):
            return None
        m = HEAD_RE.match(s[m2.start():])
        if not m:
            return None

    num, tail = cn2int(m.group(1)), m.group(2)
    if not num:
        return None
    # 尾部长尾（OCR 粘连）截到标准节名
    for cn_name in sorted(CANON_NAME, key=len, reverse=True):
        if tail.startswith(cn_name):
            tail = cn_name
            break

    if not tail:                                   # 形态③：标题名在下一段
        t2 = norm(next_text)
        if not t2 or len(t2) > 20 or is_dotleader(next_text):
            return None
        return num, t2
    if any(c in tail for c in CROSSREF_BAD):       # 第三节…"之"十一、
        return None
    if re.match(rf"^[{CN}0-9]+[、.]", tail):       # 第八节七.17 长期股权投资
        return None
    if "章" in tail or tail.startswith("规定"):    # 第十章规定的…
        return None
    if tail in CANON_NAME:
        return num, tail
    if len(tail) <= 20 and not re.search(r"[0-9、]", tail):
        return num, tail
    return None


def detect_chapters(items):
    """
    items: [{'text','page'}...]，已按阅读顺序。
    返回 [(start_item_idx, chapter_no, chapter_name, page)]。
    """
    cands = []
    for i, it in enumerate(items):
        nx = items[i + 1]["text"] if i + 1 < len(items) else ""
        h = heading_cand(it["text"], nx)
        if h:
            cands.append((i, it["page"], h[0], h[1]))

    # 闸4a：单页 >=3 个候选 → 目录页，整页丢弃
    per_page = Counter(pg for _, pg, _, _ in cands)
    cands = [c for c in cands if per_page[c[1]] < 3]

    # 闸4b：编号单调递增（允许 +1..+3 跳号容错 OCR/漏检），回跳与重复丢弃
    seq, cur = [], 0
    for c in cands:
        if cur < c[2] <= cur + 3:
            seq.append(c)
            cur = c[2]
    return seq


def chapter_label(no, name):
    return f"第{no}节 {name}"


# ------------------------------------------------------------------
#  切块
# ------------------------------------------------------------------
def split_long(text, limit=HARD_MAX):
    """超长单行先按句末、再按逗号、最后硬切，保证原子不超过 limit。"""
    if len(text) <= limit:
        return [text]
    pieces = []
    for part in re.split(r"(?<=[。！？；!?;…])", text):
        if not part:
            continue
        if len(part) <= limit:
            pieces.append(part)
        else:
            for sub in re.split(r"(?<=[，,、])", part):
                while len(sub) > limit:
                    pieces.append(sub[:limit])
                    sub = sub[limit:]
                if sub:
                    pieces.append(sub)
    # 合并相邻短片段，避免碎片
    merged = []
    for p in pieces:
        if merged and len(merged[-1]) + len(p) <= limit:
            merged[-1] += p
        else:
            merged.append(p)
    return merged


NEW_UNIT_RE = re.compile(
    r"^(第[一二三四五六七八九十]+[节章节]"
    r"|[一二三四五六七八九十]+[、.]"
    r"|（[一二三四五六七八九十0-9]+）"
    r"|\([一二三四五六七八九十0-9]+\)"
    r"|\d+[、.])"
)


def merge_paragraphs(items, max_para=420):
    """
    把 PDF 的视觉折行还原成段落。

    必要性：PDF 里一个段落被换行切成多行，每行都是一个原子，而句子末尾的"。"只出现在
    段落最后一行——不回并的话，句末收尾率只有 ~35%。回并后原子 ≈ 段落，句末自然对齐。
    表格行（kind='table'）与疑似标题行不参与合并。
    """
    out = []
    for it in items:
        cur = out[-1] if out else None
        mergeable = (
            cur is not None
            and it.get("kind") == "line"
            and cur.get("kind") == "line"
            and not cur["text"].rstrip().endswith(SENT_END)
            and len(cur["text"]) < max_para
            and len(it["text"]) >= 10
            and not NEW_UNIT_RE.match(it["text"])
        )
        if mergeable:
            cur["text"] += it["text"]
            cur["page_end"] = it["page"]
        else:
            out.append({"text": it["text"], "page": it["page"],
                        "page_end": it["page"], "kind": it.get("kind", "line")})
    return out


def make_atoms(items):
    atoms = []
    off = 0
    for it in merge_paragraphs(items):
        for piece in split_long(it["text"]):
            atoms.append({"text": piece, "page": it["page"],
                          "page_end": it.get("page_end", it["page"]), "start": off})
            off += len(piece) + 1        # +1 是 join 时的 '\n'
    for a in atoms:
        a["end"] = a["start"] + len(a["text"])
    return atoms


def chunk_ranges(atoms):
    """返回 [(i, j)]，闭区间，表示 atoms[i..j] 构成一块。"""
    n = len(atoms)
    if n == 0:
        return []
    out = []
    pos = 0
    while pos < n:
        # j = 最后一个「累计长度仍不超过 TARGET」的原子
        j = pos
        while j + 1 < n and atoms[j + 1]["end"] - atoms[pos]["start"] <= TARGET:
            j += 1
        # 在 [j-SENT_WIN, j] 内回找句末，让块尽量在句号/分号处收尾
        cut = None
        limit = atoms[j]["end"] - SENT_WIN
        k = j
        while k >= pos and atoms[k]["end"] >= limit:
            if atoms[k]["text"].rstrip().endswith(SENT_END):
                cut = k
                break
            k -= 1
        if cut is None:
            cut = j                      # 窗口内无句末，就切在 800 以内
        out.append((pos, cut))
        if cut >= n - 1:
            break
        target = atoms[cut]["end"] - OVERLAP
        p = pos + 1
        while p < cut and atoms[p]["end"] < target:
            p += 1
        pos = p if p > pos else pos + 1
    return out


def build_chunks(pdf_id, meta, items):
    chapters = detect_chapters(items)
    segs = []
    if not chapters or chapters[0][0] > 0:
        segs.append((0, chapters[0][0] if chapters else len(items), None, None,
                     items[0]["page"] if items else 1))
    for idx, (start, page, no, name) in enumerate(chapters):
        end = chapters[idx + 1][0] if idx + 1 < len(chapters) else len(items)
        segs.append((start, end, no, name, page))

    chunks = []
    gidx = 0
    for start, end, no, name, page in segs:
        seg_items = items[start:end]
        if not seg_items:
            continue
        is_front = no is None
        if is_front and not INCLUDE_FRONT:
            continue
        chapter = "前置" if is_front else chapter_label(no, name)
        atoms = make_atoms(seg_items)
        ranges = chunk_ranges(atoms)
        sec_idx = 0
        for si, (i, j) in enumerate(ranges):
            body = "\n".join(a["text"] for a in atoms[i:j + 1]).strip()
            if not body:
                continue
            # 尾块过短则并入上一块
            if len(body) < MIN_TAIL and chunks and not is_front:
                prev = chunks[-1]
                prev["text"] = prev["text"] + "\n" + body
                prev["char_len"] = len(prev["text"])
                prev["page_end"] = atoms[j]["page"]
                prev["chunk_sha1"] = hashlib.sha1(prev["text"].encode("utf-8")).hexdigest()
                continue
            gidx += 1
            sec_idx += 1
            p_start = atoms[i]["page"]
            p_end = atoms[j].get("page_end", atoms[j]["page"])
            # 注意：块可能跨表格边界——只含闭合标记的"续块"也必须算表格块，
            # 否则会被当成纯正文去算句末收尾率，把指标拖坏（踩过一次）。
            has_table = ("【表格】" in body) or ("【/表格】" in body)
            embed_text = (f"{meta['company']} {meta['report_year']}年年度报告 {chapter}\n{body}")
            chunks.append({
                "chunk_id": f"{pdf_id}#{gidx:04d}",
                "pdf_id": pdf_id,
                "company": meta["company"],
                "stock_code": meta["stock_code"],
                "report_year": meta["report_year"],
                "chapter": chapter,
                "chapter_no": 0 if is_front else no,
                "is_front": is_front,
                "page": p_start,
                "page_start": p_start,
                "page_end": p_end,
                "chunk_index": gidx,
                "sec_chunk_index": sec_idx,
                "char_len": len(body),
                "has_table": has_table,
                "n_table_rows": body.count(" | "),
                "text": body,
                "embed_text": embed_text,
                "chunk_sha1": hashlib.sha1(body.encode("utf-8")).hexdigest(),
            })

    # 丢掉纯噪声碎片（如孤立的页码/表头残留），再重排序号保证连续
    chunks = [c for c in chunks if c["char_len"] >= MIN_KEEP]
    for i, c in enumerate(chunks, 1):
        c["chunk_index"] = i
        c["chunk_id"] = f"{pdf_id}#{i:04d}"
    per_sec = {}
    for c in chunks:
        key = c["chapter"]
        per_sec[key] = per_sec.get(key, 0) + 1
        c["sec_chunk_index"] = per_sec[key]
    return chunks, chapters


def process_one(rec):
    src = TEXT_DIR / f"{rec['pdf_id']}.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for pg in data["pages"]:
        for it in pg["items"]:
            if it["text"].strip():
                items.append({"text": it["text"], "page": pg["page"],
                              "kind": it.get("kind", "line")})
    chunks, chapters = build_chunks(rec["pdf_id"], data["meta"], items)
    return chunks, chapters, data


def main(argv=None):
    setup_console()
    ap = argparse.ArgumentParser(description="年报章节切块")
    ap.add_argument("--only", help="指定 pdf_id，逗号分隔")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stats", action="store_true", help="只统计不写文件")
    args = ap.parse_args(argv)

    if not (DERIVED / "manifest.jsonl").exists():
        print("找不到 manifest.jsonl，先跑 build_manifest.py")
        return 1
    with open(DERIVED / "manifest.jsonl", encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        recs = [r for r in recs if r["pdf_id"] in want]
    if args.limit:
        recs = recs[:args.limit]

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    meta_all, total_chunks, total_chars = {}, 0, 0
    lens, sec_counts = [], Counter()
    no_chapter = []

    for n, rec in enumerate(recs, 1):
        pid = rec["pdf_id"]
        src = TEXT_DIR / f"{pid}.json"
        if not src.exists():
            print(f"[{n}/{len(recs)}] {pid} 缺少抽取产物，跳过")
            continue
        t0 = time.time()
        chunks, chapters, data = process_one(rec)
        if not chapters:
            no_chapter.append(pid)
        sec_counts[len(chapters)] += 1

        if not args.stats:
            out = CHUNK_DIR / f"{pid}.jsonl"
            tmp = out.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            tmp.replace(out)

        blob = "".join(c["text"] for c in chunks)
        meta_all[pid] = {
            "chunks": len(chunks),
            "chars": len(blob),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "chapters": [{"no": c[2], "name": c[3], "page": c[1]} for c in chapters],
        }
        total_chunks += len(chunks)
        total_chars += len(blob)
        lens.extend(c["char_len"] for c in chunks)
        ch = " ".join(f"{c[2] or '前置'}@{c[1]}" for c in chapters[:12])
        print(f"[{n}/{len(recs)}] {pid} {len(chapters)}节 {len(chunks)}块 "
              f"{time.time() - t0:.1f}s  [{ch}]")

    lens.sort()

    def pct(p):
        return lens[min(len(lens) - 1, int(len(lens) * p))] if lens else 0

    print("-" * 60)
    print(f"块数 {total_chunks}  字数 {total_chars}  文件 {len(meta_all)}")
    if lens:
        print(f"块长 min={lens[0]} P50={pct(.5)} P90={pct(.9)} max={lens[-1]}")
    print(f"节数分布: {dict(sorted(sec_counts.items()))}")
    if no_chapter:
        print(f"[警告] 未检出任何章节的 {len(no_chapter)} 份: {no_chapter[:8]}")

    if not args.stats:
        mpath = CHUNK_DIR / "_meta.json"
        prev = {}
        if mpath.exists():
            try:
                prev = json.load(open(mpath, encoding="utf-8"))
            except Exception:
                prev = {}
        old = prev.get("reports", {})
        # 局部运行时合并，不覆盖其它文件的记录
        old.update(meta_all)
        obj = {
            "params": CHUNK_PARAMS,
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "reports": old,
            "total_chunks": sum(v["chunks"] for v in old.values()),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print(f"[meta] → {mpath}（{len(old)} 份 / {obj['total_chunks']} 块）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
