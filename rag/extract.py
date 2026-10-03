# -*- coding: utf-8 -*-
"""阶段1：逐页抽取文字 + 表格行列还原，产出 _derived/text/{pdf_id}.json。

文字层健康的页用坐标聚类还原表格；坏页/扫描页回退 Tesseract OCR。
页眉页脚用「位置带 + 频次」双闸门剔除（见 ann_common 顶部说明）。

用法:
    python extract.py                  # 全量（已存在的跳过）
    python extract.py --only 000703_恒逸石化_2024
    python extract.py --force          # 忽略已有产物重算
"""
import argparse
import json
import sys
import time

import pymupdf

from ann_common import (DERIVED, MANIFEST, TEXT_DIR, collect_furniture,
                        extract_page, page_word_rows, page_dict_lines, attach_sizes)


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_manifest():
    if not MANIFEST.exists():
        print(f"[错误] 找不到 {MANIFEST}，先跑 build_manifest.py")
        return []
    with open(MANIFEST, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def extract_one(rec):
    pdf_path = rec["pdf_path"]
    doc = pymupdf.open(pdf_path)
    try:
        topset, botset, body_size = collect_furniture(doc)
        pages, stats = [], {"text": 0, "ocr": 0, "items": 0, "chars": 0, "tables": 0}
        for i, page in enumerate(doc):
            items, source, err = extract_page(page, i + 1, topset, botset)
            pages.append({
                "page": i + 1,
                "source": source,
                "items": items,
                "ocr_error": err,
            })
            if source == "text":
                stats["text"] += 1
            else:
                stats["ocr"] += 1
            stats["items"] += len(items)
            stats["chars"] += sum(len(it["text"]) for it in items)
            stats["tables"] += sum(1 for it in items if it["text"] == "【表格】")
        stats["header_lines"] = len(topset) + len(botset)
        stats["body_size"] = body_size
    finally:
        doc.close()

    return {
        "pdf_id": rec["pdf_id"],
        "meta": {k: v for k, v in rec.items() if k != "pdf_path"},
        "pages": pages,
        "stats": stats,
    }


def main(argv=None):
    setup_console()
    ap = argparse.ArgumentParser(description="年报抽取")
    ap.add_argument("--only", help="只处理指定 pdf_id，逗号分隔")
    ap.add_argument("--limit", type=int, help="只处理前 N 份（调试用）")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重算")
    args = ap.parse_args(argv)

    recs = load_manifest()
    if not recs:
        return 1
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        recs = [r for r in recs if r["pdf_id"] in want]
    if args.limit:
        recs = recs[:args.limit]

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    total = {"text": 0, "ocr": 0, "chars": 0, "tables": 0}
    done = skipped = failed = 0

    for n, rec in enumerate(recs, 1):
        pid = rec["pdf_id"]
        out = TEXT_DIR / f"{pid}.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        t0 = time.time()
        try:
            data = extract_one(rec)
        except Exception as e:
            failed += 1
            print(f"[{n}/{len(recs)}] {pid} ✗ {type(e).__name__}: {e}")
            continue

        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(out)

        st = data["stats"]
        for k in total:
            total[k] += st[k]
        done += 1
        flag = f" OCR={st['ocr']}" if st["ocr"] else ""
        print(f"[{n}/{len(recs)}] {pid} {len(data['pages'])}页 {st['chars']}字 "
              f"表{st['tables']}个{flag} {time.time() - t0:.1f}s")

    print("-" * 60)
    print(f"抽取完成：新处理 {done}，跳过 {skipped}，失败 {failed}")
    print(f"累计：{total['chars']} 字 / 表格 {total['tables']} 个 / OCR 页 {total['ocr']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
