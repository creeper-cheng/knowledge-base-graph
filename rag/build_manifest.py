# -*- coding: utf-8 -*-
"""阶段0：把 年报_合成纤维制造/索引.csv 转成 manifest.jsonl。

索引.csv 是下载阶段留下的账本，只有 状态∈{成功,已存在} 的 92 行才有 PDF。
注意它是 utf-8-sig（带 BOM），首列名是 '﻿序号'。
"""
import csv
import json
import sys

from ann_common import DERIVED, INDEX_CSV, MANIFEST, PDF_DIR

OK_STATUS = {"成功", "已存在"}


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    setup_console()
    if not INDEX_CSV.exists():
        print(f"[错误] 找不到 {INDEX_CSV}")
        return 1

    with open(INDEX_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    recs, missing, skipped = [], [], 0
    seen = set()
    for r in rows:
        status = (r.get("状态") or "").strip()
        code = (r.get("股票代码") or "").strip()
        name = (r.get("股票简称") or "").strip()
        year = (r.get("报告期") or "").strip()
        if status not in OK_STATUS:
            skipped += 1
            continue
        pdf_id = f"{code}_{name}_{year}"
        if pdf_id in seen:
            print(f"[警告] pdf_id 重复，跳过: {pdf_id}")
            continue
        seen.add(pdf_id)

        rel = (r.get("文件路径") or "").strip()
        p = DERIVED.parent.parent / rel if rel else None   # 索引里的路径是相对仓库根
        if p is None or not p.exists():
            cand = PDF_DIR / f"{pdf_id}.pdf"
            if cand.exists():
                p = cand
            else:
                missing.append((pdf_id, rel))
                continue

        recs.append({
            "pdf_id": pdf_id,
            "stock_code": code,
            "company": name,
            "exchange": (r.get("交易所") or "").strip(),
            "report_year": int(year) if year.isdigit() else 0,
            "pdf_path": str(p),
            "anno_id": (r.get("公告ID") or "").strip(),
        })

    recs.sort(key=lambda x: (x["stock_code"], x["report_year"]))
    DERIVED.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[manifest] 写出 {len(recs)} 条 → {MANIFEST}")
    print(f"[manifest] 索引共 {len(rows)} 行，按状态跳过 {skipped} 行")
    if missing:
        print(f"[警告] {len(missing)} 条找不到 PDF 文件：")
        for pid, rel in missing[:10]:
            print(f"    {pid}  <-  {rel}")
    years = {}
    for r_ in recs:
        years[r_["report_year"]] = years.get(r_["report_year"], 0) + 1
    print(f"[manifest] 报告期分布: {dict(sorted(years.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
