# -*- coding: utf-8 -*-
"""切块质量验收：跑计划里的 B/C 层断言，不需要向量。

用法: python check_chunks.py [--only pid,pid]
"""
import argparse
import json
import re
import sys
from collections import Counter

from ann_common import CHUNK_DIR

SENT_END = tuple("。！？；!?;…")
OVERLAP_TOL = 300


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main(argv=None):
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args(argv)

    files = sorted(CHUNK_DIR.glob("*.jsonl"))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files if f.stem in want]
    if not files:
        print("没有切块产物")
        return 1

    all_len, issues = [], Counter()
    tot_chunks = 0
    per_file = []

    for f in files:
        cs = load(f)
        if not cs:
            issues["空文件"] += 1
            continue
        tot_chunks += len(cs)
        n = len(cs)

        # 标签完整性
        if not all(c.get("company") and c.get("chapter") and c.get("page")
                   and c.get("chunk_index") for c in cs):
            issues["标签缺失"] += 1
        # chunk_index 连续
        if [c["chunk_index"] for c in cs] != list(range(1, n + 1)):
            issues["块序号不连续"] += 1
        # 页码单调不减 & page_start<=page_end
        if any(cs[i]["page"] > cs[i + 1]["page"] for i in range(n - 1)):
            issues["页码回退"] += 1
        if any(c["page_start"] > c["page_end"] for c in cs):
            issues["起止页倒挂"] += 1
        # 标题名里不应该混进交叉引用
        if any("章" in c["chapter"] for c in cs):
            issues["章节名含'章'"] += 1
        # 嵌入文本必须以 公司+年份+章节 开头
        bad_embed = sum(1 for c in cs if not c["embed_text"].startswith(c["company"]))
        if bad_embed:
            issues["embed前缀错"] += bad_embed

        # 句末收尾率：只在「纯正文块」上算。含表格块以单元格结尾是正常的，
        # 强行要求句号反而说明表格还原坏了。每章节末块天然不完整，也排除。
        last_of_sec = {c["chapter"]: c["chunk_index"] for c in cs}
        ends, total_end = 0, 0
        for c in cs:
            if last_of_sec[c["chapter"]] == c["chunk_index"] or c["has_table"]:
                continue
            total_end += 1
            if c["text"].rstrip().endswith(SENT_END):
                ends += 1
        end_rate = ends / total_end if total_end else 1.0

        # 重叠完整性：上一块尾部 120 字应该出现在下一块开头
        ov_ok, ov_tot = 0, 0
        for i in range(n - 1):
            if cs[i]["chapter"] != cs[i + 1]["chapter"]:
                continue      # 章节边界不要求重叠
            ov_tot += 1
            tail = cs[i]["text"][-120:]
            if tail and tail in cs[i + 1]["text"][:OVERLAP_TOL + 120]:
                ov_ok += 1
        ov_rate = ov_ok / ov_tot if ov_tot else 1.0

        all_len.extend(c["char_len"] for c in cs)
        per_file.append({
            "pid": f.stem, "n": n,
            "chapters": len({c["chapter"] for c in cs}),
            "front": sum(1 for c in cs if c["is_front"]),
            "tables": sum(1 for c in cs if c["has_table"]),
            "end_rate": end_rate, "ov_rate": ov_rate,
            "max": max(c["char_len"] for c in cs),
        })

    all_len.sort()

    def pct(p):
        return all_len[min(len(all_len) - 1, int(len(all_len) * p))]

    print("=" * 78)
    print(f"{'文件':<28}{'块数':>6}{'节':>4}{'前置':>5}{'含表':>6}{'句末率':>8}{'重叠率':>8}{'最长':>6}")
    for r in per_file:
        print(f"{r['pid']:<28}{r['n']:>6}{r['chapters']:>4}{r['front']:>5}"
              f"{r['tables']:>6}{r['end_rate']:>7.0%}{r['ov_rate']:>8.0%}{r['max']:>6}")
    print("-" * 78)
    print(f"总块数 {tot_chunks}   文件 {len(per_file)}")
    print(f"块长 min={all_len[0]} P10={pct(.1)} P50={pct(.5)} P90={pct(.9)} max={all_len[-1]}")
    print(f"块长 ≤800 占比 = {sum(1 for x in all_len if x <= 800) / len(all_len):.1%}")
    print()
    hard = {"标签缺失", "块序号不连续", "页码回退", "起止页倒挂", "章节名含'章'", "embed前缀错"}
    if issues:
        for k, v in issues.items():
            flag = "✗" if k in hard else "⚠"
            print(f"  {flag} {k}: {v}")
    else:
        print("  ✓ 结构性断言全部通过")
    bad_end = [r for r in per_file if r["end_rate"] < 0.90]
    if bad_end:
        print(f"  ⚠ 句末收尾率 <90% 的文件: {[(r['pid'], round(r['end_rate'], 2)) for r in bad_end]}")
    bad_ov = [r for r in per_file if r["ov_rate"] < 0.99]
    if bad_ov:
        print(f"  ⚠ 重叠率 <99% 的文件: {[(r['pid'], round(r['ov_rate'], 2)) for r in bad_ov]}")
    print("=" * 78)
    return 1 if any(k in hard for k in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
