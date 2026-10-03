# -*- coding: utf-8 -*-
"""年报处理共用层：路径、抽取原语复用、页眉页脚剔除、统一 item 流构建。

三条设计要点，都是实测踩出来的，别改：

1. **页眉页脚必须「位置带内统计频次」**。601233 正文里 `√适用□不适用` 出现 146 次、
   `□适用√不适用` 123 次、`单位：元币种：人民币` 112 次——都远超常规频次阈值，
   但都是有语义必须保留的正文。所以频次只在页面顶部/底部 12% 的位置带内统计，
   正文中部的行永远进不了候选集。`02_extract.find_running_lines` 是全页频次，**不能用**。

2. **表格用坐标聚类还原**，不用 `page.find_tables()`：后者会漏掉无框线财务表、
   把一整段正文误判成表格、合并单元格输出一堆 None。

3. **text 与 rows 不能都拼进正文流**，否则同一内容重复计入。统一成一种 item 流：
   每个 item 要么是单 cell 的正文行，要么是多 cell 的表格行。
"""
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pymupdf

WORK = Path(__file__).resolve().parent
REPO = WORK.parent
PDF_DIR = REPO / "年报_合成纤维制造"
DERIVED = PDF_DIR / "_derived"
INDEX_CSV = PDF_DIR / "索引.csv"
MANIFEST = DERIVED / "manifest.jsonl"
TEXT_DIR = DERIVED / "text"
CHUNK_DIR = DERIVED / "chunks"
VECTOR_DIR = DERIVED / "vectors"

# ------------------------------------------------------------------
#  页质量判定与 OCR（原本在 02_extract.py，现在内联，让 rag/ 自包含）
# ------------------------------------------------------------------
MIN_CHARS = 80          # 少于这个字数视为无有效文字
GARBLE_RATIO = 0.85     # 有效字符占比低于此值视为乱码
OCR_DPI = 300
ROW_TOL = 4.0           # 同一行的 y 容差（旧算法用，保留作参照）
CELL_GAP = 8.0          # 单元格之间的 x 间距阈值（同上）

TESS = os.environ.get("TESSERACT_EXE", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = os.environ.get("TESSDATA_DIR", str(REPO / "_work" / "tessdata"))

# 允许出现的字符：CJK 汉字、中日韩标点、全角符号、ASCII 及常用标点
GOOD_CHAR = re.compile(
    r"[\u4e00-\u9fff\uff00-\uffef\u3000-\u303fA-Za-z0-9 \n\r\t.,;:%()\[\]（）【】・·—–\-+/=]"
)
# Tesseract 会给每个汉字之间插空格，需要去掉
CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

# 新的表格还原参数（先合单元格、再聚行）
MERGE_GAP = 6.0       # 纵向间隔小于此值的两个词，可能是同一单元格的折行
X_OVERLAP_MIN = 0.5   # 横向重叠比例下限（分母取较窄者），判定"同一列"
ROW_OVERLAP = 0.35    # 单元格纵向重叠比例超过此值视为同一行


def page_quality(text):
    if len(text) < MIN_CHARS:
        return "empty"
    if len(GOOD_CHAR.findall(text)) / len(text) < GARBLE_RATIO:
        return "garbled"
    return "ok"


def ocr_page(page):
    """渲染成位图后交给 Tesseract。

    直接调 CLI——pytesseract 解码 GBK 报错信息时会崩，这个坑踩过。
    """
    pix = page.get_pixmap(dpi=OCR_DPI)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"
        pix.save(png)
        proc = subprocess.run(
            [TESS, str(png), "stdout", "-l", "chi_sim", "--tessdata-dir", TESSDATA],
            capture_output=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if proc.returncode != 0:
            return "", (proc.stderr or "")[:200]
        return CJK_SPACE.sub("", proc.stdout), ""

# ------------------------------------------------------------------
#  页眉页脚：位置带 + 频次双闸门
# ------------------------------------------------------------------
TOP_BAND = 0.12      # 页面顶部 12% 视为页眉带
BOT_BAND = 0.88      # 页面底部 12% 视为页脚带
FURN_RATIO = 0.5     # 在位置带内重复出现于 50% 以上页面才算页眉/页脚
FURN_MIN = 3

PAGE_RE = re.compile(
    r"^\s*(\d{1,4}|\d{1,4}\s*/\s*\d{1,4}|第\s*\d+\s*页|共\s*\d+\s*页)\s*$"
)


def canon(text):
    """数字塌缩 + 去空白，让跨年份的页码/页眉变体合并计数。"""
    return re.sub(r"\d+", "#", re.sub(r"[\s　]+", "", text or ""))


def is_dotleader(text):
    """目录行的引导点：`第一节 重要提示、目录和释义 ......... 3`"""
    return bool(re.match(r"^.{2,60}(\.{4,}|…{2,})\s*\d+$", (text or "").strip()))


class PageLine:
    """一行：bbox + 文本 + 字号。用于页眉页脚判定与章节标题识别。"""
    __slots__ = ("y0", "y1", "x0", "x1", "text", "size", "H")

    def __init__(self, y0, y1, x0, x1, text, size, H):
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        self.text, self.size, self.H = text, size, H


def page_dict_lines(page):
    """从 get_text("dict") 提取带字号的文本行（仅文字层页有内容）。"""
    out = []
    H = page.rect.height or 1.0
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            size = max((s["size"] for s in ln["spans"]), default=0.0)
            out.append(PageLine(y0, y1, x0, x1, txt, size, H))
    return out


def _has_digit(s):
    return any(ch.isdigit() for ch in s)


def _join_wrapped(prev, nxt):
    """折行拼接规则：数字/小数点/百分号后直接接，中文之间直接接，英文词之间才加空格。

    年报表格里数字常被折成 `3,925,279,749.` + `60`，中文标签折成
    `归属于上市公司` + `股东的净利润` + `（元）`——都必须无缝拼接。
    """
    if not prev or not nxt:
        return ""
    a, b = prev[-1], nxt[0]
    if a in ".,%" or a.isdigit() or ("一" <= a <= "鿿"):
        return ""
    if b.isdigit() or b in ".,%" or ("一" <= b <= "鿿"):
        return ""
    return " "


def page_word_rows(page):
    """
    表格「按行列还原」。返回 [{y0,y1,x0,x1,cells:[...], size}]。

    算法分两步，顺序很关键：
      1. **先把纵向相邻、横向重叠的词合并成「单元格」**——年报表格的单元格经常
         折行（`3,925,279,749.` / `60`）或纵向跨多行（`归属于上市公司` /
         `股东的净利润` / `（元）`），不先合并就会把它们当成不同行的碎片。
      2. **再按纵向重叠把单元格聚成「行」**。

    ⚠ 旧实现是「按视觉行聚行 + 行内按 x 间距切列」，在行与行 y 不对齐的表上会散架：
    实测泰和新材 2025 第 9 页的「主要会计数据」表（7 列 + 二级表头 + 全折行），
    标签列被分到别的行、表头被切碎，导致「归属于上市公司股东的净利润」这一行
    彻底对不上数字。改成先合单元格后，该表还原正确（-52.81% 与两期数字自洽）。

    **不要用 page.find_tables()**：它漏无框线财务表、把整段正文误判成表格、
    合并单元格输出一堆 None。
    """
    try:
        words = page.get_text("words")
    except Exception:
        return []
    if not words:
        return []

    cells = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        best = None
        for c in cells:
            gap = y0 - c["y1"]
            if gap < -1.0 or gap > MERGE_GAP:
                continue
            ov = min(x1, c["x1"]) - max(x0, c["x0"])
            mn = min(x1 - x0, c["x1"] - c["x0"])
            if mn > 0 and ov / mn >= X_OVERLAP_MIN:
                # 取纵向最靠下的那个候选，保证接在最末尾的折行后面
                if best is None or c["y1"] > best["y1"]:
                    best = c
        if best is not None:
            best["text"] += _join_wrapped(best["text"], txt) + txt
            best["x0"] = min(best["x0"], x0)
            best["x1"] = max(best["x1"], x1)
            best["y0"] = min(best["y0"], y0)
            best["y1"] = max(best["y1"], y1)
        else:
            cells.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": txt})

    cells.sort(key=lambda c: (c["y0"], c["x0"]))
    rows = []
    for c in cells:
        h = max(1e-6, c["y1"] - c["y0"])
        placed = False
        for r in rows:
            for d in r["cells"]:
                ov = min(c["y1"], d["y1"]) - max(c["y0"], d["y0"])
                if ov > 0 and ov / min(h, max(1e-6, d["y1"] - d["y0"])) >= ROW_OVERLAP:
                    r["cells"].append(c)
                    r["y0"] = min(r["y0"], c["y0"])
                    r["y1"] = max(r["y1"], c["y1"])
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append({"y0": c["y0"], "y1": c["y1"], "cells": [c]})

    rows.sort(key=lambda r: r["y0"])
    out = []
    for r in rows:
        cs = sorted(r["cells"], key=lambda c: c["x0"])
        out.append({
            "y0": r["y0"], "y1": r["y1"],
            "x0": cs[0]["x0"], "x1": cs[-1]["x1"],
            "cells": [c["text"] for c in cs],
            "size": 0.0,
        })

    # 收尾：标签折到第三行时会跟数字分家，形态是
    #   标签行(单格、无数字) → 数字行(整行都是数字) → 尾巴行(单格、无数字)
    # 把它拼回一行。实测这是「现金流量表补充资料」里最常见的残留问题。
    fixed, i, n = [], 0, len(out)
    while i < n:
        r = out[i]
        if (len(r["cells"]) == 1 and not _has_digit(r["cells"][0])
                and i + 1 < n and out[i + 1]["cells"]
                and all(_has_digit(c) for c in out[i + 1]["cells"])
                and out[i + 1]["y0"] - r["y1"] < 22):
            nxt = out[i + 1]
            cells = [r["cells"][0]] + list(nxt["cells"])
            y1, x0, x1 = max(r["y1"], nxt["y1"]), min(r["x0"], nxt["x0"]), max(r["x1"], nxt["x1"])
            j = i + 2
            if (j < n and len(out[j]["cells"]) == 1 and not _has_digit(out[j]["cells"][0])
                    and out[j]["y0"] - y1 < 22):
                tail = out[j]["cells"][0]
                cells[0] += _join_wrapped(cells[0], tail) + tail
                y1 = max(y1, out[j]["y1"])
                x1 = max(x1, out[j]["x1"])
                j += 1
            fixed.append({"y0": r["y0"], "y1": y1, "x0": x0, "x1": x1,
                          "cells": cells, "size": 0.0})
            i = j
            continue
        fixed.append(r)
        i += 1
    return fixed


def attach_sizes(rows, dlines):
    """给每个 word-row 补上字号：取与之 y 重叠最多的 dict-line 的字号。"""
    if not dlines:
        return rows
    for r in rows:
        best, best_ov = 0.0, 0.0
        for d in dlines:
            ov = min(r["y1"], d.y1) - max(r["y0"], d.y0)
            if ov > best_ov:
                best_ov, best = ov, d.size
        r["size"] = best
    return rows


def collect_furniture(doc):
    """
    在**位置带内**统计跨页重复行，得到页眉/页脚集合。
    返回 (topset, botset, body_size)——body_size 是全文中位字号，供章节标题加分用。
    """
    top, bot, sizes, npages = Counter(), Counter(), Counter(), 0
    for page in doc:
        npages += 1
        for ln in page_dict_lines(page):
            sizes[round(ln.size, 1)] += len(ln.text)
            k = canon(ln.text)
            if len(k) < 4 or len(k) > 100:
                continue
            if ln.y1 < ln.H * TOP_BAND:
                top[k] += 1
            elif ln.y0 > ln.H * BOT_BAND:
                bot[k] += 1
    thr = max(FURN_MIN, FURN_RATIO * npages)
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    return ({k for k, c in top.items() if c >= thr},
            {k for k, c in bot.items() if c >= thr},
            body_size)


def is_furniture_row(row, topset, botset, H):
    """行是否落在位置带内、且是跨页重复行或页码。"""
    txt = " ".join(row["cells"]).strip()
    if not txt:
        return True
    if row["y1"] < H * TOP_BAND:
        return canon(txt) in topset or bool(PAGE_RE.match(txt))
    if row["y0"] > H * BOT_BAND:
        return canon(txt) in botset or bool(PAGE_RE.match(txt))
    return False


# ------------------------------------------------------------------
#  统一 item 流
# ------------------------------------------------------------------
TABLE_OPEN = "【表格】"
TABLE_CLOSE = "【/表格】"
BLANK_CELL = "—"
TABLE_GAP_MULT = 1.8   # 相邻行 y 间距超过 1.8 倍中位行距 → 表格块断开


def _row_dy(rows):
    """中位行距，用于判断表格块是否断开。"""
    if len(rows) < 2:
        return 0.0
    dys = sorted(rows[i + 1]["y0"] - rows[i]["y0"] for i in range(len(rows) - 1))
    return dys[len(dys) // 2]


def _multi_rows_to_items(block):
    """把连续的多 cell 行序列化成表格 item。列宽按该块最大列数补齐。"""
    ncol = max(len(r["cells"]) for r in block)
    items = [{"kind": "table", "text": TABLE_OPEN, "page": block[0]["page"],
              "size": block[0].get("size", 0.0)}]
    for r in block:
        cells = [(c or "").strip() or BLANK_CELL for c in r["cells"]]
        cells += [BLANK_CELL] * (ncol - len(cells))
        items.append({"kind": "table", "text": " | ".join(cells), "page": r["page"],
                      "size": r.get("size", 0.0)})
    items.append({"kind": "table", "text": TABLE_CLOSE, "page": block[-1]["page"],
                  "size": block[-1].get("size", 0.0)})
    return items


def rows_to_items(rows, pageno):
    """
    把一页的 word-rows 转成 item 流。
    判定规则：单 cell → 正文行；连续 >=2 个多 cell 行 → 表格；孤立的多 cell 行 → 当正文。
    """
    for r in rows:
        r["page"] = pageno
    items, i, n = [], 0, len(rows)
    while i < n:
        if len(rows[i]["cells"]) >= 2:
            j = i
            while j + 1 < n and len(rows[j + 1]["cells"]) >= 2:
                j += 1
            block = rows[i:j + 1]
            # 行距突变也算断开
            med = _row_dy(block)
            if med > 0:
                cut = len(block)
                for k in range(1, len(block)):
                    if block[k]["y0"] - block[k - 1]["y0"] > med * TABLE_GAP_MULT:
                        cut = k
                        break
                block = block[:cut]
                j = i + cut - 1

            if len(block) >= 2:
                items.extend(_multi_rows_to_items(block))
                i = j + 1
                continue
            # 孤立多 cell 行 → 当正文
            txt = " ".join(c for c in rows[i]["cells"] if c.strip())
            items.append({"kind": "line", "text": txt, "page": pageno,
                          "size": rows[i].get("size", 0.0)})
            i += 1
        else:
            items.append({"kind": "line", "text": rows[i]["cells"][0], "page": pageno,
                          "size": rows[i].get("size", 0.0)})
            i += 1
    return items


def ocr_text_to_items(text, pageno):
    """OCR 页没有坐标/字号，退化为按行切。"""
    allrows = [{"cells": [ln.strip()], "y0": i, "y1": i, "size": 0.0}
               for i, ln in enumerate(text.splitlines()) if ln.strip()]
    return rows_to_items(allrows, pageno)


def _clean_ratio(t):
    return len(GOOD_CHAR.findall(t)) / max(1, len(t))


def extract_page(page, pageno, topset, botset):
    """
    抽取一页 → (items, source, err)。
    文字层健康则用坐标聚类；否则回退 OCR。

    注意：page_quality 把 <80 字判为 "empty" 直接送 OCR，
    但年报里"第八节 优先股相关情况 / □适用 √不适用"这种分节页本来就只有十几个字，
    文字层是好端端的。误送 OCR 会丢坐标（页眉剔不掉、标题行首被页码污染）。
    所以这里补一条：短但干净的文字层仍然按 text 处理。
    """
    H = page.rect.height or 1.0
    raw = page.get_text()
    quality = page_quality(raw)
    stripped = raw.strip()
    if quality != "ok" and len(stripped) >= 8 and _clean_ratio(stripped) >= GARBLE_RATIO:
        quality = "ok"

    if quality == "ok":
        dlines = page_dict_lines(page)
        rows = attach_sizes(page_word_rows(page), dlines)
        rows = [r for r in rows if not is_furniture_row(r, topset, botset, H)]
        return rows_to_items(rows, pageno), "text", ""

    text, err = ocr_page(page)
    return ocr_text_to_items(text, pageno), f"ocr({quality})", err
