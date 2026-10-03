# -*- coding: utf-8 -*-
"""
巨潮资讯 上市公司年度报告批量下载工具
========================================
数据来源: www.cninfo.com.cn （巨潮资讯网，证监会指定信息披露平台，覆盖沪深北）
PDF下载:  static.cninfo.com.cn

用途: 批量下载指定行业上市公司的「年度报告正文」PDF。

使用方法:
  1. 修改下方"配置区"的参数（一般只需改 REPORT_YEARS / SAVE_DIR）
  2. 先用 --dry-run 核对命中的公告（不下载，最快）
  3. 确认无误后直接运行: python download_annual_reports.py
  4. PDF 与 索引.csv 保存到 年报_合成纤维制造/

常用命令:
  python download_annual_reports.py --self-test      # 离线自测，验证正则与代码分派
  python download_annual_reports.py --dry-run        # 只查不下载
  python download_annual_reports.py --only 000703,601233
  python download_annual_reports.py --retry-failed   # 只重跑上次失败的
  python download_annual_reports.py --verify         # 巡检已下载 PDF 的完整性
  python download_annual_reports.py --dump-list      # 导出公司名单 CSV 模板

⚠ 关键坑（已实测）:
  * FY2025 年报是 2026-04 才披露的，所以报告期只能按【标题里的年份】判定，
    绝不能按公告日期筛，否则最新一年会整个漏掉。SE_DATE 必须放宽覆盖披露日。
  * 巨潮接口 column 必须与交易所匹配（沪=sse / 深=szse），否则静默返回 0 条。
  * stock 参数必须是 "代码,orgId" 两段，只传 6 位代码返回 0 条。
  * orgId 格式不统一（gssz0000703 / gssh0600519 / 9900019547 都有），只能查表。
"""

import os
import re
import sys
import csv
import json
import math
import time
import random
import argparse
import requests

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


# ------------------------------------------------------------------
#  控制台编码（必须在任何 print 之前执行；Windows 默认 GBK 会让中文标题炸掉）
# ------------------------------------------------------------------
def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", "") or "").lower()
            if stream is not None and enc not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # 重定向到不支持 reconfigure 的对象时静默降级


setup_console()


# ============================================================
#  ★★★ 配置区 — 只需修改这里 ★★★
# ============================================================

# 要下载哪几个「报告期」的年报（按标题里的年份筛，不是按公告日期）
REPORT_YEARS = [2023, 2024, 2025]

# PDF / 索引保存目录
SAVE_DIR = "年报_合成纤维制造"

# 公司名单 CSV（不存在时用内置种子名单）
COMPANY_CSV = "companies_合成纤维制造.csv"

# 公告查询日期范围。必须覆盖 FY2025 在 2026-04 的披露！
SE_DATE = "2024-01-01~2026-12-31"

# 每页条数（巨潮硬上限 30，改大无效）
PAGE_SIZE = 30
# 分页安全阀，防止接口异常导致死循环
MAX_PAGES = 40

# 是否下载北交所公司（bj 参数组合未实测，默认关）
ALLOW_BJ = False

# 是否下载"边界"口径公司（恒逸石化/荣盛石化等石化为主、粘胶类）
INCLUDE_BORDERLINE = True

# 是否让请求走系统/环境代理。
# 巨潮资讯是国内站点，走代理不但慢，下载几十 MB 的年报时极易 ReadTimeout。
# requests 在 Windows 上默认会读注册表里的系统代理（Clash 会把系统代理设成 127.0.0.1:7897），
# 所以这里默认关闭 trust_env，直连巨潮。
USE_PROXY = False

# 限速：随机区间（秒），防止被封 IP
PDF_DELAY = (0.8, 1.5)
QUERY_DELAY = (0.3, 0.7)
COMPANY_DELAY = (0.5, 1.0)

# 重试
MAX_RETRY = 3
BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0
TIMEOUT_QUERY = 20
TIMEOUT_PDF = 90  # 年报可达 20MB+，别用 30

# 完整性判定
MIN_PDF_BYTES = 200 * 1024  # 低于 200KB 可疑（多半是摘要或错误页）
SIZE_TOLERANCE = 0.95  # 已有文件 >= 接口标注大小*0.95 视为完整
CHECK_PDF_MAGIC = True  # 校验 %PDF- 头 + %%EOF 尾

# 行为开关
SKIP_EXISTING = True  # 已存在且完整则跳过（实现断点续传）
KEEP_ALL_REVISIONS = False  # True=修订版和原版都下；False=每年只留最新
VERBOSE = True  # 打印每个公告的候选明细

# ============================================================
#  以下为程序逻辑，一般不需要修改
# ============================================================

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STOCK_JSON_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
PDF_BASE = "http://static.cninfo.com.cn/"
CATEGORY = "category_ndbg_szsh"  # 巨潮：年度报告
ORGID_CACHE = "_cache/orgid_map.json"
ORGID_MAX_AGE_DAYS = 3

INDEX_FIELDS = ["序号", "股票代码", "股票简称", "交易所", "报告期", "公告标题",
                "公告日期", "公告ID", "文件路径", "字节数", "状态", "备注"]
FAIL_FIELDS = ["股票代码", "股票简称", "报告期", "公告标题", "公告ID",
               "PDF链接", "错误信息", "尝试次数", "记录时间"]
COMPANY_FIELDS = ["代码", "简称", "交易所", "orgId", "启用", "口径", "备注"]

TZ_CN = timezone(timedelta(hours=8))

STATUS_OK = "成功"
STATUS_EXIST = "已存在"
STATUS_NO_ANN = "无公告"
STATUS_NO_MATCH = "无匹配"
STATUS_FAIL = "失败"
STATUS_SKIP = "跳过"
ALL_STATUS = {STATUS_OK, STATUS_EXIST, STATUS_NO_ANN, STATUS_NO_MATCH, STATUS_FAIL, STATUS_SKIP}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# 代码前缀 -> 交易所
MARKET_PREFIX = {
    "600": "SH", "601": "SH", "603": "SH", "605": "SH", "688": "SH", "689": "SH",
    "000": "SZ", "001": "SZ", "002": "SZ", "003": "SZ", "300": "SZ", "301": "SZ", "302": "SZ",
    "920": "BJ", "430": "BJ",
}
# 交易所 -> (column, plate)
MARKET_PARAMS = {
    "SH": ("sse", "sh"),
    "SZ": ("szse", "sz"),
    "BJ": ("bjse", "bj"),  # 未实测，仅 --bj 时使用
}

# 内置种子名单：(代码, 简称, 口径)
# 口径: 核心 = 合成纤维制造(C282)主流标的；边界 = 石化为主或粘胶(C281)，可自行删行
DEFAULT_COMPANIES = [
    # ---- 涤纶 / 聚酯 (C2822) ----
    ("601233", "桐昆股份", "核心", "涤纶长丝 POY/FDY/DTY"),
    ("603225", "新凤鸣", "核心", "涤纶长丝"),
    ("603332", "苏州龙杰", "核心", "差别化涤纶长丝"),
    ("301057", "汇隆新材", "核心", "差别化/有色涤纶长丝"),
    ("002998", "优彩资源", "核心", "再生涤纶短纤"),
    ("603406", "天富龙", "核心", "再生涤纶短纤"),
    ("600527", "江南高纤", "核心", "涤纶毛条/复合短纤"),
    ("002206", "海利得", "核心", "涤纶工业丝/帘子布"),
    ("002427", "尤夫股份", "核心", "涤纶工业丝"),
    ("000936", "华西股份", "核心", "涤纶短纤"),
    # ---- 锦纶 / 尼龙 (C2821) ----
    ("600810", "神马股份", "核心", "尼龙66/己二酸"),
    ("000782", "恒申新材", "核心", "锦纶（原美达股份）"),
    ("601113", "华鼎股份", "核心", "锦纶DTY"),
    ("605166", "聚合顺", "核心", "尼龙6切片 PA6"),
    ("603382", "海阳科技", "核心", "尼龙/锦纶"),
    # ---- 氨纶 (C2826) ----
    ("002064", "华峰化学", "核心", "氨纶龙头"),
    ("002254", "泰和新材", "核心", "氨纶+芳纶"),
    ("000949", "新乡化纤", "核心", "粘胶长丝+氨纶"),
    # ---- 维纶 / PVA (C2824) ----
    ("600063", "皖维高新", "核心", "PVA/维纶原料"),
    # ---- 丙纶 (C2825) ----
    ("300876", "蒙泰高新", "核心", "丙纶长丝"),
    # ---- 碳纤维 / 高性能纤维 (C2829) ----
    ("300699", "光威复材", "核心", "碳纤维及复材"),
    ("300777", "中简科技", "核心", "高性能碳纤维"),
    ("688295", "中复神鹰", "核心", "碳纤维"),
    ("688722", "同益中", "核心", "超高分子量聚乙烯纤维"),
    ("000420", "吉林化纤", "核心", "粘胶+碳纤维"),
    # ---- 生物基 ----
    ("688065", "凯赛生物", "核心", "生物基聚酰胺"),
    ("688203", "海正生材", "核心", "聚乳酸 PLA"),
    # ---- 边界口径（默认启用，可删）----
    ("000703", "恒逸石化", "边界", "石化为主，按最大收入原则未必落 C28"),
    ("002493", "荣盛石化", "边界", "石化为主，口径有分歧"),
    ("600889", "南京化纤", "边界", "粘胶，属 C281 非合成纤维"),
    ("002172", "澳洋健康", "边界", "粘胶/医疗，业务已转型"),
    ("000677", "恒天海龙", "边界", "粘胶，ST"),
    # ---- 北交所（默认停用）----
    ("920077", "吉林碳谷", "北交所", "碳纤维原丝，北交所，默认不下载"),
]


# ------------------------------------------------------------------
#  工具函数
# ------------------------------------------------------------------
WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def clean_filename(name, maxlen=120):
    """去除文件名非法字符，并处理 Windows 的结尾点/空格与保留名问题"""
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
        name = str(name).replace(ch, '')
    name = re.sub(r"\s+", " ", name).strip().rstrip(". ")
    if name.split(".")[0].upper() in WIN_RESERVED:
        name = "_" + name
    return name[:maxlen]


def ts_to_date(ms):
    """巨潮 announcementTime 是 epoch 毫秒，必须显式按 UTC+8 解析，否则会掉一天"""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=TZ_CN).strftime("%Y-%m-%d")
    except Exception:
        return ""


def log(msg):
    print(msg, flush=True)


def sleep_range(rng):
    time.sleep(random.uniform(rng[0], rng[1]))


# ------------------------------------------------------------------
#  年报识别（纯函数，可离线自测）
# ------------------------------------------------------------------
RE_HTML = re.compile(r"<[^>]*>")
RE_EXCLUDE = re.compile(
    r"摘要"
    r"|英文|English|Annual\s*Report"
    r"|更正|补充|澄清|问询|提示|说明会|路演"
    r"|半年度|季度|一季|三季"
    r"|审计|鉴证|审阅|内部控制|财务报表附注"
    r"|监事会|董事会|独立董事|股东大会"
    r"|业绩预告|业绩快报|社会责任|ESG|环境、社会"
    r"|H股|港股",
    re.IGNORECASE,
)
RE_ABOUT_NOTICE = re.compile(r"^关于.{0,40}?(的)?(公告|通知|说明|提示性公告|更正公告|回复|问询函)$")
# 全称"年度报告"：可以用在任何位置（如"皖维高新2023年年度报告全文"）
RE_ANNUAL_LONG = re.compile(r"(?<!\d)(20\d{2})[\s_]*年?[\s_]*年度报告")
# 简写"年报"：仅在标题末尾才算正文（如"600889_南京化纤_2025年_年报"）。
# 必须锚定行尾，否则"2024年报点评"这类也会误命中。
RE_ANNUAL_SHORT = re.compile(r"(?<!\d)(20\d{2})[\s_]*年?[\s_]*年报[\s_]*(?:[（(][^）)]*[）)])?[\s_]*$")


def normalize_title(t):
    """剥掉巨潮全文检索的高亮标签，统一空白"""
    t = RE_HTML.sub("", t or "")
    t = t.replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def match_annual_report(title):
    """
    判断标题是否「年度报告正文」，是则返回报告期年份，否则 None。
    顺序不可颠倒：先排除、再结构兜底、最后匹配年份。
    """
    t = normalize_title(title)
    if not t:
        return None
    if RE_EXCLUDE.search(t):
        return None
    if RE_ABOUT_NOTICE.match(t):
        return None
    years = set(RE_ANNUAL_LONG.findall(t)) | set(RE_ANNUAL_SHORT.findall(t))
    if len(years) != 1:  # 0 个=不是年报；>=2 个=年份歧义，交人工
        return None
    return int(years.pop())


def pick_annual_reports(anns, years):
    """
    按报告期分桶，每年取披露时间最新的一条（修订版必然后发）。
    返回 {年份: 记录}，被舍弃的候选挂在记录的 _alts 上，绝不静默丢弃。
    """
    buckets = {}
    for a in anns:
        y = match_annual_report(a["title"])
        if y is not None and y in years:
            buckets.setdefault(y, []).append(a)

    picked = {}
    for y, cands in buckets.items():
        # 主排序：公告时间戳倒序
        # 次排序：标题不带"关于"前缀的优先（更正/说明类公告一般带）
        # 三排序：附件大的优先（同秒重复上传时更可能是正文）
        cands.sort(key=lambda a: (a["ts_ms"], not a["title"].startswith("关于"), a["size_kb"]),
                   reverse=True)
        top = dict(cands[0])
        top["_alts"] = cands[1:]
        picked[y] = top
    return picked


# ------------------------------------------------------------------
#  公司名单
# ------------------------------------------------------------------
@dataclass
class Company:
    code: str
    name: str
    market: str = ""      # 空 = 按代码前缀自动判定
    org_id: str = ""      # 空 = 查 szse_stock.json
    enabled: bool = True
    note: str = ""
    quota: str = ""       # 口径：核心 / 边界 / 北交所


def _norm_code(raw):
    s = (raw or "").strip().strip("'\"")
    s = re.sub(r"\.(SH|SZ|BJ)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\D", "", s)
    return s.zfill(6) if 0 < len(s) <= 6 else ""


def _norm_bool(raw, default=True):
    s = str(raw or "").strip().lower()
    if s in ("", "1", "true", "yes", "y", "是", "启用"):
        return default
    if s in ("0", "false", "no", "n", "否", "停用"):
        return False
    return default


def default_companies():
    rows = []
    for code, name, quota, note in DEFAULT_COMPANIES:
        rows.append(Company(code=code, name=name, enabled=(quota != "北交所"),
                            note=note, quota=quota))
    return rows


def load_companies(path, allow_bj=False, include_borderline=True):
    """CSV 优先；不存在则用内置种子。脚本只读 CSV，永不覆写。"""
    if not os.path.exists(path):
        log(f"[名单] 未找到 {path}，使用内置种子名单（可运行 --dump-list 导出为 CSV 编辑）")
        comps = default_companies()
    else:
        comps = []
        seen = set()
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for i, raw in enumerate(reader, start=2):
                if not raw:
                    continue
                first = (list(raw.values())[0] or "").strip() if raw else ""
                if first.startswith("#"):
                    continue
                code = _norm_code(raw.get("代码"))
                if not code:
                    log(f"[警告] 第{i}行代码非法，已跳过: {raw.get('代码')!r}")
                    continue
                if code in seen:
                    log(f"[警告] 代码重复已忽略: {code}（第{i}行）")
                    continue
                seen.add(code)
                comps.append(Company(
                    code=code,
                    name=(raw.get("简称") or "").strip() or code,
                    market=(raw.get("交易所") or "").strip().upper(),
                    org_id=(raw.get("orgId") or "").strip(),
                    enabled=_norm_bool(raw.get("启用"), True),
                    note=(raw.get("备注") or "").strip(),
                    quota=(raw.get("口径") or "").strip(),
                ))
        if not comps:
            log(f"[警告] {path} 解析后为空，回退内置种子名单")
            comps = default_companies()

    # 口径过滤（CSV 里没有"口径"列时按备注里的关键词判断）
    out = []
    for c in comps:
        quota = c.quota or ("边界" if "边界" in c.note else "")
        if quota == "边界" and not include_borderline:
            continue
        if quota == "北交所" and not allow_bj:
            continue
        out.append(c)
    return out


def dump_default_list(path, force=False):
    if os.path.exists(path) and not force:
        log(f"[错误] {path} 已存在。要覆盖请加 --force（避免误删你手工编辑的名单）")
        return 1
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPANY_FIELDS)
        w.writeheader()
        for code, name, quota, note in DEFAULT_COMPANIES:
            w.writerow({
                "代码": code, "简称": name, "交易所": "", "orgId": "",
                "启用": "0" if quota == "北交所" else "1",
                "备注": note, "口径": quota,
            })
    log(f"[名单] 已导出模板: {os.path.abspath(path)}")
    return 0


def detect_market(code):
    if code[:3] in MARKET_PREFIX:
        return MARKET_PREFIX[code[:3]]
    if code[:2] in {"83", "87"} or code[:1] in {"4", "8"}:
        return "BJ"
    return "UNKNOWN"


def market_params(market):
    return MARKET_PARAMS.get(market)


# ------------------------------------------------------------------
#  HTTP
# ------------------------------------------------------------------
def make_session():
    s = requests.Session()
    if not USE_PROXY:
        # 屏蔽环境变量与 Windows 注册表里的系统代理（Clash 等）
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
    })
    return s


def request_with_retry(session, method, url, kind="query", headers=None, **kw):
    """指数退避重试；4xx（除 408/429）不重试"""
    timeout = TIMEOUT_QUERY if kind == "query" else TIMEOUT_PDF
    last_err = ""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = session.request(method, url, headers=headers, timeout=timeout, **kw)
            if r.status_code == 200:
                return r, attempt
            if r.status_code in (408, 429) or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
            else:
                return None, attempt  # 4xx 重试没意义
        except requests.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRY:
            wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 0.6), BACKOFF_CAP)
            if VERBOSE:
                log(f"      ↻ 第{attempt}次失败({last_err})，{wait:.1f}s 后重试")
            time.sleep(wait)
    return None, MAX_RETRY


def load_orgid_map(session, cache_path):
    """拉 szse_stock.json（实为全 A 股总表）→ {code: orgId}；带本地缓存，断网可复跑"""
    cache_file = os.path.join(SAVE_DIR, cache_path)
    cached = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
        except Exception:
            cached = None

    if cached:
        age_days = (time.time() - cached.get("fetched_at", 0)) / 86400
        if age_days < ORGID_MAX_AGE_DAYS:
            log(f"[orgId] 使用本地缓存（{age_days:.1f} 天前，共 {len(cached.get('map', {}))} 条）")
            return cached["map"]

    log("[orgId] 正在拉取股票列表 ...")
    r, _ = request_with_retry(session, "GET", STOCK_JSON_URL, kind="query")
    if r is None:
        if cached:
            log("[警告] 拉取失败，沿用过期缓存")
            return cached["map"]
        log("[错误] 拉取股票列表失败，且无可用缓存")
        return {}
    try:
        data = r.json()
    except Exception as e:
        log(f"[错误] 股票列表 JSON 解析失败: {e}")
        return cached["map"] if cached else {}

    m = {}
    for item in data.get("stockList") or []:
        code = (item.get("code") or "").strip()
        org = (item.get("orgId") or "").strip()
        if code and org:
            m[code] = org
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "map": m}, f, ensure_ascii=False)
    log(f"[orgId] 已获取 {len(m)} 条并落缓存")
    return m


def query_announcements(session, code, org_id, column, plate, se_date):
    """
    拉取某公司全部年报类公告。
    返回 (归一化记录列表, totalRecordNum, 错误信息)
    """
    url = QUERY_URL
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    }
    out, total, page = [], 0, 1
    while page <= MAX_PAGES:
        payload = {
            "pageNum": page,
            "pageSize": PAGE_SIZE,
            "column": column,
            "tabName": "fulltext",
            "plate": plate,
            "stock": f"{code},{org_id}",  # 必须两段，只传代码返回 0 条
            "searchkey": "",
            "secid": "",
            "category": CATEGORY,
            "trade": "",
            "seDate": se_date,
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        r, _ = request_with_retry(session, "POST", url, kind="query", headers=headers, data=payload)
        if r is None:
            return out, total, "查询请求失败"
        try:
            j = r.json()
        except Exception as e:
            return out, total, f"响应非 JSON: {e}"

        total = j.get("totalRecordNum") or 0
        anns = j.get("announcements") or []  # 无结果时是 null，不是 []
        if not anns:
            break
        for a in anns:
            out.append({
                "title": normalize_title(a.get("announcementTitle")),
                "ts_ms": int(a.get("announcementTime") or 0),
                "date": ts_to_date(a.get("announcementTime")),
                "url": PDF_BASE + (a.get("adjunctUrl") or "").lstrip("/"),
                "raw_url": a.get("adjunctUrl") or "",
                "size_kb": int(a.get("adjunctSize") or 0),
                "ann_id": str(a.get("announcementId") or ""),
            })

        if page * PAGE_SIZE >= total:
            break
        page += 1
        sleep_range(QUERY_DELAY)
    return out, total, ""


# ------------------------------------------------------------------
#  下载
# ------------------------------------------------------------------
def build_output_path(code, name, year, title):
    fn = clean_filename(f"{code}_{name}_{year}年年度报告")
    if "修订" in title or "更新" in title:
        fn += "_修订版"
    return os.path.join(SAVE_DIR, fn + ".pdf")


def check_existing(path, expect_kb):
    """二次防线：兼容历史上用别的工具下过的文件"""
    if not os.path.exists(path):
        return False
    n = os.path.getsize(path)
    if n < MIN_PDF_BYTES:
        return False
    return expect_kb <= 0 or n >= expect_kb * 1024 * SIZE_TOLERANCE


def _cleanup(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def download_pdf(session, url, dest, expect_kb):
    """
    原子落盘：.part + os.replace → 「文件存在 ⟺ 文件完整」。
    整个下载（含流式读取）都带重试——大文件读到一半断流是最常见的失败模式，
    必须重试而不是让脚本崩掉。
    """
    if SKIP_EXISTING and check_existing(dest, expect_kb):
        return STATUS_EXIST, os.path.getsize(dest), ""

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last_err = ""

    for attempt in range(1, MAX_RETRY + 1):
        try:
            with session.get(url, timeout=TIMEOUT_PDF, stream=True) as r:
                if r.status_code != 200:
                    code = r.status_code
                    if not (code in (408, 429) or code >= 500):
                        return STATUS_FAIL, 0, f"HTTP {code}"
                    last_err = f"HTTP {code}"
                else:
                    ctype = (r.headers.get("Content-Type") or "").lower()
                    if "pdf" not in ctype:
                        return STATUS_FAIL, 0, f"Content-Type 非 PDF: {r.headers.get('Content-Type')}"

                    size = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(64 * 1024):
                            if chunk:
                                f.write(chunk)
                                size += len(chunk)

                    # 完整性三连：大小 / 魔数 / EOF
                    with open(tmp, "rb") as f:
                        head = f.read(5)
                        f.seek(max(0, size - 2048))
                        tail = f.read()
                    if size < MIN_PDF_BYTES:
                        _cleanup(tmp)
                        return STATUS_FAIL, size, f"文件过小（{size} 字节），疑似摘要或错误页"
                    if CHECK_PDF_MAGIC and (head != b"%PDF-" or b"%%EOF" not in tail):
                        _cleanup(tmp)
                        return STATUS_FAIL, size, "文件不完整（尾部无 %%EOF）"

                    os.replace(tmp, dest)
                    return STATUS_OK, size, ""
        except KeyboardInterrupt:
            _cleanup(tmp)
            raise
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            _cleanup(tmp)

        if attempt < MAX_RETRY:
            wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 0.6), BACKOFF_CAP)
            if VERBOSE:
                log(f"      ↻ 下载中断（{last_err}），{wait:.0f}s 后重试 {attempt}/{MAX_RETRY - 1}")
            time.sleep(wait)

    _cleanup(tmp)
    return STATUS_FAIL, 0, f"下载失败（已重试 {MAX_RETRY} 次）: {last_err}"


# ------------------------------------------------------------------
#  落盘
# ------------------------------------------------------------------
def write_csv_atomic(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def merge_csv(new_rows, path, key_fields, drop_keys=None):
    """
    把本次结果并入已有 CSV。
    用于 --only / --retry-failed 这类局部运行：不能让它们把全量索引覆盖成只剩几行。
    drop_keys 里的键会先被移除（例如重跑成功后要把它从失败清单里剔除）。
    """
    old = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                old = list(csv.DictReader(f))
        except Exception:
            old = []
    drop = drop_keys or set()

    def key(r):
        return tuple(str(r.get(k, "")) for k in key_fields)

    merged = {}
    for r in old:
        if key(r) in drop:
            continue
        merged[key(r)] = r
    for r in new_rows:
        merged[key(r)] = r
    return list(merged.values())


def append_log(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_retry_pairs(path):
    """读失败清单 → {(代码, 报告期)}"""
    if not os.path.exists(path):
        return set()
    pairs = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("股票代码") or "").strip()
            year = (row.get("报告期") or "").strip()
            if code and year.isdigit():
                pairs.add((code, int(year)))
    return pairs


# ------------------------------------------------------------------
#  自测
# ------------------------------------------------------------------
def self_test():
    ok = fail = 0
    cases = [
        ("2024年年度报告", 2024),
        ("2024年度报告", 2024),
        ("2024 年年度报告", 2024),
        ("恒逸石化2024年年度报告", 2024),
        ("2024年年度报告（修订版）", 2024),
        ("2024年年度报告摘要", None),
        ("2024 Annual Report", None),
        ("2024年年度报告(英文版)", None),
        ("2024年半年度报告", None),
        ("关于2024年年度报告的更正公告", None),
        ("关于举行2024年年度报告网上说明会的公告", None),
        ("2019年年度报告", 2019),
        ("<em>2023年年度报告</em>", 2023),
        ("2024年年度报告及2023年年度报告更正的说明", None),
        # 简写"年报"形态（实测南京化纤 600889 的真实标题）
        ("600889_南京化纤_2025年_年报", 2025),
        ("600889_南京化纤_2025年_年报_摘要", None),
        ("2024年报点评", None),
        ("关于2024年报的问询函", None),
    ]
    for title, expect in cases:
        got = match_annual_report(title)
        if got == expect:
            ok += 1
        else:
            fail += 1
            log(f"  ✗ 匹配失败: {title!r} 期望={expect} 实际={got}")

    market_cases = [("600519", "SH"), ("000703", "SZ"), ("300777", "SZ"),
                    ("688295", "SH"), ("601233", "SH"), ("920077", "BJ")]
    for code, expect in market_cases:
        got = detect_market(code)
        if got == expect:
            ok += 1
        else:
            fail += 1
            log(f"  ✗ 分派失败: {code} 期望={expect} 实际={got}")

    # orgId 断言（只校验本地缓存，不联网）
    cache_file = os.path.join(SAVE_DIR, ORGID_CACHE)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                m = json.load(f).get("map", {})
            for code, expect in [("000703", "gssz0000703"),
                                 ("600519", "gssh0600519"),
                                 ("601233", "9900019547")]:
                got = m.get(code)
                if got == expect:
                    ok += 1
                elif got is None:
                    log(f"  - orgId 缓存无 {code}（跳过）")
                else:
                    fail += 1
                    log(f"  ✗ orgId 不一致: {code} 期望={expect} 实际={got}")
        except Exception as e:
            log(f"  - orgId 缓存读取失败（跳过）: {e}")
    else:
        log("  - 尚未生成 orgId 缓存，先跑一次 --dry-run 可启用 orgId 断言")

    log(f"self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


def verify_output():
    """巡检已下载 PDF 的完整性，并比对索引"""
    index_path = os.path.join(SAVE_DIR, "索引.csv")
    if not os.path.exists(index_path):
        log(f"[错误] 找不到 {index_path}")
        return 1
    with open(index_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ok = fail = 0
    total_bytes = 0
    checked = 0
    for r in rows:
        if r.get("状态") not in (STATUS_OK, STATUS_EXIST):
            continue
        checked += 1
        p = r.get("文件路径") or ""
        if not os.path.exists(p):
            fail += 1
            log(f"  ✗ 索引有记录但文件不存在: {p}")
            continue
        n = os.path.getsize(p)
        total_bytes += n
        with open(p, "rb") as f:
            head = f.read(5)
            f.seek(max(0, n - 2048))
            tail = f.read()
        reason = ""
        if head != b"%PDF-":
            reason = f"文件头异常 {head!r}"
        elif b"%%EOF" not in tail:
            reason = "文件尾缺 %%EOF"
        elif n < MIN_PDF_BYTES:
            reason = f"文件过小 {n} 字节"
        else:
            expect_kb = int(r.get("字节数") or 0)
            if expect_kb and abs(n - expect_kb) / n > 0.05:
                reason = f"与索引记录大小不符（索引 {expect_kb} 实际 {n}）"
        if reason:
            fail += 1
            log(f"  ✗ {os.path.basename(p)}: {reason}")
        else:
            ok += 1

    disk = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(".pdf")] if os.path.isdir(SAVE_DIR) else []
    log("=" * 60)
    log(f"  校验 {ok}/{checked} 通过，失败 {fail}")
    log(f"  磁盘 PDF 文件数: {len(disk)}，索引有效记录数: {checked}")
    log(f"  总体积: {total_bytes / 1024 / 1024:.1f} MB")
    if len(disk) != checked:
        log(f"  ⚠ 磁盘文件数与索引记录数不一致，差 {len(disk) - checked}")
    log("=" * 60)
    return 1 if fail else 0


# ------------------------------------------------------------------
#  主流程
# ------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="巨潮资讯 上市公司年度报告批量下载工具")
    p.add_argument("--dry-run", action="store_true", help="只查询并打印命中，不下载")
    p.add_argument("--years", help="覆盖报告期，如 2024,2025")
    p.add_argument("--only", help="只处理指定代码，如 000703,601233")
    p.add_argument("--retry-failed", action="store_true", help="只重跑失败清单里的")
    p.add_argument("--verify", action="store_true", help="巡检已下载 PDF")
    p.add_argument("--self-test", action="store_true", help="离线自测正则与代码分派")
    p.add_argument("--dump-list", action="store_true", help="导出公司名单 CSV 模板")
    p.add_argument("--list", dest="list_path", default=COMPANY_CSV, help="公司名单 CSV 路径")
    p.add_argument("--bj", action="store_true", help="包含北交所公司")
    p.add_argument("--resume-from", help="从指定代码开始")
    p.add_argument("--force", action="store_true", help="--dump-list 时覆盖已存在文件")
    p.add_argument("--verbose", action="store_true", default=VERBOSE)
    return p.parse_args(argv)


def main(argv=None):
    global VERBOSE
    args = parse_args(argv)
    VERBOSE = args.verbose

    if args.self_test:
        return self_test()
    if args.verify:
        return verify_output()
    if args.dump_list:
        return dump_default_list(args.list_path, args.force)

    years = REPORT_YEARS
    if args.years:
        years = [int(y) for y in args.years.split(",") if y.strip().isdigit()]
    only = set(_norm_code(c) for c in args.only.split(",")) if args.only else None

    os.makedirs(SAVE_DIR, exist_ok=True)
    index_path = os.path.join(SAVE_DIR, "索引.csv")
    preview_path = os.path.join(SAVE_DIR, "索引_预览.csv")
    fail_path = os.path.join(SAVE_DIR, "失败清单.csv")
    log_path = os.path.join(SAVE_DIR, "下载日志.txt")
    retry_pairs = load_retry_pairs(fail_path) if args.retry_failed else None
    # 局部运行：结果要并入已有索引，不能把全量索引覆盖掉
    partial = bool(only) or args.retry_failed

    # 清理上次中断残留
    stale = 0
    for fn in os.listdir(SAVE_DIR):
        if fn.endswith(".pdf.part"):
            try:
                os.remove(os.path.join(SAVE_DIR, fn))
                stale += 1
            except OSError:
                pass
    if stale:
        log(f"[清理] 移除 {stale} 个残留 .part 文件")

    log("=" * 60)
    log("  巨潮资讯 上市公司年度报告批量下载")
    log("=" * 60)
    log(f"  报告期: {years}   保存: {SAVE_DIR}/")
    if args.dry_run:
        log("  模式: DRY-RUN（只查询不下载）")
    if args.retry_failed:
        log(f"  模式: 只重试失败项（{len(retry_pairs)} 格）")

    session = make_session()

    # 公司名单
    comps = load_companies(args.list_path, allow_bj=args.bj, include_borderline=INCLUDE_BORDERLINE)
    if only:
        comps = [c for c in comps if c.code in only]
    if args.resume_from:
        start = _norm_code(args.resume_from)
        codes = [c.code for c in comps]
        if start in codes:
            comps = comps[codes.index(start):]
    enabled = [c for c in comps if c.enabled]
    log(f"[名单] 载入 {len(comps)} 家（启用 {len(enabled)}，停用 {len(comps) - len(enabled)}）")

    # orgId
    org_map = load_orgid_map(session, ORGID_CACHE)
    for c in enabled:
        if not c.org_id:
            c.org_id = org_map.get(c.code, "")

    if not args.dry_run:
        hit = sum(1 for c in enabled if c.org_id)
        log(f"[orgId] 命中 {hit}/{len(enabled)}，缺失 {len(enabled) - hit}")

    total_cells = len(enabled) * len(years)
    log(f"[查询] 共 {len(enabled)} 家 × {len(years)} 年 = {total_cells} 格")
    log("-" * 60)

    index_rows, fail_rows = [], []
    seq = 0
    n_ok = n_exist = n_fail = 0
    consecutive_query_fail = 0

    try:
        for ci, comp in enumerate(enabled, 1):
            market = comp.market or detect_market(comp.code)
            params = market_params(market)

            def emit(year, status, note="", ann=None, size=0, path=""):
                nonlocal seq
                seq += 1
                index_rows.append({
                    "序号": seq, "股票代码": comp.code, "股票简称": comp.name,
                    "交易所": market, "报告期": year,
                    "公告标题": (ann or {}).get("title", ""),
                    "公告日期": (ann or {}).get("date", ""),
                    "公告ID": (ann or {}).get("ann_id", ""),
                    "文件路径": path, "字节数": size,
                    "状态": status, "备注": note,
                })

            # 前置校验
            if not comp.org_id:
                for y in years:
                    emit(y, STATUS_SKIP, "orgId 未找到（可能已退市/新上市/列表未更新）")
                log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} — 跳过（无 orgId）")
                continue
            if params is None:
                for y in years:
                    emit(y, STATUS_SKIP, f"交易所无法判定（{market}）")
                log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} — 跳过（{market}）")
                continue

            column, plate = params
            try:
                anns, total_hit, err = query_announcements(
                    session, comp.code, comp.org_id, column, plate, SE_DATE)
            except Exception as e:
                anns, total_hit, err = [], 0, f"{type(e).__name__}: {e}"

            if err:
                consecutive_query_fail += 1
                for y in years:
                    emit(y, STATUS_FAIL, err)
                    fail_rows.append({
                        "股票代码": comp.code, "股票简称": comp.name, "报告期": y,
                        "公告标题": "", "公告ID": "", "PDF链接": "",
                        "错误信息": err, "尝试次数": MAX_RETRY,
                        "记录时间": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                n_fail += len(years)
                log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} — 查询失败: {err}")
                if consecutive_query_fail >= 5:
                    log("[熔断] 连续 5 家公司查询失败，疑似网络或接口故障。")
                    log("       已保存进度，恢复后可用 --retry-failed 续跑。")
                    break
                continue

            consecutive_query_fail = 0

            picked = pick_annual_reports(anns, set(years))

            for y in years:
                key = (comp.code, y)
                if retry_pairs is not None and key not in retry_pairs:
                    continue
                if not anns:
                    emit(y, STATUS_NO_ANN, f"该区间共查询到 {total_hit} 条年报类公告")
                    continue
                if y not in picked:
                    titles = " / ".join(a["title"] for a in anns[:4])
                    emit(y, STATUS_NO_MATCH, f"有 {len(anns)} 条候选但无匹配。样本: {titles}")
                    continue

                ann = picked[y]
                alts = ann.pop("_alts", [])
                note = ""
                if alts:
                    note = "；另有 %d 条同年度候选: %s" % (
                        len(alts), "; ".join(f"{a['date']}({a['title']})" for a in alts[:3]))
                if ann["size_kb"] and ann["size_kb"] < 300:
                    note += "；⚠附件偏小，疑似摘要"

                path = build_output_path(comp.code, comp.name, y, ann["title"])
                if args.dry_run:
                    log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} {y}年 → "
                        f"{ann['date']} | {ann['title']} | {ann['size_kb']}KB")
                    emit(y, STATUS_SKIP, "dry-run" + note, ann=ann)
                    continue

                status, size, err = download_pdf(session, ann["url"], path, ann["size_kb"])
                if status == STATUS_OK:
                    n_ok += 1
                    log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} {y}年年度报告 "
                        f"✓ 成功 ({size / 1024 / 1024:.1f}MB)")
                    sleep_range(PDF_DELAY)
                elif status == STATUS_EXIST:
                    n_exist += 1
                    log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} {y}年年度报告 — 已存在")
                else:
                    n_fail += 1
                    log(f"[{ci}/{len(enabled)}] {comp.code} {comp.name} {y}年年度报告 ✗ {err}")
                    fail_rows.append({
                        "股票代码": comp.code, "股票简称": comp.name, "报告期": y,
                        "公告标题": ann["title"], "公告ID": ann["ann_id"],
                        "PDF链接": ann["url"], "错误信息": err,
                        "尝试次数": MAX_RETRY,
                        "记录时间": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
                    })

                emit(y, status, note, ann=ann, size=size, path=path)

            sleep_range(COMPANY_DELAY)

    except KeyboardInterrupt:
        log("\n\n[中断] 收到 Ctrl+C，正在保存进度 ...")
    finally:
        if args.dry_run:
            # dry-run 写"预览索引"，不覆盖正式索引
            write_csv_atomic(index_rows, preview_path, INDEX_FIELDS)
        else:
            if partial:
                # 重跑过的格子要先从旧失败清单里移除，再并入本次结果
                evaluated = {(r["股票代码"], str(r["报告期"])) for r in index_rows}
                merged_idx = merge_csv(index_rows, index_path, ["股票代码", "报告期"])
                merged_fail = merge_csv(fail_rows, fail_path, ["股票代码", "报告期"],
                                        drop_keys=evaluated)
            else:
                merged_idx, merged_fail = list(index_rows), list(fail_rows)
            merged_idx.sort(key=lambda r: (r.get("股票代码", ""), str(r.get("报告期", ""))))
            for i, r in enumerate(merged_idx, 1):
                r["序号"] = i
            index_rows[:] = merged_idx
            write_csv_atomic(merged_idx, index_path, INDEX_FIELDS)
            write_csv_atomic(merged_fail, fail_path, FAIL_FIELDS)
        append_log(log_path,
                   f"{datetime.now(TZ_CN):%Y-%m-%d %H:%M:%S} | 成功{n_ok} 已存在{n_exist} "
                   f"失败{n_fail} | 报告期{years} | dry_run={args.dry_run}")

    log("-" * 60)
    if args.dry_run:
        log(f"  预览索引: {len(index_rows)} 条  → {os.path.abspath(preview_path)}")
    else:
        log(f"  索引记录: {len(index_rows)} 条  → {os.path.abspath(index_path)}")
        log(f"  成功 {n_ok} / 已存在 {n_exist} / 失败 {n_fail}")
        if fail_rows:
            log(f"  失败清单 → {os.path.abspath(fail_path)}（可用 --retry-failed 续跑）")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
