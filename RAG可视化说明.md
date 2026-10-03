# RAG 过程可视化

把 A 股年报向量库的**内部过程**画出来：散点图看聚类、点击看原文、提问后分四段依次演示
「问题 → 向量 → 找邻居 → 变答案」，最后交 DeepSeek 生成带引用的回答。

---

## 快速开始

```
1) 建索引（只跑一次）     双击 build_index.bat
2) 启动服务（一条命令）   双击 run_rag.bat
3) 浏览器自动打开          http://localhost:8765/
```

默认监听 `0.0.0.0`，端口从 `8765` 起自动挑一个空闲的（`8766 → 8767 → 5173 → 8000`），
启动时会打印实际地址。首次运行 Windows 会弹防火墙授权，**允许即可**（只是想局域网访问才需要，
本机访问不受影响）。

命令行等价写法：

```bat
qwen3-embedding\.venv\Scripts\python.exe _work\07_embed.py --sample   :: 向量化
python _work\08_index.py                                              :: 合并向量矩阵
python _work\10_build_index.py                                        :: UMAP + BM25 + 偏移表
python _work\11_app.py --port 8765                                    :: 起服务
```

## DeepSeek 配置

本机原本没有任何 DeepSeek 凭据，需要在启动前设置环境变量。最省事的做法是在仓库根目录建一个
`deepseek.env`（`run_rag.bat` 会自动读进来，每行 `KEY=VALUE`）：

```
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-chat
```

**没配也能用**——第 ①②③ 步是纯本地检索，照常演示；第 ④ 步会明确提示要设哪些变量，不会崩。

---

## 界面上的四个阶段（有时间先后，不是一次全出来）

| 阶段 | 你会看到 |
|---|---|
| **① 问题 → 1024 维向量** | 前 8 个分量带符号逐个蹦出（等宽字体对齐），配一个前 48 维的正负条形图，附 L2 范数与编码耗时 |
| **② 向量落在哪里** | 相机平滑移到该点，蓝色脉冲圈标出位置 |
| **③ 召回邻居** | 逐条从查询点生长连线、标出每个块的余弦相似度；右侧列表同步填充 |
| **④ 生成回答** | 交 DeepSeek 流式生成，答案里的 `[n]` 是可点的引用 chip，点了会定位并展开该块原文 |

**散点图**是全部文本块的向量降到二维的结果。滚轮缩放、拖拽平移、双击复位；
**点任意一个点**打开右侧抽屉看该块原文；左下角图例可按章节／公司／年份切换着色，点击可筛除。

### 想验证「确实分了段」而不是一次性渲染

浏览器控制台执行：

```js
__stageLog.map(e => [e.stage, Math.round(e.t)])
```

会看到 `①向量 / ②落点 / ③邻居 / ④回答` 的时间戳依次递增、相邻间隔 ≥600ms。
`①preview` 那一条还存了实际显示的 8 个数值，可与接口返回逐位比对。

URL 加 `?slow=3` 可把所有停顿放大 3 倍，方便录屏或肉眼确认。

---

## 环境为什么分两个 Python

本机 torch 只装在 `qwen3-embedding/.venv`，jieba 只在主环境，两边都没有 Web 框架。

解法是**常驻子进程 + 行协议**，而不是把 torch 拖进服务进程：

```
主环境 (pythoncore-3.14)           venv (qwen3-embedding/.venv)
  ├─ HTTP 服务  11_app.py    ⇄     └─ 常驻编码器  12_embed_worker.py
  ├─ 建索引     10_build_index.py        （模型加载一次驻留，stdin/stdout JSON 行协议）
  └─ DeepSeek 调用
```

这样 UI 能秒开、模型加载不阻塞服务、worker 挂了可以单独重启，也不用动 venv。

## 文件

| 文件 | 作用 |
|---|---|
| `run_rag.bat` / `build_index.bat` | 一条命令启动 / 建索引 |
| `_work/10_build_index.py` | UMAP 降维 + BM25 倒排 + chunk 字节偏移表 |
| `_work/11_app.py` | 标准库 HTTP 服务（零依赖）+ 全部接口 + DeepSeek 流式 |
| `_work/12_embed_worker.py` | venv 侧常驻编码器 |
| `_work/web/index.html` | 单文件前端（原生 JS + 内联 CSS，零 CDN） |
| `_work/_smoke.py` | 端到端冒烟测试，14 项断言 |
| `年报_合成纤维制造/_derived/viz/` | 可视化产物（coords/groups/bm25/offsets/build_meta） |

## 接口

| 路径 | 说明 |
|---|---|
| `GET /api/health` | 索引/模型/DeepSeek 就绪状态、语料规模 |
| `GET /api/meta` | 语料统计、章节与公司表、是否样本模式 |
| `GET /api/coords` | 二进制：32B 头 + xy(float32) + 章节/公司/标记(u8) |
| `GET /api/chunk?row=` | 该块原文，用偏移表 O(1) seek |
| `POST /api/retrieve` | 检索：返回向量预览、查询落点、top-k 邻居与相似度 |
| `GET /api/vec?qid=` | 完整 1024 维向量（"复制完整向量"用） |
| `POST /api/answer` | NDJSON 流：start / token / citations / done / error |
| `GET /api/kw?q=` | BM25 关键词检索（只给搜索框，**不参与 RAG 召回**） |

## 几个设计上的说明

**召回只用向量**，BM25 倒排索引单独建、只服务关键词搜索框。

**查询向量在散点图上的落点**取 top-k 邻居二维坐标的**相似度加权重心**。
真正的做法是保留整个 UMAP reducer 用 `transform()` 外推，但那要 pickle 几百 MB；
加权重心既廉价，又与"以它为中心召回周围这几个点"的叙事天然一致。界面上如实标注了这一点。

**向量不许量化**。实测 Qwen3-Embedding-0.6B 用 PyTorch 动态 int8 量化后，
与 fp32 的平均余弦只剩 **0.637**（元凶是 FFN），等于换了模型而且**不报错**。
所以索引与查询两侧都必须 fp32。

**跨页块允许、跨章节块不允许**；块长 ≤800 字、重合 ≥120 字、尽量在句末收尾
（纯正文块句末率 85–95%；含表格块的"不结尾"是正确的，表格行本来就以单元格结尾）。

## 已知限制

- 监听 `0.0.0.0` 且**无鉴权**，仅限内网演示，**不要暴露到公网**。
- 后端向量化占满 CPU 时，单次查询编码会从几百毫秒涨到 4–5 秒（前端有"模型加载中"提示但不会提示变慢）。
  等后台向量化跑完就恢复正常。
- `?slow=` 只影响前端停顿，不影响后端耗时。
