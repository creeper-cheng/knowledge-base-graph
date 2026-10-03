# knowledge-base-graph

把 A 股年报做成可检索的知识库，并把 RAG 的内部过程可视化。

代码在 [`rag/`](rag/README.md)，那里有完整的设计说明与踩坑记录。

## 这是什么

- **散点图**：全部文本块的向量降到二维，按章节着色，一眼看出聚成哪几团
- **点开看原文**：点任意一点展开该块的原文（公司 / 年份 / 章节 / 页码）
- **单文档问答**：提问后分段演示 ① 向量前 8 个真实数值 → ② 向量落在散点图的位置
  → ③ 逐条画出召回的邻居与相似度 → ④ DeepSeek 流式生成带 `[n]` 引用的回答
- **跨文档聚合**：逐家扫描全部年报、各抽一份结构化结论再汇总
  （「哪些公司计提了存货跌价准备，它们之后的产能计划是什么」这类问题，
  top-k 检索天然答不了，必须 map-reduce）

## 快速开始

```bat
build_index.bat     先建索引（抽取 → 切块 → 向量化 → UMAP/BM25）
run_rag.bat         起服务并打开浏览器
```

需要先准备两样东西（都不入库）：

1. 年报 PDF 放在 `年报_合成纤维制造/`，用 `download_annual_reports.py` 从巨潮资讯下载
2. `qwen3-embedding/` 下的嵌入模型，按它的 README 从 hf-mirror 拉取

DeepSeek 的 key 放在仓库根目录 `deepseek.env`（已在 .gitignore 里）：

```
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-chat
```

没配也能跑——单文档问答的前三步是纯本地检索，第四步会提示怎么配。

## 数据与模型为什么不入库

年报 PDF 约 350 MB、抽取文本约 800 MB、切块 122 MB、向量 50 MB，
模型权重 1.2 GB。这些都可以从源头重新生成或下载，
放进 git 只会把仓库撑爆（且 GitHub 单文件超 100 MB 会被拒）。
