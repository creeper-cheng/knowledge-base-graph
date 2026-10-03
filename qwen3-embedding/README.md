# Qwen3-Embedding-0.6B 本地运行

已在本机跑通（CPU 推理，无 CUDA）。

## 实测环境

| 项 | 值 |
|---|---|
| 设备 | CPU（Intel Iris Xe 核显，无 CUDA，走 `device=cpu`） |
| Python | 3.14.7（venv 在 `.venv/`） |
| torch / transformers | 2.14.1 / 5.18.0 |
| 模型 | `Qwen/Qwen3-Embedding-0.6B`，595.8M 参数，**1024 维**向量 |
| 加载耗时 | ~5–7s |
| 编码速度 | **~110ms/条**（短句，batch=8，float32） |

## 用法

```bash
run.bat                                          # 演示：加载模型 + 语义检索排序
run.bat --text "你好，世界" "Hello, world"        # 编码任意文本
run.bat --compare "猫" "狗" "量子力学"             # 两两余弦相似度矩阵
```

或者直接调：

```bash
.venv\Scripts\python.exe qwen3_embed.py
```

在代码里复用：

```python
from qwen3_embed import QwenEmbedder

emb = QwenEmbedder()
vecs = emb.encode(["一段文本", "另一段"], batch_size=8)

# 检索场景：查询侧加指令前缀，文档侧不加，效果明显更好
q = emb.encode(["央行降准的影响"], is_query=True,
               task="Given a web search query, retrieve relevant passages that answer the query")
docs = emb.encode(["中国人民银行宣布下调存款准备金率……"])
scores = q @ docs.T          # 向量已 L2 归一化，点积即余弦相似度
```

## 关于下载

`huggingface.co` 本机直连不通（TLS 握手失败），脚本已写死走 **hf-mirror.com** 镜像，
模型缓存在 `models/`（1.2G）。首次运行自动下载，之后自动切离线模式。

`qwen3_embed.py` 顶部用 `os.environ.setdefault` 设这些变量，所以在 shell 里显式设了
`HF_ENDPOINT` / `HF_HUB_OFFLINE` 也会被尊重、不会被覆盖。

## 两个容易踩的坑

1. **必须 `padding_side='left'`**。Qwen3-Embedding 用「最后一个非 padding token」做池化，
   右 padding 会把整个 batch 的向量全取成 padding 位，结果静默错误、不报错。
   `last_token_pool()` 虽然两种都处理了，但 tokenizer 仍按官方的左 padding 初始化。
2. **查询要加 `Instruct: ...\nQuery: ...` 前缀，文档不加**。非对称检索下不加指令，
   召回质量会明显下降。

## 目录

```
qwen3-embedding/
├── qwen3_embed.py    # 推理脚本（也可当模块 import）
├── run.bat           # Windows 启动器
├── .venv/            # 虚拟环境，约 2.5G
└── models/           # HF 缓存，含模型权重约 1.2G
```

`.venv/` 和 `models/` 已在 `.gitignore` 里，不入库。整个目录约 4G，全在 C 盘项目内，删掉即还原。
