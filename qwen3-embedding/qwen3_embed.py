"""Qwen3-Embedding-0.6B 本地推理（CPU）。

用法:
    python qwen3_embed.py                     # 跑内置演示：加载模型 + 检索排序
    python qwen3_embed.py --text "你好，世界"   # 编码任意文本，打印维度/耗时/前几个值
    python qwen3_embed.py --compare "猫" "狗" "量子力学"   # 两两余弦相似度矩阵

镜像与缓存路径在下面写死，避免每次都要配环境变量。
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_HF_HOME = os.path.join(_HERE, "models")
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# 模型已经在本地缓存过就直接离线跑：省掉每次加载时的联网校验（快 ~1.5s，没网也能跑）。
# 首次运行缓存还不存在，会自动走 hf-mirror 下载。
if os.path.isdir(os.path.join(_HF_HOME, "hub", "models--Qwen--Qwen3-Embedding-0.6B")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding 官方取池化方式：取最后一个非 padding token 的隐状态。

    注意：配合 padding_side='left' 时，padding 在左边，有效 token 靠右，
    所以最后一个位置永远是有效 token；右 padding 则要按下标逐个取。
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def format_query(task_description: str, query: str) -> str:
    """Qwen3-Embedding 的 query 指令模板。文档侧不加前缀。"""
    return f"Instruct: {task_description}\nQuery: {query}"


class QwenEmbedder:
    def __init__(self, model_id: str = MODEL_ID, dtype: torch.dtype = torch.float32, verbose: bool = True):
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[load] {model_id}")
            print(f"[load] device={self.device} dtype={dtype} params={n_params/1e6:.1f}M "
                  f"耗时={time.time() - t0:.1f}s")

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_length: int = 8192,
        is_query: bool = False,
        task: str | None = None,
    ) -> torch.Tensor:
        """返回 L2 归一化后的向量，shape=(len(texts), hidden)，余弦相似度直接用点积算。"""
        if is_query and task is not None:
            texts = [format_query(task, t) for t in texts]

        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb.float(), p=2, dim=1)
            chunks.append(emb.cpu())
        return torch.cat(chunks, dim=0)


DEMO_DOCS = [
    "苹果公司 2024 财年第四季度营收 949 亿美元，同比增长 6%，服务业务创历史新高。",
    "合成纤维制造行业在 2025 年面临原料价格波动，涤纶长丝产能利用率降至 78%。",
    "中国人民银行宣布下调金融机构存款准备金率 0.5 个百分点，释放长期流动性约 1 万亿元。",
    "研究发现，每天摄入适量坚果与心血管疾病风险降低呈正相关。",
    "该论文提出一种基于注意力机制的稀疏检索方法，在 BEIR 基准上超过了 BM25。",
]


def run_demo(embedder: QwenEmbedder) -> None:
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = ["央行降准对市场有什么影响？", "什么模型在检索基准上打败了 BM25？"]

    print(f"\n[demo] 语料 {len(DEMO_DOCS)} 条，检索 {len(queries)} 个问题\n")

    t0 = time.time()
    doc_emb = embedder.encode(DEMO_DOCS, batch_size=8)
    t_doc = time.time() - t0
    t0 = time.time()
    q_emb = embedder.encode(queries, batch_size=8, is_query=True, task=task)
    t_q = time.time() - t0

    print(f"[demo] 向量维度={doc_emb.shape[1]}  "
          f"文档编码 {t_doc:.2f}s ({t_doc/len(DEMO_DOCS)*1000:.0f}ms/条)  "
          f"查询编码 {t_q:.2f}s")

    sims = q_emb @ doc_emb.T
    for qi, q in enumerate(queries):
        print(f"\n查询: {q}")
        ranked = sims[qi].argsort(descending=True)
        for rank, di in enumerate(ranked[:3], 1):
            print(f"  {rank}. [{sims[qi][di]:.4f}] {DEMO_DOCS[di][:52]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B 本地推理")
    parser.add_argument("--text", nargs="+", help="直接编码这些文本并打印向量信息")
    parser.add_argument("--compare", nargs="+", help="打印这些文本两两之间的余弦相似度矩阵")
    args = parser.parse_args()

    embedder = QwenEmbedder()

    if args.text:
        t0 = time.time()
        emb = embedder.encode(args.text)
        dt = time.time() - t0
        print(f"\n[encode] {len(args.text)} 条，维度={emb.shape[1]}，耗时={dt:.3f}s")
        for text, vec in zip(args.text, emb):
            print(f"  {text[:40]!r} -> norm={vec.norm():.4f} 前5维={vec[:5].tolist()}")
            print(f"      前5维和={vec[:5].sum():.4f}")
        return

    if args.compare:
        emb = embedder.encode(args.compare)
        sims = emb @ emb.T
        width = max(len(t) for t in args.compare) + 2
        print()
        print(" " * width + "".join(f"{t[:8]:>10}" for t in args.compare))
        for i, text in enumerate(args.compare):
            row = "".join(f"{sims[i][j]:>10.4f}" for j in range(len(args.compare)))
            print(f"{text[:width-2]:<{width}}" + row)
        return

    run_demo(embedder)


if __name__ == "__main__":
    main()
