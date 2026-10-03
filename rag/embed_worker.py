# -*- coding: utf-8 -*-
r"""常驻编码 worker —— 跑在 qwen3-embedding\.venv 里（torch 只装在那儿）。

为什么要单独一个常驻进程：主环境没有 torch，而服务进程又不该把 torch 拖进来
（GIL 争用、启动被 4-7 秒模型加载阻塞、worker 挂了要重启整个服务）。
所以模型在这个子进程里加载一次、常驻，用 stdin/stdout 行协议跟服务通信。

协议（一行一个 JSON）：
    启动完成 → {"ready": true, "dim": 1024, "model_id": "...", "load_s": 2.4}
    收 {"q": "查询文本"} → 发 {"vec": [...1024 floats...], "ms": 123.4}
    出错 → {"error": "..."}
    收 {"cmd": "quit"} → 退出

⚠ stdout 只走协议，任何日志都必须写 stderr，否则会污染协议流。
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "qwen3-embedding"))

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK = ("Given a Chinese A-share annual report query, "
                "retrieve the passage that answers it")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import torch
    from qwen3_embed import QwenEmbedder

    threads = int(os.environ.get("EMB_THREADS", "16"))
    torch.set_num_threads(threads)
    task = os.environ.get("EMB_TASK", DEFAULT_TASK)

    t0 = time.time()
    emb = QwenEmbedder(dtype=torch.float32, verbose=False)
    # 预热一次，把首查的额外延迟（内核选择/JIT）提前吃掉
    emb.encode(["预热"], is_query=True, task=task)
    load_s = round(time.time() - t0, 2)
    log(f"[worker] ready in {load_s}s threads={threads}")

    out = sys.stdout
    out.write(json.dumps({"ready": True, "dim": 1024, "model_id": MODEL_ID,
                          "load_s": load_s}) + "\n")
    out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            out.write(json.dumps({"error": f"bad json: {e}"}) + "\n")
            out.flush()
            continue
        if req.get("cmd") == "quit":
            log("[worker] quit")
            break
        try:
            t = time.time()
            vec = emb.encode([req.get("q", "")], is_query=True, task=task)[0]
            out.write(json.dumps({
                "vec": vec.numpy().tolist(),
                "ms": round((time.time() - t) * 1000, 1),
            }) + "\n")
        except Exception as e:
            out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
