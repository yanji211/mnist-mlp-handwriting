"""
手写数字识别 · 平板输入后端

- GET  /            返回 index.html（画图 + 预测界面）
- POST /predict     接收 784 个 0~255 灰度值，返回预测数字与 10 类概率

依赖：仅 numpy（已在隔离 venv 中）。
启动：python app.py [--port 8000] [--model model.npz]

平板访问：让平板与电脑连同一 WiFi，浏览器打开 http://<电脑局域网IP>:8000
"""

import argparse
import json
import os
import random
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from model import MLP

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = None
BASE_MODEL_PATH = None
CORRECTIONS = []          # list of (np.ndarray(784,), int) —— 用户纠正并记住的样本
CORR_PATH = os.path.join(THIS_DIR, "corrections.npz")


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def load_corrections():
    """从磁盘读取已记住的纠错样本（若存在）。"""
    global CORRECTIONS
    CORRECTIONS = []
    if os.path.exists(CORR_PATH):
        d = np.load(CORR_PATH)
        X, Y = d["X"], d["Y"]
        CORRECTIONS = [(X[i].astype(np.float32), int(Y[i])) for i in range(len(Y))]


def save_corrections():
    """把当前记住的纠错样本落盘。"""
    if CORRECTIONS:
        X = np.stack([c[0] for c in CORRECTIONS]).astype(np.float32)   # (N, 784)
        Y = np.array([c[1] for c in CORRECTIONS], dtype=np.int64)
        np.savez(CORR_PATH, X=X, Y=Y)
    elif os.path.exists(CORR_PATH):
        os.remove(CORR_PATH)


def replay_corrections():
    """
    用经验回放把已记住的纠错样本重新学一遍，重建个性化模型。
    每次启动都从「基础模型」出发重放，保证结果可复现、可重置。
    """
    for (px, py) in CORRECTIONS:
        MODEL.learn_example(px.reshape(1, 784) / 255.0, py, lr=0.005, steps=2)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            path = os.path.join(THIS_DIR, "index.html")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, b"index.html not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw)
        except Exception as e:
            self._send(400, json.dumps({"error": "请求解析失败: " + str(e)}).encode("utf-8"))
            return

        if self.path == "/predict":
            self._do_predict(data)
        elif self.path == "/learn":
            self._do_learn(data)
        elif self.path == "/reset_learn":
            self._do_reset_learn()
        else:
            self._send(404, b"not found", "text/plain")

    def _build_layers(self, x):
        probs, acts = MODEL.run(x)
        probs = np.asarray(probs).reshape(-1)
        return probs, {
            "input": acts[0], "h1": acts[1], "h2": acts[2],
            "h3": acts[3], "output": acts[4],
        }

    def _do_predict(self, data):
        try:
            pixels = np.asarray(data.get("pixels", []), dtype=np.float32)
            if pixels.shape != (784,):
                raise ValueError(f"需要 784 个像素，收到 {pixels.shape}")
            x = pixels.reshape(1, 784) / 255.0
            probs, layers = self._build_layers(x)
            pred = int(np.argmax(probs))
            self._send(200, json.dumps({
                "prediction": pred,
                "probabilities": [round(float(p), 4) for p in probs],
                "layers": layers,
                "sizes": MODEL.sizes,
                "remembered": len(CORRECTIONS),
            }).encode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

    def _do_learn(self, data):
        """识别错误 -> 用户纠正 -> 就地更新权重并记住样本。"""
        try:
            pixels = np.asarray(data.get("pixels", []), dtype=np.float32)
            label = int(data.get("label"))
            if pixels.shape != (784,):
                raise ValueError(f"需要 784 个像素，收到 {pixels.shape}")
            if not (0 <= label <= 9):
                raise ValueError("label 必须是 0~9")
            x = pixels.reshape(1, 784) / 255.0

            # 1) 在新样本上做几步 SGD（小学习率，避免灾难性遗忘）
            MODEL.learn_example(x, label, lr=0.01, steps=4)
            # 2) 经验回放：随机重学几个历史纠错，稳定个性化效果
            if len(CORRECTIONS) > 0:
                past = random.sample(CORRECTIONS, min(4, len(CORRECTIONS)))
                for (px, py) in past:
                    MODEL.learn_example(px.reshape(1, 784) / 255.0, py, lr=0.005, steps=2)

            # 3) 记住这个样本并落盘（基础模型 model.npz 不动）
            CORRECTIONS.append((pixels.astype(np.float32), label))
            save_corrections()

            # 4) 返回更新后的预测与可视化数据
            probs, layers = self._build_layers(x)
            pred = int(np.argmax(probs))
            self._send(200, json.dumps({
                "prediction": pred,
                "probabilities": [round(float(p), 4) for p in probs],
                "layers": layers,
                "sizes": MODEL.sizes,
                "remembered": len(CORRECTIONS),
            }).encode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

    def _do_reset_learn(self):
        """清空所有纠错记忆，恢复到基础模型。"""
        global CORRECTIONS, MODEL
        CORRECTIONS = []
        save_corrections()
        MODEL = MLP.load(BASE_MODEL_PATH)   # 重新加载未被污染的基准模型
        self._send(200, json.dumps({"ok": True, "remembered": 0}).encode("utf-8"))

    def log_message(self, *args):
        pass  # 安静一点


def main():
    global MODEL, BASE_MODEL_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.path.join(THIS_DIR, "model.npz"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"未找到模型 {args.model}，请先运行 python train.py")
    BASE_MODEL_PATH = args.model
    MODEL = MLP.load(args.model)
    print("[model] 已加载", args.model, "结构", MODEL.sizes)

    # 加载并回放已记住的纠错样本，重建个性化模型
    load_corrections()
    if CORRECTIONS:
        replay_corrections()
        print(f"[learn] 已回放 {len(CORRECTIONS)} 个纠错样本，个性化模型就绪")
    else:
        print("[learn] 暂无纠错记忆")

    ip = get_local_ip()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] 本机:   http://localhost:{args.port}")
    print(f"[serve] 平板:   http://{ip}:{args.port}   (同一 WiFi)")
    print("[serve] 按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 已停止")


if __name__ == "__main__":
    main()
