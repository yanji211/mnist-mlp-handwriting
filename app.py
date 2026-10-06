"""
手写数字识别 · 平板输入后端（MLP + CNN 双模型对比）

保留原 MLP 的全部接口不变：
  - GET  /                返回 index.html
  - POST /predict         仅 MLP 预测（向后兼容）
  - POST /learn           纠错学习（同时作用于已加载的 CNN，若可用）
  - POST /reset_learn     重置个性化学习（同时重置 CNN）
  - POST /predict_both    同时返回 MLP 与 CNN 的预测 + 可视化，供前端并排对比

依赖：仅 numpy（已在隔离 venv / 便携 runtime 中）。
启动：python app.py [--port 8000] [--model model.npz] [--cnn cnn_model.npz]
"""

import argparse
import json
import os
import random
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from model import MLP
from cnn import CNN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {"mlp": None, "cnn": None}        # mlp 必加载；cnn 可选（对比用）
BASE_PATHS = {"mlp": None, "cnn": None}
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
    """把当前记住的纠错样本落盘。

    注意：便携运行时（WorkBuddy 自带 sitecustomize）会劫持 os.remove 为
    “安全删除”，在回收站不可用的环境下会抛 OSError。因此这里统一用 np.savez
    覆盖写入，空列表时写入空数组，避免依赖 os.remove，保证 /reset_learn 在
    任何环境下都能正常清空记忆。
    """
    if CORRECTIONS:
        X = np.stack([c[0] for c in CORRECTIONS]).astype(np.float32)   # (N, 784)
        Y = np.array([c[1] for c in CORRECTIONS], dtype=np.int64)
    else:
        X = np.zeros((0, 784), dtype=np.float32)
        Y = np.zeros((0,), dtype=np.int64)
    np.savez(CORR_PATH, X=X, Y=Y)


def replay_corrections():
    """
    用经验回放把已记住的纠错样本重新学一遍，重建个性化模型。
    每次启动都从「基础模型」出发重放，保证结果可复现、可重置。
    同时作用于 MLP 与 CNN（若已加载）。
    """
    for (px, py) in CORRECTIONS:
        for kind, m in MODELS.items():
            if m is not None:
                m.learn_example(px.reshape(1, 784) / 255.0, py, lr=0.005, steps=2)


def _meta():
    """读取两个模型的测试集准确率，供前端对比展示。"""
    meta = {"mlp_acc": None, "cnn_acc": None, "mlp_arch": "MLP: 784→256→128→64→10",
            "cnn_arch": "CNN: Conv(8)→Pool→Conv(16)→Pool→FC(10)"}
    # MLP 准确率：取 train_log.json 最后一轮
    log_path = os.path.join(THIS_DIR, "train_log.json")
    if os.path.exists(log_path):
        try:
            log = json.load(open(log_path))
            if log:
                meta["mlp_acc"] = float(log[-1]["test_acc"])
        except Exception:
            pass
    # CNN 准确率：取 models_meta.json
    mpath = os.path.join(THIS_DIR, "models_meta.json")
    if os.path.exists(mpath):
        try:
            d = json.load(open(mpath))
            if "cnn_acc" in d:
                meta["cnn_acc"] = float(d["cnn_acc"])
        except Exception:
            pass
    return meta


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
            raw = self.rfile.read(length) if length > 0 else b""
            data = json.loads(raw) if raw else {}
        except Exception as e:
            self._send(400, json.dumps({"error": "请求解析失败: " + str(e)}).encode("utf-8"))
            return

        try:
            if self.path == "/predict":
                self._do_predict(data)
            elif self.path == "/predict_both":
                self._do_predict_both(data)
            elif self.path == "/learn":
                self._do_learn(data)
            elif self.path == "/reset_learn":
                self._do_reset_learn()
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:
            # 任何 handler 内部异常都返回 500 JSON，而不是让连接被静默断开
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"))

    # ---------- MLP 可视化（原逻辑不变）----------
    def _build_mlp(self, x):
        probs, acts = MODELS["mlp"].run(x)
        probs = np.asarray(probs).reshape(-1)
        return probs, {
            "input": acts[0], "h1": acts[1], "h2": acts[2],
            "h3": acts[3], "output": acts[4],
        }

    def _do_predict(self, data):
        """仅 MLP（向后兼容老前端 / 接口）。"""
        try:
            pixels = np.asarray(data.get("pixels", []), dtype=np.float32)
            if pixels.shape != (784,):
                raise ValueError(f"需要 784 个像素，收到 {pixels.shape}")
            x = pixels.reshape(1, 784) / 255.0
            probs, layers = self._build_mlp(x)
            pred = int(np.argmax(probs))
            self._send(200, json.dumps({
                "prediction": pred,
                "probabilities": [round(float(p), 4) for p in probs],
                "layers": layers,
                "sizes": MODELS["mlp"].sizes,
                "remembered": len(CORRECTIONS),
            }).encode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

    def _do_predict_both(self, data):
        """MLP + CNN 同时预测与可视化，供前端并排对比。"""
        try:
            pixels = np.asarray(data.get("pixels", []), dtype=np.float32)
            if pixels.shape != (784,):
                raise ValueError(f"需要 784 个像素，收到 {pixels.shape}")
            x = pixels.reshape(1, 784) / 255.0

            # MLP
            mlp_probs, mlp_layers = self._build_mlp(x)
            mlp_pred = int(np.argmax(mlp_probs))
            mlp_out = {
                "prediction": mlp_pred,
                "probabilities": [round(float(p), 4) for p in mlp_probs],
                "layers": mlp_layers,
                "sizes": MODELS["mlp"].sizes,
            }

            # CNN（可选）
            cnn_out = {"present": False}
            if MODELS["cnn"] is not None:
                cnn_probs, cnn_vis = MODELS["cnn"].run(x)
                cnn_probs = np.asarray(cnn_probs).reshape(-1)
                cnn_pred = int(np.argmax(cnn_probs))
                cnn_out = {
                    "present": True,
                    "prediction": cnn_pred,
                    "probabilities": [round(float(p), 4) for p in cnn_probs],
                    "conv1": cnn_vis["conv1"],
                    "conv1_shape": cnn_vis["conv1_shape"],
                    "conv2": cnn_vis["conv2"],
                    "conv2_shape": cnn_vis["conv2_shape"],
                    "arch": cnn_vis["arch"],
                }

            agree = cnn_out["present"] and (mlp_pred == cnn_out["prediction"])
            self._send(200, json.dumps({
                "mlp": mlp_out,
                "cnn": cnn_out,
                "agree": bool(agree),
                "remembered": len(CORRECTIONS),
                "meta": _meta(),
            }).encode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

    def _do_learn(self, data):
        """识别错误 -> 用户纠正 -> 就地更新权重并记住样本（MLP 与 CNN 同时）。"""
        try:
            pixels = np.asarray(data.get("pixels", []), dtype=np.float32)
            label = int(data.get("label"))
            if pixels.shape != (784,):
                raise ValueError(f"需要 784 个像素，收到 {pixels.shape}")
            if not (0 <= label <= 9):
                raise ValueError("label 必须是 0~9")
            x = pixels.reshape(1, 784) / 255.0

            # 1) 对每个已加载模型做几步 SGD
            for kind, m in MODELS.items():
                if m is not None:
                    m.learn_example(x, label, lr=0.01, steps=4)

            # 2) 经验回放：随机重学几个历史纠错，稳定个性化效果
            if len(CORRECTIONS) > 0:
                past = random.sample(CORRECTIONS, min(4, len(CORRECTIONS)))
                for (px, py) in past:
                    for kind, m in MODELS.items():
                        if m is not None:
                            m.learn_example(px.reshape(1, 784) / 255.0, py, lr=0.005, steps=2)

            # 3) 记住这个样本并落盘（基础模型不动）
            CORRECTIONS.append((pixels.astype(np.float32), label))
            save_corrections()

            # 4) 返回更新后的预测与可视化数据（优先返回 MLP 视角，兼容老渲染）
            probs, layers = self._build_mlp(x)
            pred = int(np.argmax(probs))
            cnn_pred = None
            if MODELS["cnn"] is not None:
                cp, _ = MODELS["cnn"].run(x)
                cnn_pred = int(np.argmax(cp))
            self._send(200, json.dumps({
                "prediction": pred,
                "probabilities": [round(float(p), 4) for p in probs],
                "layers": layers,
                "sizes": MODELS["mlp"].sizes,
                "cnn_prediction": cnn_pred,
                "remembered": len(CORRECTIONS),
            }).encode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

    def _do_reset_learn(self):
        """清空所有纠错记忆，恢复到基础模型（MLP 与 CNN 同时）。"""
        global CORRECTIONS, MODELS
        CORRECTIONS = []
        save_corrections()
        # 重新加载未被污染的基础模型
        if BASE_PATHS["mlp"] and os.path.exists(BASE_PATHS["mlp"]):
            MODELS["mlp"] = MLP.load(BASE_PATHS["mlp"])
        if BASE_PATHS["cnn"] and os.path.exists(BASE_PATHS["cnn"]):
            MODELS["cnn"] = CNN.load(BASE_PATHS["cnn"])
        self._send(200, json.dumps({"ok": True, "remembered": 0}).encode("utf-8"))

    def log_message(self, *args):
        pass  # 安静一点


def main():
    global MODELS, BASE_PATHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.path.join(THIS_DIR, "model.npz"))
    ap.add_argument("--cnn", default=os.path.join(THIS_DIR, "cnn_model.npz"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"未找到 MLP 模型 {args.model}，请先运行 python train.py")
    BASE_PATHS["mlp"] = args.model
    MODELS["mlp"] = MLP.load(args.model)
    print("[model] 已加载 MLP", args.model, "结构", MODELS["mlp"].sizes)

    if os.path.exists(args.cnn):
        BASE_PATHS["cnn"] = args.cnn
        MODELS["cnn"] = CNN.load(args.cnn)
        print("[model] 已加载 CNN（对比模型）", args.cnn)
    else:
        print("[model] 未找到 CNN 模型（可选）；仅 MLP 可用，运行 python train_cnn.py 可启用对比")

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
