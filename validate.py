"""
端到端验证：
1) 加载 model.npz，复现测试集准确率；
2) 在进程内起 HTTP 服务，对若干真实测试图发起 /predict 请求，
   确认「前端预处理等价输入 -> 后端推理」闭环可用。
"""
import json
import os
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from model import MLP
import app as appmod

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
MODEL_PATH = os.path.join(PROJ, "model.npz")


def load_idx_images(path):
    with open(path, "rb") as f:
        f.read(16)  # magic + 3 integers
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28 * 28).astype(np.float32)


def load_idx_labels(path):
    with open(path, "rb") as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)


def main():
    # ---- 1. 加载模型并复现测试集精度 ----
    assert os.path.exists(MODEL_PATH), "请先运行 train.py 生成 model.npz"
    model = MLP.load(MODEL_PATH)
    print("[load] 结构", model.sizes)

    X_test = load_idx_images(os.path.join(DATA, "t10k-images-idx3-ubyte"))
    y_test = load_idx_labels(os.path.join(DATA, "t10k-labels-idx1-ubyte"))

    probs = model.forward(X_test / 255.0)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_test).mean()
    print(f"[acc] 测试集 {len(y_test)} 张，准确率 = {acc*100:.2f}%")

    # ---- 2. 起 HTTP 服务，对真实图发起 /predict ----
    port = 8731
    appmod.MODEL = model
    httpd = ThreadingHTTPServer(("127.0.0.1", port), appmod.Handler)
    httpd_thread = __import__("threading").Thread(target=httpd.serve_forever, daemon=True)
    httpd_thread.start()
    print(f"[serve] 测试服务已起 http://127.0.0.1:{port}/predict")

    # 挑 10 张不同数字各一张，做推理闭环测试
    picks = []
    seen = set()
    for i in range(len(y_test)):
        d = y_test[i]
        if d not in seen:
            picks.append(i)
            seen.add(d)
        if len(picks) == 10:
            break

    ok = 0
    for i in picks:
        payload = json.dumps({"pixels": X_test[i].tolist()}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
        correct = resp["prediction"] == int(y_test[i])
        ok += 1 if correct else 0
        print(f"  图#{i:05d} 真值={int(y_test[i])} 预测={resp['prediction']} "
              f"top1={max(resp['probabilities'])*100:.1f}% {'OK' if correct else 'X'}")

    print(f"[loop] 10 张全数字闭环测试：{ok}/10 正确")
    httpd.shutdown()

    print("\n=== 验证结论 ===")
    print(f"模型加载: OK | 测试集准确率: {acc*100:.2f}% | HTTP 闭环: {ok}/10")
    if acc > 0.95 and ok == 10:
        print("整体通过：可交付（`app.py` 可直接给平板用）。")
    else:
        print("存在未达标项，请检查。")


if __name__ == "__main__":
    main()
