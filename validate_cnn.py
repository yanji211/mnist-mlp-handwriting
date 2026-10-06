"""
端到端验证 CNN 对比功能：
1) 加载 MLP 与 CNN 两个模型，复现各自测试集准确率；
2) 起 HTTP 服务，对真实测试图发起 /predict_both，确认
   MLP / CNN 预测、可视化数据、一致结论、准确率元信息 都正常返回。
"""
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from model import MLP
from cnn import CNN
import app as appmod

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
MODEL_PATH = os.path.join(PROJ, "model.npz")
CNN_PATH = os.path.join(PROJ, "cnn_model.npz")


def load_idx_images(path):
    with open(path, "rb") as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28 * 28).astype(np.float32)


def load_idx_labels(path):
    with open(path, "rb") as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)


def main():
    assert os.path.exists(MODEL_PATH), "请先运行 train.py 生成 model.npz"
    assert os.path.exists(CNN_PATH), "请先运行 train_cnn.py 生成 cnn_model.npz"

    mlp = MLP.load(MODEL_PATH)
    cnn = CNN.load(CNN_PATH)
    X_test = load_idx_images(os.path.join(DATA, "t10k-images-idx3-ubyte"))
    y_test = load_idx_labels(os.path.join(DATA, "t10k-labels-idx1-ubyte"))

    mp = mlp.predict(X_test / 255.0)
    cp = cnn.predict(X_test / 255.0)
    print(f"[acc] MLP 测试集准确率 = {np.mean(mp == y_test)*100:.2f}%")
    print(f"[acc] CNN 测试集准确率 = {np.mean(cp == y_test)*100:.2f}%")
    agree = np.mean(mp == cp)
    print(f"[agree] 两个模型预测一致率 = {agree*100:.2f}%")

    # 起 HTTP 服务测试 /predict_both
    port = 8732
    appmod.MODELS["mlp"] = mlp
    appmod.MODELS["cnn"] = cnn
    appmod.BASE_PATHS["mlp"] = MODEL_PATH
    appmod.BASE_PATHS["cnn"] = CNN_PATH
    httpd = ThreadingHTTPServer(("127.0.0.1", port), appmod.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    picks = []
    seen = set()
    for i in range(len(y_test)):
        if y_test[i] not in seen:
            picks.append(i); seen.add(int(y_test[i]))
        if len(picks) == 10:
            break

    ok = 0
    for i in picks:
        payload = json.dumps({"pixels": X_test[i].tolist()}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict_both",
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
        mlp_p = resp["mlp"]["prediction"]
        cnn_p = resp["cnn"]["prediction"] if resp["cnn"]["present"] else None
        # 结构正确性校验（与模型准确率无关）
        has_mlp_layers = "input" in resp["mlp"]["layers"] and "output" in resp["mlp"]["layers"]
        has_cnn_feat = resp["cnn"]["present"] and len(resp["cnn"]["conv1"]) == 8 \
                       and len(resp["cnn"]["conv1"][0]) == 28 * 28
        agree_consistent = resp["agree"] == (mlp_p == cnn_p)
        struct_ok = has_mlp_layers and has_cnn_feat and agree_consistent \
                    and isinstance(mlp_p, int) and (cnn_p is None or isinstance(cnn_p, int))
        ok += 1 if struct_ok else 0
        print(f"  图#{i:05d} 真值={int(y_test[i])} MLP={mlp_p} "
              f"CNN={cnn_p if cnn_p is not None else '-'} "
              f"agree={resp['agree']} {'OK' if struct_ok else 'X'}")
    httpd.shutdown()

    print("\n=== 验证结论 ===")
    print(f"MLP 准确率 {np.mean(mp==y_test)*100:.2f}% | CNN 准确率 {np.mean(cp==y_test)*100:.2f}% | "
          f"两模型一致率 {agree*100:.2f}%")
    print(f"/predict_both 结构闭环 {ok}/10 （仅校验返回结构/一致性与预测合法性，不含模型准确率）")
    if ok == 10 and np.mean(mp == y_test) > 0.94 and np.mean(cp == y_test) > 0.94:
        print("整体通过：MLP + CNN 对比功能可交付（`app.py` 可直接给平板用）。")
    else:
        print("存在未达标项，请检查。")


if __name__ == "__main__":
    main()
