"""
训练 CNN（纯 NumPy）识别 MNIST 手写数字。

结构（见 cnn.py）：
    输入(1×28×28) → Conv(8,3×3)+ReLU → MaxPool(2) →
                    Conv(16,3×3)+ReLU → MaxPool(2) → Flatten(784) → FC(10)+softmax

用法：
    python train_cnn.py [--epochs 12] [--batch 128] [--lr 0.1] [--data data]

产出：
    cnn_model.npz       训练好的权重
    cnn_train_log.json  每轮 train_loss / train_acc / test_acc
    models_meta.json    记录 MLP / CNN 测试集准确率，供前端对比展示
"""

import argparse
import json
import os
import struct

import numpy as np

from download_data import download
from cnn import CNN


# ---------------- 读取 MNIST idx 文件（与 train.py 一致）----------------
def _read_images(path):
    with open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, r * c).astype(np.float32)


def _read_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_dir="data"):
    X_train = _read_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
    y_train = _read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    X_test = _read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    y_test = _read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    X_train /= 255.0
    X_test /= 255.0
    return (X_train, y_train), (X_test, y_test)


def one_hot(y, k=10):
    out = np.zeros((y.shape[0], k), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train(epochs=12, batch=128, lr=0.1, momentum=0.9, data_dir="data"):
    if not os.path.exists(os.path.join(data_dir, "train-images-idx3-ubyte")):
        print("未找到 MNIST，先下载……")
        download(data_dir)

    (X_train, y_train), (X_test, y_test) = load_mnist(data_dir)
    print(f"[data] 训练 {X_train.shape[0]} 张，测试 {X_test.shape[0]} 张")

    model = CNN()
    N = X_train.shape[0]
    n_batches = (N + batch - 1) // batch
    log = []

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_train, y_train = X_train[perm], y_train[perm]
        epoch_loss, correct = 0.0, 0

        for i in range(n_batches):
            xb = X_train[i * batch:(i + 1) * batch]
            yb = y_train[i * batch:(i + 1) * batch]
            yb_oh = one_hot(yb)

            probs = model.forward(xb)               # ŷ
            grads = model.backward(yb)             # 用缓存算梯度

            eps = 1e-8
            loss = -np.mean(np.sum(yb_oh * np.log(probs + eps), axis=1))
            epoch_loss += loss * xb.shape[0]
            correct += int(np.sum(np.argmax(probs, axis=1) == yb))

            # SGD + momentum（直接更新各层参数）
            for key in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"):
                model.__dict__[key] -= lr * grads[key]

        train_loss = float(epoch_loss / N)
        train_acc = float(correct / N)
        test_pred = model.predict(X_test)
        test_acc = float(np.mean(test_pred == y_test))
        print(f"epoch {epoch:2d}/{epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc*100:.2f}%  test_acc={test_acc*100:.2f}%")
        log.append({"epoch": epoch, "loss": train_loss,
                    "train_acc": train_acc, "test_acc": test_acc})

    model.save("cnn_model.npz")
    with open("cnn_train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # 更新 models_meta.json（与 MLP 的准确率并列，供前端对比）
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
    meta["cnn_acc"] = test_acc
    meta["cnn_arch"] = "CNN: Conv(8)→Pool→Conv(16)→Pool→FC(10)"
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)

    print(f"[done] 已保存 cnn_model.npz（测试集准确率 {test_acc*100:.2f}%）与 cnn_train_log.json")
    return model, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr, data_dir=args.data)
