"""对 model.MLP 做数值梯度校验，确认 forward/backward 正确（不依赖 MNIST）。"""
import numpy as np
from model import MLP, softmax


def loss_of(model, X, y):
    probs = model.forward(X)
    yoh = np.zeros((1, probs.shape[1]), dtype=np.float32)
    yoh[0, y] = 1.0
    return float(-np.sum(yoh * np.log(probs[0] + 1e-8)))


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, 4)).astype(np.float32)
    y = 2  # 类别
    model = MLP(sizes=(4, 5, 3), seed=1)

    # 解析梯度
    model.forward(X)
    grads = model.backward(np.array([y]))

    # 数值梯度
    eps = 1e-3
    max_err = 0.0
    for name in ("W", "b"):
        for l in range(1, model.n_layers + 1):
            param = model.W[l] if name == "W" else model.b[l]
            flat = param.reshape(-1).copy()
            gflat = (grads[name][l]).reshape(-1)
            for i in range(flat.shape[0]):
                orig = flat[i]
                flat[i] = orig + eps
                if name == "W":
                    model.W[l] = flat.reshape(param.shape)
                else:
                    model.b[l] = flat.reshape(param.shape)
                lp = loss_of(model, X, y)
                flat[i] = orig - eps
                if name == "W":
                    model.W[l] = flat.reshape(param.shape)
                else:
                    model.b[l] = flat.reshape(param.shape)
                lm = loss_of(model, X, y)
                flat[i] = orig
                if name == "W":
                    model.W[l] = flat.reshape(param.shape)
                else:
                    model.b[l] = flat.reshape(param.shape)
                num = (lp - lm) / (2 * eps)
                ana = gflat[i]
                rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-8)
                max_err = max(max_err, rel)
    print(f"max relative gradient error = {max_err:.2e}")
    assert max_err < 1e-2, "梯度校验未通过！"
    print("梯度校验通过 ✓ (forward/backward 实现正确)")


if __name__ == "__main__":
    main()
