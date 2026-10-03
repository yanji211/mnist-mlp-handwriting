"""
三层神经网络（3 个隐藏神经元层）识别 MNIST 手写数字。

结构：  输入(784) -> 隐藏①(256) -> 隐藏②(128) -> 隐藏③(64) -> 输出(10)
激活：  隐藏层 ReLU，输出层 softmax
损失：  交叉熵

本文件用纯 NumPy 实现，代码与《三层网络识别手写数字_教学文档.md》中的公式
逐行对应，方便学生把「数学」和「代码」对起来看。

各符号对应文档：
  z^l = W^l · a^{l-1} + b^l
  a^l = ReLU(z^l)            (隐藏层)
  ŷ   = softmax(z^4)        (输出层)
  δ^4 = ŷ - y
  δ^l = (W^{l+1})^T · δ^{l+1} ⊙ ReLU'(z^l)
  ∂L/∂W^l = δ^l · (a^{l-1})^T
"""

import numpy as np


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator):
    """He 初始化：适配 ReLU，方差 = 2 / fan_in。"""
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_out, fan_in)) * std


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """ReLU 的导数：z>0 时为 1，否则为 0。"""
    return (x > 0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    # 数值稳定：减去每行最大值
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


class MLP:
    """3 隐藏层多层感知机。"""

    def __init__(self, sizes=(784, 256, 128, 64, 10), seed=42):
        self.sizes = list(sizes)          # [784, 256, 128, 64, 10]
        self.n_layers = len(sizes) - 1    # 4 个权重矩阵（含输出层）
        rng = np.random.default_rng(seed)

        # 初始化权重与偏置：W[l] 形状 (sizes[l+1], sizes[l])
        self.W = [None]                       # 占位，使下标从 1 开始
        self.b = [None]
        for l in range(1, self.n_layers + 1):
            fan_in, fan_out = sizes[l - 1], sizes[l]
            self.W.append(he_init(fan_in, fan_out, rng))
            self.b.append(np.zeros((1, fan_out)))

        # 缓存（forward 时填充，backward 时使用）
        self.cache = {}

    # ---------- 前向传播 ----------
    def forward(self, X: np.ndarray):
        """
        X: shape (N, 784)。返回 softmax 概率 (N, 10)。
        同时把每层的中间量存进 self.cache 供 backward 使用。
        """
        X = np.asarray(X, dtype=np.float32)
        activations = [X]   # a^0 = x
        zs = [None]         # z^0 不存在，占位
        a = X
        for l in range(1, self.n_layers + 1):
            z = a @ self.W[l].T + self.b[l]      # z^l = W^l·a^{l-1} + b^l
            if l == self.n_layers:
                a = softmax(z)                   # 输出层用 softmax
            else:
                a = relu(z)                       # 隐藏层用 ReLU
            zs.append(z)
            activations.append(a)
        self.cache = {"activations": activations, "zs": zs}
        return activations[-1]                    # ŷ

    # ---------- 反向传播 ----------
    def backward(self, y_true: np.ndarray):
        """
        y_true: 整型标签 (N,) 或 one-hot (N, 10)。
        返回 grads 字典：grads['W'][l], grads['b'][l]（下标从 1 开始）。
        """
        activations = self.cache["activations"]   # a^0..a^4
        zs = self.cache["zs"]                     # z^1..z^4
        N = activations[0].shape[0]

        if y_true.ndim == 1:                      # 转 one-hot
            y = np.zeros((N, self.sizes[-1]), dtype=np.float32)
            y[np.arange(N), y_true.astype(int)] = 1.0
        else:
            y = y_true.astype(np.float32)

        grads = {"W": [None] * (self.n_layers + 1),
                 "b": [None] * (self.n_layers + 1)}

        # 输出层：δ^4 = ŷ - y  （softmax + 交叉熵的优雅结论）
        delta = activations[-1] - y               # (N, 10)

        # 从输出层往回逐层递推
        for l in range(self.n_layers, 0, -1):
            a_prev = activations[l - 1]           # a^{l-1}
            dW = delta.T @ a_prev / N             # ∂L/∂W^l = δ^l · (a^{l-1})^T / N
            db = np.sum(delta, axis=0, keepdims=True) / N
            grads["W"][l] = dW
            grads["b"][l] = db
            if l > 1:
                # δ^{l-1} = (W^l)^T · δ^l ⊙ ReLU'(z^{l-1})
                delta = (delta @ self.W[l]) * relu_grad(zs[l - 1])
        return grads

    # ---------- 预测 ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    # ---------- 增量学习（纠错） ----------
    def learn_example(self, x, y_true, lr: float = 0.01, steps: int = 4):
        """
        对一个样本 (x, y_true) 做几步 SGD 更新，就地修改权重。

        这是「识别错了 -> 用户纠正 -> 模型记住」的核心：
        复用 forward/backward 得到梯度，再沿负梯度方向更新 W、b。
        学习率 lr 故意取小、步数 steps 少，避免单个纠错样本把
        已学好的知识「灾难性遗忘」。

        返回更新后该样本的 softmax 概率 (10,)。
        """
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        y_true = int(y_true)
        for _ in range(steps):
            self.forward(x)                       # 填充 self.cache
            grads = self.backward(np.array([y_true], dtype=int))
            for l in range(1, self.n_layers + 1):
                self.W[l] -= lr * grads["W"][l]
                self.b[l] -= lr * grads["b"][l]
        return self.forward(x)[0]

    def run(self, X: np.ndarray):
        """
        前向传播并返回每一层的激活值（用于前端可视化）。
        自包含、不依赖共享 cache，多线程调用也安全。
        返回 (probs, [a0_flat, a1_flat, ... a4_flat])，
        其中 a0 是输入(784)，a1~a3 是隐藏层 ReLU 激活，a4 是输出 softmax。
        """
        X = np.asarray(X, dtype=np.float32)
        a = X
        acts = [a]
        for l in range(1, self.n_layers + 1):
            z = a @ self.W[l].T + self.b[l]
            a = softmax(z) if l == self.n_layers else relu(z)
            acts.append(a)
        probs = acts[-1]
        flat = [np.asarray(x).ravel().tolist() for x in acts]
        return probs, flat

    # ---------- 存取 ----------
    def save(self, path: str):
        # 各层权重/偏置形状不同，必须用 object 数组逐个赋值，
        # 否则 np.array(list) 会尝试拼成规则数组而失败。
        W = np.empty(self.n_layers, dtype=object)
        b = np.empty(self.n_layers, dtype=object)
        for i in range(self.n_layers):
            W[i] = self.W[i + 1]
            b[i] = self.b[i + 1]
        np.savez(path, sizes=np.array(self.sizes), W=W, b=b)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        sizes = tuple(int(s) for s in data["sizes"])
        model = cls(sizes=sizes)
        model.W = [None] + [data["W"][i] for i in range(len(sizes) - 1)]
        model.b = [None] + [data["b"][i] for i in range(len(sizes) - 1)]
        return model
