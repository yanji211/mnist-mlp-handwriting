"""
纯 NumPy 实现的卷积神经网络（CNN）识别 MNIST 手写数字。

与 model.py（3 隐藏层 MLP）保持「功能一致」：同样的输入接口
（一张图 = 784 个 0~1 灰度值）、同样的输出（10 类 softmax 概率），
以及 forward / predict / predict_proba / run / save / load / learn_example
这套完全相同的公开方法，方便前后端无缝切换、并排对比。

结构（保留空间结构的卷积网络）：
    输入(1×28×28)
    → Conv1(8 个 3×3, pad=1) + ReLU → MaxPool(2×2)   (8×14×14)
    → Conv2(16 个 3×3, pad=1) + ReLU → MaxPool(2×2)  (16×7×7)
    → Flatten(784) → FC(784→10) + softmax           (10)

卷积用 im2col 展开成矩阵乘法（标准高效写法），反向传播逐层回传。
前向时缓存各层中间量，backward/learn_example 复用这些缓存算梯度。

教学对应：
    z = conv(x, W) + b        （卷积 = 局部连接 + 权值共享）
    a = ReLU(z)
    p = maxpool(a)             （下采样，平移不变性）
    最后 FC + softmax 与 MLP 输出层一致
"""

import numpy as np


# ---------------- 基础工具 ----------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He 初始化，适配 ReLU。"""
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_out, fan_in)) * std


# ---------------- 卷积 / 池化的 im2col 实现 ----------------
def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """
    x: (N, C, H, W) -> (N*oh*ow, C*kh*kw)
    把每个卷积窗口拉成一行，卷积即可化为矩阵乘。
    """
    N, C, H, W = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    H, W = x.shape[2], x.shape[3]
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    cols = np.zeros((N, C, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[:, :, i:i + oh * stride:stride, j:j + ow * stride:stride]
    # 重排成 (N*oh*ow, C*kh*kw)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * oh * ow, C * kh * kw)
    return cols


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """
    im2col 的逆：把梯度列累加回 (N, C, H, W)。
    cols: (N*oh*ow, C*kh*kw)
    """
    N, C, H, W = x_shape
    if pad > 0:
        Hp, Wp = H + 2 * pad, W + 2 * pad
    else:
        Hp, Wp = H, W
    oh = (Hp - kh) // stride + 1
    ow = (Wp - kw) // stride + 1
    cols = cols.reshape(N, oh, ow, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    dx = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i + oh * stride:stride, j:j + ow * stride:stride] += cols[:, :, i, j, :, :]
    if pad > 0:
        dx = dx[:, :, pad:H + pad, pad:W + pad]
    return dx


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    2D 卷积。x: (N,C,H,W)，w: (F,C,kh,kw)，b: (F,)
    返回 (N, F, oh, ow)。
    """
    N, C, H, W = x.shape
    F = w.shape[0]
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    H, W = x.shape[2], x.shape[3]
    oh = (H - w.shape[2]) // stride + 1
    ow = (W - w.shape[3]) // stride + 1
    cols = im2col(x, w.shape[2], w.shape[3], stride, pad=0)  # (N*oh*ow, C*kh*kw)
    w_col = w.reshape(F, -1)                                 # (F, C*kh*kw)
    out = cols @ w_col.T + b                                 # (N*oh*ow, F)
    # im2col 行序是 (n, oy, ox)，F 在最后一维；先排成 (N,oh,ow,F) 再换轴到 (N,F,oh,ow)
    return out.reshape(N, oh, ow, F).transpose(0, 3, 1, 2)


def conv2d_backward(dout: np.ndarray, x: np.ndarray, w: np.ndarray,
                    stride: int = 1, pad: int = 0):
    """
    卷积反向。返回 (dW, db, dx)。
    dout: (N, F, oh, ow)
    x:    未 padding 的输入 (N, C, H, W)
    w:    (F, C, kh, kw)
    """
    N, C, H, W = x.shape
    F = w.shape[0]
    kh, kw = w.shape[2], w.shape[3]
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    else:
        x_pad = x
    Hp, Wp = x_pad.shape[2], x_pad.shape[3]
    oh = dout.shape[2]
    ow = dout.shape[3]
    cols = im2col(x_pad, kh, kw, stride, pad=0)          # (M, C*kh*kw), M=N*oh*ow
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(cols.shape[0], F)  # (M, F)

    N = dout.shape[0]                                    # 按样本数取均值（与 MLP 一致）
    dW = ((cols.T @ dout_flat).T / N).reshape(F, C, kh, kw)  # (F, C, kh, kw)
    db = dout_flat.sum(axis=0) / N                       # (F,)
    dcols = dout_flat @ w.reshape(F, -1)                # (M, C*kh*kw)
    dx = col2im(dcols, x.shape, kh, kw, stride, pad)    # (N, C, H, W)
    return dW, db, dx


def maxpool2d(x: np.ndarray, size: int = 2):
    """
    2×2 最大池化。返回 (out, mask)。
    mask: (N, C, H, W) 的 0/1，标记每个窗口最大值的位置。
    窗口按 (N, C, oh, ow, size, size) 布局，idx 取每个窗口内最大值的下标。
    """
    N, C, H, W = x.shape
    oh, ow = H // size, W // size
    # 把 2×2 窗口拉成最后一维：(N,C,oh,ow,size,size)
    xr = x.reshape(N, C, oh, size, ow, size).transpose(0, 1, 2, 4, 3, 5)
    xf = xr.reshape(N, C, oh, ow, size * size)
    idx = np.argmax(xf, axis=4)                       # (N, C, oh, ow)
    out = np.max(xf, axis=4)                          # (N, C, oh, ow)
    M = N * C * oh * ow
    mw = np.zeros((M, size * size), dtype=bool)
    mw[np.arange(M), idx.ravel()] = True              # 每个窗口仅最大值位置为 1
    mask = mw.reshape(N, C, oh, ow, size, size).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)
    return out, mask


def maxpool2d_backward(dout: np.ndarray, mask: np.ndarray, size: int = 2) -> np.ndarray:
    """
    最大池化反向：把梯度送回被选中（最大值）的位置。
    必须把 (N,C,H,W) 的 mask 还原成 (M, size*size) 的窗口布局（与 forward 互逆）。
    """
    N, C, oh, ow = dout.shape
    H, W = oh * size, ow * size
    M = N * C * oh * ow
    mw = mask.reshape(N, C, oh, size, ow, size).transpose(0, 1, 2, 4, 3, 5).reshape(M, size * size)
    dout_flat = dout.reshape(M)
    dwin = mw.astype(dout.dtype) * dout_flat[:, None]   # (M, size*size)
    dx = dwin.reshape(N, C, oh, ow, size, size).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)
    return dx


# ---------------- CNN 主类 ----------------
class CNN:
    """小型 CNN：2 个卷积块 + 1 个全连接输出层。"""

    def __init__(self, seed: int = 123):
        rng = np.random.default_rng(seed)
        # 卷积块 1：1 -> 8 通道
        self.conv1_w = he_init(1 * 3 * 3, 8, rng).reshape(8, 1, 3, 3).astype(np.float32)
        self.conv1_b = np.zeros(8, dtype=np.float32)
        # 卷积块 2：8 -> 16 通道
        self.conv2_w = he_init(8 * 3 * 3, 16, rng).reshape(16, 8, 3, 3).astype(np.float32)
        self.conv2_b = np.zeros(16, dtype=np.float32)
        # 全连接：16*7*7=784 -> 10
        self.fc_w = he_init(784, 10, rng).astype(np.float32)
        self.fc_b = np.zeros(10, dtype=np.float32)
        self.cache = {}

    # ---------- 前向传播 ----------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, 784) 灰度值，已归一化到 [0,1]。返回 (N, 10) softmax 概率。
        """
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        x = X.reshape(N, 1, 28, 28)

        z1 = conv2d(x, self.conv1_w, self.conv1_b, pad=1)   # (N,8,28,28)
        a1 = relu(z1)
        p1, m1 = maxpool2d(a1, 2)                           # (N,8,14,14)

        z2 = conv2d(p1, self.conv2_w, self.conv2_b, pad=1) # (N,16,14,14)
        a2 = relu(z2)
        p2, m2 = maxpool2d(a2, 2)                           # (N,16,7,7)

        flat = p2.reshape(N, -1)                            # (N,784)
        z3 = flat @ self.fc_w.T + self.fc_b                # (N,10)
        probs = softmax(z3)

        self.cache = {
            "x": x, "z1": z1, "a1": a1, "p1": p1, "m1": m1,
            "z2": z2, "a2": a2, "p2": p2, "m2": m2, "flat": flat,
        }
        return probs

    # ---------- 反向传播 ----------
    def backward(self, y_true: np.ndarray):
        """
        y_true: 整型标签 (N,) 或 one-hot (N,10)。
        返回 grads 字典（均值梯度，已除以 N）。
        """
        c = self.cache
        N = c["x"].shape[0]
        if y_true.ndim == 1:
            y = np.zeros((N, 10), dtype=np.float32)
            y[np.arange(N), y_true.astype(int)] = 1.0
        else:
            y = y_true.astype(np.float32)

        # 输出层：δ = ŷ - y（softmax + 交叉熵的优雅结论，这里用 cache 重算 ŷ）
        probs = softmax(c["flat"] @ self.fc_w.T + self.fc_b)
        dz3 = probs - y                                       # (N,10)

        dW_fc = dz3.T @ c["flat"] / N                         # (10,784)
        db_fc = np.sum(dz3, axis=0) / N
        dflat = dz3 @ self.fc_w                               # (N,784)
        dp2 = dflat.reshape(c["p2"].shape)                    # (N,16,7,7)

        da2 = maxpool2d_backward(dp2, c["m2"], 2)             # (N,16,14,14)
        dz2 = da2 * relu_grad(c["z2"])
        dW2, db2, dp1 = conv2d_backward(dz2, c["p1"], self.conv2_w, pad=1)

        da1 = maxpool2d_backward(dp1, c["m1"], 2)             # (N,8,28,28)
        dz1 = da1 * relu_grad(c["z1"])
        dW1, db1, _ = conv2d_backward(dz1, c["x"], self.conv1_w, pad=1)

        grads = {
            "conv1_w": dW1, "conv1_b": db1,
            "conv2_w": dW2, "conv2_b": db2,
            "fc_w": dW_fc, "fc_b": db_fc,
        }
        return grads

    # ---------- 预测 ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    # ---------- 增量学习（纠错，与 MLP.learn_example 同接口）----------
    def learn_example(self, x, y_true, lr: float = 0.01, steps: int = 4):
        """
        对单个样本做几步 SGD，就地更新权重。返回该样本更新后的 softmax 概率 (10,)。
        """
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        y_true = int(y_true)
        for _ in range(steps):
            probs = self.forward(x)
            grads = self.backward(np.array([y_true], dtype=int))
            self.conv1_w -= lr * grads["conv1_w"]
            self.conv1_b -= lr * grads["conv1_b"]
            self.conv2_w -= lr * grads["conv2_w"]
            self.conv2_b -= lr * grads["conv2_b"]
            self.fc_w -= lr * grads["fc_w"]
            self.fc_b -= lr * grads["fc_b"]
        return self.forward(x)[0]

    # ---------- 自包含前向（供前端可视化，线程安全）----------
    def run(self, X: np.ndarray):
        """
        返回 (probs, vis)，vis 含卷积特征图（展平的一维列表），供前端渲染。
        conv1: 8 张 28×28；conv2: 16 张 14×14。
        """
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        x = X.reshape(N, 1, 28, 28)
        z1 = conv2d(x, self.conv1_w, self.conv1_b, pad=1)
        a1 = relu(z1)
        p1, _ = maxpool2d(a1, 2)
        z2 = conv2d(p1, self.conv2_w, self.conv2_b, pad=1)
        a2 = relu(z2)
        p2, _ = maxpool2d(a2, 2)
        flat = p2.reshape(N, -1)
        probs = softmax(flat @ self.fc_w.T + self.fc_b)

        conv1 = [np.asarray(a1[0, f]).ravel().tolist() for f in range(a1.shape[1])]
        conv2 = [np.asarray(a2[0, f]).ravel().tolist() for f in range(a2.shape[1])]
        vis = {
            "conv1": conv1,        # 8 × (28*28)
            "conv1_shape": [28, 28],
            "conv2": conv2,        # 16 × (14*14)
            "conv2_shape": [14, 14],
            "arch": "CNN: Conv(8)→Pool→Conv(16)→Pool→FC(10)",
        }
        return probs, vis

    # ---------- 存取 ----------
    def save(self, path: str):
        np.savez(path,
                 conv1_w=self.conv1_w, conv1_b=self.conv1_b,
                 conv2_w=self.conv2_w, conv2_b=self.conv2_b,
                 fc_w=self.fc_w, fc_b=self.fc_b)

    @classmethod
    def load(cls, path: str):
        d = np.load(path)
        m = cls.__new__(cls)
        m.conv1_w = d["conv1_w"].astype(np.float32)
        m.conv1_b = d["conv1_b"].astype(np.float32)
        m.conv2_w = d["conv2_w"].astype(np.float32)
        m.conv2_b = d["conv2_b"].astype(np.float32)
        m.fc_w = d["fc_w"].astype(np.float32)
        m.fc_b = d["fc_b"].astype(np.float32)
        m.cache = {}
        return m


# ---------------- 数值梯度校验（内部自检）----------------
def _grad_check():
    """对 conv2d / maxpool 反向做数值校验。"""
    rng = np.random.default_rng(0)
    N, C, F, H, W, kh, kw = 2, 1, 3, 6, 6, 3, 3
    x = rng.standard_normal((N, C, H, W)).astype(np.float32)
    w = rng.standard_normal((F, C, kh, kw)).astype(np.float32)
    b = np.zeros(F, dtype=np.float32)

    def loss_via_conv(xv, wv, bv):
        out = conv2d(xv, wv, bv, pad=1)
        return float(np.sum(out ** 2))

    # 解析梯度
    out = conv2d(x, w, b, pad=1)
    dout = 2 * out
    dW, db, dx = conv2d_backward(dout, x, w, pad=1)

    eps = 1e-2
    max_err = 0.0
    # 权重梯度
    wf = w.reshape(-1).copy()
    gwf = dW.reshape(-1)
    for i in range(wf.shape[0]):
        orig = wf[i]
        wf[i] = orig + eps; lp = loss_via_conv(x, wf.reshape(F, C, kh, kw), b)
        wf[i] = orig - eps; lm = loss_via_conv(x, wf.reshape(F, C, kh, kw), b)
        wf[i] = orig
        num = (lp - lm) / (2 * eps)
        rel = abs(num - gwf[i]) / (abs(num) + abs(gwf[i]) + 1e-8)
        max_err = max(max_err, rel)
    # 输入梯度
    xf = x.reshape(-1).copy()
    gxf = dx.reshape(-1)
    for i in range(xf.shape[0]):
        orig = xf[i]
        xf[i] = orig + eps
        lp = loss_via_conv(xf.reshape(N, C, H, W), w, b)
        xf[i] = orig - eps
        lm = loss_via_conv(xf.reshape(N, C, H, W), w, b)
        xf[i] = orig
        num = (lp - lm) / (2 * eps)
        rel = abs(num - gxf[i]) / (abs(num) + abs(gxf[i]) + 1e-8)
        max_err = max(max_err, rel)
    print(f"[cnn grad-check] max relative error = {max_err:.2e}")
    assert max_err < 1e-2, "CNN 卷积梯度校验未通过！"
    print("CNN 卷积梯度校验通过 ✓")
    return max_err


if __name__ == "__main__":
    _grad_check()
