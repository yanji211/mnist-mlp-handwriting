"""
下载 MNIST 训练素材（4 个文件），带多镜像回退。

最终在 data/ 下生成 4 个解压后的 idx 文件：
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte

镜像优先级（网络不通会自动尝试下一个）：
  1. torchvision 官方 S3 镜像
  2. GitHub cvdfoundation/mnist
"""

import gzip
import os
import urllib.request

FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
}

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/cvdfoundation/mnist/master/",
]


def _download_one(url: str, dest: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-mlp"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def download(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    for gz_name, raw_name in FILES.items():
        raw_path = os.path.join(data_dir, raw_name)
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
            print(f"[skip] 已存在: {raw_name}")
            continue

        gz_path = os.path.join(data_dir, gz_name)
        ok = False
        for mirror in MIRRORS:
            url = mirror + gz_name
            try:
                print(f"[get ] {url}")
                _download_one(url, gz_path)
                ok = True
                break
            except Exception as e:   # 换下一个镜像
                print(f"       失败: {e}")
        if not ok:
            raise RuntimeError(f"所有镜像都无法下载 {gz_name}")

        # 解压
        with gzip.open(gz_path, "rb") as fin, open(raw_path, "wb") as fout:
            fout.write(fin.read())
        os.remove(gz_path)
        print(f"[ok ] 解压: {raw_name}")


if __name__ == "__main__":
    download()
