"""Классификатор рукописных букв (A-Z) с нуля, без фреймворков.

Архитектура: вход 784 (28x28) -> скрытый слой 128 (ReLU) -> softmax 26.
Обучение: минибатч-градиентный спуск + перекрёстная энтропия.
"""
import gzip
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_CLASSES = 26
HIDDEN = 200  # нейронов скрытого слоя
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "letters.npz")


def read_idx(gz_path):
    """Читает gzipped IDX-файл (формат MNIST/EMNIST).
    У картинок (idx3) заголовок 16 байт, у меток (idx1) — 8."""
    with gzip.open(gz_path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    header = 16 if magic == 0x803 else 8
    return np.frombuffer(raw, dtype=np.uint8, offset=header)


def load_split(name):
    """Загружает train или test часть EMNIST-letters в вертикальной ориентации."""
    images = read_idx(os.path.join(DATA, f"letters-{name}-images.gz"))
    labels = read_idx(os.path.join(DATA, f"letters-{name}-labels.gz"))
    n = labels.size
    images = images.reshape(n, 28, 28).astype(np.float32) / 255.0
    images = np.transpose(images, (0, 2, 1))[:, ::-1, :]
    images = images.reshape(n, 784)
    labels = (labels - 1).astype(int)  # метки 1..26 -> 0..25
    return images, labels


def load_letters():
    Xtr, ytr = load_split("train")
    try:
        Xte, yte = load_split("test")
    except FileNotFoundError:
        Xte, yte = Xtr, ytr
    return Xtr, ytr, Xte, yte


def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def train(epochs=25, batch=128, lr=0.3, seed=42):
    X, y, Xte, yte = load_letters()
    rng = np.random.default_rng(seed)
    m, d = X.shape
    # инициализация весов
    W1 = rng.standard_normal((d, HIDDEN)) * np.sqrt(2.0 / d)
    b1 = np.zeros((1, HIDDEN))
    W2 = rng.standard_normal((HIDDEN, N_CLASSES)) * np.sqrt(2.0 / HIDDEN)
    b2 = np.zeros((1, N_CLASSES))

    idx = np.arange(m)
    for epoch in range(epochs):
        rng.shuffle(idx)
        total_loss = 0.0
        steps = 0
        for start in range(0, m, batch):
            batch_idx = idx[start:start + batch]
            xb = X[batch_idx]
            yb = np.eye(N_CLASSES)[y[batch_idx]]

            # forward
            z1 = xb @ W1 + b1
            a1 = np.maximum(z1, 0)                      # ReLU
            z2 = a1 @ W2 + b2
            p = softmax(z2)                             # вероятности классов

            # loss (mean cross-entropy)
            total_loss += -np.mean(np.sum(yb * np.log(p + 1e-12), axis=1))
            steps += 1

            # backward
            dz2 = (p - yb) / batch
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            dz1 = (dz2 @ W2.T) * (z1 > 0)
            dW1 = xb.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        acc = accuracy(W1, b1, W2, b2, X, y)
        te_acc = accuracy(W1, b1, W2, b2, Xte, yte)
        print(f"  epoch {epoch+1:3d}/{epochs}  loss={total_loss/steps:.3f}  "
              f"train_acc={acc:.4f}  test_acc={te_acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez_compressed(MODEL_PATH, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Сохранил веса: {MODEL_PATH}")
    return W1, b1, W2, b2


def predict(images):
    """images: (N, 784). Возвращает (индексы классов, вероятности)."""
    W1, b1, W2, b2 = load_weights()
    z1 = images @ W1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2
    p = softmax(z2)
    return p.argmax(axis=1), p


def load_weights():
    if not os.path.exists(MODEL_PATH):
        print("Веса не найдены — обучаю...")
        train()
    w = np.load(MODEL_PATH)
    return w["W1"], w["b1"], w["W2"], w["b2"]


def accuracy(W1, b1, W2, b2, X, y):
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0)
    p = softmax(a1 @ W2 + b2)
    return np.mean(p.argmax(axis=1) == y)


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


if __name__ == "__main__":
    print("Обучаю распознавание букв A-Z на EMNIST (124 800 train / 20 800 test)...")
    train()