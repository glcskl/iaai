"""Нейросеть с нуля. Только numpy, никаких фреймворков.

Что тут происходит:
- сеть: вход -> скрытый слой (ReLU) -> выход (sigmoid)
- forward propagation: считаем предсказания
- loss: бинарная перекрёстная энтропия
- backward propagation: градиенты по правилу цепочки
- update: градиентный спуск
"""
import numpy as np


class NeuralNet:
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        rng = np.random.default_rng(seed)
        # веса инициализируем маленькими случайными числами
        self.W1 = rng.standard_normal((n_input, n_hidden)) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_output)) * 0.5
        self.b2 = np.zeros((1, n_output))

    def _relu(self, x):
        return np.maximum(x, 0)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # защита от переполнения
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """Прямой проход: X -> предсказания."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def predict(self, X):
        """Примо: (X > 0.5).astype(int), чтобы легко сравнивать с ответами."""
        return self.forward(X) > 0.5

    def loss(self, a2, y):
        """Бинарная перекрёстная энтропия. Добавляем эпсилон, чтобы log(0) не взорвался."""
        eps = 1e-12
        a2 = np.clip(a2, eps, 1 - eps)
        return -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))

    def backward(self, X, y):
        """Обратное распространение ошибки."""
        m = X.shape[0]
        dz2 = self.a2 - y                     # производная loss по z2 (с учётом сигмоиды)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)   # производная ReLU
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=5000, lr=0.5, verbose=True):
        """Полный цикл обучения."""
        self.loss_history = []
        for epoch in range(epochs):
            a2 = self.forward(X)
            loss = self.loss(a2, y)
            self.loss_history.append(loss)
            dW1, db1, dW2, db2 = self.backward(X, y)
            # шаг градиентного спуска
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:5d}  loss = {loss:.4f}")
        return self.loss_history