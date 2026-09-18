"""Обучаем сеть с нуля на задаче XOR.

XOR — это «заклятие» для линейных моделей (они её не могут):
    вход A, вход B -> ответ
    0,0 -> 0
    0,1 -> 1
    1,0 -> 1
    1,1 -> 0
Только сеть с хотя бы одним скрытым слоем её осилит.
"""
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNet

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

net = NeuralNet(n_input=2, n_hidden=4, n_output=1, seed=42)
print("Обучаю на XOR...")
losses = net.train(X, y, epochs=3000, lr=1.0)

print("\nПредсказания после обучения:")
for (a, b), expected in zip(X, y):
    out = net.forward(np.array([[a, b]]))[0][0]
    print(f"  XOR({a},{b}) -> {out:.3f}  (ожидали {expected[0]}, сеть говорит {'1' if out>0.5 else '0'})")

ok = sum(
    1 for (a, b), expected in zip(X, y)
    if net.predict(np.array([[a, b]]))[0][0] == expected[0]
)
print(f"\nТочно: {ok}/4")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Обучение сети на XOR (с нуля, без фреймворков)")
plt.xlabel("эпоха")
plt.ylabel("loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("xor_training.png", dpi=120)
print("\nГрафик обучения сохранён: xor_training.png")