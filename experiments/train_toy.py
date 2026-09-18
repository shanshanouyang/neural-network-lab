import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sklearn.datasets import make_moons
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

net = SimpleNetwork(input_dim=2, hidden_dim=4)

for epoch in range(1000):
    y_hat = net.forward(X)
    loss = binary_cross_entropy(y, y_hat)
    grads = net.backward(y)
    net.update(*grads, lr=0.1)

    if epoch % 100 == 0:
        acc = np.mean((y_hat.flatten() > 0.5) == y)
        print(f"epoch {epoch}, loss={loss:.4f}, acc={acc:.4f}")