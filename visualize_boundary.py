import numpy as np
from sklearn.datasets import make_moons
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy
from engine.visualize import plot_decision_boundary

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
net = SimpleNetwork(input_dim=2, hidden_dim=8, seed=42)

for epoch in range(2000):
    y_hat = net.forward(X)
    loss = binary_cross_entropy(y, y_hat)
    grads = net.backward(y)
    net.update(*grads, lr=0.1)
    if epoch % 500 == 0:
        acc = np.mean((y_hat.flatten() > 0.5) == y)
        print(f"epoch {epoch}, loss={loss:.4f}, acc={acc:.4f}")

plot_decision_boundary(
    net, X, y,
    title="Trained Network Decision Boundary",
    save_path="decision_boundary.png"
)