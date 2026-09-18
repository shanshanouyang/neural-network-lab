import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sklearn.datasets import make_moons
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy


def train_with_regularization(X, y, reg_type, reg_lambda, epochs=2000, lr=0.1, seed=42):
    net = SimpleNetwork(input_dim=2, hidden_dim=8, seed=seed)
    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        y_hat = net.forward(X)
        data_loss = binary_cross_entropy(y, y_hat)
        reg_loss = net.regularization_loss(reg_type, reg_lambda)
        total_loss = data_loss + reg_loss

        grads = net.backward(y, reg_type=reg_type, reg_lambda=reg_lambda)
        net.update(*grads, lr=lr)

        acc = np.mean((y_hat.flatten() > 0.5) == y)
        history['loss'].append(total_loss)
        history['acc'].append(acc)

    return net, history


if __name__ == "__main__":
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

    configs = [
        ("No Regularization", None, 0.0),
        ("L1 Regularization", "l1", 0.001),
        ("L2 Regularization", "l2", 0.01),
    ]

    results = []
    for name, reg_type, reg_lambda in configs:
        net, history = train_with_regularization(X, y, reg_type, reg_lambda)
        final_acc = history['acc'][-1]
        final_loss = history['loss'][-1]
        sparsity = net.weight_sparsity()
        results.append((name, final_acc, final_loss, sparsity))

    print(f"{'Config':<20} {'Final Acc':<12} {'Final Loss':<12} {'Weight Sparsity':<15}")
    print("-" * 60)
    for name, acc, loss, sparsity in results:
        print(f"{name:<20} {acc:<12.4f} {loss:<12.4f} {sparsity:<15.4f}")