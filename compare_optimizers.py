import numpy as np
from sklearn.datasets import make_moons
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy
from engine.optimizers import SGD, Momentum, Adam


def train_with_optimizer(X, y, optimizer, epochs=1000, seed=42, target_acc=0.85):
    net = SimpleNetwork(input_dim=2, hidden_dim=8, seed=seed)
    history = {'loss': [], 'acc': []}
    epochs_to_target = None

    for epoch in range(epochs):
        y_hat = net.forward(X)
        loss = binary_cross_entropy(y, y_hat)
        acc = np.mean((y_hat.flatten() > 0.5) == y)

        grads = net.backward(y)
        optimizer.update(net, *grads)

        history['loss'].append(loss)
        history['acc'].append(acc)

        if epochs_to_target is None and acc >= target_acc:
            epochs_to_target = epoch

    return net, history, epochs_to_target


if __name__ == "__main__":
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

    optimizers = [
        ("SGD", SGD(lr=0.1)),
        ("Momentum", Momentum(lr=0.1, beta=0.9)),
        ("Adam", Adam(lr=0.01)),
    ]

    results = []
    for name, opt in optimizers:
        net, history, epochs_to_target = train_with_optimizer(X, y, opt)
        final_acc = history['acc'][-1]
        final_loss = history['loss'][-1]
        results.append((name, final_acc, final_loss, epochs_to_target))

    print(f"{'Optimizer':<12} {'Final Acc':<12} {'Final Loss':<12} {'Epochs to 85% acc':<20}")
    print("-" * 60)
    for name, acc, loss, epochs_to_target in results:
        et_str = str(epochs_to_target) if epochs_to_target is not None else "not reached"
        print(f"{name:<12} {acc:<12.4f} {loss:<12.4f} {et_str:<20}")