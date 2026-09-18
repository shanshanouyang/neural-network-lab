import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from engine.mnist_network import MNISTNetwork, categorical_cross_entropy

print("Fetching MNIST dataset (this may take a minute on first run)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values from [0, 255] to [0, 1]
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

n_classes = 10
y_train_onehot = np.zeros((len(y_train), n_classes))
y_train_onehot[np.arange(len(y_train)), y_train] = 1

net = MNISTNetwork(input_dim=784, hidden1=128, hidden2=64, output_dim=10, seed=42)

batch_size = 64
epochs = 20
lr = 0.1
n_samples = X_train.shape[0]

print(f"Training on {n_samples} samples, testing on {len(X_test)} samples")
print(f"Architecture: 784 -> 128 -> 64 -> 10, lr={lr}, batch_size={batch_size}")
print()

for epoch in range(epochs):
    perm = np.random.permutation(n_samples)
    X_shuffled = X_train[perm]
    y_shuffled = y_train_onehot[perm]

    epoch_loss = 0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        y_hat = net.forward(X_batch)
        loss = categorical_cross_entropy(y_batch, y_hat)
        grads = net.backward(y_batch)
        net.update(*grads, lr=lr)

        epoch_loss += loss
        n_batches += 1

    train_preds = net.predict(X_train)
    train_acc = np.mean(train_preds == y_train)

    test_preds = net.predict(X_test)
    test_acc = np.mean(test_preds == y_test)

    print(f"epoch {epoch}, avg_loss={epoch_loss / n_batches:.4f}, "
          f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

print()
print(f"Final test accuracy: {test_acc:.4f}")