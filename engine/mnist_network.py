import numpy as np
from .activations import relu, relu_derivative


def softmax(z):
    """Numerically stable softmax over the last axis."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-8):
    """y_true_onehot and y_pred both have shape (n_samples, n_classes)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))


class MNISTNetwork:
    """
    Fully connected network: 784 -> 128 -> 64 -> 10
    ReLU activations on hidden layers, softmax on output, categorical cross-entropy loss.
    """

    def __init__(self, input_dim=784, hidden1=128, hidden2=64, output_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        # He initialization, appropriate for ReLU activations
        self.W1 = rng.standard_normal((input_dim, hidden1)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.standard_normal((hidden1, hidden2)) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.standard_normal((hidden2, output_dim)) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.y_hat = softmax(self.Z3)
        return self.y_hat

    def backward(self, y_true_onehot):
        n = y_true_onehot.shape[0]

        # For softmax + categorical cross-entropy, the combined gradient
        # simplifies to (y_hat - y_true), same structural pattern as
        # sigmoid + binary cross-entropy.
        dZ3 = (self.y_hat - y_true_onehot) / n
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2, dW3, db3

    def update(self, dW1, db1, dW2, db2, dW3, db3, lr=0.01):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)