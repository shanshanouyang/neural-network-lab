import numpy as np
from .activations import relu, relu_derivative, sigmoid
from .losses import binary_cross_entropy

class SimpleNetwork:
    def __init__(self, input_dim=2, hidden_dim=4, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, 1)) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.y_hat = sigmoid(self.Z2)
        return self.y_hat

    def backward(self, y_true):
        n = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)

        dZ2 = (self.y_hat - y_true) / n

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)

        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr=0.1):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2