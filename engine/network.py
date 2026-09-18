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

    def backward(self, y_true, reg_type=None, reg_lambda=0.0):
        """
        reg_type: None, 'l1', or 'l2'
        reg_lambda: regularization strength
        """
        n = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)
        dZ2 = (self.y_hat - y_true) / n
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        if reg_type == 'l2':
            dW1 = dW1 + reg_lambda * self.W1
            dW2 = dW2 + reg_lambda * self.W2
        elif reg_type == 'l1':
            dW1 = dW1 + reg_lambda * np.sign(self.W1)
            dW2 = dW2 + reg_lambda * np.sign(self.W2)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr=0.1):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def regularization_loss(self, reg_type=None, reg_lambda=0.0):
        if reg_type == 'l2':
            return (reg_lambda / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        elif reg_type == 'l1':
            return reg_lambda * (np.sum(np.abs(self.W1)) + np.sum(np.abs(self.W2)))
        return 0.0

    def weight_sparsity(self, threshold=1e-3):
        """Fraction of weights (in W1 and W2 combined) close to zero."""
        all_weights = np.concatenate([self.W1.flatten(), self.W2.flatten()])
        return np.mean(np.abs(all_weights) < threshold)