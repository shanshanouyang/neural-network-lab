import numpy as np


class SGD:
    """Plain stochastic gradient descent."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, net, dW1, db1, dW2, db2):
        net.W1 -= self.lr * dW1
        net.b1 -= self.lr * db1
        net.W2 -= self.lr * dW2
        net.b2 -= self.lr * db2


class Momentum:
    """SGD with momentum: accumulates a velocity term for each parameter."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def update(self, net, dW1, db1, dW2, db2):
        if self.v is None:
            self.v = {
                'W1': np.zeros_like(net.W1),
                'b1': np.zeros_like(net.b1),
                'W2': np.zeros_like(net.W2),
                'b2': np.zeros_like(net.b2),
            }

        self.v['W1'] = self.beta * self.v['W1'] + (1 - self.beta) * dW1
        self.v['b1'] = self.beta * self.v['b1'] + (1 - self.beta) * db1
        self.v['W2'] = self.beta * self.v['W2'] + (1 - self.beta) * dW2
        self.v['b2'] = self.beta * self.v['b2'] + (1 - self.beta) * db2

        net.W1 -= self.lr * self.v['W1']
        net.b1 -= self.lr * self.v['b1']
        net.W2 -= self.lr * self.v['W2']
        net.b2 -= self.lr * self.v['b2']


class Adam:
    """Adam optimizer: maintains per-parameter first and second moment estimates."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, net, dW1, db1, dW2, db2):
        if self.m is None:
            self.m = {
                'W1': np.zeros_like(net.W1), 'b1': np.zeros_like(net.b1),
                'W2': np.zeros_like(net.W2), 'b2': np.zeros_like(net.b2),
            }
            self.v = {
                'W1': np.zeros_like(net.W1), 'b1': np.zeros_like(net.b1),
                'W2': np.zeros_like(net.W2), 'b2': np.zeros_like(net.b2),
            }

        self.t += 1
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        params = {'W1': net.W1, 'b1': net.b1, 'W2': net.W2, 'b2': net.b2}

        for key in grads:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)