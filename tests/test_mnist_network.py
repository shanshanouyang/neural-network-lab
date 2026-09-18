import numpy as np
import pytest
from engine.mnist_network import MNISTNetwork, softmax, categorical_cross_entropy


def make_toy_multiclass_data(n_samples=8, input_dim=10, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, input_dim))
    y_labels = rng.integers(0, n_classes, size=n_samples)
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_labels] = 1
    return X, y_onehot


def test_softmax_output_sums_to_one():
    """Each row of a softmax output should sum to 1."""
    z = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    probs = softmax(z)
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0)


def test_softmax_numerical_stability():
    """Softmax should not overflow even with very large input values."""
    z = np.array([[1000.0, 1001.0, 1002.0]])
    probs = softmax(z)
    assert not np.any(np.isnan(probs))
    assert np.allclose(np.sum(probs), 1.0)


def test_forward_output_shape():
    """Forward pass should return an output with shape (n_samples, n_classes)."""
    X, y = make_toy_multiclass_data(n_samples=8, input_dim=10, n_classes=4)
    net = MNISTNetwork(input_dim=10, hidden1=6, hidden2=5, output_dim=4, seed=1)
    y_hat = net.forward(X)
    assert y_hat.shape == (8, 4)


def test_forward_output_is_valid_probability_distribution():
    """Each row of the network's output should sum to 1 (since it is a softmax)."""
    X, y = make_toy_multiclass_data(n_samples=8, input_dim=10, n_classes=4)
    net = MNISTNetwork(input_dim=10, hidden1=6, hidden2=5, output_dim=4, seed=1)
    y_hat = net.forward(X)
    row_sums = np.sum(y_hat, axis=1)
    assert np.allclose(row_sums, 1.0)


def test_backward_output_shapes():
    """Each gradient should have the same shape as its corresponding parameter."""
    X, y = make_toy_multiclass_data(n_samples=8, input_dim=10, n_classes=4)
    net = MNISTNetwork(input_dim=10, hidden1=6, hidden2=5, output_dim=4, seed=1)
    net.forward(X)
    dW1, db1, dW2, db2, dW3, db3 = net.backward(y)

    assert dW1.shape == net.W1.shape
    assert db1.shape == net.b1.shape
    assert dW2.shape == net.W2.shape
    assert db2.shape == net.b2.shape
    assert dW3.shape == net.W3.shape
    assert db3.shape == net.b3.shape


def test_gradient_check_w3():
    """
    Numerical gradient check for W3 (output layer weights), the layer
    closest to the loss and simplest to verify.
    """
    X, y = make_toy_multiclass_data(n_samples=6, input_dim=8, n_classes=3)
    net = MNISTNetwork(input_dim=8, hidden1=5, hidden2=4, output_dim=3, seed=2)

    net.forward(X)
    dW1, db1, dW2, db2, dW3, db3 = net.backward(y)

    def loss_fn():
        y_hat = net.forward(X)
        return categorical_cross_entropy(y, y_hat)

    eps = 1e-6
    numerical_dW3 = np.zeros_like(net.W3)
    for i in range(net.W3.shape[0]):
        for j in range(net.W3.shape[1]):
            original = net.W3[i, j]
            net.W3[i, j] = original + eps
            loss_plus = loss_fn()
            net.W3[i, j] = original - eps
            loss_minus = loss_fn()
            net.W3[i, j] = original
            numerical_dW3[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_error = np.max(np.abs(dW3 - numerical_dW3))
    assert max_error < 1e-5, f"W3 gradient check failed with max error {max_error:.2e}"


def test_gradient_check_w1():
    """
    Numerical gradient check for W1 (first hidden layer weights), the
    layer furthest from the loss with the longest chain rule path,
    making it the most likely place for a backprop bug to surface.
    """
    X, y = make_toy_multiclass_data(n_samples=6, input_dim=8, n_classes=3)
    net = MNISTNetwork(input_dim=8, hidden1=5, hidden2=4, output_dim=3, seed=2)

    net.forward(X)
    dW1, db1, dW2, db2, dW3, db3 = net.backward(y)

    def loss_fn():
        y_hat = net.forward(X)
        return categorical_cross_entropy(y, y_hat)

    eps = 1e-6
    numerical_dW1 = np.zeros_like(net.W1)
    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            original = net.W1[i, j]
            net.W1[i, j] = original + eps
            loss_plus = loss_fn()
            net.W1[i, j] = original - eps
            loss_minus = loss_fn()
            net.W1[i, j] = original
            numerical_dW1[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_error = np.max(np.abs(dW1 - numerical_dW1))
    assert max_error < 1e-5, f"W1 gradient check failed with max error {max_error:.2e}"


def test_loss_decreases_after_training_steps():
    """A few mini-batch gradient descent steps should reduce the loss
    on a small toy multiclass dataset."""
    X, y = make_toy_multiclass_data(n_samples=50, input_dim=10, n_classes=4)
    net = MNISTNetwork(input_dim=10, hidden1=16, hidden2=8, output_dim=4, seed=3)

    y_hat = net.forward(X)
    loss_before = categorical_cross_entropy(y, y_hat)

    for _ in range(20):
        y_hat = net.forward(X)
        grads = net.backward(y)
        net.update(*grads, lr=0.1)

    y_hat_after = net.forward(X)
    loss_after = categorical_cross_entropy(y, y_hat_after)

    assert loss_after < loss_before


def test_predict_returns_valid_class_indices():
    """predict() should return integer class indices within the valid range."""
    X, y = make_toy_multiclass_data(n_samples=10, input_dim=10, n_classes=4)
    net = MNISTNetwork(input_dim=10, hidden1=6, hidden2=5, output_dim=4, seed=1)
    predictions = net.predict(X)

    assert predictions.shape == (10,)
    assert np.all(predictions >= 0) and np.all(predictions < 4)