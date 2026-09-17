import numpy as np
import pytest
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy
from engine.gradient_check import check_gradients, numerical_gradient, relative_error


def make_toy_data(n_samples=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2))
    y = rng.integers(0, 2, size=n_samples)
    return X, y


def test_forward_output_shape():
    """Forward pass should return an output with shape (n_samples, 1)."""
    X, y = make_toy_data(n_samples=5)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)
    y_hat = net.forward(X)
    assert y_hat.shape == (5, 1)


def test_forward_output_range():
    """Sigmoid output should always be strictly between 0 and 1."""
    X, y = make_toy_data(n_samples=10)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)
    y_hat = net.forward(X)
    assert np.all(y_hat > 0) and np.all(y_hat < 1)


def test_backward_output_shapes():
    """Each gradient should have the same shape as its corresponding parameter."""
    X, y = make_toy_data(n_samples=5)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)
    net.forward(X)
    dW1, db1, dW2, db2 = net.backward(y)

    assert dW1.shape == net.W1.shape
    assert db1.shape == net.b1.shape
    assert dW2.shape == net.W2.shape
    assert db2.shape == net.b2.shape


def test_gradient_check_all_parameters():
    """
    Full numerical gradient check: analytic gradients from backward()
    must closely match numerically estimated gradients.
    """
    X, y = make_toy_data(n_samples=5)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)

    passed = check_gradients(net, X, y, tolerance=1e-5, verbose=False)
    assert passed


def test_gradient_check_individual_tolerance():
    """
    Same check as above, but asserts on each parameter's max relative
    error individually so a failure points directly at which parameter
    is wrong, rather than just a single pass/fail flag.
    """
    X, y = make_toy_data(n_samples=5)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)

    net.forward(X)
    dW1, db1, dW2, db2 = net.backward(y)

    analytic_grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    for name, analytic in analytic_grads.items():
        numerical = numerical_gradient(net, X, y, name)
        max_error = np.max(relative_error(analytic, numerical))
        assert max_error < 1e-5, f"{name} gradient check failed with error {max_error:.2e}"


def test_loss_decreases_after_update():
    """A single gradient descent step with a small learning rate should
    strictly decrease the loss."""
    X, y = make_toy_data(n_samples=20)
    net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)

    y_hat = net.forward(X)
    loss_before = binary_cross_entropy(y, y_hat)

    grads = net.backward(y)
    net.update(*grads, lr=0.01)

    y_hat_after = net.forward(X)
    loss_after = binary_cross_entropy(y, y_hat_after)

    assert loss_after < loss_before


def test_binary_cross_entropy_shape_mismatch_regression():
    """
    Regression test for a real bug found during development: y_true with
    shape (n,) and y_pred with shape (n, 1) silently broadcast into an
    (n, n) array instead of raising an error, producing a wrong loss
    value without any exception.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.2], [0.8], [0.3], [0.7]])

    loss = binary_cross_entropy(y_true, y_pred)

    # Manually compute the correct loss using properly aligned shapes
    y_true_aligned = y_true.reshape(-1, 1)
    expected = -np.mean(
        y_true_aligned * np.log(y_pred) + (1 - y_true_aligned) * np.log(1 - y_pred)
    )

    assert np.isclose(loss, expected)