import numpy as np
from .losses import binary_cross_entropy


def numerical_gradient(net, X, y, param_name, eps=1e-6):
    """
    Compute the numerical gradient for a given parameter (e.g. 'W1', 'b1', 'W2', 'b2').
    Returns an array with the same shape as the parameter, containing the
    numerically estimated gradient at each position.
    """
    param = getattr(net, param_name)
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + eps
        y_hat_plus = net.forward(X)
        loss_plus = binary_cross_entropy(y, y_hat_plus)

        param[idx] = original_value - eps
        y_hat_minus = net.forward(X)
        loss_minus = binary_cross_entropy(y, y_hat_minus)

        param[idx] = original_value  # restore original value

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    return grad


def relative_error(analytic, numerical, eps=1e-8):
    """
    Compute the relative error between the analytic and numerical gradients.
    This is more reliable than a raw absolute difference, since it stays
    meaningful whether the gradient magnitude is large or small.
    """
    numerator = np.abs(analytic - numerical)
    denominator = np.maximum(np.abs(analytic), np.abs(numerical)) + eps
    return numerator / denominator


def check_gradients(net, X, y, tolerance=1e-5, verbose=True):
    """
    Run a full gradient check on all of the network's parameters
    (W1, b1, W2, b2). Returns True if every parameter passes.
    """
    net.forward(X)
    dW1, db1, dW2, db2 = net.backward(y)

    analytic_grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    all_passed = True

    for name, analytic in analytic_grads.items():
        numerical = numerical_gradient(net, X, y, name)
        errors = relative_error(analytic, numerical)
        max_error = np.max(errors)
        passed = max_error < tolerance

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}: max relative error = {max_error:.2e}")

        if not passed:
            all_passed = False

    return all_passed